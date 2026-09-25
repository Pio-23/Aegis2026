# LLM Pilot class for rover. Provides high-level autopilot functions and holds

# AEGIS Senior Design, Created 10/22/2025

from email.mime import message
from fileinput import filename
import os
from urllib import response

from matplotlib import lines
import openai

import json
import time
import base64

from lidar import scan


class Autopilot:

    system_context = """
    You are the autonomous navigation controller for the AEGIS rover.

    You are not a conversational assistant. Your job is to autonomously observe
    the environment, choose ONE safe action that makes useful exploration progress, execute that action
    using the available tools, and briefly report what you are doing.

    AVAILABLE SENSORS

    1. LiDAR
        - Use LiDAR as the primary source for obstacle distance and geometry.
        - LiDAR tells you whether paths are physically blocked or clear.
        - Never intentionally move toward a direction that LiDAR clearly identifies
     as unsafe.

    2. Camera
        - Use the camera for visual and semantic understanding.
        - Use it to identify things LiDAR cannot explain well, including:
          doors, hallways, walls, furniture, terrain, openings, paths, and objects.
        - Combine camera information with LiDAR instead of treating them separately.

    AUTONOMOUS DECISION PROCESS

    At startup:

    1. Perform a LiDAR scan.
    2. Examine the LiDAR summary.
    3. Capture a camera image when visual context would improve the decision.
    4. Combine LiDAR geometry and camera observations.
    5. Choose ONE safest reasonable action.
    6. Execute that action.
    7. After movement, reassess the environment before making another major move.

    Do not ask the operator what action to take.

    Do not say:
        - "Would you like me to..."
        - "What would you like me to do?"
        - "Should I scan again?"
        - "Should I move?"
        - "If you want, I can..."

    Instead, decide autonomously.

    GOOD:
        "LiDAR shows the front is blocked. I am capturing a camera image to
        understand the opening on the left."

    Then call capture_camera.

    GOOD:
        "The rear path is clear and visually unobstructed. I am reversing slowly."

    Then call move_rover.

    BAD:
        "Would you like me to reverse or scan again?"

    SENSOR FUSION RULES

    - LiDAR determines physical clearance.
    - Camera provides visual meaning and context.
    - When both sensors agree that a direction is safe, prefer that direction.
    - If LiDAR is ambiguous, use the camera before moving.
    - If the camera is ambiguous, rely on LiDAR for collision safety.
    - Never override a clearly unsafe LiDAR reading only because the camera
        appears clear.
    - If sensor information conflicts, gather additional information instead
        of guessing.

    LIDAR CLEARANCE STATES

    - "blocked" means the near obstacle distance is 0.40 m or less.
    Do not intentionally move toward a blocked direction.

    - "caution" means the near obstacle distance is greater than 0.40 m
    but no more than 0.75 m.
    A caution direction may be usable, but movement should be slow and
    should agree with the camera and ultrasonic sensors.

    - "clear" means the near obstacle distance is greater than 0.75 m.

    - Prefer clear directions over caution directions when reasonable.
    - A caution direction is not automatically blocked.

    SENSOR STATE INTERPRETATION

    - lidar.scanning only indicates whether a LiDAR scan is currently
    being captured.
    - lidar.scanning = false does NOT mean LiDAR is unavailable.
    - A completed LiDAR summary remains valid until new sensor information
    indicates that the environment has changed or a new scan is required.

    - If a camera image has successfully been provided in the current
    observation cycle, treat that image as valid camera information.
    - Do not claim that the camera is disconnected when a current image
    was successfully captured and provided.

    - Always prefer the most recent completed LiDAR summary, camera image,
    and fresh ultrasonic telemetry over temporary sensor activity flags.

    MOVEMENT RULES

    - Issue only ONE physical movement command at a time.
    - Prefer slow and conservative movement near obstacles.
    - After moving, reassess before committing to another significant movement.
    - Do not repeatedly issue movement commands without updated sensor information.
    - Never move toward a blocked direction.
    - If there is no safe movement, stay stationary.

    ROVER MOTION MODEL

    - The rover uses skid-steer differential drive.
    - It cannot move sideways or strafe.
    - To travel toward the left or right, first use TURN to rotate the
    rover toward that direction, then use MOVE.
    - TURN means spin/rotate the rover in place.
    - Turning requires more motor torque than straight movement.
    - Do not request extremely low turn speeds.

    TURN EXECUTION BEHAVIOR

    - TURN rotates the rover in place; it does not move sideways.
    - LEFT and RIGHT turns are stationary skid-steer rotations.
    - A normal TURN segment currently lasts about 4.5 seconds at full turn power.
    - Based on physical testing, 1.5 seconds produced roughly a 10-degree turn,
    so a 4.5-second turn is expected to produce roughly a 30-degree heading change.
    - This angle is approximate and can vary with traction and battery level.
    - If a larger heading change is needed, issue another TURN after reassessing.
    - After facing an open direction, use MOVE to travel forward.

    EXPLORATION OBJECTIVE

    Your primary navigation objective is to explore the environment safely.

    - Prefer actions that move the rover into new, previously unexplored space.
    - When the path ahead is clear, generally prefer continuing forward rather
    than reversing or repeatedly changing direction.
    - Do not immediately undo the previous movement unless new sensor information
    indicates that continuing is unsafe or unproductive.
    - Avoid oscillating between forward and reverse movements.
    - Avoid repeatedly turning left and right without making forward progress.
    - Use reverse primarily to escape an obstacle, dead end, or unsafe position,
    not as a routine exploration movement.
    - When encountering an obstacle, turn toward a safer open direction and then
    continue forward into that new area.
    - Prefer making steady progress through the environment instead of remaining
    near the same location.
    - Remember recent movement commands and avoid returning immediately to the
    position you just came from unless necessary for safety.
    - If the front is clear and there is no navigation reason to turn or reverse,
    continue exploring forward.

    RE-SCANNING RULES

    Do not continuously repeat LiDAR scans without a reason.

    If one scan reports no safe path:
    1. Capture a camera image.
    2. Analyze LiDAR and camera together.
    3. If still unsafe, perform at most one additional LiDAR scan to verify.
    4. If there is still no safe path, use no_op and remain stationary.

    Do not enter an endless scan loop.

    COMMUNICATION STYLE

    Briefly state:
    - what you detected,
    - what you decided,
    - what you are doing.

    Then use the appropriate tool.

    Do not present choices to the operator.
    Do not wait for operator confirmation.
    """
    context_msg: dict[str, str] = {"role": "system", "content": system_context}

    # Define available tools for the rover autopilot
    aegis_tools = [
        {
            "type": "function",
            "function": {
                "name": "scan_environment",
                "description": (
                    "Captures a high-res LiDAR scan of the rover's surroundings."
                    "Use this to gather detailed environmental data whenever"
                    "telemetry suggests anything worth scanning, or if it has been"
                    "a while since the last scan. Make sure to make it as cheap as possible"
                    "by making the scan only 200 rings"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                },
            }
        },
        {
             "type": "function",
              "function": {
                 "name": "capture_camera",
                 "description": (
                     "Capture a still image from the rover camera when visual "
                     "information would help understand the environment."
                  ),
                   "parameters": {
                       "type": "object",
                       "properties": {},
                       "required": []
                   }
             }
        },
        {
            "type": "function",
            "function": {
                "name": "move_rover",
                "description": (
                    "Issues movement commands to the rover."
                    "Use this to move or turn the rover."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "op": {
                            "type": "string",
                            "enum": ["TURN", "MOVE"],
                            "description": (
                                "Type of movement."
                                "Either TURN (rotational) or MOVE (linear)."
                            )
                        },
                        "spd": {
                             "type": "number",
                              "minimum": -1,
                              "maximum": 1,
                               "description": (
                                    "Speed factor in range [-1, 1]. "
                                    "Required for MOVE and TURN. "
                                    "Positive only for TURN."
                                 )
},
                        "turn_dir": {
                            "type": "string",
                            "enum": ["LEFT", "RIGHT"],
                            "description": (
                                "Turn direction. Only used when op is TURN."
                            )
                        }
                    },
                    "required": ["op", "spd"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "no_op",
                "description": (
                    "Signal that no safe/clear action can be taken now."
                ),
                "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type":"string", "description":"Why action is deferred."},
                    "confidence": {
                    "type":"number", "minimum":0, "maximum":1,
                    "description":"Model confidence that doing nothing is correct."
                    },
                    "needs": {
                    "type":"array",
                    "items":{"type":"string"},
                    "description":"Specific data/information needed to proceed."
                    }
                },
                "required": ["reason"]
                }
            }
        }
    ]

    def __init__(self) -> None:

        self.memory_depth = 21  # Number of past interactions to remember
        self.memory = [self.context_msg]
        self.model_name = "gpt-5-nano"  # LLM model to use

        # Connect API account to client
        self.client = openai.OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
        )

    def encode_image(self, filename):
        with open(filename, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def summarize_telemetry(self, telemetry):
        """
        Return only the ultrasonic sensor data needed for navigation.
        """

        if not isinstance(telemetry, dict):
            return {
                "ultrasonic": {
                    "front_left_cm": None,
                    "front_center_cm": None,
                    "front_right_cm": None,
                    "rear_cm": None,
                    "lidar_mount_cm": None
                },
                "telemetry_valid": False
            }

        ultrasonics = telemetry.get("ultrasonics", {})

        return {
            "ultrasonic": {
                "front_left_cm": ultrasonics.get("left_cm"),
                "front_center_cm": ultrasonics.get("center_cm"),
                "front_right_cm": ultrasonics.get("right_cm"),
                "rear_cm": ultrasonics.get("rear_cm"),
                "lidar_mount_cm": ultrasonics.get("lidar_cm")
            },
            "telemetry_valid": True
        }
    

    def decide_actions(self, telemetry, scanner, ugv_cam, dump_folder, get_current_telemetry):
        """
        Ask GPT what to do.
        If GPT requests a LiDAR scan, perform the scan,
        summarize it, send the result back to GPT,
        then ask GPT again.
        """

        sensor_summary = self.summarize_telemetry(telemetry)
        
        self .update_memory({
            "role": "user",
            "content": json.dumps({
                "telemetry": telemetry,
                "sensor_summary": sensor_summary,
            })
        })

        MAX_TOOL_STEPS =5

        for step in range(MAX_TOOL_STEPS):

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.memory,
                tools=Autopilot.aegis_tools,
                tool_choice="required",
                parallel_tool_calls=False
            )

            message = response.choices[0].message

            self.update_memory(message)

            if message.content:
                print(message.content)

            if not message.tool_calls:
                return []

            for call in message.tool_calls:

                name = call.function.name

                print(
                    f"[AI] GPT requested tool: {name} "
                    f"(id: {call.id})"
                )

            # ==========================================
            # LIDAR
            # ==========================================

                if name == "scan_environment":
                    print("Starting LiDAR scan...")

                    lidar_filename = scanner.scan(filepath=dump_folder)
                    print(f"Scan saved: {lidar_filename}")

                    summary = self.summarize_scan_file(lidar_filename)

                    print("[DEBUG] LiDAR summary:")
                    print(json.dumps(summary, indent=2))

                    self.memory.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps({
                            "status": "scan_saved",
                            "file_path": lidar_filename,
                            "summary": summary
                        })
                    })

                    print(
                        f"[AI] GPT requested LiDAR scan. Scan complete and summarized."
                        f"for {call.id}" 
                    )

                # ==========================================
                # AUTOMATIC CAMERA CAPTURE AFTER LIDAR
                # ==========================================

                    if ugv_cam is not None and ugv_cam.connected:
                        print("[AI] Capturing camera image after LiDAR scan.")

                        camera_filename = ugv_cam.capture_image(
                            filepath=dump_folder
                        )

                        if camera_filename is not None:
                            print(f"[AI] Camera image saved: {camera_filename}")

                            image_base64 = self.encode_image(camera_filename)

                            self.memory.append({
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": (
                                            "This is the latest rover camera image. "
                                            "Analyze it together with the latest LiDAR scan. "
                                            "information. Identify obstacles, openings, doors, hallways,"
                                            "terrain, and anything useful for navigation. "
                                            "Use what you see to decide the safest next action."
                                        )
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": "data:image/jpeg;base64," + image_base64,
                                            "detail": "low"
                                        }
                                    }
                                ]
                            })

                            print(
                                f"[AI] GPT requested camera capture after LiDAR scan. "
                                f"Image saved and sent to GPT for {call.id}"
                            )

                        else: 
                            print("[AI] Camera capture failed after LiDAR scan.")

                    else: 
                        print("[AI] Camera not connected, skipping capture after LiDAR scan.")


                    fresh_telemetry = get_current_telemetry()

                    if isinstance(fresh_telemetry, dict):
                        fresh_sensor_summary = self.summarize_telemetry(fresh_telemetry)

                        self.update_memory({
                            "role": "user",
                            "content": json.dumps({
                                "telemetry": fresh_telemetry,
                                "sensor_summary": fresh_sensor_summary,
                                "observation": ("This is the latest telemetry after the LiDAR scan and camera scan"
                                "use this ultrasonic readings to decide the next movement."
                                )
                            })
                        })

                        print("[AI] Updated memory with fresh telemetry after LiDAR scan and camera capture.")

                    else:
                        print("[AI] Failed to retrieve fresh telemetry after LiDAR scan and camera capture.")

                    continue

                # ==========================================
                # CAMERA
                # ==========================================
                
                elif name == "capture_camera":

                    print("[AI] GPT requested camera capture.")

                    if ugv_cam is None or not ugv_cam.connected:

                        tool_result = {
                            "status": "failed",
                            "reason": "Camera is not connected"
                        }

                        self.memory.append({
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": json.dumps(tool_result)
                        })

                        continue
                
                    filename = ugv_cam.capture_image(
                        filepath=dump_folder
                    )

                    if filename is None:

                        self.memory.append({
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": json.dumps({
                                "status": "failed",
                                "reason": "Camera capture failed"
                            })
                        })

                        continue

                    print(
                        f"[AI] Camera image saved: {filename}"
                    )

                    self.memory.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps({
                            "status": "image_saved",
                            "file_path": filename
                        })
                    })

                    image_base64 = self.encode_image(filename)

                    self.memory.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "This is the latest rover camera image. "
                                    "Analyze it together with the latest LiDAR scan. "
                                    "information. Identify obstacles, openings, doors, hallways,"
                                    "terrain, and anything useful for navigation. "
                                    "Use what you see to decide the safest next action."
                                )
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "data:image/jpeg;base64," + image_base64,
                                    "detail": "low"
                                }
                            }
                        ]
                    })

                    print(
                        f"[AI] GPT requested camera capture. Image saved and sent to GPT."
                        f"for {call.id}"
                    )

                    continue

                    # ==========================================
                    # MOVE
                    # ==========================================
                elif name == "move_rover":

                    print(
                        "[AI] GPT requested rover movement"
                        "sending movement to UART"
                    )
                    
                    return [call]

                    # ==========================================
                    # NO-OP
                    # ==========================================
                elif name == "no_op":

                    print(
                        "[AI] GPT requested no-op. "
                    )

                    return [call]

                    # ==========================================
                    # UNKNOWN TOOL
                    # ==========================================    
                else:
                    print(
                        f"[ERR] GPT requested unknown tool: {name}. "
                    )

                    self.memory.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps({
                            "status": "failed",
                            "reason": f"Unknown tool: {name}"
                        })
                    })

        print(
            "[AI] GPT did not request any actionable tools. "
            "Remaining stationary and waiting for new telemetry."
        )

        return []

    def summarize_scan_file(self, filename):
        import math
        import numpy as np

        zones = {
            "front": [],
            "front_right": [],
            "right": [],
            "back_right": [],
            "back": [],
            "back_left": [],
            "left": [],
            "front_left": []
        }

        total_points = 0

        # --------------------------------------------------
        # TEMPORARY ROVER SELF-MASK
        # Measurements are from the center of the LiDAR.
        #
        # +X = front
        # -X = back
        # +Y = left
        # -Y = right
        # --------------------------------------------------

        FRONT_LIMIT = 0.1524   # 0.5 ft
        REAR_LIMIT = 0.6096    # 2.0 ft
        LEFT_LIMIT = 0.4572    # 1.5 ft
        RIGHT_LIMIT = 0.4572   # 1.5 ft

        # Floor appears about 0.30 m below the LiDAR.
        # Ignore points at or below this height.
        FLOOR_CUTOFF_Z = -0.27

        # Clearance thresholds
        BLOCKED_DISTANCE = 0.40
        CLEAR_DISTANCE = 0.75

        with open(filename, "r") as f:
            lines = f.readlines()

        for line in lines:

            values = line.strip().split()

            if len(values) < 4:
                continue

            try:
                x = float(values[0])
                y = float(values[1])
                z = float(values[2])
                intensity = float(values[3])

            except ValueError:
                continue

            # --------------------------------------------------
            # Ignore points belonging to the rover itself
            # --------------------------------------------------

            if (
                -REAR_LIMIT <= x <= FRONT_LIMIT
                and
                -RIGHT_LIMIT <= y <= LEFT_LIMIT
            ):
                continue

            # Ignore floor-level LiDAR returns
            if z <= FLOOR_CUTOFF_Z:
                continue

            distance = math.sqrt(
                x**2 +
                y**2 +
                z**2
            )

            if distance <= 0:
                continue

            total_points += 1

            # --------------------------------------------------
            # Determine direction around rover
            # --------------------------------------------------

            angle = math.degrees(
                math.atan2(y, x)
            )

            if angle < 0:
                angle += 360

            if angle >= 337.5 or angle < 22.5:
                zones["front"].append(distance)

            elif 22.5 <= angle < 67.5:
                zones["front_left"].append(distance)

            elif 67.5 <= angle < 112.5:
                zones["left"].append(distance)

            elif 112.5 <= angle < 157.5:
                zones["back_left"].append(distance)

            elif 157.5 <= angle < 202.5:
                zones["back"].append(distance)

            elif 202.5 <= angle < 247.5:
                zones["back_right"].append(distance)

            elif 247.5 <= angle < 292.5:
                zones["right"].append(distance)

            elif 292.5 <= angle < 337.5:
                zones["front_right"].append(distance)

        if total_points == 0:
            return {
                "status": "empty_scan",
                "scan_ok": False
            }

        zone_summary = {}

        # --------------------------------------------------
        # Summarize each direction
        # --------------------------------------------------

        for zone_name, distances in zones.items():

            if distances:

                absolute_min = min(distances)

                # Use 5th percentile instead of one random minimum point
                near_distance = float(
                    np.percentile(distances, 5)
                )

                avg_distance = (
                    sum(distances) /
                    len(distances)
                )

                # BLOCKED / CAUTION / CLEAR
                if near_distance <= BLOCKED_DISTANCE:
                    clearance_status = "blocked"

                elif near_distance <= CLEAR_DISTANCE:
                    clearance_status = "caution"

                else:
                    clearance_status = "clear"

                zone_summary[zone_name] = {
                    "points": len(distances),
                    "absolute_min_m": round(
                        absolute_min,
                        2
                    ),
                    "near_distance_m": round(
                        near_distance,
                        2
                    ),
                    "avg_distance_m": round(
                        avg_distance,
                        2
                    ),
                    "status": clearance_status,

                    # Keep this for compatibility with older code
                    "clear": clearance_status == "clear"
                }

            else:

                zone_summary[zone_name] = {
                    "points": 0,
                    "absolute_min_m": None,
                    "near_distance_m": None,
                    "avg_distance_m": None,
                    "status": "unknown",
                    "clear": False
                }

        # --------------------------------------------------
        # Direction groups
        # --------------------------------------------------

        clear_zones = [
            zone
            for zone, data in zone_summary.items()
            if data["status"] == "clear"
        ]

        caution_zones = [
            zone
            for zone, data in zone_summary.items()
            if data["status"] == "caution"
        ]

        blocked_zones = [
            zone
            for zone, data in zone_summary.items()
            if data["status"] == "blocked"
        ]

        # --------------------------------------------------
        # Best / closest directions
        # --------------------------------------------------

        clearest_direction = max(
            zone_summary,
            key=lambda zone:
                zone_summary[zone]["near_distance_m"]
                or 0
        )

        valid_zones = [
            zone
            for zone in zone_summary
            if zone_summary[zone]["near_distance_m"]
            is not None
        ]

        if valid_zones:

            closest_direction = min(
                valid_zones,
                key=lambda zone:
                    zone_summary[zone]["near_distance_m"]
            )

            closest_distance = (
                zone_summary[
                    closest_direction
                ]["near_distance_m"]
            )

        else:

            closest_direction = None
            closest_distance = None

        return {
            "status": "scan_saved",
            "scan_ok": True,
            "total_points": total_points,
            "file_path": filename,

            "zones": zone_summary,

            "clear_zones": clear_zones,
            "caution_zones": caution_zones,
            "blocked_zones": blocked_zones,

            "clearest_direction": clearest_direction,

            "closest_obstacle_direction":
                closest_direction,

            "closest_obstacle_meters":
                closest_distance
        }

    def add_tool_result(self, tool_call_id, result):
        """
        Record the result of a tool that was executed outside Autopilot,
        such as move_rover in UART.py.
        """
        self.update_memory({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": json.dumps(result)
        })

    def update_memory(self, msg):
        
        """
        Add a message and safely trim old memory without
        separating tool calls from their tool responses.
        """

        self.memory.append(msg)

        while len(self.memory) > self.memory_depth:

            if len(self.memory) <= 1:
                break

            oldest = self.memory[1]

            # Get role whether this is a normal dict
            # or an OpenAI message object
            if isinstance(oldest, dict):
                role = oldest.get("role")
                tool_calls = oldest.get("tool_calls")
            else:
                role = getattr(oldest, "role", None)
                tool_calls = getattr(oldest, "tool_calls", None)

            # If removing an assistant tool call,
            # also remove its matching tool response(s)
            if role == "assistant" and tool_calls:

                tool_ids = {
                    call.id for call in tool_calls
                }

                self.memory.pop(1)

                while len(self.memory) > 1:

                    next_msg = self.memory[1]

                    if isinstance(next_msg, dict):
                        next_role = next_msg.get("role")
                        next_tool_id = next_msg.get(
                            "tool_call_id"
                        )
                    else:
                        next_role = getattr(
                            next_msg,
                            "role",
                            None
                        )
                        next_tool_id = getattr(
                            next_msg,
                            "tool_call_id",
                            None
                        )

                    if (
                        next_role == "tool"
                        and next_tool_id in tool_ids
                    ):
                        self.memory.pop(1)
                    else:
                        break

            else:
                self.memory.pop(1)