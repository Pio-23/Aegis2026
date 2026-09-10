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
    the environment, choose the safest reasonable action, execute that action
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

    MOVEMENT RULES

    - Issue only ONE physical movement command at a time.
    - Prefer slow and conservative movement near obstacles.
    - After moving, reassess before committing to another significant movement.
    - Do not repeatedly issue movement commands without updated sensor information.
    - Never move toward a blocked direction.
    - If there is no safe movement, stay stationary.

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

    def decide_actions(self, telemetry, scanner, ugv_cam, dump_folder):
        """
        Ask GPT what to do.
        If GPT requests a LiDAR scan, perform the scan,
        summarize it, send the result back to GPT,
        then ask GPT again.
        """

        self .update_memory({
            "role": "user",
            "content": json.dumps({
                "telemetry": telemetry,
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

            self.memory.append(message)

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

                    filename = scanner.scan(filepath=dump_folder)
                    print(f"Scan saved: {filename}")

                    summary = self.summarize_scan_file(filename)

                    self.memory.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps({
                            "status": "scan_saved",
                            "file_path": filename,
                            "summary": summary
                        })
                    })

                    print(
                        f"[AI] GPT requested LiDAR scan. Scan complete and summarized."
                        f"for {call.id}" 
                    )

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
            distance = math.sqrt(x**2 + y**2 + z**2)
            if distance <= 0:
                continue

            total_points += 1

            # Angle around rover, assuming:
            # +X = front
            # +Y = left
            # -Y = right
            angle = math.degrees(math.atan2(y, x))
            
            # Convert angle to 0-360
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

        for zone_name, distances in zones.items():
            if distances:
                zone_summary[zone_name] = {
                    "points": len(distances),
                    "min_distance_m": round(min(distances), 2),
                    "avg_distance_m": round(sum(distances) / len(distances), 2),
                    "clear": min(distances) > 0.40
                }
            else:
                zone_summary[zone_name] = {
                    "points": 0,
                    "min_distance_m": None,
                    "avg_distance_m": None,
                    "clear": False
                }

        clear_zones = [
            zone for zone, data in zone_summary.items()
            if data["clear"]
        ]

        blocked_zones = [
            zone for zone, data in zone_summary.items()
            if data["min_distance_m"] is not None and data["min_distance_m"] <= 0.40
        ]

        clearest_direction = max(
            zone_summary,
            key=lambda z: zone_summary[z]["avg_distance_m"] or 0
        )

        closest_direction = min(
            [z for z in zone_summary if zone_summary[z]["min_distance_m"] is not None],
            key=lambda z: zone_summary[z]["min_distance_m"]
        )
        return {
            "status": "scan_saved",
            "scan_ok": True,
            "total_points": total_points,
            "file_path": filename,
            "zones": zone_summary,
            "clear_zones": clear_zones,
            "blocked_zones": blocked_zones,
            "clearest_direction": clearest_direction,
            "closest_obstacle_direction": closest_direction,
            "closest_obstacle_meters": zone_summary[closest_direction]["min_distance_m"]
        }

    def validate_action(self, toolcall, telemetry=None) -> bool:
        """
        Validate the proposed action for safety and feasibility.
        Returns True if valid, False otherwise.
        """

        
        name = toolcall.function.name                           # type: ignore
        args = json.loads(toolcall.function.arguments or "{}")  # type: ignore
        
    
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


    def update_memory(self, msg : dict) -> None:
        """
        Update the internal memory with a new message.
        """
        if (len(self.memory) > self.memory_depth):
            self.memory.pop(1)  # Remove oldest, keep system context
            
        self.memory.append(msg)