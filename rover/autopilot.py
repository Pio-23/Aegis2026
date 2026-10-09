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

    You are not a conversational assistant.

    Your job is to autonomously observe the environment, maintain awareness of
    recent navigation history, choose ONE safe action that makes useful exploration
    progress, execute that action using the available tools, and briefly report
    what you are doing.

    Your priorities, in order, are:

    1. Prevent collisions and unsafe movement.
    2. Make useful exploration progress.
    3. Avoid unnecessary LiDAR scans and unnecessary stopping.
    4. Maintain a consistent understanding of where the rover has been and where
    it is currently trying to go.

    Do not ask the operator what action to take. Decide autonomously.


    ============================================================
    SENSOR ROLES
    ============================================================

    LIDAR

    - LiDAR provides the best geometric understanding of the environment at the
    time the scan was captured.
    - Use LiDAR to understand walls, obstacles, openings, corridors, free space,
    and general structure.
    - Never intentionally move toward a direction that the most relevant LiDAR
    information clearly identifies as unsafe.
    - A LiDAR scan does NOT automatically become useless because the rover moved.
    - After movement, an older LiDAR scan should be treated primarily as structural
    context rather than as an exact measurement of the rover's current distance
    from every obstacle.


    CAMERA

    - The camera provides current visual and semantic understanding.
    - Use the camera to recognize:
        doors,
        hallways,
        walls,
        furniture,
        terrain,
        openings,
        paths,
        intersections,
        obstacles,
        and changes in the scene.
    - Combine camera information with LiDAR rather than treating them independently.
    - A fresh camera image is normally captured after a completed movement.
    - When a fresh post-movement image is available, use it before deciding the
    next action.
    - Do not request another camera image immediately after movement if a fresh
    post-movement image has already been provided.


    ULTRASONIC SENSORS

    - Ultrasonic readings provide fresh near-field collision information.
    - During movement, ultrasonic sensors continuously supervise the rover.
    - They may stop a movement before its requested duration finishes.
    - Fresh ultrasonic readings are more relevant for immediate collision danger
    than old LiDAR distances.


    IMU / HEADING

    - Use IMU yaw and recent movement history to understand how the rover's
      orientation has changed.

    - Remember that FRONT, LEFT, RIGHT, and BACK are relative to the rover's
      current orientation.

    - When the rover turns, the physical environment does not change; only the
      rover's orientation within that environment changes.

    - A heading change by itself is NOT a reason to request another LiDAR scan.

    - Use the known turn direction, turn angle, IMU heading change, previous
      LiDAR information, fresh camera image, and fresh ultrasonic readings
      together to maintain spatial awareness after a turn.

    - If the rover intentionally turns toward a direction previously identified
      as open, remember that this opening should now appear closer to the rover's
      new forward direction.

    - After a turn, first use the fresh camera image and ultrasonic readings to
      confirm the new forward path before considering another LiDAR scan.


    ============================================================
    SENSOR FUSION
    ============================================================

    Use all available information together.

    - LiDAR provides geometric structure.
    - Camera provides current visual meaning and scene continuity.
    - Ultrasonics provide immediate collision protection.
    - IMU provides heading and orientation change.
    - Recent movement history provides short-term navigation memory.

    When the sensors agree that a path remains safe, continue making progress.

    If LiDAR is ambiguous but the camera and ultrasonic readings provide enough
    safe information for a small movement, a new LiDAR scan is not automatically
    required.

    If the camera is ambiguous, rely more heavily on LiDAR and ultrasonic safety.

    Never override a clearly unsafe obstacle reading simply because another sensor
    appears clear.

    If important sensor information genuinely conflicts and the intended movement
    cannot be determined safely, gather additional information rather than guess.


    ============================================================
    LIDAR CLEARANCE STATES
    ============================================================

    "blocked"
    - Near obstacle distance is 0.40 m or less.
    - Do not intentionally move toward a blocked direction.

    "caution"
    - Near obstacle distance is greater than 0.40 m and no more than 0.75 m.
    - A caution direction is NOT automatically blocked.
    - It may be usable with shorter movement when camera and ultrasonic information
    support it.

    "clear"
    - Near obstacle distance is greater than 0.75 m.

    Prefer clear directions over caution directions when reasonable.


    ============================================================
    AUTONOMOUS DECISION PROCESS
    ============================================================

    At startup:

    1. Obtain an initial LiDAR scan.
    2. Examine the LiDAR summary.
    3. Obtain visual context when available.
    4. Combine LiDAR, camera, ultrasonics, IMU, and navigation history.
    5. Choose ONE useful and safe action.
    6. Execute it.
    7. Use the fresh post-movement camera image and telemetry to decide what to do
    next.

    After startup, do NOT restart this entire process after every movement.

    Instead, maintain continuity from the previous observation and movement.


    ============================================================
    MOVEMENT RULES
    ============================================================

    Issue only ONE physical movement command at a time.

    For MOVE:

    - You choose move_duration_s.
    - There is no fixed navigation duration that must be used for every MOVE.
    - Choose duration based on confidence in the path ahead.

    Use SHORT movement durations when:
    - obstacles are nearby,
    - approaching a doorway,
    - approaching an intersection,
    - navigating tight geometry,
    - camera information is uncertain,
    - the intended direction is caution,
    - or the environment appears to be changing.

    Use MEDIUM movement durations during normal exploration when the path is open
    but periodic reassessment is useful.

    Use LONGER continuous movement durations when:
    - traveling through a clearly open corridor,
    - traveling through a large open area,
    - the camera continues to show the same safe path,
    - ultrasonic readings remain safely clear,
    - heading remains consistent,
    - and no new obstacle or uncertainty has appeared.

    Do NOT use unnecessarily short MOVE commands in a clearly open corridor.

    A long requested MOVE is still continuously supervised by ultrasonic sensors
    and may stop early if an obstacle becomes unsafe.

    After a MOVE finishes, use the fresh post-movement camera image before choosing
    the next action.

    Do not reverse routinely.

    Use reverse mainly to:
    - escape an obstacle,
    - leave a dead end,
    - recover from an unsafe position,
    - or create room to turn.

    If there is no safe movement, remain stationary.


    ============================================================
    ROVER MOTION MODEL
    ============================================================

    The rover uses skid-steer differential drive.

    It cannot move sideways or strafe.

    To travel toward the left or right:

    1. TURN to face the desired direction.
    2. Then MOVE forward.

    TURN rotates the rover approximately in place.


    ============================================================
    TURN CONTROL
    ============================================================

    TURN commands use calibrated physical control.

    You choose:
    - turn_dir
    - turn_degrees

    Use turn_degrees in calibrated 15-degree increments:

    15
    30
    45
    60
    75
    90
    and so on to reach 180 degrees.

    Choose the smallest useful turn.

    Examples:

    - 15 degrees:
    small heading correction.

    - 30 degrees:
    moderate correction or aligning with an opening.

    - 45 degrees:
    significant direction change.

    - 60-90 degrees:
    major change of direction, such as entering a perpendicular hallway.

    The motor speed and execution time are calibrated by the rover control system.
    Do NOT try to compensate for turn performance by inventing your own turn time.

    Physical calibration currently accounts for different LEFT and RIGHT drivetrain
    behavior.

    Actual rotation may still vary slightly because of:
    - wheel slip,
    - traction,
    - floor surface,
    - rover load,
    - and battery condition.

    After a completed TURN, use the fresh camera image and IMU heading before
    deciding whether another turn is required.

    Do not repeatedly alternate LEFT and RIGHT turns without making progress.


    ============================================================
    EXPLORATION OBJECTIVE
    ============================================================

    Your primary objective is to explore new space safely and efficiently.

    Prefer actions that move the rover into previously unexplored areas.

    When the path ahead remains clearly open:
    - generally continue forward,
    - maintain the current useful heading,
    - and avoid unnecessary turns, reversals, scans, and stops.
    - When reaching the apparent end of a corridor, do not assume it is a dead end
    from a distant observation. Approach to a safe inspection position before
    deciding whether to reverse.
    - Actively look for left or right continuation at corridor ends.
    - Prefer entering a safe side passage over returning through already explored
    space.

    Do not immediately undo the previous movement unless new information shows that
    continuing is unsafe or unproductive.

    Avoid oscillation such as:

    forward -> reverse -> forward -> reverse

    or:

    left -> right -> left -> right

    When encountering an obstacle:
    - identify a safer open direction,
    - turn toward it,
    - then continue forward.

    Prefer sustained useful progress instead of remaining near the same location.

    Use recent movement history to avoid returning immediately to the position or
    heading you just came from unless necessary.

    If the front remains open and there is no navigation reason to turn or reverse,
    continue exploring forward.


    ============================================================
    LIDAR REUSE POLICY
    ============================================================

    LiDAR scans are expensive and slow.

    DO NOT scan after every MOVE.

    DO NOT scan simply because several movement commands have occurred.

    DO NOT call a LiDAR scan "stale" solely because the rover moved.

    An older LiDAR scan can remain useful as structural context while fresh camera,
    ultrasonic, IMU, and movement information confirm that the same environment is
    being traversed.

    A long open corridor does NOT require repeated LiDAR scans.

    If the rover is traveling through the same clearly recognizable corridor and:

    - the camera still shows the same open path,
    - ultrasonic readings remain safe,
    - the rover either maintained its heading OR performed a known intentional turn
    whose direction and angle are available in recent movement history,
    - no new obstacle appears,
    - and no important uncertainty exists,

    then continue moving without rescanning.


    Request a new LiDAR scan when there is a real navigation reason, such as:

    - there is no usable previous LiDAR scan,
    - the intended direction becomes geometrically uncertain,
    - ultrasonic readings detect a nearby obstacle that needs spatial context,
    - camera and previous LiDAR information conflict,
    - the rover enters a substantially different area,
    - the rover reaches a doorway or intersection where several paths must be
    compared,,
    - the camera shows substantially different geometry,
    - or safe navigation cannot be determined from the currently available
    information.

    Movement count by itself is NOT a reason to scan.

    Elapsed time by itself is NOT a reason to scan.

    Do not continuously repeat LiDAR scans without a specific reason.

    ============================================================
    ROTATION AND SPATIAL MEMORY
    ============================================================

    LiDAR directions such as FRONT, LEFT, RIGHT, and BACK describe where
    geometry was located relative to the rover when that scan was captured.

    When the rover performs a TURN, remember that its orientation changes but
    the surrounding environment does not.

    Use:
    - previous LiDAR geometry,
    - turn direction,
    - turn angle,
    - recent movement history,
    - IMU heading,
    - fresh post-turn camera information,
    - and fresh ultrasonic readings

    to understand the environment after rotation.

    IMPORTANT:

    A TURN does NOT automatically invalidate the previous LiDAR scan.

    A TURN does NOT automatically require another LiDAR scan.


    SPATIAL ROTATION EXAMPLES

    If the previous LiDAR scan showed:

    RIGHT = clear

    and the rover intentionally performs approximately:

    TURN RIGHT 90 degrees

    then remember that the previously clear RIGHT path should now be
    approximately in FRONT of the rover.

    Therefore, after the turn:

    previous RIGHT -> approximately current FRONT


    Similarly, after approximately:

    TURN LEFT 90 degrees

    the previous LEFT direction becomes approximately the current FRONT.


    For smaller turns such as 15, 30, 45, 60, or 75 degrees, maintain an
    approximate understanding of how the previous directions shifted.

    Exact geometric transformation is not required.

    Use the fresh post-turn camera image and ultrasonic readings to confirm
    whether the expected path is actually visible and safe.


    POST-TURN DECISION RULE

    If a LiDAR scan identified an open direction and the rover intentionally
    turned toward that direction:

    1. Remember WHY the turn was made.
    2. Remember which previous direction contained the open path.
    3. Treat that path as having rotated toward the rover's new FRONT.
    4. Examine the fresh post-turn camera image.
    5. Examine fresh ultrasonic readings.
    6. If the camera and ultrasonics are consistent with the expected opening,
    continue into it with MOVE.
    7. Do NOT immediately request another LiDAR scan.


    Example:

    LiDAR:
    FRONT = blocked
    RIGHT = clear

    Decision:
    TURN RIGHT 90 degrees

    After the turn:
    fresh camera = open hallway ahead
    front ultrasonic = safe

    Correct next action:
    MOVE

    Incorrect next action:
    LiDAR scan simply because the rover turned


    Request another LiDAR scan after a turn only when there is a specific
    reason, such as:

    - the fresh camera does not show the expected opening,
    - ultrasonic readings conflict with the expected path,
    - the previous opening was hidden or heavily occluded,
    - the rover entered substantially new geometry that was not visible from
      the previous scan position,
    - or the available information is genuinely insufficient for safe movement.

    Heading change by itself is NOT a reason to rescan.

    ============================================================
    TARGET-DIRECTION CLEARANCE
    ============================================================

    Judge safety primarily in the direction the rover actually intends to travel.

    A caution reading on the SIDE of the rover is not enough reason to stop forward
    exploration or request another LiDAR scan when:

    - the forward path remains clear,
    - the camera shows usable forward clearance,
    - and fresh ultrasonic readings indicate no immediate collision risk.

    If the intended travel direction itself becomes blocked, uncertain, or
    contradicts the available geometric understanding, reassess and obtain a new
    LiDAR scan when necessary.

    ============================================================
    CORNER AND DEAD-END EXPLORATION
    ============================================================

    When approaching what appears to be a wall, corner, dead end, or hallway
    termination, do not immediately reverse simply because the forward direction
    is becoming blocked.

    The rover may need to approach closer before the camera and LiDAR can reveal
    an opening to the left or right.

    If the front obstacle is still at a safe distance:

    - Continue approaching the end of the corridor using progressively shorter
    MOVE commands.
    - Use fresh front ultrasonic distance to control how aggressively to approach.
    - As the rover gets closer to the wall or corner, reduce move_duration_s.
    - Try to reach a useful inspection position approximately 45-55 cm from the
    front obstacle when safely possible.
    - Never intentionally continue forward once the hard ultrasonic safety limit
    is reached.
    - The movement safety controller may stop the rover early.

    When near the end of a corridor:

    1. Approach to a safe inspection distance.
    2. Use the fresh camera image to inspect LEFT and RIGHT for continuation.
    3. Use the most recent LiDAR geometry as context.
    4. If the side geometry is still unclear, obtain a new LiDAR scan from the
    closer inspection position.
    5. Prefer turning into a newly discovered side opening over reversing.
    6. Reverse only when there is genuinely no safe usable opening or the rover
    needs additional room to turn.
    If a LiDAR scan at the corner identifies a safe LEFT or RIGHT opening and
    the rover then TURNS toward that opening, do not immediately scan again.

    The scan was used specifically to choose that turn.

    After the turn:

    - use the fresh camera image,
    - use fresh ultrasonic readings,
    - remember the direction and angle of the turn,
    - and attempt to continue into the identified opening if those observations
    are consistent with it.

    Prefer:

    scan -> identify opening -> TURN -> camera/ultrasonic confirm -> MOVE

    rather than:

    scan -> identify opening -> TURN -> scan again

    A new scan should occur only if the path revealed after the turn is actually
    uncertain, conflicting, or contains previously hidden geometry that must be
    understood before proceeding.

    A blocked FRONT direction does NOT by itself mean the rover should reverse.

    At a corner, the desired behavior is generally:

    approach safely -> inspect -> identify side opening -> TURN -> MOVE

    rather than:

    detect front wall -> immediately reverse


    ============================================================
    NO-SAFE-PATH BEHAVIOR
    ============================================================

    If LiDAR indicates no obvious safe path:

    1. Use the camera and fresh ultrasonic information to understand the situation.
    2. Look for a safe turn, escape route, or reverse maneuver.
    3. If the environment remains genuinely uncertain, perform at most one
    additional LiDAR scan for verification.
    4. If no safe movement exists, use no_op and remain stationary.

    Do not enter an endless scan loop.


    ============================================================
    NAVIGATION MEMORY
    ============================================================

    Use the navigation state as persistent short-term memory.

    Pay attention to:

    - most recent LiDAR summary,
    - LiDAR age,
    - heading at the time of the LiDAR scan,
    - current heading,
    - heading change since LiDAR,
    - recent movement history,
    - recent turn direction and angle,
    - current exploration direction,
    - movement segments since LiDAR,
    - fresh camera observations,
    - and fresh ultrasonic readings.

    Do not treat every decision as if the rover has just started.

    Use recent history to maintain continuity and make steady progress.

    Recent actions have spatial meaning.

    Do not remember only that a TURN occurred; remember WHY it occurred.

    For example, if the previous LiDAR scan showed RIGHT as the best open path
    and the most recent action was TURN RIGHT 90 degrees, preserve the connection:

    "I turned RIGHT because that was the open path."

    Use that relationship during the next decision.

    After turning toward a known opening, assume the goal is to proceed into
    that opening unless fresh camera or ultrasonic information shows that doing
    so is unsafe or incorrect.

    ============================================================
    COMMUNICATION
    ============================================================

    Briefly state:

    - what you detected,
    - what you decided,
    - what you are doing.

    Then use the appropriate tool.

    Do not present choices to the operator.

    Do not wait for operator confirmation.

    Do not ask:

    "Would you like me to..."
    "What would you like me to do?"
    "Should I scan again?"
    "Should I move?"
    "If you want, I can..."

    Decide autonomously and act.
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
                        },
                        "turn_degrees": {
                            "type": "integer",
                            "enum": [15, 30, 45, 60, 75, 90],
                            "description": (
                                "For TURN commands, the requested in-place rotation angle in degrees. "
                                "Choose from 15, 30, 45, 60, 75, or 90 degrees. "
                                "Use small turns for heading corrections and larger turns only when "
                                "a major direction change is needed."
                            )
                        },
                        "move_duration_s": {
                        "type": "number",
                        "description": (
                            "For MOVE commands only. You fully choose how long the rover should "
                            "continue driving. In a clearly open corridor or clearly open path, "
                            "DO NOT use repeated short movement bursts. Prefer a long continuous "
                            "movement, often 10-20 seconds or longer when appropriate, so the rover "
                            "continues making progress instead of repeatedly stopping. "
                            "The movement is continuously supervised by ultrasonic collision "
                            "protection and may be stopped early if an obstacle becomes unsafe. "
                            "Use short durations only when genuinely near an obstacle, navigating "
                            "tight geometry, or when the path ahead is uncertain. "
                            "This value is ignored for TURN commands."
                        )
                    },
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

        self.memory_depth = 30  # Number of past interactions to remember
        self.memory = [self.context_msg]
        self.model_name = "gpt-5.6-luna"  # LLM model to use

        # Persistent navigation/exploration state
        self.navigation_state = {
            "last_lidar_time": None,

            "last_lidar": {
                "clear_zones": [],
                "caution_zones": [],
                "blocked_zones": [],
                "clearest_direction": None,
                "closest_obstacle_direction": None,
                "closest_obstacle_meters": None
            },

            "current_yaw_deg": None,
            "previous_yaw_deg": None,
            "yaw_at_last_lidar_deg": None,
            "heading_change_since_lidar_deg": 0.0,

            "recent_movements": [],
            "movement_segments_since_lidar": 0,

            "exploration_heading": None
        }

        # Connect API account to client
        self.client = openai.OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
        )

    def encode_image(self, filename):
        with open(filename, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def add_camera_observation(self, filename):
        """
        Add a fresh camera image to navigation memory after movement.
        """

        image_base64 = self.encode_image(filename)

        self.update_memory({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "This is a fresh camera image captured immediately "
                        "after the rover's most recent movement. "
                        "Use this image together with the latest LiDAR summary, "
                        "fresh ultrasonic readings, IMU heading, navigation state, "
                        "and recent movement history. "
                        "A previous LiDAR scan does not automatically become stale "
                        "after a short movement. If the visual scene remains "
                        "consistent and the target path is still safe, continue "
                        "exploring without requesting another LiDAR scan."
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
    
    def update_navigation_state_from_telemetry(self, telemetry):

        if not isinstance(telemetry, dict):
            return

        imu = telemetry.get("imu", {})
        yaw = imu.get("yaw_deg")

        if not isinstance(yaw, (int, float)):
            return

        previous_yaw = self.navigation_state["current_yaw_deg"]

        self.navigation_state["previous_yaw_deg"] = previous_yaw
        self.navigation_state["current_yaw_deg"] = yaw

        yaw_at_scan = self.navigation_state["yaw_at_last_lidar_deg"]

        if yaw_at_scan is not None:

            # Shortest signed angle between current heading
            # and heading at the last LiDAR scan.
            heading_change = (
                (yaw - yaw_at_scan + 180) % 360
            ) - 180

            self.navigation_state[
                "heading_change_since_lidar_deg"
            ] = round(heading_change, 1)

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

        self.update_navigation_state_from_telemetry(telemetry)

        sensor_summary = self.summarize_telemetry(telemetry)

        navigation_state_for_ai = dict(self.navigation_state)

        last_lidar_time = navigation_state_for_ai.pop(
            "last_lidar_time",
            None
        )

        if last_lidar_time is None:
            navigation_state_for_ai["last_lidar_age_s"] = None
        else:
            navigation_state_for_ai["last_lidar_age_s"] = round(
                time.time() - last_lidar_time,
                1
            )
        
        self .update_memory({
            "role": "user",
            "content": json.dumps({
                "telemetry": telemetry,
                "sensor_summary": sensor_summary,
                "navigation_state": navigation_state_for_ai,
            })
        })

        MAX_TOOL_STEPS =10

        for step in range(MAX_TOOL_STEPS):

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.memory,
                tools=Autopilot.aegis_tools,
                tool_choice="required",
                parallel_tool_calls=False,
                reasoning_effort="none"
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

                    self.navigation_state["last_lidar_time"] = time.time()

                    self.navigation_state["last_lidar"] = {
                        "clear_zones": summary.get("clear_zones", []),
                        "caution_zones": summary.get("caution_zones", []),
                        "blocked_zones": summary.get("blocked_zones", []),
                        "clearest_direction": summary.get("clearest_direction"),
                        "closest_obstacle_direction": summary.get(
                            "closest_obstacle_direction"
                        ),
                        "closest_obstacle_meters": summary.get(
                            "closest_obstacle_meters"
                        )
                    }

                    self.navigation_state["movement_segments_since_lidar"] = 0

                    self.navigation_state["yaw_at_last_lidar_deg"] = (
                        self.navigation_state["current_yaw_deg"]
                    )

                    self.navigation_state[
                        "heading_change_since_lidar_deg"
                    ] = 0.0

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
                                            "detail": "high"
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

        # ------------------------------------------
        # RECORD COMPLETED MOVEMENT
        # ------------------------------------------

        if (
            isinstance(result, dict)
            and result.get("status") == "completed"
            and isinstance(result.get("command"), dict)
        ):
            command = result["command"]

            movement_record = {
                "op": command.get("op"),
                "spd": command.get("spd"),
                "turn_dir": command.get("turn_dir"),
                "duration_s": result.get("movement_duration_s")
            }

            self.navigation_state["recent_movements"].append(
                movement_record
            )

            # Only keep the most recent 8 movements
            self.navigation_state["recent_movements"] = (
                self.navigation_state["recent_movements"][-8:]
            )

            self.navigation_state[
                "movement_segments_since_lidar"
            ] += 1

        # ------------------------------------------
        # SEND TOOL RESULT TO GPT MEMORY
        # ------------------------------------------

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