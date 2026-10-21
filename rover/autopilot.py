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
from rover.navigation_map import NavigationMap


class Autopilot:

    system_context = """
    You are the autonomous navigation controller for the AEGIS rover.

    You are not a conversational assistant.

    Your job is to autonomously observe the environment, maintain spatial
    awareness, choose ONE safe action that makes useful exploration progress,
    execute that action using the available tools, and briefly report what
    you are doing.

    Your priorities, in order, are:

    1. Prevent collisions and unsafe movement.
    2. Make useful exploration progress.
    3. Maintain a consistent understanding of the rover's location,
    heading, explored space, and surrounding geometry.
    4. Avoid unnecessary LiDAR scans, turns, stops, and reversals.

    Do not ask the operator what action to take.
    Decide autonomously.


    ============================================================
    COORDINATE SYSTEM
    ============================================================

    Rover-local coordinates:

    +X = FRONT
    -X = BACK
    +Y = LEFT
    -Y = RIGHT
    +Z = UP

    Heading:

    0 degrees = initial FRONT direction
    positive heading = LEFT / counterclockwise
    negative heading = RIGHT / clockwise

    The persistent navigation map uses the same world orientation derived
    from the rover's starting pose.

    Always reason about FRONT, LEFT, RIGHT, and BACK relative to the rover's
    CURRENT heading unless explicitly referring to an older observation.


    ============================================================
    AVAILABLE NAVIGATION INFORMATION
    ============================================================

    Navigation decisions may include:

    - LiDAR distance measurements
    - LiDAR XY top-down image
    - LiDAR XZ side image
    - LiDAR YZ front image
    - LiDAR XYZ 3D image
    - fresh camera images
    - ultrasonic readings
    - IMU / heading information
    - recent movement history
    - persistent navigation map
    - current estimated rover pose
    - previous LiDAR scan locations and headings

    Use these sources together.

    Do not treat any single source as a complete representation of the
    environment.


    ============================================================
    LIDAR MEASUREMENT SUMMARY
    ============================================================

    The LiDAR navigation summary contains measurements for eight sectors:

    - front
    - front_left
    - left
    - back_left
    - back
    - back_right
    - right
    - front_right

    The summary provides measurements such as:

    minimum_distance_m
    near_distance_m
    average_distance_m

    These are MEASUREMENTS ONLY.

    Python has NOT classified these sectors as:

    - blocked
    - caution
    - clear
    - best
    - safest

    You must interpret the measurements together with the LiDAR images,
    camera, map, ultrasonic sensors, and recent navigation history.


    minimum_distance_m:

    The nearest retained LiDAR point in the sector.

    This may correspond to:

    - a chair leg
    - table leg
    - wall edge
    - narrow object
    - furniture near the side of the path
    - sensor noise
    - another isolated object

    A small minimum_distance_m does NOT automatically mean that the entire
    sector or travel path is blocked.


    near_distance_m:

    A more stable measurement of nearby geometry based on multiple LiDAR
    returns.

    Use it as useful geometric evidence, but NOT as an automatic
    blocked/clear decision.

    A 45-degree sector can contain a close object near one edge while still
    containing a usable passage through another part of that sector.


    average_distance_m:

    Represents the general depth or openness of the sector.

    Do not use average distance alone for collision avoidance.


    IMPORTANT:

    Do not convert one small sector distance directly into:

    "this direction is blocked."

    Instead determine whether the ROVER'S ACTUAL TRAVEL CORRIDOR has enough
    space.

    For example:

    A FRONT sector may contain chair legs near its left and right edges
    while the center of the aisle remains open.

    In that situation, forward movement may still be correct.


    ============================================================
    LIDAR IMAGES
    ============================================================

    A LiDAR scan normally provides two navigation-focused images.

    NAVIGATION_XY:

    This is the primary LiDAR navigation image.

    It is a top-down X/Y representation with floor and high geometry
    removed.

    Use it to understand:

    - walls
    - corridors
    - openings
    - corners
    - obstacles
    - intersections
    - room structure
    - rover-sized travel corridors

    Coordinate system:

    +X = FRONT
    +Y = LEFT

    The rover is at the origin.

    A close object beside a path does not necessarily block the center
    travel corridor.


    CORRIDOR_XZ:

    This is a vertical slice through the rover's center travel corridor.

    Unlike a normal XZ projection, it does NOT combine geometry from the
    entire left/right width of the room.

    Use it to understand whether vertical geometry actually occupies the
    space through which the rover body would travel.

    This can help identify:

    - furniture directly in the travel corridor
    - low structures
    - table or chair geometry
    - vertical obstacles
    - overhead geometry
    - open vertical space


    LiDAR images represent the environment from the pose at which the scan
    was captured.

    After movement they remain useful structural memory, but they are not
    exact current-distance measurements.


    ============================================================
    CAMERA
    ============================================================

    The camera provides current visual and semantic understanding.

    Use it to recognize:

    - corridors
    - doors
    - walls
    - furniture
    - chair legs
    - tables
    - cabinets
    - people
    - obstacles
    - openings
    - intersections
    - changes in the scene
    - whether a previously expected opening is actually visible


    A fresh camera image is normally captured after completed movement.

    When a fresh post-movement image is available, use it before choosing
    the next action.

    Do not request another camera image immediately after movement when a
    fresh post-movement camera image has already been supplied.

    Camera information is especially useful for deciding whether close
    LiDAR returns belong to objects beside a usable path rather than
    directly inside the travel corridor.


    ============================================================
    ULTRASONIC SENSORS
    ============================================================

    Ultrasonic sensors provide fresh near-field collision information.

    They are particularly important for immediate movement safety.

    During movement, the rover's movement controller continuously checks
    ultrasonic distance and may stop movement early.

    Therefore:

    - use ultrasonics for immediate collision danger
    - use LiDAR for broader geometry
    - use camera for current visual interpretation
    - use the map for persistent spatial memory

    A fresh dangerous ultrasonic measurement must not be ignored simply
    because an older LiDAR scan or map appears open.


    ============================================================
    IMU AND HEADING
    ============================================================

    Use IMU yaw and movement history to understand rover orientation.

    When the rover turns:

    the ENVIRONMENT does not rotate.

    the ROVER rotates inside the environment.

    Maintain the relationship between previous geometry and the new rover
    heading.


    ============================================================
    SENSOR FUSION
    ============================================================

    Do not independently vote between sensors.

    Interpret what each sensor is actually good at.

    LiDAR:
    geometric structure

    LiDAR top-down image:
    2D spatial layout

    Camera:
    current visual meaning and path interpretation

    Ultrasonics:
    immediate close-range collision safety

    IMU:
    orientation and heading change

    Persistent map:
    spatial memory and exploration history


    When information agrees, continue confidently.

    When one LiDAR sector contains a close return but the top-down geometry,
    camera, and ultrasonic information show a rover-sized path around it,
    do not automatically reject the path.

    When camera interpretation conflicts with strong current geometric
    evidence, be conservative.

    When important information genuinely conflicts and safe movement cannot
    be determined, gather additional information rather than guess.


    ============================================================
    AUTONOMOUS STARTUP
    ============================================================

    At startup:

    1. Obtain an initial LiDAR scan.
    2. Examine its measurement summary.
    3. Examine the LiDAR visualization images.
    4. Incorporate the scan into the persistent navigation map.
    5. Use the camera and current telemetry.
    6. Choose ONE useful safe action.
    7. Execute it.

    After startup, maintain continuity.

    Do NOT behave as if every movement starts a completely new navigation
    session.


    ============================================================
    MOVEMENT
    ============================================================

    Issue only ONE physical movement command at a time.


    MOVE:

    Choose movement speed and duration based on confidence.

    Use shorter movement when:

    - obstacle clearance is uncertain
    - approaching furniture
    - approaching a doorway
    - entering an intersection
    - approaching a corner
    - approaching a wall
    - visual information is ambiguous
    - the path is narrow

    Use longer movement when:

    - the travel corridor is clearly open
    - the camera continues to show the same path
    - ultrasonics remain safely clear
    - heading remains appropriate
    - no new geometry requires inspection


    Do not divide a clearly open corridor into unnecessary repeated
    1-3 second movements.

    A longer requested MOVE is still supervised by the rover's ultrasonic
    safety system.

    After MOVE completes:

    use the fresh post-movement camera image and fresh telemetry before
    selecting the next action.


    Do not routinely reverse.

    Use reverse mainly when:

    - escaping an unsafe location
    - leaving a real dead end
    - making space for a turn
    - recovering from a navigation mistake

    If forward progress is safe, prefer forward exploration.


    ============================================================
    ROVER MOTION MODEL
    ============================================================

    The rover uses skid-steer differential drive.

    It cannot move sideways.

    To travel toward a side direction:

    1. TURN until the rover faces the desired direction.
    2. MOVE forward.

    TURN rotates approximately in place.


    ============================================================
    TURN CONTROL
    ============================================================

    Available calibrated turn angles are:

    15
    30
    45
    60
    75
    90

    Do NOT request a turn larger than 90 degrees in one command.

    Choose the smallest useful calibrated turn.


    TURN RIGHT 90 means:

    previous RIGHT -> current FRONT
    previous FRONT -> current LEFT
    previous LEFT -> current BACK
    previous BACK -> current RIGHT


    TURN LEFT 90 means:

    previous LEFT -> current FRONT
    previous FRONT -> current RIGHT
    previous RIGHT -> current BACK
    previous BACK -> current LEFT


    IMPORTANT:

    A previous BACK direction requires approximately 180 degrees of total
    rotation.

    A single 90-degree turn does NOT face the previous BACK direction.

    If a reversal of approximately 180 degrees is necessary, use two
    calibrated turns, normally two 90-degree turns.

    Reassess using available camera, ultrasonics, heading, and navigation
    context between turns when appropriate.


    For smaller turns:

    15 = small correction
    30 = moderate correction
    45 = significant diagonal direction change
    60-90 = major direction change

    Do not invent your own turn timing.

    The rover control system handles calibrated LEFT and RIGHT timing.


    ============================================================
    SPATIAL MEMORY AFTER A TURN
    ============================================================

    A turn does NOT invalidate the previous LiDAR scan.

    A turn does NOT automatically require another LiDAR scan.

    Remember WHY the rover turned.


    Example:

    Before turn:

    previous RIGHT contains an opening.

    Action:

    TURN RIGHT 90

    Afterward:

    that previous RIGHT opening should now be approximately in current
    FRONT.

    Use:

    - fresh camera
    - fresh ultrasonic readings
    - IMU / heading
    - persistent map

    to confirm that the expected path is now ahead.

    If those observations agree, MOVE into the opening.

    Do not immediately scan again merely because a turn occurred.


    For 15, 30, 45, 60, or 75 degree turns, maintain an approximate
    understanding of how previously observed geometry shifted relative to
    the rover.


    ============================================================
    EXPLORATION
    ============================================================

    The objective is to safely explore new space.

    Prefer useful unexplored paths over repeatedly returning through
    already explored space when both are safe.

    If the current forward corridor remains open:

    continue forward.

    Do not turn simply for the sake of changing direction.

    Avoid oscillation such as:

    forward -> reverse -> forward -> reverse

    or:

    left -> right -> left -> right


    Do not immediately undo the previous movement unless new information
    shows it was unsafe or unproductive.

    Use the persistent map and movement history to recognize places and
    headings already explored.


    ============================================================
    TARGET TRAVEL CORRIDOR
    ============================================================

    Judge movement safety based primarily on the actual region through
    which the rover body will travel.

    Do not treat every obstacle inside a broad directional sector as if it
    occupies the center of the rover's trajectory.

    For forward movement, determine whether there appears to be a continuous
    rover-sized route ahead.

    Examples of objects that may exist beside a safe forward path:

    - chair legs
    - desk legs
    - cabinets beside an aisle
    - wall edges
    - furniture beside a hallway

    Use LiDAR top-down geometry and camera imagery to distinguish side
    objects from obstacles directly in the travel corridor.


    ============================================================
    CORNERS AND CORRIDOR ENDS
    ============================================================

    When approaching a wall, corner, corridor ending, or apparent dead end:

    do not immediately reverse.

    Approach to a safe inspection position when possible.

    As front clearance decreases:

    reduce movement duration.

    Use:

    - camera
    - LiDAR structural memory
    - persistent map
    - ultrasonic readings

    to look for a continuation to the left or right.

    If geometry around the corner remains genuinely unknown, a new LiDAR
    scan from the closer position may be useful.

    Prefer:

    approach -> inspect -> identify opening -> TURN -> MOVE

    rather than:

    detect wall -> immediately reverse


    ============================================================
    LIDAR REUSE POLICY
    ============================================================

    LiDAR scans are expensive and slow.

    DO NOT scan after every MOVE.

    DO NOT scan after every TURN.

    DO NOT scan because a certain amount of time passed.

    DO NOT scan solely because several movements occurred.

    An older scan remains useful as structural memory.


    Continue without rescanning when:

    - the rover is still following recognizable geometry
    - camera information remains consistent
    - ultrasonic readings remain safe
    - heading changes are understood
    - the persistent map provides sufficient context
    - no important new geometric uncertainty exists


    Request a new LiDAR scan when there is a REAL geometric reason, such as:

    - no usable LiDAR scan exists
    - entering substantially new or unseen geometry
    - approaching an intersection where paths cannot be understood from
    existing information
    - encountering an obstacle that requires broader spatial understanding
    - camera and existing LiDAR/map information conflict
    - a corner hides geometry necessary for the next decision
    - current path geometry cannot be determined safely


    A new heading alone is NOT a reason to scan.

    Movement count alone is NOT a reason to scan.

    Elapsed time alone is NOT a reason to scan.


    ============================================================
    USING NEW LIDAR SCANS
    ============================================================

    When a new LiDAR scan is obtained:

    1. Read the numeric distance measurements.
    2. Examine the XY top-down image carefully.
    3. Use the other LiDAR images for additional structural information.
    4. Compare the geometry with the camera image.
    5. Add the scan to the persistent navigation map.
    6. Compare the new scan position and heading with previous map history.
    7. Decide which actual rover-sized path is useful and safe.

    Do not simply choose whichever sector has the numerically largest
    near_distance_m or average_distance_m.

    A very large distance behind the rover does NOT automatically make BACK
    the preferred exploration direction.

    Consider exploration progress and current intended route.


    ============================================================
    PERSISTENT MAP VS NEW OBSERVATIONS
    ============================================================

    When the map and current observations agree:

    use the map confidently as spatial memory.

    When they differ slightly:

    remember that dead-reckoning drift and measurement noise are possible.

    When they differ substantially:

    prefer current safety observations and consider whether a new LiDAR
    scan or future localization correction is required.

    Never intentionally drive into a fresh obstacle because the persistent
    map says the area was previously empty.


    ============================================================
    NO SAFE PATH
    ============================================================

    If no safe path is apparent:

    1. Inspect current camera information.
    2. Inspect fresh ultrasonic readings.
    3. Use recent LiDAR geometry and the persistent map.
    4. Look for a safe turn or reverse escape.
    5. If necessary, perform one useful LiDAR scan to resolve uncertainty.
    6. If no safe movement exists, use no_op and remain stationary.

    Do not enter an endless LiDAR scanning loop.


    ============================================================
    NAVIGATION MEMORY
    ============================================================

    Treat navigation state and the persistent map as memory.

    Remember:

    - current estimated pose
    - current heading
    - previous LiDAR scan poses
    - previous scan geometry
    - recent movement history
    - which directions were intentionally explored
    - why the previous turn occurred
    - where known openings were located
    - where the rover has already traveled


    Do not remember only:

    "I turned right."

    Remember:

    "I turned right because an opening was observed on the right, so after
    the turn that opening should now be approximately ahead."


    ============================================================
    DECISION DISCIPLINE
    ============================================================

    Before every physical action, internally determine:

    1. Where am I approximately on the persistent map?
    2. What is my current heading?
    3. What was the purpose of the previous movement?
    4. What does the most relevant LiDAR geometry show?
    5. What do the current camera and ultrasonics show?
    6. Is there a rover-sized travel corridor?
    7. Does this action make useful exploration progress?
    8. Is a new LiDAR scan genuinely necessary?

    Then choose ONE action.


    ============================================================
    COMMUNICATION
    ============================================================

    Before using a tool, briefly state:

    - what you observe
    - how the available sensors/map support that interpretation
    - what action you are taking

    Keep the explanation concise.

    Do not present choices to the operator.

    Do not ask:

    "Would you like me to..."
    "What should I do?"
    "Should I move?"
    "Should I scan?"
    "If you want, I can..."

    Decide and act autonomously.
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

        self.memory_depth = 50  # Number of past interactions to remember
        self.memory = [self.context_msg]
        self.model_name = "gpt-5.6-luna"  # LLM model to use

        # Persistent navigation/exploration state
        self.navigation_state = {
            "last_lidar_time": None,

            "last_lidar": {
                "zones": {}
            },

            "current_yaw_deg": None,
            "previous_yaw_deg": None,
            "yaw_at_last_lidar_deg": None,
            "heading_change_since_lidar_deg": 0.0,

            "recent_movements": [],
            "movement_segments_since_lidar": 0,

            "exploration_heading": None
        }

        self.navigation_map = NavigationMap()
        self.navigation_map_file = None

        # Connect API account to client
        self.client = openai.OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
        )

    def encode_image(self, filename):
        with open(filename, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def add_lidar_visualizations(self, lidar_views):
        """
        Send navigation-focused LiDAR images to Luna.

        Normal autonomous navigation uses:

            navigation_xy
                Height-filtered top-down geometry.

            corridor_xz
                Vertical geometry only through the rover's
                center travel corridor.

        A 3D image may exist during manual debug runs, but
        is not required for normal Luna navigation.
        """

        if not isinstance(lidar_views, dict) or not lidar_views:

            print(
                "[AI] No LiDAR visualizations available."
            )

            return

        content = [
            {
                "type": "text",
                "text": (
                    "These are navigation-focused visualizations "
                    "of the latest AEGIS LiDAR scan.\n\n"

                    "Coordinate system:\n"
                    "+X = FRONT\n"
                    "+Y = LEFT\n"
                    "+Z = UP\n"
                    "The rover is at the origin.\n\n"

                    "NAVIGATION_XY:\n"
                    "This is the main top-down navigation view. "
                    "Floor and high geometry have been removed. "
                    "Use it to understand walls, obstacles, openings, "
                    "corridors, corners, and rover-sized paths.\n\n"

                    "CORRIDOR_XZ:\n"
                    "This is NOT the entire room collapsed into a side "
                    "view. It contains only geometry near the rover's "
                    "center travel corridor. Use it to determine whether "
                    "objects, overhangs, furniture, or other geometry "
                    "occupy the vertical space through which the rover "
                    "would actually travel.\n\n"

                    "Directional obstacle_distance_m values are supporting "
                    "distance measurements only. A close return somewhere "
                    "inside a broad sector does not automatically mean the "
                    "actual rover travel corridor is blocked.\n\n"

                    "Use these images together with the camera, ultrasonic "
                    "sensors, persistent navigation map, current heading, "
                    "and recent movement history."
                )
            }
        ]

        view_labels = {
            "navigation_xy":
                "NAVIGATION_XY - TOP-DOWN NAVIGATION VIEW",

            "corridor_xz":
                "CORRIDOR_XZ - CENTER TRAVEL-CORRIDOR SLICE"
        }

        view_order = [
            "navigation_xy",
            "corridor_xz"
        ]

        images_added = 0

        for view_name in view_order:

            filename = lidar_views.get(
                view_name
            )

            if not filename:
                print(
                    f"[AI] LiDAR view missing: "
                    f"{view_name}"
                )
                continue

            if not os.path.exists(
                filename
            ):
                print(
                    "[AI] LiDAR visualization "
                    f"file not found: {filename}"
                )
                continue

            image_base64 = self.encode_image(
                filename
            )

            content.append({
                "type": "text",
                "text":
                    view_labels[view_name]
            })

            content.append({
                "type": "image_url",
                "image_url": {
                    "url":
                        "data:image/png;base64,"
                        + image_base64,

                    "detail":
                        "high"
                }
            })

            images_added += 1

        if images_added == 0:

            print(
                "[AI] No LiDAR visualization "
                "images could be loaded."
            )

            return

        self.update_memory({
            "role": "user",
            "content": content
        })

        print(
            f"[AI] Sent {images_added} "
            "LiDAR navigation images to Luna."
        )

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

        self.navigation_map_file = os.path.join(
            dump_folder,
            "navigation_map_latest.png"
        )

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
                "navigation_map": self.navigation_map.get_ai_context()
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
                    
                    lidar_filename = scanner.scan(filepath=dump_folder, visualize=True)
                    print(f"Scan saved: {lidar_filename}")

                    lidar_views = scanner.last_visualizations

                    print("[AI] LiDAR visualization files:")
                    for view_name, view_file in lidar_views.items():
                        print(f"    {view_name}: {view_file}")
                        
                    summary = self.summarize_scan_file(lidar_filename)

                    # THEN print it
                    print("\n[LIDAR DEBUG] Environment summary:")
                    print(json.dumps(summary, indent=2))
                    print()

                    self.navigation_state["last_lidar_time"] = time.time()  

                    self.navigation_state["last_lidar"] = {
                        "zones": summary.get("zones", {})
                    }

                    self.navigation_state["movement_segments_since_lidar"] = 0

                    self.navigation_state["yaw_at_last_lidar_deg"] = (
                        self.navigation_state["current_yaw_deg"]
                    )

                    self.navigation_state[
                        "heading_change_since_lidar_deg"
                    ] = 0.0

                    self.navigation_map.add_lidar_scan(
                        filename=lidar_filename,
                        summary=summary,
                        visualization_files=lidar_views
                    )

                    if self.navigation_map_file:

                        self.navigation_map.render_map_png(
                            self.navigation_map_file
                        )

                    self.memory.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps({
                            "status": "scan_saved",
                            "file_path": lidar_filename,
                            "summary": summary,
                            "navigation_map": self.navigation_map.get_ai_context()
                        })
                    })

                    self.add_lidar_visualizations(
                        lidar_views
                    )
                        
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
        #
        # Measurements are from the center of the LiDAR.
        #
        # +X = front
        # -X = back
        # +Y = left
        # -Y = right
        # --------------------------------------------------

        FRONT_LIMIT = 0.1524
        REAR_LIMIT = 0.6096
        LEFT_LIMIT = 0.4572
        RIGHT_LIMIT = 0.4572

        ROVER_MIN_Z = -0.30
        ROVER_MAX_Z = 0.40

        # --------------------------------------------------
        # NAVIGATION HEIGHT FILTER
        #
        # Z is ONLY used to remove floor and high geometry.
        #
        # Navigation distance itself is calculated using
        # only X and Y.
        # --------------------------------------------------

        MIN_NAV_Z = -0.27
        MAX_NAV_Z = 1.00

        # --------------------------------------------------
        # OBSTACLE DISTANCE
        #
        # Use the 5th percentile rather than the single
        # closest point.
        #
        # This makes the measurement less sensitive to
        # one random/noisy LiDAR return.
        # --------------------------------------------------

        OBSTACLE_PERCENTILE = 5.0

        # --------------------------------------------------
        # READ CLOUD
        #
        # Stream line-by-line so the entire million-point
        # file is not loaded into RAM at once.
        # --------------------------------------------------

        with open(filename, "r") as f:

            for line in f:

                values = line.split()

                if len(values) < 3:
                    continue

                try:
                    x = float(values[0])
                    y = float(values[1])
                    z = float(values[2])

                except ValueError:
                    continue

                # ------------------------------------------
                # Remove floor and geometry that is too high
                # to matter for normal rover navigation.
                # ------------------------------------------

                if z <= MIN_NAV_Z or z >= MAX_NAV_Z:
                    continue

                # ------------------------------------------
                # Ignore points belonging to the rover body.
                # ------------------------------------------

                if (
                    -REAR_LIMIT <= x <= FRONT_LIMIT
                    and
                    -RIGHT_LIMIT <= y <= LEFT_LIMIT
                    and
                    ROVER_MIN_Z <= z <= ROVER_MAX_Z
                ):
                    continue

                # ------------------------------------------
                # PURE X/Y NAVIGATION DISTANCE
                # ------------------------------------------

                distance = math.hypot(
                    x,
                    y
                )

                if distance <= 0:
                    continue

                total_points += 1

                # ------------------------------------------
                # Determine direction around rover
                # ------------------------------------------

                angle = math.degrees(
                    math.atan2(y, x)
                )

                if angle < 0:
                    angle += 360

                if angle >= 337.5 or angle < 22.5:
                    zone = "front"

                elif 22.5 <= angle < 67.5:
                    zone = "front_left"

                elif 67.5 <= angle < 112.5:
                    zone = "left"

                elif 112.5 <= angle < 157.5:
                    zone = "back_left"

                elif 157.5 <= angle < 202.5:
                    zone = "back"

                elif 202.5 <= angle < 247.5:
                    zone = "back_right"

                elif 247.5 <= angle < 292.5:
                    zone = "right"

                else:
                    zone = "front_right"

                zones[zone].append(
                    distance
                )

        # --------------------------------------------------
        # EMPTY SCAN
        # --------------------------------------------------

        if total_points == 0:

            return {
                "status": "empty_scan",
                "scan_ok": False
            }

        # --------------------------------------------------
        # DISTANCE MEASUREMENTS ONLY
        #
        # No:
        # - blocked
        # - caution
        # - clear
        # - average distance
        # - minimum distance
        # - best direction
        #
        # Luna gets ONE useful obstacle-distance estimate
        # for each direction.
        # --------------------------------------------------

        zone_summary = {}

        for zone_name, distances in zones.items():

            if not distances:

                zone_summary[zone_name] = {
                    "obstacle_distance_m": None
                }

                continue

            obstacle_distance = float(
                np.percentile(
                    distances,
                    OBSTACLE_PERCENTILE
                )
            )

            zone_summary[zone_name] = {
                "obstacle_distance_m":
                    round(
                        obstacle_distance,
                        2
                    )
            }

        # --------------------------------------------------
        # RETURN
        # --------------------------------------------------

        return {
            "status": "scan_saved",
            "scan_ok": True,

            "navigation_points":
                total_points,

            "file_path":
                filename,

            "zones":
                zone_summary
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
                "turn_degrees": command.get("turn_degrees"),
                "duration_s": result.get("movement_duration_s")
            }

            op = command.get("op")

            if op == "MOVE":
                self.navigation_map.apply_move(
                    duration_s=result.get("movement_duration_s"),
                    spd=command.get("spd", 0.0)
                )

            elif op == "TURN":
                self.navigation_map.apply_turn(
                    turn_dir=command.get("turn_dir"),
                    turn_degrees=command.get("turn_degrees")
                )

            if self.navigation_map_file:

                self.navigation_map.render_map_png(
                    self.navigation_map_file
                )
                
            self.navigation_state["recent_movements"].append(
                movement_record
            )

            # Only keep the most recent 8 movements
            self.navigation_state["recent_movements"] = (
                self.navigation_state["recent_movements"][-20:]
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