# LLM Pilot class for rover. Provides high-level autopilot functions and holds

# AEGIS Senior Design, Created 10/22/2025

import os
import json
import time
import base64

import openai

from rover.navigation_map import NavigationMap


# Select the new 3D LiDAR scene for Luna.
# Disabled until offline payload testing passes.
ENABLE_LIDAR_QUERY_TOOL = False
ENABLE_AUTONOMOUS_MOVEMENT = True
ENABLE_LIDAR_MAP_INGEST = True

class Autopilot:

    system_context = """
    You are Luna, the autonomous navigation controller for AEGIS.

    Your job is to explore the environment autonomously using the
    available scan_environment, capture_camera, move_rover and no_op tools.
    You make the navigation decisions. Do not wait for manual instructions.

    AEGIS uses an existing continuous autonomous control loop.
    When observations support a feasible movement, select a useful,
    appropriately sized MOVE or TURN using move_rover.
    Reassess the environment between movement decisions.
    Do not repeatedly choose no_op solely because calibration is approximate.

    LIDAR COORDINATES:
    Raw STL27L frame: +Y FRONT, +X RIGHT, -X LEFT, -Y BACK, +Z UP.
    NavigationMap frame: +X FRONT, +Y LEFT.
    The implemented provisional conversion is map_x=lidar_y,
    map_y=-lidar_x before applying the rover heading.
    Physical mounting calibration and pose estimates remain approximate.
    Never confuse the two frames.

    FULL-360 SCAN REORIENTATION:
    The STL27L scan measures all directions around the rover.
    A turn IN PLACE does not invalidate the existing full scan.
    The saved four directional images are labeled at SCAN TIME.
    They do not automatically become current-facing images.
    Use the estimated IMU heading change to reinterpret them; IMU readings may be delayed or inaccurate.

    Positive heading change means LEFT.
    If the rover turned LEFT by D degrees, its current FRONT
    corresponds to bearing -D degrees (clockwise-positive)
    in the previous LiDAR scan coordinate system.
    The old scan FRONT now lies D degrees RIGHT of current FRONT.
    For RIGHT turns, reverse these signs.

    The live navigation context gives an estimated rotation, not a guaranteed physical angle.
    Do not call the old scan 'heading-misaligned' or rescan solely
    because the rover turned. Reuse its full 360-degree geometry.
    Turning changes orientation, not the measured room geometry.
    A genuinely new sensor position or newly revealed area can
    justify another scan; heading change alone cannot.

    LIDAR INPUTS:
    view_01_right: 105-degree 3D perspective facing RIGHT.
    view_02_front: 105-degree 3D perspective facing FRONT.
    view_03_left: 105-degree 3D perspective facing LEFT.
    view_04_back: 105-degree 3D perspective facing BACK.
    view_05_roofcut: rich top-down geometry, Z -0.49 to +1.50 m.
    view_06_navigation_dark: scan-local rover-oriented occupancy view.
    view_07_navigation_white: NavigationMap occupancy representation.
    perception_layers: height-aware measured LiDAR returns.

    geometry_regions.json contains connected surface groups and
    eight-direction individual-return measurements.
    Region centers are not full region boundaries.
    Missing measurements are UNKNOWN, not confirmed clearance.
    Near-origin measurements can include rover-body returns or noise.
    Use measured geometry rather than treating render colors as objects.

    CAMERA AND SENSORS:
    Camera observations help identify surroundings and openings.
    The front-center ultrasonic has produced plausible readings.
    Rear and lidar-mount ultrasonic channels have produced invalid values.
    Side ultrasonic readings may also be unreliable.
    Ignore implausible sensor values; do not interpret them as open space.
    Current sensor readings and fresh observations take priority over
    outdated scan interpretations.

    SCAN REUSE POLICY (CRITICAL):
    A full 3D LiDAR scan takes about 80 seconds and is expensive.
    Treat its geometry as reusable spatial memory, not a one-step input.

    Reuse each scan over MULTIPLE movements and turns.
    Account for the rover's estimated pose, heading and movement history.
    Use the persistent map and camera to update the scene between scans.
    A turn does not automatically invalidate previous LiDAR geometry.

    DO NOT request another full scan merely because:
    - One movement finished, including a short forward movement.
    - The rover turned or its heading changed.
    - A camera image changed slightly.
    - The scan has aged while the rover remained near the same pose.
    - An ultrasonic sensor reports an inconsistent reading.
    - You are uncertain but have not exhausted existing observations.

    Request a new full scan ONLY when it adds essential new geometry:
    - Reaching a corner, doorway, wall or previously occluded area.
    - Entering substantially new space not covered by the old scan.
    - A major environmental change invalidates previous geometry.
    - Existing geometry cannot distinguish viable movement options.

    Before scanning, explicitly ask:
    "What information is missing that the existing scan cannot provide?"
    If there is no specific answer, DO NOT rescan.
    A repeated scan from an unchanged pose usually adds little information.
    If no movement is supportable, remain stationary instead of
    repeatedly requesting redundant scans.

    Ultrasonic measurements are secondary evidence.
    Cross-check plausible readings with LiDAR geometry and camera.
    A reading around 0.9 m does not automatically mean STOP or RESCAN.
    Do not assume a faulty reading means the path is clear.

    EFFICIENT EXPLORATION:
    Make meaningful progress along observed, sufficiently open corridors.
    Avoid unnecessary tiny movements when there is room to travel farther.
    Prefer approximately 3-5 second movement segments when available
    verified corridor distance and stopping margin support them.
    Use shorter segments near objects, corners or uncertain openings.
    Never drive beyond the observed clearance or rely on missing
    LiDAR returns as proof of free space.
    Choose productive turns when they offer a better observed route.
    Reuse the existing map and spatial memory after each action.

    STRAIGHT CORRIDOR EXPLORATION:
    Prefer useful forward progress through sufficiently wide,
    observed corridors instead of unnecessary turns.

    Maintain an exploration objective across multiple decisions.
    Continue toward that objective while the route remains supported
    by LiDAR geometry, recent camera views, and current telemetry.

    In spacious corridors, prefer successive supervised forward
    segments of about 4-6 seconds when clearance supports them.
    Reassess after each segment and continue straight when appropriate.
    Shorten segments near walls, furniture, corners, or uncertainty.

    A completed movement is not a reason to rescan.
    Reuse the latest geometry and update position and heading.
    Only rescan when existing observations cannot support the route.

    A gap in a camera image is not proof of rover-sized clearance.
    Furniture can obstruct the rover above ultrasonic sensor height.
    If boxed in, prioritize finding a supported escape direction
    instead of repeatedly commanding forward movement.

    Actual IMU turn results take priority over requested angles.
    After a significant turn overshoot, reassess the new heading
    before continuing.

    SIDE-OPENING EXPLORATION:
    When FRONT and BACK are constrained, do not assume the
    rover is trapped. Evaluate LEFT and RIGHT using the full
    360-degree LiDAR scan and stored spatial geometry.

    Front-left and front-right ultrasonics face generally
    forward. They are NOT measurements of the entire left
    or right side of the rover.

    The forward camera cannot see all lateral escape routes.
    A wall ahead does not mean there is no route to either side.

    If LiDAR geometry supports an adequately wide side opening
    and the rover has room to rotate, prefer turning toward that
    opening over repeatedly choosing no_op or rescanning.

    Use measured turns and camera observations to reassess
    the route. Do not assume an incorrect or delayed IMU
    measurement represents the actual physical rotation.

    One full 360-degree scan remains useful after a turn.
    Reuse previously observed side geometry whenever possible.
    Scan again only for genuinely new or unresolved geometry.

    The objective is productive exploration, not remaining
    stationary merely because forward motion is obstructed.

    HALLWAY CRUISE V2:
    Prioritize efficient, continuous exploration of large spaces.

    In a sufficiently open, observed hallway, prefer forward MOVE
    segments of 6 to 8 seconds at moderate speed around 0.30-0.35.
    Choose duration based on observed clearance and stopping margin.
    Never use longer movement merely because a hallway looks open.

    Preserve the current route objective across decisions.
    After each movement, update estimated pose and recent history.
    If the corridor remains sufficiently clear, CONTINUE STRAIGHT
    rather than turning unnecessarily or requesting another scan.

    A hallway should be explored through consecutive supervised
    forward movements until approaching a corner, wall, doorway
    or genuine change in its geometry.

    Avoid left-right-left-right oscillation without progress.
    When forward movement is obstructed, use existing 360-degree
    LiDAR geometry to consider side openings.
    Turn toward a supported alternative route, then evaluate
    forward travel along that new heading.

    Reuse the same LiDAR scan across multiple movements and turns.
    Turning alone is not a reason to rescan.
    A new scan is useful when significant new geometry becomes
    necessary for navigation.

    Keep physical turn uncertainty in mind: IMU yaw may lag.
    Do not assume the requested turn and actual turn are identical.
    Do not force movement into furniture, walls or unknown clearance.

    AUTONOMOUS MOVEMENT:
    AEGIS is a skid-steer rover; it cannot translate sideways.
    Choose one action at a time.
    Available turn angles: 15, 30, 45, 60, 75, 90 degrees.
    Use short, conservative movements near nearby structures.
    Consider rover footprint, observed obstacles, heading, and scan age.
    Do not move into an observed obstacle or an unsupported direction.
    Existing UART collision checks and emergency stop take priority.
    If observations do not support movement, use no_op or gather data.
    Avoid repeating the same explanation or unnecessary scans.

    You are the autonomous controller, not just the scan interpreter.
    The separate LiDAR analysis describes geometry; it does not itself
    issue motor commands. You choose subsequent navigation actions
    through the existing tool interface.

    Briefly describe observations and the next action.
    """
    context_msg: dict[str, str] = {"role": "system", "content": system_context}

    # Define available tools for the rover autopilot
    aegis_tools = [
        {
            "type": "function",
            "function": {
                "name": "scan_environment",
                "description": (
                    "Capture a high-resolution LiDAR scan when new geometric "
                    "information is genuinely needed. Do not scan simply because "
                    "time has passed, because the rover moved, or because the rover "
                    "turned. Reuse previous LiDAR geometry, camera observations, "
                    "ultrasonics, heading history, and the persistent map whenever "
                    "they provide enough information for safe navigation."
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
                "name": "query_lidar_geometry",
                "description": (
                    "Inspect actual measured 3D LiDAR points from the "
                    "latest processed scan in a selected direction and "
                    "height layer. Returns measured distances and XYZ "
                    "evidence, not verified collision clearance. "
                    "Angles are relative to the rover at SCAN TIME. "
                    "This tool never moves the rover."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "angle_deg": {
                            "type": "number",
                            "minimum": -180,
                            "maximum": 180,
                            "description": (
                                "0=front, +90=left, -90=right, "
                                "180=back, relative to scan pose."
                            )
                        },
                        "half_width_deg": {
                            "type": "number",
                            "exclusiveMinimum": 0,
                            "maximum": 20,
                            "description": (
                                "Degrees to include on each side "
                                "of the requested angle."
                            )
                        },
                        "layer": {
                            "type": "string",
                            "enum": [
                                "floor_candidate",
                                "low",
                                "mid",
                                "high",
                                "overhead"
                            ]
                        }
                    },
                    "required": [
                        "angle_deg",
                        "half_width_deg",
                        "layer"
                    ],
                    "additionalProperties": False
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

            "current_yaw_deg": None,

            "yaw_at_last_lidar_deg": None,
            "heading_change_since_lidar_deg": 0.0,

            "recent_movements": [],

           "defer_next_scan_after_turn": False,

        }

        self.navigation_map = NavigationMap()
        self.latest_lidar_geometry_file = None

        # Latest direct raw LiDAR sample sent to Luna.
        #
        # Only ONE raw sample is kept in model memory because each
        # ~16.8k-point sample uses roughly 400k input tokens.
        self.latest_raw_lidar_message = None

        # Connect API account to client
        self.client = openai.OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
        )



    def add_lidar_evidence(self, evidence):
        """Luna analyzes 4 images once per new LiDAR scan."""

        from pathlib import Path

        summary_file = Path(
            evidence["summary_file"]
        )

        summary = json.loads(
            summary_file.read_text(
                encoding="utf-8"
            )
        )

        image_paths = evidence["views"]

        prompt = (
            "Analyze one completed AEGIS LiDAR scan. "
            "Four directional images are depth-colored "
            "LiDAR point clouds, not photographs. "
            "Other images provide rich roof-cut geometry, "
            "two navigation maps, and perception layers. "
            "Empty-looking space is UNKNOWN, not verified clear. "
            "Provisional sensor axes: +Y front, +X right, "
            "-X left, -Y back. "
            "The legacy navigation map uses +X front, "
            "and the transform is not yet verified. "
            "Orange means near returns, not necessarily "
            "the rover. Missing returns mean UNKNOWN. "
            "Describe visible structures and compare "
            "them to measured regions. "
            "Do not invent object identities, infer "
            "safe corridors, or suggest motor commands. "
            "Be concise and state uncertainties.\n\n"
        )

        content = [{
            "type": "text",
            "text": (
                prompt
                + json.dumps(
                    summary,
                    separators=(",", ":")
                )
            )
        }]

        for name in (
            "front",
            "right",
            "left",
            "back",
            "roofcut_topdown",
            "map_topdown",
            "map_white",
            "perception_layers"
        ):
            image_path = image_paths[name]

            if not os.path.isfile(image_path):
                raise FileNotFoundError(
                    image_path
                )

            content.extend([
                {
                    "type": "text",
                    "text": {
                        "front": "3D LiDAR FRONT, 105-degree view",
                        "right": "3D LiDAR RIGHT, 105-degree view",
                        "left": "3D LiDAR LEFT, 105-degree view",
                        "back": "3D LiDAR BACK, 105-degree view",
                        "roofcut_topdown": (
                            "All-return roof-cut top-down geometry, "
                            "sensor-relative Z -0.49 to +1.50 m."
                        ),
                        "map_topdown": (
                            "Dark scan-local navigation map, "
                            "front up, right right, 10 cm cells."
                        ),
                        "map_white": (
                            "White persistent navigation map, "
                            "including available pose history."
                        ),
                        "perception_layers": (
                            "Height-aware 3D LiDAR perception layers."
                        )
                    }[name]
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": (
                            "data:image/png;base64,"
                            + self.encode_image(
                                image_path
                            )
                        ),
                        "detail": "high"
                    }
                }
            ])

        # One analysis call, WITHOUT rover tools.
        response = (
            self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Analyze LiDAR evidence read-only. "
                            "Never control hardware."
                        )
                    },
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                max_completion_tokens=800,
                reasoning_effort="none"
            )
        )

        description = (
            response.choices[0].message.content
            or ""
        )

        if not description.strip():
            raise RuntimeError(
                "Luna returned no LiDAR description"
            )

        source = Path(
            evidence["source_cloud"]
        )

        saved_description = (
            summary_file.parent /
            f"{source.stem}_luna_vision.txt"
        )

        saved_description.write_text(
            description,
            encoding="utf-8"
        )

        # Normal decisions receive text, not the
        # four encoded images over and over.
        message = {
            "role": "user",
            "content": (
                "LATEST COMPLETED LIDAR OBSERVATION. "
                "FRAME NOT YET CALIBRATED. "
                "NOT MOVEMENT AUTHORIZATION.\n"
                + json.dumps(
                    summary,
                    separators=(",", ":")
                )
                + "\nLUNA VISUAL INTERPRETATION:\n"
                + description
            )
        }

        previous = getattr(
            self,
            "latest_lidar_evidence_message",
            None
        )

        if previous is not None:
            self.memory = [
                item
                for item in self.memory
                if item is not previous
            ]

        self.update_memory(message)

        self.latest_lidar_evidence_message = (
            message
        )

        print(
            "[AI] Luna analyzed eight LiDAR images."
        )

        print(
            "[AI] Description saved:",
            saved_description
        )



    def encode_image(self, filename):
        with open(filename, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")








    def add_camera_observation(
        self,
        filename,
        context="post_movement",
        detail="low"
    ):
        """
        Add a camera image to Luna's navigation memory.

        context:
            post_movement
            post_lidar
            requested
        """

        if not filename or not os.path.exists(filename):
            print(
                "[AI] Camera observation file unavailable."
            )
            return

        context_text = {
            "post_movement": (
                "Fresh camera image captured immediately after the "
                "rover's most recent movement. Use it to verify the "
                "current travel corridor, obstacles, openings, and "
                "scene continuity."
            ),

            "post_lidar": (
                "Fresh camera image captured immediately after the "
                "latest LiDAR scan. Compare it with the eight-view "
                "LiDAR package, ultrasonic readings, and the persistent "
                "map to interpret the current environment."
            ),

            "requested": (
                "Fresh camera image requested for navigation. "
                "Use it with the current LiDAR information, "
                "ultrasonics, map, heading, and movement history."
            )
        }.get(
            context,
            "Fresh rover camera image for navigation."
        )

        image_base64 = self.encode_image(
            filename
        )

        self.update_memory({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": context_text
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url":
                            "data:image/jpeg;base64,"
                            + image_base64,
                        "detail":
                            detail
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
    

    def _query_lidar_geometry(
        self, angle_deg, half_width_deg=2.0, layer="mid"
    ):
        """Inspect measured XYZ points without controlling movement."""
        import math
        import numpy as np
        from rover.aegis_lidar_scene import query_sector

        allowed_layers = {
            "floor_candidate", "low", "mid", "high", "overhead"
        }

        try:
            angle = float(angle_deg)
            half_width = float(half_width_deg)
        except (TypeError, ValueError):
            return {
                "status": "invalid_query",
                "reason": "Angle and width must be numbers"
            }

        if (
            not math.isfinite(angle)
            or not -180 <= angle <= 180
            or not math.isfinite(half_width)
            or not 0 < half_width <= 20
            or layer not in allowed_layers
        ):
            return {
                "status": "invalid_query",
                "reason": "Invalid angle, half-width, or layer"
            }

        filename = getattr(
            self, "latest_lidar_geometry_file", None
        )

        if not filename or not os.path.isfile(filename):
            return {
                "status": "no_geometry",
                "reason": "No processed full scan available"
            }

        try:
            with np.load(filename, allow_pickle=False) as npz:
                points = npz["points_xyz_intensity"]
                floor_z = float(npz["floor_z_sensor_frame"])

            measurement = query_sector(
                points, floor_z, angle, half_width, layer
            )
        except (OSError, ValueError, KeyError) as exc:
            return {
                "status": "query_error",
                "reason": str(exc)
            }

        state = getattr(self, "navigation_state", {}) or {}
        scan_time = state.get("last_lidar_time")

        age = (
            round(max(0.0, time.time() - scan_time), 1)
            if isinstance(scan_time, (int, float))
            else None
        )

        return {
            "status": "ok",
            "frame": "rover-local at scan time",
            "scan_age_s": age,
            "heading_change_since_scan_deg": state.get(
                "heading_change_since_lidar_deg"
            ),
            "direction_deg": angle,
            "half_width_deg": half_width,
            "height_layer": layer,
            "floor_z_estimated_m": floor_z,
            "measurement": measurement,
            "warning": (
                "Measured returns only. Missing returns are "
                "UNKNOWN, not clear. No movement authorization."
            )
        }



    def _handle_lidar_scan(self, call, scanner, ugv_cam, dump_folder, get_current_telemetry):

        if self.navigation_state.get(
            "defer_next_scan_after_turn",
            False
        ):
            print(
                "[AI] LiDAR scan deferred once after TURN. "
                "Reuse existing geometry, camera, ultrasonics, "
                "heading, and map first."
            )

            # Only defer ONE scan request.
            self.navigation_state[
                "defer_next_scan_after_turn"
            ] = False

            self.update_memory({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps({
                    "status": "scan_deferred",
                    "reason": (
                        "One immediate post-turn LiDAR request was deferred. "
                        "First reconsider the existing LiDAR geometry, "
                        "fresh camera image, ultrasonics, heading, and map. "
                        "If important geometric uncertainty still remains "
                        "after reconsideration, a subsequent scan request "
                        "is permitted."
                    )
                })
            })

            return
    
        print("Starting LiDAR scan...")

        from rover.aegis_scan_package import (
            build_scan_package
        )

        lidar_filename = scanner.scan(
            filepath=dump_folder,
            visualize=True
        )

        if (
            not lidar_filename
            or not os.path.isfile(lidar_filename)
        ):
            raise RuntimeError(
                "LiDAR scan did not save correctly"
            )

        evidence = build_scan_package(
            lidar_filename
        )

        if "perception_layers" not in evidence["views"]:
            raise RuntimeError(
                "Fresh perception layer missing from scan package"
            )

        lidar_views = {
            **evidence["views"],
            "measured_regions": evidence["summary_file"]
        }

        # The legacy NPZ query input is no longer
        # generated by this experimental path.
        self.latest_lidar_geometry_file = None

        print(
            "[AI] New measured regions:",
            evidence["region_count"]
        )

        print(
            "[AI] Perception preparation time:",
            evidence["render_and_group_seconds"]
        )


        print(
            "[AI] LiDAR input files:"
        )

        for view_name, view_file in lidar_views.items():
            print(
                f"    {view_name}: {view_file}"
            )

        summary = None

        print()

        self.navigation_state[
            "last_lidar_time"
        ] = time.time()

        self.navigation_state[
            "yaw_at_last_lidar_deg"
        ] = (
            self.navigation_state[
                "current_yaw_deg"
            ]
        )

        self.navigation_state[
            "heading_change_since_lidar_deg"
        ] = 0.0

        # Pause incompatible map ingestion until calibrated.
        if ENABLE_LIDAR_MAP_INGEST:
            self.navigation_map.add_lidar_scan(
                filename=lidar_filename,
                summary=summary,
                visualization_files=lidar_views,
                scan_frame="lidar_provisional"
            )
            if self.navigation_map_file:
                self.navigation_map.render_map_png(
                    self.navigation_map_file
                )

            self.navigation_map.render_map_png(
                evidence["views"]["map_white"]
            )
        else:
            print(
                "[MAP] LiDAR map ingest paused: "
                "coordinate frames unverified."
            )

        self.update_memory({
            "role": "tool",
            "tool_call_id": call.id,
            "content": json.dumps({
                "status":
                    "scan_saved",

                "file_path":
                    lidar_filename,

                "lidar_geometry":
                    (
                        "This scan is provided as four depth-colored 3D LiDAR views "
                        "and measured connected-surface regions. "
                        "The sensor frame is provisional (+Y front, +X right), "
                        "and the legacy navigation map is not synchronized. "
                        "These observations inform the autonomous decision loop; "
                        "the existing movement checks still apply."
                    ),

                "navigation_map":
                    self.navigation_map.get_ai_context()
            })
        })

        self.add_lidar_evidence(evidence)

        print(
            "[AI] GPT requested LiDAR scan. "
            "Scan complete and summarized. "
            f"for {call.id}"
        )

        # ------------------------------------------
        # AUTOMATIC CAMERA CAPTURE AFTER LIDAR
        # ------------------------------------------

        if (
            ugv_cam is not None
            and ugv_cam.connected
        ):

            print(
                "[AI] Capturing camera image "
                "after LiDAR scan."
            )

            camera_filename = (
                ugv_cam.capture_image(
                    filepath=dump_folder
                )
            )

            if camera_filename is not None:

                print(
                    f"[AI] Camera image saved: "
                    f"{camera_filename}"
                )

                self.add_camera_observation(
                    camera_filename,
                    context="post_lidar",
                    detail="high"
                )

                print(
                    "[AI] Fresh post-LiDAR camera image "
                    "added to navigation memory."
                )

            else:

                print(
                    "[AI] Camera capture failed "
                    "after LiDAR scan."
                )

        else:

            print(
                "[AI] Camera not connected, "
                "skipping capture after LiDAR scan."
            )

        # ------------------------------------------
        # FRESH SENSOR STATE
        # ------------------------------------------

        fresh_telemetry = (
            get_current_telemetry()
        )

        if isinstance(
            fresh_telemetry,
            dict
        ):

            self.update_navigation_state_from_telemetry(
                fresh_telemetry
            )

            fresh_sensor_summary = (
                self.summarize_telemetry(
                    fresh_telemetry
                )
            )

            self.update_memory({
                "role": "user",
                "content": json.dumps({
                    "fresh_sensor_summary":
                        fresh_sensor_summary,

                    "current_yaw_deg":
                        self.navigation_state[
                            "current_yaw_deg"
                        ],

                    "observation":
                        (
                            "Fresh sensor state immediately "
                            "after the LiDAR scan and camera "
                            "capture."
                        )
                })
            })

            print(
                "[AI] Updated memory with fresh "
                "post-LiDAR sensor state."
            )

        else:

            print(
                "[AI] Failed to retrieve fresh telemetry "
                "after LiDAR scan."
            )


    def _handle_camera_capture(
        self,
        call,
        ugv_cam,
        dump_folder
    ):
        """
        Handle an explicit camera request from Luna.
        """

        print(
            "[AI] GPT requested camera capture."
        )

        # ------------------------------------------
        # CAMERA AVAILABLE?
        # ------------------------------------------

        if (
            ugv_cam is None
            or not ugv_cam.connected
        ):

            self.update_memory({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps({
                    "status": "failed",
                    "reason": "Camera is not connected"
                })
            })

            return

        # ------------------------------------------
        # CAPTURE IMAGE
        # ------------------------------------------

        filename = ugv_cam.capture_image(
            filepath=dump_folder
        )

        if filename is None:

            self.update_memory({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps({
                    "status": "failed",
                    "reason": "Camera capture failed"
                })
            })

            return

        print(
            f"[AI] Camera image saved: {filename}"
        )

        # ------------------------------------------
        # TOOL RESULT
        # ------------------------------------------

        self.update_memory({
            "role": "tool",
            "tool_call_id": call.id,
            "content": json.dumps({
                "status": "image_saved",
                "file_path": filename
            })
        })

        # ------------------------------------------
        # SEND IMAGE TO LUNA
        # ------------------------------------------

        self.add_camera_observation(
            filename,
            context="requested",
            detail="low"
        )

        print(
            "[AI] GPT requested camera capture. "
            f"Image saved and sent to GPT for {call.id}"
        )


    def _handle_lidar_geometry_query(self, call):
        """Handle a read-only LiDAR measurement request."""

        try:
            args = json.loads(
                call.function.arguments or "{}"
            )

            if not isinstance(args, dict):
                raise ValueError("Expected a JSON object")

            result = self._query_lidar_geometry(
                angle_deg=args.get("angle_deg"),
                half_width_deg=args.get(
                    "half_width_deg", 2.0
                ),
                layer=args.get("layer", "mid")
            )

        except (ValueError, TypeError) as exc:
            result = {
                "status": "invalid_query",
                "reason": str(exc)
            }

        self.update_memory({
            "role": "tool",
            "tool_call_id": call.id,
            "content": json.dumps(result)
        })


    def _handle_unknown_tool(
        self,
        call,
        name
    ):
        """
        Record an unsupported tool request.
        """

        print(
            f"[ERR] GPT requested unknown tool: {name}."
        )

        self.update_memory({
            "role": "tool",
            "tool_call_id": call.id,
            "content": json.dumps({
                "status": "failed",
                "reason": f"Unknown tool: {name}"
            })
        })


    def _build_navigation_context(
        self,
        telemetry
    ):
        """
        Build the compact navigation state sent to Luna.

        Full rover telemetry is still recorded elsewhere.
        Luna receives only navigation-relevant information.
        """

        self.update_navigation_state_from_telemetry(
            telemetry
        )

        sensor_summary = self.summarize_telemetry(
            telemetry
        )

        navigation_state = dict(
            self.navigation_state
        )

        delta = navigation_state.get(
            "heading_change_since_lidar_deg"
        )
        if isinstance(delta, (int, float)):
            navigation_state["lidar_heading_reuse"] = {
                "heading_change_left_positive_deg": delta,
                "current_front_in_old_scan_clockwise_deg":
                    round(-delta, 1),
                "old_scan_front_from_current_clockwise_deg":
                    round(delta, 1),
                "full_360_scan_still_usable_after_turn": True
            }

        last_lidar_time = navigation_state.pop(
            "last_lidar_time",
            None
        )

        if last_lidar_time is None:

            navigation_state[
                "last_lidar_age_s"
            ] = None

        else:

            navigation_state[
                "last_lidar_age_s"
            ] = round(
                time.time()
                - last_lidar_time,
                1
            )

        return {
            "sensor_summary":
                sensor_summary,

            "navigation_state":
                navigation_state,

            "navigation_map":
                self.navigation_map.get_ai_context()
        }

    def _request_model_action(self):
        """
        Ask the current navigation model for the next tool action.

        Keeping the model request in one place will make it easier
        to replace the cloud model with a local model later.
        """

        available_tools = [
            tool for tool in Autopilot.aegis_tools
            if (
                (ENABLE_LIDAR_QUERY_TOOL or tool["function"]["name"] != "query_lidar_geometry")
                and (ENABLE_AUTONOMOUS_MOVEMENT or tool["function"]["name"] != "move_rover")
            )
        ]

        return self.client.chat.completions.create(
            model=self.model_name,
            messages=self.memory,
            tools=available_tools,
            tool_choice="required",
            parallel_tool_calls=False,
            reasoning_effort="none"
        )


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

        navigation_context = (
            self._build_navigation_context(
                telemetry
            )
        )

        self.update_memory({
            "role": "user",
            "content": json.dumps(
                navigation_context
            )
        })

        MAX_TOOL_STEPS = 10

        for step in range(MAX_TOOL_STEPS):

            response = self._request_model_action()

            message = response.choices[0].message

            self.update_memory(message)

            if message.content:
                print(message.content)

            if not message.tool_calls:
                return []

            for call in message.tool_calls:

                name = call.function.name

                # AEGIS GEMMA LIVE SHADOW HOOK V1
                # Prediction and logging only; never executes movement.
                if name in ("move_rover", "no_op"):
                    try:
                        from rover.local_ai import submit_shadow

                        latest = get_current_telemetry()
                        if not isinstance(latest, dict):
                            latest = telemetry

                        shadow_context = (
                            self._build_navigation_context(latest)
                        )

                        try:
                            luna_args = json.loads(
                                call.function.arguments or "{}"
                            )
                        except (ValueError, TypeError):
                            luna_args = {
                                "raw": str(call.function.arguments)
                            }

                        submit_shadow(
                            dump_folder,
                            shadow_context,
                            {
                                "tool": name,
                                "arguments": luna_args
                            }
                        )
                    except Exception as exc:
                        print(
                            "[GEMMA SHADOW] Observer skipped:",
                            str(exc)[:200]
                        )

                print(
                    f"[AI] GPT requested tool: {name} "
                    f"(id: {call.id})"
                )
                    # ==========================================
                    # LIDAR
                    # ==========================================

                if name == "scan_environment":

                    self._handle_lidar_scan(
                        call=call,
                        scanner=scanner,
                        ugv_cam=ugv_cam,
                        dump_folder=dump_folder,
                        get_current_telemetry=get_current_telemetry
                    )

                    continue


                    # ==========================================
                    # LIDAR GEOMETRY QUERY
                    # ==========================================

                elif name == "query_lidar_geometry":

                    self._handle_lidar_geometry_query(
                        call
                    )

                    continue

                    # ==========================================
                    # CAMERA
                    # ==========================================
                
                elif name == "capture_camera":

                    self._handle_camera_capture(
                        call=call,
                        ugv_cam=ugv_cam,
                        dump_folder=dump_folder
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
                    self._handle_unknown_tool(
                        call=call,
                        name=name
                    )

        print(
            "[AI] GPT did not request any actionable tools. "
            "Remaining stationary and waiting for new telemetry."
        )

        return []


    def add_tool_result(self, tool_call_id, result):

        # ------------------------------------------
        # RECORD COMPLETED MOVEMENT
        # ------------------------------------------

        if (
            isinstance(result, dict)
            and result.get("status") in ("completed", "partial")
            and isinstance(result.get("command"), dict)
        ):
            command = result["command"]

            movement_record = {
                "op": command.get("op"),
                "spd": command.get("spd"),
                "turn_dir": command.get("turn_dir"),
                "requested_turn_degrees": command.get("turn_degrees"),
                "actual_turn_degrees": result.get("actual_turn_degrees"),
                "duration_s": result.get("movement_duration_s")
            }

            op = command.get("op")

            if op == "MOVE":
                self.navigation_map.apply_move(
                    duration_s=result.get("movement_duration_s"),
                    spd=command.get("spd", 0.0)
                )

            elif op == "TURN":
                actual_turn_deg = result.get(
                    "actual_turn_degrees"
                )

                # Normal IMU-controlled turns use the measured angle.
                # The requested angle is used only if no measured angle
                # exists, such as the old timed fallback mode.
                if actual_turn_deg is None:
                    actual_turn_deg = command.get(
                        "turn_degrees"
                    )

                self.navigation_map.apply_turn(
                    turn_dir=command.get("turn_dir"),
                    turn_degrees=actual_turn_deg
                )

                self.navigation_state[
                    "defer_next_scan_after_turn"
                ] = True

            if self.navigation_map_file:

                self.navigation_map.render_map_png(
                    self.navigation_map_file
                )
                
            self.navigation_state["recent_movements"].append(
                movement_record
            )

            # Only keep the most recent 20 movements
            self.navigation_state["recent_movements"] = (
                self.navigation_state["recent_movements"][-20:]
            )


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