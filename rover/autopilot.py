# LLM Pilot class for rover. Provides high-level autopilot functions and holds

# AEGIS Senior Design, Created 10/22/2025

import os
import json
import time
import base64

import openai

from rover.navigation_map import NavigationMap

from rover.aegis_corridor_geometry import query_forward_corridor

DEBUG_LIDAR_INTERPRETATION = True

# Select the new 3D LiDAR scene for Luna.
# Disabled until offline payload testing passes.
USE_3D_SCENE_FOR_LUNA = False

ENABLE_LIDAR_QUERY_TOOL = False
ENABLE_GEMMA_GEOMETRY_SHADOW = False

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

    - a direct sampled raw STL27L point-cloud input
    - direct LiDAR top-down navigation geometry
    - direct LiDAR polar/range geometry
    - direct 10-degree LiDAR angle/range point measurements
    - left and right LiDAR travel-corridor views
    - fresh camera images
    - ultrasonic readings
    - IMU / heading information
    - recent movement history
    - persistent navigation map
    - current estimated rover pose
    - previous LiDAR scan locations and headings
    - measured encoder odometry when available

    Use these sources together.

    Do not treat any single source as a complete representation of the
    environment.

    Different sensor inputs provide different forms of evidence. Use their
    strengths together before choosing a physical movement.


    ============================================================
    LIDAR GEOMETRY
    ============================================================

    The raw STL27L point cloud is the source of truth for LiDAR geometric
    measurements.

    The rover provides several direct representations derived from the same
    LiDAR scan:

    - a sampled subset of direct raw X Y Z INTENSITY measurements
    - a top-down navigation-height point-cloud view
    - a polar/range point-cloud view
    - 10-degree angular measurement bins containing point counts by distance
    - a left travel-corridor XZ view
    - a right travel-corridor XZ view

    The raw sampled point input normally contains approximately 16,800 actual
    measured LiDAR points distributed across the full scan.

    Each raw point contains:

    X Y Z INTENSITY

    X, Y, and Z are Cartesian coordinates in meters.

    INTENSITY is LiDAR return intensity and is NOT another spatial coordinate.

    The sampled raw points are supporting direct sensor evidence. They do not
    replace the full-scan LiDAR representations.

    Python does NOT decide which direction is blocked, clear, safe, unsafe,
    or preferable.

    Python may organize measured LiDAR points by angle, distance, height, or
    visualization format so that the geometry can be interpreted efficiently.

    These transformations are measurements and representations, not navigation
    conclusions.

    Interpret the measured geometry yourself.

    The 10-degree measurements describe how many actual LiDAR returns were
    observed at different horizontal ranges inside each 10-degree angular
    region.

    They do NOT contain a Python-selected best direction.

    Use neighboring angular regions together.

    Do not choose or reject a path because of one isolated close point or one
    isolated distant point.

    Instead determine whether the combined LiDAR evidence indicates:

    - substantial geometry
    - isolated objects
    - continuous walls or structures
    - continuous open angular regions
    - openings
    - rover-sized travel corridors
    - geometry beside a path rather than inside it
    - geometry occupying the rover's actual travel height and width

    Compare the raw sampled measurements, point-density distributions,
    neighboring 10-degree measurements, top-down geometry, polar/range view,
    and travel-corridor views.

    Then cross-check that interpretation with:

    - fresh camera imagery
    - ultrasonic readings
    - current heading
    - persistent navigation map
    - recent movement history
    - encoder odometry when available

    Ultrasonic sensors remain the primary immediate near-field collision
    measurement.

    If a fresh LiDAR scan has already been obtained from the current pose and the
    rover has not materially changed its position or viewpoint, do NOT request
    another LiDAR scan simply because the same obstacle remains present.

    Repeated measurements of the same static geometry do not provide useful new
    navigation information.

    Use the fresh scan to choose a movement, escape maneuver, or NO_OP if every
    movement is genuinely unsafe.

    ============================================================
    LIDAR REPRESENTATIONS
    ============================================================

    NAVIGATION_XY:

    This is the main top-down LiDAR navigation view.

    It contains measured LiDAR geometry within the navigation-relevant height
    range so that floor and geometry too high to affect normal rover travel do
    not dominate the navigation view.

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

    A close object beside a path does not necessarily block the rover's actual
    travel corridor.


    POLAR_RANGE:

    This is a direct angle-versus-horizontal-distance representation of
    measured LiDAR returns.

    0 degrees = FRONT
    positive angles = LEFT
    negative angles = RIGHT
    180 degrees = BACK

    Radial distance represents horizontal distance from the rover.

    Use this view together with the 10-degree measurements to understand how
    geometry is distributed by direction and range.

    Do not interpret one isolated return as an entire blocked direction.

    Examine continuous structures and distributions across neighboring angles.


    10-DEGREE LIDAR MEASUREMENTS:

    The complete 360-degree navigation cloud is divided into 10-degree angular
    regions.

    For each region, the rover provides measured point counts across distance
    ranges.

    These values are direct organized measurements, not a blocked/clear
    classification.

    Use them especially when comparing small directional differences such as:

    - straight ahead
    - slightly left of forward
    - slightly right of forward

    Prefer broad, consistent geometric evidence across neighboring bins rather
    than one unusually close or unusually distant return.


    LEFT_CORRIDOR_XZ:

    This view contains vertical LiDAR geometry through the LEFT half of the
    rover's center travel corridor.

    Use it to determine whether measured geometry occupies the vertical space
    the left portion of the rover body would need while moving forward.


    RIGHT_CORRIDOR_XZ:

    This view contains vertical LiDAR geometry through the RIGHT half of the
    rover's center travel corridor.

    Use it to determine whether measured geometry occupies the vertical space
    the right portion of the rover body would need while moving forward.


    The LEFT and RIGHT corridor views can help identify:

    - furniture inside the travel corridor
    - low structures
    - table or chair geometry
    - vertical obstacles
    - overhangs
    - open vertical travel space
    - an obstacle affecting only one side of the rover's path


    RAW LIDAR POINT SAMPLE:

    A sampled subset of approximately 16,800 direct raw STL27L measurements may
    also be provided.

    Use these raw points as independent geometric evidence.

    The raw sample is useful for checking whether conclusions from the organized
    LiDAR representations are supported by actual measured point geometry.

    Do not assume that the raw sample alone perfectly represents every detail of
    the full scan.

    When the raw sample and the full-scan representations disagree on a small
    directional preference, use the full-scan top-down, polar/range, 10-degree
    measurements, corridor views, camera, and other sensors to resolve the
    uncertainty.

    LiDAR representations describe the environment from the pose at which the
    scan was captured.

    After movement, an older LiDAR scan remains useful structural memory but is
    not an exact current-distance measurement.


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

    - Ultrasonic readings provide fresh near-field collision information.

    - The physical FRONT CENTER ultrasonic sensor is currently unreliable and may
    report 0 or invalid values.

    - IGNORE front_center_cm for navigation decisions until further notice.

    - Use lidar_mount_cm as the rover's effective FRONT CENTER proximity reading.
    The LiDAR-mounted ultrasonic faces approximately the same forward direction
    as the normal center sensor, but is mounted slightly higher.

    - Therefore interpret the ultrasonic sensors as:

        front_left_cm   = front-left proximity
        lidar_mount_cm  = front-center proximity
        front_right_cm  = front-right proximity
        rear_cm         = rear proximity

    - Do NOT treat front_center_cm = 0 as an obstacle, missing clearance, or a
    reason to stop, rescan, or use no_op.

    - A zero or invalid front_center_cm is an expected known sensor fault.

    - For current forward-clearance decisions, use lidar_mount_cm together with
    front_left_cm and front_right_cm.

    - Because lidar_mount_cm is mounted slightly higher than the original center
    sensor, remember that it may be less reliable for extremely low obstacles.
    Use the camera, LiDAR geometry, and the left/right ultrasonic sensors as
    additional confirmation.

    - During movement, ultrasonic sensors continuously supervise the rover.

    - Fresh ultrasonic information is more relevant for immediate collision danger
    than old LiDAR observations.


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

    Interpret what each sensor and representation is actually good at.


    Raw sampled LiDAR points:
    direct measured geometric evidence from the latest STL27L scan


    NAVIGATION_XY:
    overall top-down spatial layout at navigation-relevant heights


    POLAR_RANGE:
    angular and horizontal-range structure around the rover


    10-degree LiDAR measurements:
    fine directional range and point-density evidence across the full
    navigation cloud


    LEFT_CORRIDOR_XZ and RIGHT_CORRIDOR_XZ:
    vertical occupancy of the rover's actual travel corridor


    Camera:
    current visual and semantic meaning, object interpretation, and visual
    validation of proposed paths


    Ultrasonics:
    immediate close-range collision safety


    IMU:
    orientation and measured heading change


    Persistent map:
    spatial memory, previous scan poses, and exploration history


    Recent movement history:
    the purpose and sequence of recent rover actions


    Encoder odometry when available:
    actual measured rover displacement


    When multiple independent measurements support the same geometric
    interpretation, confidence may increase.

    Before physical movement, cross-check the proposed travel path against the
    most relevant available sources.

    A close LiDAR return does not automatically mean an entire direction is
    unusable.

    Determine whether the return represents:

    - a continuous blocking structure
    - an isolated object
    - geometry beside the intended path
    - rover or sensor geometry
    - floor or overhead geometry
    - an object occupying only one side of the travel corridor

    When the top-down geometry, polar/range view, neighboring 10-degree
    measurements, corridor views, camera, and ultrasonics support a rover-sized
    path, do not reject that path because of one isolated LiDAR return.

    Likewise, do not trust a visually open camera path when strong current LiDAR
    geometry or ultrasonic measurements show a real obstacle in the rover's
    travel corridor.

    Camera information does not override strong geometric collision evidence.

    LiDAR information does not automatically override fresh near-field
    ultrasonic collision evidence.

    When raw sampled LiDAR points disagree slightly with the full-scan
    representations, remember that the raw points are only a sampled subset.
    Use the full-scan representations and the other navigation sensors to
    resolve the disagreement.

    When important information genuinely conflicts and safe movement cannot be
    determined, gather the specific additional information needed rather than
    guessing.


    ============================================================
    AUTONOMOUS STARTUP
    ============================================================

    At startup:

    1. Obtain an initial LiDAR scan.
    2. Examine the direct LiDAR top-down geometry.
    3. Examine the polar/range geometry.
    4. Examine the 10-degree angle/range measurements.
    5. Examine the left and right travel-corridor views.
    6. Use the direct raw LiDAR sample as supporting measured evidence.
    7. Incorporate the scan into the persistent navigation map.
    8. Examine the fresh camera image and fresh telemetry.
    9. Cross-check the proposed path with ultrasonics and current heading.
    10. Choose ONE useful safe action.
    11. Execute only that action.

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
    OBSTACLE ESCAPE BEHAVIOR
    ============================================================

    A close obstacle in one portion of the rover's forward sensing region does
    NOT mean that all rover movement is unsafe.

    If forward travel is constrained, actively evaluate alternative physical
    actions using the available geometry and sensors.

    Possible alternatives include:

    - turning away from the obstacle
    - making a smaller heading correction
    - reversing a short distance when rear clearance is sufficient
    - returning toward a recently known open corridor
    - repositioning to obtain a better view of an opening

    For example:

    If FRONT-LEFT is very close but FRONT-RIGHT and the surrounding geometry
    indicate open space, consider turning or repositioning toward the right.

    If forward geometry is too constrained but REAR clearance is sufficient,
    consider a short supervised reverse to create maneuvering space.

    A close reading on one ultrasonic sensor should restrict movement into the
    region measured by that sensor. It should NOT automatically prohibit every
    possible rover movement.


    ============================================================
    NO_OP POLICY
    ============================================================

    NO_OP is not the default response to a static obstacle.

    Do not repeatedly remain stationary simply because the current forward path
    is blocked.

    Before choosing NO_OP, evaluate whether a safe:

    - reverse
    - left turn
    - right turn
    - small heading correction
    - return toward known open space

    is available.

    NO_OP is appropriate when:

    - all immediately available physical movements appear unsafe
    - a temporary or moving obstacle should be allowed to clear
    - required sensor information is currently unavailable but expected shortly
    - remaining stationary is genuinely safer than every available movement

    If the obstacle is static and another safe maneuver exists, prefer the safe
    maneuver over repeatedly choosing NO_OP.


    ============================================================
    ESCAPING TIGHT AREAS
    ============================================================

    When the rover reaches a tight furniture cluster or dead-end:

    1. Identify which part of the rover's travel region is constrained.
    2. Check rear clearance.
    3. Check whether turning away from the obstacle is geometrically reasonable.
    4. Use recent movement history to remember the direction from which the rover
       entered the constrained area.
    5. If necessary, reverse toward that recently known open region.
    6. Reorient only after sufficient maneuvering clearance exists.

    Do not become trapped in a repeated:

        NO_OP -> NO_OP -> NO_OP

    loop when a safe escape maneuver exists.

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

    TURN commands are executed using BNO085 closed-loop yaw feedback.

    The requested turn angle is a target. The rover measures the actual
    completed rotation and uses the measured turn angle to update its
    heading and persistent navigation map.

    Do not assume that the requested angle and actual completed angle
    are exactly identical.


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

    1. Examine NAVIGATION_XY for overall navigation-height geometry.
    2. Examine POLAR_RANGE for angular and distance structure.
    3. Examine the 10-degree angle/range point measurements.
    4. Compare neighboring 10-degree regions rather than isolated returns.
    5. Examine LEFT_CORRIDOR_XZ and RIGHT_CORRIDOR_XZ for actual vertical
       travel-corridor occupancy.
    6. Examine the sampled raw LiDAR points as independent direct measurement
       evidence.
    7. Compare the LiDAR interpretation with the fresh camera image.
    8. Cross-check immediate clearance using the fresh ultrasonic readings.
    9. Use IMU heading to maintain the correct relationship between rover
       orientation and observed geometry.
    10. Incorporate the scan pose and geometry into the persistent navigation
        map.
    11. Compare the new observation with previous map history and recent
        movement.
    12. Decide which actual rover-sized path is useful and safe.

    Python does not provide a blocked/clear/best-direction LiDAR conclusion.

    Determine the useful travel direction from the measured sensor evidence
    yourself.

    For broad navigation, identify continuous open geometry.

    For fine steering decisions such as slightly left versus slightly right,
    pay particular attention to neighboring 10-degree measurements,
    POLAR_RANGE, and the left/right corridor views.

    Use the sampled raw LiDAR points to validate geometric reasoning, but do not
    treat the sampled subset as more complete than the full-scan
    representations.

    A large measured distance in one direction does NOT automatically make that
    direction the preferred exploration route.

    Also consider:

    - rover-sized clearance
    - current camera evidence
    - ultrasonic safety
    - current heading
    - existing exploration progress
    - persistent map history
    - the purpose of the current route

    If the available evidence agrees that a rover-sized route is safe and useful,
    continue exploration without unnecessary rescanning.

    If the evidence materially conflicts and the conflict affects movement
    safety, obtain the specific additional observation needed before moving.


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

    aegis_tools.append({
        "type": "function",
        "function": {
            "name": "query_forward_corridor",
            "description": (
                "Read measured low/mid-height XYZ returns in the experimental "
                "forward 0.5-2m, left/center/right corridor, with a 10cm grid. "
                "No-return cells are UNKNOWN, not traversable. "
                "No motion permission."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            }
        }
    })

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

    def encode_image(self, filename):
        with open(filename, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    
    def debug_lidar_interpretation(
        self,
        lidar_views
    ):
        """
        Development-only diagnostic.

        Ask Luna to interpret only the latest direct LiDAR
        geometry images.

        This diagnostic intentionally does NOT use:

            camera
            ultrasonics
            persistent map
            movement history
            raw 16.8k XYZ sample

        The main autopilot receives all of those separately.
        """

        if not isinstance(
            lidar_views,
            dict
        ):
            return

        view_order = [
            "navigation_xy",
            "polar_range",
            "left_corridor_xz",
            "right_corridor_xz"
        ]

        view_labels = {
            "navigation_xy":
                (
                    "NAVIGATION_XY - top-down "
                    "navigation-height LiDAR geometry"
                ),

            "polar_range":
                (
                    "POLAR_RANGE - direct angle versus "
                    "horizontal-distance LiDAR view"
                ),

            "left_corridor_xz":
                (
                    "LEFT_CORRIDOR_XZ - vertical geometry "
                    "through the left half of the travel corridor"
                ),

            "right_corridor_xz":
                (
                    "RIGHT_CORRIDOR_XZ - vertical geometry "
                    "through the right half of the travel corridor"
                )
        }

        content = [
            {
                "type": "text",
                "text": (
                    "Analyze ONLY these LiDAR geometry images.\n\n"

                    "Coordinate system:\n"
                    "+X = FRONT\n"
                    "-X = BACK\n"
                    "+Y = LEFT\n"
                    "-Y = RIGHT\n"
                    "+Z = UP\n\n"

                    "The rover is at the origin.\n\n"

                    "Do not use camera data, ultrasonics, "
                    "persistent-map history, or previous "
                    "conversation information.\n\n"

                    "Describe the measured geometry, nearby "
                    "structures, broad openings, rover-sized "
                    "travel corridors, and uncertainty.\n\n"

                    "Do not classify an entire direction from "
                    "one isolated point."
                )
            }
        ]

        images_added = 0

        for view_name in view_order:

            filename = lidar_views.get(
                view_name
            )

            if (
                not filename
                or not os.path.exists(
                    filename
                )
            ):

                print(
                    "[LIDAR LUNA DEBUG] Missing view: "
                    f"{view_name}"
                )

                continue

            image_base64 = self.encode_image(
                filename
            )

            content.append({
                "type": "text",
                "text":
                    view_labels[
                        view_name
                    ]
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
                "[LIDAR LUNA DEBUG] "
                "No LiDAR images available."
            )

            return


        debug_messages = [
            {
                "role": "system",

                "content": (
                    "You are inspecting AEGIS rover "
                    "LiDAR geometry for development "
                    "debugging. Analyze only the supplied "
                    "LiDAR measurements."
                )
            },
            {
                "role": "user",
                "content": content
            }
        ]


        try:

            response = (
                self.client
                .chat.completions
                .create(
                    model=self.model_name,
                    messages=debug_messages,
                    reasoning_effort="none"
                )
            )

            interpretation = (
                response
                .choices[0]
                .message
                .content
            )

            print()
            print("=" * 60)

            print(
                "[LIDAR LUNA DEBUG - "
                "DIRECT GEOMETRY ONLY]"
            )

            print("=" * 60)

            print(
                interpretation
                if interpretation
                else "No interpretation returned."
            )

            print("=" * 60)
            print()


        except Exception as e:

            print(
                "[LIDAR LUNA DEBUG] "
                f"Analysis failed: {e}"
            )


    def create_raw_lidar_ai_sample(
        self,
        lidar_filename,
        stride=64
    ):
        """
        Create a uniformly sampled raw LiDAR file for Luna.

        The original full-resolution cloud is NOT modified.

        Every 64th measured point is retained.

        Current cloud format:

            X Y Z INTENSITY

        This produces approximately 16.8k points from the
        current ~1.07 million-point / 400-ring scan.

        This is direct sensor data. No blocked/clear/best-path
        interpretation is performed here.
        """

        if (
            not lidar_filename
            or not os.path.exists(
                lidar_filename
            )
        ):
            return None

        output_dir = os.path.dirname(
            lidar_filename
        )

        timestamp = time.strftime(
            "%Y%m%d_%H%M%S"
        )

        output_file = os.path.join(
            output_dir,
            f"lidar_raw_ai_sample_{timestamp}.txt"
        )

        source_points = 0
        sampled_points = 0

        with open(
            lidar_filename,
            "r"
        ) as source:

            with open(
                output_file,
                "w"
            ) as destination:

                for line_number, line in enumerate(
                    source
                ):

                    source_points += 1

                    if (
                        line_number % stride
                        != 0
                    ):
                        continue

                    values = line.split()

                    # Current STL27L converted cloud:
                    #
                    # X Y Z INTENSITY
                    if len(values) < 4:
                        continue

                    destination.write(
                        line
                    )

                    sampled_points += 1

        print(
            "[AI] Raw LiDAR sample created: "
            f"{sampled_points:,} / "
            f"{source_points:,} points"
        )

        print(
            "[AI] Raw LiDAR sample file: "
            f"{output_file}"
        )

        return output_file

    def add_raw_lidar_sample_to_memory(
        self,
        filename
    ):
        """
        Send the latest sampled raw STL27L point cloud to Luna.

        Only the newest raw sample is retained in model memory.

        Current point format:

            X Y Z INTENSITY

        X/Y/Z are Cartesian coordinates in meters.
        INTENSITY is LiDAR return intensity.

        No navigation classification is performed here.
        """

        if (
            not filename
            or not os.path.exists(filename)
        ):

            print(
                "[AI] Raw LiDAR sample unavailable."
            )

            return

        # --------------------------------------------------
        # REMOVE PREVIOUS RAW LIDAR SAMPLE FROM MEMORY
        # --------------------------------------------------

        if (
            self.latest_raw_lidar_message
            is not None
        ):

            old_message = (
                self.latest_raw_lidar_message
            )

            self.memory = [
                msg
                for msg in self.memory
                if msg is not old_message
            ]

            self.latest_raw_lidar_message = None

            print(
                "[AI] Removed previous raw LiDAR "
                "sample from model memory."
            )

        # --------------------------------------------------
        # READ NEW DIRECT RAW SAMPLE
        # --------------------------------------------------

        try:

            with open(
                filename,
                "r"
            ) as f:

                raw_points = f.read()

        except Exception as e:

            print(
                "[AI] Failed to read raw LiDAR "
                f"sample: {e}"
            )

            return

        point_count = (
            raw_points.count("\n")
        )

        if (
            raw_points
            and not raw_points.endswith("\n")
        ):

            point_count += 1

        # --------------------------------------------------
        # BUILD MODEL MESSAGE
        # --------------------------------------------------

        message = {
            "role": "user",

            "content": (
                "DIRECT RAW LIDAR POINT SAMPLE\n\n"

                "This is a uniformly sampled subset of the "
                "latest STL27L point cloud.\n\n"

                "The full-resolution cloud remains stored "
                "locally. These points are direct sensor "
                "measurements and have NOT been classified "
                "as blocked, clear, safe, unsafe, or "
                "preferable by Python.\n\n"

                "Each line contains:\n\n"

                "X Y Z INTENSITY\n\n"

                "X, Y, and Z are Cartesian coordinates "
                "in meters.\n"

                "INTENSITY is LiDAR return intensity and "
                "is NOT another spatial coordinate.\n\n"

                "Coordinate system:\n"
                "+X = FRONT\n"
                "-X = BACK\n"
                "+Y = LEFT\n"
                "-Y = RIGHT\n"
                "+Z = UP\n\n"

                f"Sampled points: {point_count:,}\n\n"

                "Use these raw measurements together with "
                "the full-cloud LiDAR images, 10-degree "
                "measurements, camera, ultrasonics, IMU, "
                "persistent map, and movement history.\n\n"

                "RAW POINTS START BELOW:\n"
                "----------------------------------------\n"

                + raw_points
            )
        }

        self.update_memory(
            message
        )

        self.latest_raw_lidar_message = (
            message
        )

        print(
            "[AI] Added latest raw LiDAR sample "
            f"to model memory: {point_count:,} points"
        )


    def add_lidar_visualizations(
        self,
        lidar_views
    ):
        """
        Send direct LiDAR geometry representations to Luna.

        These are measurement representations only.

        Python does NOT decide:

            blocked
            clear
            safe
            unsafe
            best direction
        """

        if (
            not isinstance(
                lidar_views,
                dict
            )
            or not lidar_views
        ):

            print(
                "[AI] No LiDAR inputs available."
            )

            return

        content = [
            {
                "type": "text",

                "text": (
                    "DIRECT LIDAR GEOMETRY INPUTS\n\n"

                    "These inputs come from the latest "
                    "AEGIS STL27L scan.\n\n"

                    "Coordinate system:\n"
                    "+X = FRONT\n"
                    "-X = BACK\n"
                    "+Y = LEFT\n"
                    "-Y = RIGHT\n"
                    "+Z = UP\n\n"

                    "The rover is at the origin.\n\n"

                    "Python has NOT classified any "
                    "direction as blocked, clear, safe, "
                    "unsafe, or preferable.\n\n"

                    "Interpret the measured geometry "
                    "yourself and cross-check it with "
                    "the camera, ultrasonics, IMU, "
                    "persistent map, and movement history."
                )
            }
        ]

        # --------------------------------------------------
        # LIDAR IMAGES
        # --------------------------------------------------

        view_labels = {


            "scene_perception_png":
                (
                    "AEGIS 3D LIDAR PERCEPTION. "
                    "Top-down measured geometry with height-aware "
                    "obstacle information and a directional distance guide. "
                    "Coordinate system: +X front, +Y left, +Z up. "
                    "The floor height is estimated, not calibrated. "
                    "Missing LiDAR returns mean UNKNOWN, not free space. "
                    "Colors represent measured geometry, not verified "
                    "safe travel corridors."
                ),

            "navigation_xy":
                (
                    "NAVIGATION_XY - DIRECT TOP-DOWN "
                    "LIDAR GEOMETRY"
                ),

            "polar_range":
                (
                    "POLAR_RANGE - DIRECT ANGLE / "
                    "DISTANCE LIDAR VIEW"
                ),

            "left_corridor_xz":
                (
                    "LEFT_CORRIDOR_XZ - LEFT HALF OF "
                    "THE ROVER TRAVEL CORRIDOR"
                ),

            "right_corridor_xz":
                (
                    "RIGHT_CORRIDOR_XZ - RIGHT HALF OF "
                    "THE ROVER TRAVEL CORRIDOR"
                )

        }

        view_order = [
            "navigation_xy",
            "polar_range",
            "left_corridor_xz",
            "right_corridor_xz"
        ]

        # Prefer the new 3D scene when explicitly enabled.
        # Otherwise retain the original LiDAR image inputs.
        if (
            USE_3D_SCENE_FOR_LUNA
            and lidar_views.get("scene_perception_png")
        ):
            view_order = [
                "scene_perception_png"
            ]

        images_added = 0

        for view_name in view_order:

            filename = lidar_views.get(
                view_name
            )

            if not filename:

                print(
                    "[AI] LiDAR view missing: "
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
                    view_labels[
                        view_name
                    ]
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


        # --------------------------------------------------
        # DIRECT 10-DEGREE LIDAR MEASUREMENTS
        # --------------------------------------------------

        measurements_added = False

        measurements_file = (
            lidar_views.get(
                "angular_measurements"
            )
        )

        if (
            measurements_file
            and os.path.exists(
                measurements_file
            )
        ):

            try:

                with open(
                    measurements_file,
                    "r"
                ) as f:

                    angular_measurements = (
                        json.load(f)
                    )

                content.append({
                    "type": "text",

                    "text": (
                        "DIRECT LIDAR 10-DEGREE "
                        "MEASUREMENTS\n\n"

                        "These contain direct measured "
                        "point counts grouped by angle "
                        "and distance.\n\n"

                        "They are NOT blocked/clear "
                        "classifications and do NOT "
                        "contain a Python-selected best "
                        "direction.\n\n"

                        "Use neighboring angular regions "
                        "together when evaluating a "
                        "rover-sized travel path.\n\n"

                        + json.dumps(
                            angular_measurements,
                            separators=(
                                ",",
                                ":"
                            )
                        )
                    )
                })

                measurements_added = True

                print(
                    "[AI] Added direct 10-degree "
                    "LiDAR measurements."
                )

            except Exception as e:

                print(
                    "[AI] Failed to load direct "
                    "LiDAR measurements: "
                    f"{e}"
                )


        # --------------------------------------------------
        # SEND DIRECT LIDAR PACKAGE
        # --------------------------------------------------

        if (
            images_added == 0
            and not measurements_added
        ):

            print(
                "[AI] No direct LiDAR geometry "
                "inputs could be loaded."
            )

            return

        self.update_memory({
            "role": "user",
            "content": content
        })

        print(
            "[AI] Sent direct LiDAR package: "
            f"{images_added} images, "
            f"10-degree measurements="
            f"{measurements_added}"
        )

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
                "latest LiDAR scan. Compare it with NAVIGATION_XY, "
                "CORRIDOR_XZ, ultrasonic readings, and the persistent "
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

        lidar_filename = scanner.scan(
            filepath=dump_folder,
            visualize=True
        )

        print(
            f"Scan saved: {lidar_filename}"
        )

        lidar_views = scanner.last_visualizations

        self.latest_lidar_geometry_file = lidar_views.get(
            "scene_full_geometry"
        )
        if ENABLE_GEMMA_GEOMETRY_SHADOW and self.latest_lidar_geometry_file:
            from threading import Thread
            from rover.aegis_gemma_geometry_shadow import run_shadow
            Thread(
                target=run_shadow,
                args=(self.latest_lidar_geometry_file,),
                daemon=True
            ).start()


        # --------------------------------------------------
        # DIRECT RAW LIDAR SAMPLE FOR LUNA
        # --------------------------------------------------

        raw_lidar_sample = (
            self.create_raw_lidar_ai_sample(
                lidar_filename,
                stride=64
            )
        )

        if raw_lidar_sample is not None:

            lidar_views[
                "raw_lidar_sample"
            ] = raw_lidar_sample


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

        self.navigation_map.add_lidar_scan(
            filename=lidar_filename,
            summary=summary,
            visualization_files=lidar_views
        )

        if self.navigation_map_file:

            self.navigation_map.render_map_png(
                self.navigation_map_file
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
                        "The latest raw STL27L scan is represented "
                        "directly through the LiDAR top-down point view, "
                        "polar/range point view, and direct 10-degree "
                        "angle/range point-count measurements. "
                        "Python has not classified directions as blocked, "
                        "clear, safe, unsafe, or preferable. "
                        "Interpret the measured geometry yourself."
                    ),

                "navigation_map":
                    self.navigation_map.get_ai_context()
            })
        })

        self.add_lidar_visualizations(
            lidar_views
        )

        # Send direct sampled raw STL27L measurements.
        #
        # This message is managed separately because only the
        # newest ~16.8k-point raw sample should remain in memory.
        self.add_raw_lidar_sample_to_memory(
            raw_lidar_sample
        )

        if DEBUG_LIDAR_INTERPRETATION:

            self.debug_lidar_interpretation(
                lidar_views
            )

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

    def _query_forward_corridor(self):
        """Read measurements from the most recent processed LiDAR scan."""
        import os
        import time
        from rover.aegis_corridor_geometry import query_forward_corridor

        path = self.latest_lidar_geometry_file

        if not path or not os.path.isfile(path):
            return {
                "status": "unavailable",
                "reason": "No processed 3D scan yet"
            }

        last = self.navigation_state.get("last_lidar_time")
        age = (
            round(time.time() - last, 1)
            if isinstance(last, (int, float))
            else None
        )

        heading = self.navigation_state.get(
            "heading_change_since_lidar_deg"
        )

        try:
            result = query_forward_corridor(path)
        except (OSError, ValueError, KeyError) as exc:
            return {
                "status": "error",
                "reason": str(exc)
            }

        return {
            "status": "ok",
            "frame": "rover-local at scan time",
            "scan_age_s": age,
            "heading_change_since_scan_deg": heading,
            "measurement": result,
            "warning": (
                "Historical scan frame, not clearance "
                "or motion authorization."
            )
        }

    def _handle_forward_corridor_query(self, call):
        """Answer a geometry tool call without accessing rover motors."""
        try:
            arguments = json.loads(
                call.function.arguments or "{}"
            )

            if arguments != {}:
                raise ValueError(
                    "query_forward_corridor takes no arguments"
                )

            result = self._query_forward_corridor()

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

    def _request_model_action(self):
        """Request model action with optional read-only 3D tools."""

        geometry_names = {
            "query_lidar_geometry",
            "query_forward_corridor"
        }

        tools = [
            tool for tool in Autopilot.aegis_tools
            if ENABLE_LIDAR_QUERY_TOOL
            or tool["function"]["name"] not in geometry_names
        ]

        return self.client.chat.completions.create(
            model=self.model_name,
            messages=self.memory,
            tools=tools,
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
                elif name == "query_forward_corridor":
                    self._handle_forward_corridor_query(call)
                    continue

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