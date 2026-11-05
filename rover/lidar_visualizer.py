import os
import time
import json

import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ----------------------------------------------------------
# VISUALIZATION SETTINGS
# ----------------------------------------------------------

# The real cloud stays FULL resolution.
#
# This limit applies ONLY to PNG rendering for Luna.
MAX_VIS_POINTS = 300_000

# Navigation height range.
MIN_NAV_Z = -0.27
MAX_NAV_Z = 1.00

# Vertical XZ slice centered through the rover.
#
# This represents approximately:
#   0.45 m left of center
#   0.45 m right of center
#
# Later replace this using the rover's measured width.
CORRIDOR_HALF_WIDTH_M = 0.45

# Luna images do not need print-quality resolution.
IMAGE_DPI = 140

# ----------------------------------------------------------
# DIRECT LIDAR MEASUREMENT SETTINGS
# ----------------------------------------------------------

# 360 degrees / 10 degrees = 36 angular bins.
#
# These bins do NOT classify directions as blocked or clear.
# They only preserve measured point density by angle/range
# for the AI to interpret itself.
ANGULAR_BIN_DEG = 10

# Distance histogram resolution.
RANGE_BIN_WIDTH_M = 0.50

# Distances beyond this are still counted separately.
# This keeps the JSON compact while preserving the fact
# that farther returns exist.
RANGE_HISTOGRAM_MAX_M = 20.0



def _downsample_xyz(
    x,
    y,
    z,
    max_points=MAX_VIS_POINTS
):
    """
    Downsample visualization points only.

    The original LiDAR cloud is NOT modified.
    """

    count = len(x)

    if count <= max_points:
        return x, y, z

    rng = np.random.default_rng(
        42
    )

    indices = rng.choice(
        count,
        size=max_points,
        replace=False
    )

    return (
        x[indices],
        y[indices],
        z[indices]
    )

def _create_angular_measurements(
    x,
    y,
    output_file
):
    """
    Create direct 10-degree LiDAR range measurements.

    This function does NOT decide:
        - blocked / clear
        - safe / unsafe
        - best direction
        - obstacle distance

    It only groups actual LiDAR returns by:

        angle
        distance

    and reports how many measured points fall into each
    angular/range cell.

    Coordinate convention:

        0 deg   = FRONT
        +90 deg = LEFT
        -90 deg = RIGHT
        180 deg = BACK
    """

    distances = np.hypot(
        x,
        y
    )

    angles_deg = np.degrees(
        np.arctan2(
            y,
            x
        )
    )

    valid = (
        np.isfinite(distances)
        & np.isfinite(angles_deg)
        & (distances > 0.001)
    )

    distances = distances[valid]
    angles_deg = angles_deg[valid]

    # Round each point to its nearest 10-degree direction.
    angle_centers = (
        np.rint(
            angles_deg / ANGULAR_BIN_DEG
        )
        * ANGULAR_BIN_DEG
    ).astype(int)

    # +180 and -180 represent the same BACK direction.
    angle_centers[
        angle_centers == -180
    ] = 180

    centers = list(
        range(
            -170,
            181,
            ANGULAR_BIN_DEG
        )
    )

    distance_edges = np.arange(
        0.0,
        RANGE_HISTOGRAM_MAX_M
        + RANGE_BIN_WIDTH_M,
        RANGE_BIN_WIDTH_M
    )

    angular_bins = []

    for center in centers:

        mask = (
            angle_centers == center
        )

        sector_distances = (
            distances[mask]
        )

        in_range = (
            sector_distances[
                sector_distances
                < RANGE_HISTOGRAM_MAX_M
            ]
        )

        histogram, _ = np.histogram(
            in_range,
            bins=distance_edges
        )

        distance_counts = {}

        for i, count in enumerate(
            histogram
        ):

            start_m = (
                distance_edges[i]
            )

            end_m = (
                distance_edges[i + 1]
            )

            label = (
                f"{start_m:.1f}-"
                f"{end_m:.1f}"
            )

            distance_counts[label] = int(
                count
            )

        beyond_max = int(
            np.count_nonzero(
                sector_distances
                >= RANGE_HISTOGRAM_MAX_M
            )
        )

        angular_bins.append({
            "angle_center_deg":
                int(center),

            "point_count":
                int(
                    len(
                        sector_distances
                    )
                ),

            "distance_point_counts":
                distance_counts,

            "points_beyond_20m":
                beyond_max
        })

    data = {
        "coordinate_system": {
            "0_deg": "FRONT",
            "positive_angles": "LEFT",
            "negative_angles": "RIGHT",
            "180_deg": "BACK"
        },

        "angular_bin_width_deg":
            ANGULAR_BIN_DEG,

        "distance_bin_width_m":
            RANGE_BIN_WIDTH_M,

        "distance_histogram_max_m":
            RANGE_HISTOGRAM_MAX_M,

        "total_points":
            int(
                len(
                    distances
                )
            ),

        "angular_bins":
            angular_bins
    }

    with open(
        output_file,
        "w"
    ) as f:

        json.dump(
            data,
            f,
            separators=(",", ":")
        )

    print(
        "[LIDAR DIRECT] "
        f"10-degree measurements saved: "
        f"{output_file}"
    )

    return output_file


def create_lidar_visualizations(
    points,
    output_dir,
    debug_3d=False
):
    """
    Create navigation-focused LiDAR visualizations.

    Full LiDAR data is preserved for:
        - persistent map
        - distance summary
        - saved cloud

    PNG rendering uses a smaller subset for speed.

    Normal Luna views:

        1. navigation_xy
           Top-down navigation geometry.

        2. corridor_xz
           Vertical slice through the rover's actual
           center travel corridor.

    Optional:

        3. xyz_3d
           Full 3D debug view for human inspection.

    Coordinate convention:

        +X = FRONT
        +Y = LEFT
        +Z = UP
    """

    start_time = time.time()

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    points = np.asarray(
        points,
        dtype=float
    )

    if (
        points.ndim != 2
        or points.shape[1] < 3
    ):
        raise ValueError(
            f"Expected LiDAR points shaped "
            f"(N, 3+) but got {points.shape}"
        )

    # ------------------------------------------------------
    # EXTRACT XYZ
    # ------------------------------------------------------

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(z)
    )

    # Remove invalid zero returns.
    nonzero = (
        np.sqrt(
            x**2 +
            y**2 +
            z**2
        )
        > 0.001
    )

    valid &= nonzero

    x = x[valid]
    y = y[valid]
    z = z[valid]

    full_point_count = len(x)

    print(
        f"[LIDAR VIS] Full valid cloud: "
        f"{full_point_count:,} points"
    )

    timestamp = time.strftime(
        "%Y%m%d_%H%M%S"
    )

    files = {}


    # ======================================================
    # 1. NAVIGATION XY TOP-DOWN
    # ======================================================

    # Only retain heights relevant to rover navigation.
    nav_mask = (
        (z > MIN_NAV_Z)
        & (z < MAX_NAV_Z)
    )

    nav_x = x[nav_mask]
    nav_y = y[nav_mask]
    nav_z = z[nav_mask]

    nav_full_count = len(
        nav_x
    )

    (
        vis_x,
        vis_y,
        vis_z
    ) = _downsample_xyz(
        nav_x,
        nav_y,
        nav_z
    )

    xy_file = os.path.join(
        output_dir,
        f"lidar_navigation_xy_{timestamp}.png"
    )

    fig, ax = plt.subplots(
        figsize=(9, 9)
    )

    ax.scatter(
        vis_y,
        vis_x,
        s=0.22,
        alpha=0.70
    )

    ax.scatter(
        [0],
        [0],
        s=100,
        marker="^",
        label="AEGIS Rover"
    )

    ax.set_xlabel(
        "Y (m)   +Y = LEFT"
    )

    ax.set_ylabel(
        "X (m)   +X = FRONT"
    )

    ax.set_title(
        "AEGIS LiDAR - Navigation Top-Down\n"
        f"{nav_full_count:,} navigation points | "
        f"{len(vis_x):,} rendered"
    )

    ax.set_aspect(
        "equal",
        adjustable="box"
    )

    # +Y = LEFT, so positive Y belongs visually
    # on the left side of the image.
    ax.invert_xaxis()

    ax.grid(True)

    ax.legend(
        loc="upper right"
    )

    fig.savefig(
        xy_file,
        dpi=IMAGE_DPI
    )

    plt.close(fig)

    files[
        "navigation_xy"
    ] = xy_file

    print(
        "[LIDAR VIS] Navigation XY saved: "
        f"{xy_file}"
    )


        # ======================================================
    # 2. DIRECT POLAR / RANGE VIEW
    # ======================================================

    (
        polar_x,
        polar_y,
        polar_z
    ) = _downsample_xyz(
        nav_x,
        nav_y,
        nav_z
    )

    polar_angles = np.arctan2(
        polar_y,
        polar_x
    )

    polar_ranges = np.hypot(
        polar_x,
        polar_y
    )

    polar_file = os.path.join(
        output_dir,
        f"lidar_polar_range_{timestamp}.png"
    )

    fig = plt.figure(
        figsize=(9, 9)
    )

    ax = fig.add_subplot(
        111,
        projection="polar"
    )

    # FRONT is visually at the top.
    ax.set_theta_zero_location(
        "N"
    )

    # Positive angle = LEFT.
    ax.set_theta_direction(
        1
    )

    ax.scatter(
        polar_angles,
        polar_ranges,
        s=0.25,
        alpha=0.70
    )

    ax.set_title(
        "AEGIS LiDAR - Direct Polar Range View\n"
        "0° = FRONT | +90° = LEFT | "
        "-90° = RIGHT | 180° = BACK"
    )

    ax.grid(True)

    fig.savefig(
        polar_file,
        dpi=IMAGE_DPI
    )

    plt.close(fig)

    files[
        "polar_range"
    ] = polar_file

    print(
        "[LIDAR VIS] Polar range saved: "
        f"{polar_file}"
    )


    # ======================================================
    # 3. DIRECT 10-DEGREE RANGE MEASUREMENTS
    # ======================================================

    angular_file = os.path.join(
        output_dir,
        f"lidar_angular_10deg_{timestamp}.json"
    )

    _create_angular_measurements(
        x=nav_x,
        y=nav_y,
        output_file=angular_file
    )

    files[
        "angular_measurements"
    ] = angular_file

    # ======================================================
    # LEFT SIDE
    # ======================================================

    # Do NOT collapse the entire room into X/Z.
    #
    # Only show geometry physically near the
    # rover's center travel corridor.
    
    left_corridor_mask = (
        (y >= 0.0)
        & (y <= CORRIDOR_HALF_WIDTH_M)
    )

    left_x = x[left_corridor_mask]
    left_y = y[left_corridor_mask]
    left_z = z[left_corridor_mask]

    left_full_count = len(left_x)

    ( left_vis_x, left_vis_y, left_vis_z) = _downsample_xyz( left_x, left_y, left_z)

    left_file = os.path.join(
        output_dir,
        f"lidar_left_corridor_xz_{timestamp}.png"
    )

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    ax.scatter(
        left_vis_x,
        left_vis_z,
        s=0.35,
        alpha=0.65
    )

    ax.scatter(
        [0],
        [0],
        s=100,
        marker="^",
        label="AEGIS Rover"
    )

    ax.set_xlabel(
        "X (m)   +X = FRONT"
    )

    ax.set_ylabel(
        "Z (m)   +Z = UP"
    )

    ax.set_title(
        "AEGIS LiDAR - LEFT Travel Corridor\n"
        f"0 <= Y <= {CORRIDOR_HALF_WIDTH_M:.2f} m | "
        f"{left_full_count:,} points | "
        f"{len(left_vis_x):,} rendered"
    )

    ax.grid(True)

    ax.legend(
        loc="upper right"
    )

    fig.savefig(
        left_file,
        dpi=IMAGE_DPI
    )

    plt.close(fig)

    files[
        "left_corridor_xz"
    ] = left_file


    
    # ======================================================
    # RIGHT SIDE
    # ======================================================

    right_corridor_mask = (
        (y <= 0.0)
        & (y >= -CORRIDOR_HALF_WIDTH_M)
    )

    right_x = x[right_corridor_mask]
    right_y = y[right_corridor_mask]
    right_z = z[right_corridor_mask]

    right_full_count = len(right_x)

    (
        right_vis_x,
        right_vis_y,
        right_vis_z
    ) = _downsample_xyz(
        right_x,
        right_y,
        right_z
    )

    right_file = os.path.join(
        output_dir,
        f"lidar_right_corridor_xz_{timestamp}.png"
    )

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    ax.scatter(
        right_vis_x,
        right_vis_z,
        s=0.35,
        alpha=0.65
    )

    ax.scatter(
        [0],
        [0],
        s=100,
        marker="^",
        label="AEGIS Rover"
    )

    ax.set_xlabel(
        "X (m)   +X = FRONT"
    )

    ax.set_ylabel(
        "Z (m)   +Z = UP"
    )

    ax.set_title(
        "AEGIS LiDAR - RIGHT Travel Corridor\n"
        f"-{CORRIDOR_HALF_WIDTH_M:.2f} <= Y <= 0 m | "
        f"{right_full_count:,} points | "
        f"{len(right_vis_x):,} rendered"
    )

    ax.grid(True)

    ax.legend(
        loc="upper right"
    )

    fig.savefig(
        right_file,
        dpi=IMAGE_DPI
    )

    plt.close(fig)

    files[
        "right_corridor_xz"
    ] = right_file


    # ======================================================
    # 3. OPTIONAL FULL 3D DEBUG VIEW
    # ======================================================

    if debug_3d:

        (
            debug_x,
            debug_y,
            debug_z
        ) = _downsample_xyz(
            x,
            y,
            z
        )

        debug_file = os.path.join(
            output_dir,
            f"lidar_xyz_debug_{timestamp}.png"
        )

        fig = plt.figure(
            figsize=(10, 8)
        )

        ax = fig.add_subplot(
            111,
            projection="3d"
        )

        ax.scatter(
            debug_x,
            debug_y,
            debug_z,
            s=0.25,
            alpha=0.45
        )

        ax.scatter(
            [0],
            [0],
            [0],
            s=100,
            marker="^",
            label="AEGIS Rover"
        )

        ax.set_xlabel(
            "X - Front"
        )

        ax.set_ylabel(
            "Y - Left"
        )

        ax.set_zlabel(
            "Z - Up"
        )

        ax.set_title(
            "AEGIS LiDAR - 3D DEBUG VIEW\n"
            f"{len(debug_x):,} rendered points"
        )

        ax.view_init(
            elev=25,
            azim=-60
        )

        ax.legend(
            loc="upper right"
        )

        fig.savefig(
            debug_file,
            dpi=IMAGE_DPI
        )

        plt.close(fig)

        files[
            "xyz_debug"
        ] = debug_file

        print(
            "[LIDAR VIS] 3D debug saved: "
            f"{debug_file}"
        )




    # ======================================================
    # FINISHED
    # ======================================================

    elapsed = (
        time.time()
        - start_time
    )

    print(
        "[LIDAR VIS] Navigation visualizations "
        f"completed in {elapsed:.2f} seconds."
    )

    return files