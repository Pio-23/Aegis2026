import os
import time

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
MAX_VIS_POINTS = 150_000

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
IMAGE_DPI = 110


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
        s=0.35,
        alpha=0.55
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
    # 2. CENTER-CORRIDOR XZ SLICE
    # ======================================================

    # Do NOT collapse the entire room into X/Z.
    #
    # Only show geometry physically near the
    # rover's center travel corridor.
    corridor_mask = (
        np.abs(y)
        <= CORRIDOR_HALF_WIDTH_M
    )

    corridor_x = x[
        corridor_mask
    ]

    corridor_y = y[
        corridor_mask
    ]

    corridor_z = z[
        corridor_mask
    ]

    corridor_full_count = len(
        corridor_x
    )

    (
        slice_x,
        slice_y,
        slice_z
    ) = _downsample_xyz(
        corridor_x,
        corridor_y,
        corridor_z
    )

    corridor_file = os.path.join(
        output_dir,
        f"lidar_corridor_xz_{timestamp}.png"
    )

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    ax.scatter(
        slice_x,
        slice_z,
        s=0.4,
        alpha=0.6
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
        "AEGIS LiDAR - Center Travel Corridor\n"
        f"|Y| <= {CORRIDOR_HALF_WIDTH_M:.2f} m | "
        f"{corridor_full_count:,} points | "
        f"{len(slice_x):,} rendered"
    )

    ax.grid(True)

    ax.legend(
        loc="upper right"
    )

    fig.savefig(
        corridor_file,
        dpi=IMAGE_DPI
    )

    plt.close(fig)

    files[
        "corridor_xz"
    ] = corridor_file

    print(
        "[LIDAR VIS] Corridor XZ saved: "
        f"{corridor_file}"
    )


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