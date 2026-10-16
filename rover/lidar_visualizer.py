import os
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


def create_lidar_visualizations(points, output_dir):
    """
    Create four visualizations from the FULL Cartesian LiDAR point cloud.

    Expected AEGIS coordinate convention:
        +X = FRONT
        +Y = LEFT
        +Z = UP

    Creates:
        XY  -> top-down view
        XZ  -> side view
        YZ  -> front/elevation view
        XYZ -> 3D perspective

    No downsampling is performed here.
    """

    start_time = time.time()

    os.makedirs(output_dir, exist_ok=True)

    points = np.asarray(points)

    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(
            f"Expected LiDAR points shaped (N, 3+) but got {points.shape}"
        )

    # ----------------------------------------------------------
    # Extract XYZ
    # ----------------------------------------------------------

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Remove only invalid numeric values.
    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(z)
    )

    x = x[valid]
    y = y[valid]
    z = z[valid]

    point_count = len(x)

    print(
        f"[LIDAR VIS] Rendering {point_count:,} valid LiDAR points..."
    )

    # One timestamp shared by all four images.
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    files = {}


    # ==========================================================
    # 1. XY TOP-DOWN VIEW
    # ==========================================================

    top_down_file = os.path.join(
        output_dir,
        f"lidar_xy_top_{timestamp}.png"
    )

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.scatter(
        y,
        x,
        s=0.15,
        alpha=0.5,
        rasterized=True
    )

    # Rover at origin.
    ax.scatter(
        [0],
        [0],
        s=120,
        marker="^",
        label="AEGIS Rover"
    )

    ax.set_xlabel("Y (m)   +Y = LEFT")
    ax.set_ylabel("X (m)   +X = FRONT")

    ax.set_title(
        f"AEGIS LiDAR - XY Top-Down View\n"
        f"{point_count:,} points"
    )

    ax.set_aspect(
        "equal",
        adjustable="box"
    )

    ax.grid(True)
    ax.legend()

    plt.tight_layout()

    plt.savefig(
        top_down_file,
        dpi=150
    )

    plt.close(fig)

    files["xy_top"] = top_down_file

    print(
        f"[LIDAR VIS] XY top-down saved: {top_down_file}"
    )


    # ==========================================================
    # 2. XZ SIDE VIEW
    # ==========================================================

    xz_file = os.path.join(
        output_dir,
        f"lidar_xz_side_{timestamp}.png"
    )

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.scatter(
        x,
        z,
        s=0.15,
        alpha=0.5,
        rasterized=True
    )

    # Rover origin.
    ax.scatter(
        [0],
        [0],
        s=120,
        marker="^",
        label="AEGIS Rover"
    )

    ax.set_xlabel("X (m)   +X = FRONT")
    ax.set_ylabel("Z (m)   +Z = UP")

    ax.set_title(
        f"AEGIS LiDAR - XZ Side View\n"
        f"{point_count:,} points"
    )

    ax.grid(True)
    ax.legend()

    plt.tight_layout()

    plt.savefig(
        xz_file,
        dpi=150
    )

    plt.close(fig)

    files["xz_side"] = xz_file

    print(
        f"[LIDAR VIS] XZ side view saved: {xz_file}"
    )


    # ==========================================================
    # 3. YZ FRONT / ELEVATION VIEW
    # ==========================================================

    yz_file = os.path.join(
        output_dir,
        f"lidar_yz_front_{timestamp}.png"
    )

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.scatter(
        y,
        z,
        s=0.15,
        alpha=0.5,
        rasterized=True
    )

    ax.scatter(
        [0],
        [0],
        s=120,
        marker="^",
        label="AEGIS Rover"
    )

    ax.set_xlabel("Y (m)   +Y = LEFT")
    ax.set_ylabel("Z (m)   +Z = UP")

    ax.set_title(
        f"AEGIS LiDAR - YZ Front View\n"
        f"{point_count:,} points"
    )

    ax.grid(True)
    ax.legend()

    plt.tight_layout()

    plt.savefig(
        yz_file,
        dpi=150
    )

    plt.close(fig)

    files["yz_front"] = yz_file

    print(
        f"[LIDAR VIS] YZ front view saved: {yz_file}"
    )


    # ==========================================================
    # 4. FULL XYZ 3D VIEW
    # ==========================================================

    view_3d_file = os.path.join(
        output_dir,
        f"lidar_xyz_3d_{timestamp}.png"
    )

    fig = plt.figure(figsize=(11, 9))

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    ax.scatter(
        x,
        y,
        z,
        s=0.10,
        alpha=0.4,
        rasterized=True
    )

    # Rover origin.
    ax.scatter(
        [0],
        [0],
        [0],
        s=120,
        marker="^",
        label="AEGIS Rover"
    )

    ax.set_xlabel("X - Front")
    ax.set_ylabel("Y - Left")
    ax.set_zlabel("Z - Up")

    ax.set_title(
        f"AEGIS LiDAR - XYZ 3D View\n"
        f"{point_count:,} points"
    )

    # Perspective angle.
    ax.view_init(
        elev=25,
        azim=-60
    )

    ax.legend()

    plt.tight_layout()

    plt.savefig(
        view_3d_file,
        dpi=150
    )

    plt.close(fig)

    files["xyz_3d"] = view_3d_file

    print(
        f"[LIDAR VIS] XYZ 3D view saved: {view_3d_file}"
    )


    # ==========================================================
    # FINISHED
    # ==========================================================

    elapsed = time.time() - start_time

    print(
        f"[LIDAR VIS] All 4 visualizations completed in "
        f"{elapsed:.2f} seconds."
    )

    return files