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

# Enable the new 3D geometric perception processor.
# Keep disabled until integration testing is complete.
ENABLE_3D_PERCEPTION = True

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
    """Generate only retained 3D perception outputs.

    Legacy XY, XZ, polar and 10-degree outputs
    are no longer generated.

    Preserve the original return keys for callers.
    """

    start_time = time.time()

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    if not ENABLE_3D_PERCEPTION:
        print(
            "[LIDAR PERCEPTION] Disabled."
        )
        return {}

    points = np.asarray(
        points,
        dtype=np.float32
    )

    if (
        points.ndim != 2
        or points.shape[1] < 4
    ):
        raise ValueError(
            "Perception requires Nx4 XYZ/intensity points"
        )

    from rover.aegis_lidar_scene import (
        create_perception_outputs
    )

    timestamp = time.strftime(
        "%Y%m%d_%H%M%S"
    )

    scene_dir = os.path.join(
        output_dir,
        f"perception_{timestamp}"
    )

    scene_files = create_perception_outputs(
        points[:, :4],
        scene_dir,
        source_name=f"AEGIS scan {timestamp}",
        map_radius_m=9.0,
        save_full_geometry=True
    )

    expected = {
        "scene_perception_png": "perception_layers",
        "scene_ranges_json": "angular_3d_ranges",
        "scene_full_geometry": "full_point_geometry"
    }

    result = {}

    for output_key, source_key in expected.items():
        filename = scene_files.get(source_key)

        if (
            not filename
            or not os.path.isfile(filename)
        ):
            raise RuntimeError(
                f"Perception output missing: {source_key}"
            )

        result[output_key] = filename

    elapsed = time.time() - start_time

    print(
        "[LIDAR PERCEPTION] "
        "Perception layers and full geometry saved. "
        f"Time: {elapsed:.2f}s"
    )

    return result
