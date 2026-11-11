
"""Read-only forward LiDAR corridor measurements.

Never authorizes rover movement.
"""

import json
import sys
import numpy as np

from rover.aegis_lidar_scene import HEIGHT_BANDS

# Experimental dimensions, NOT a calibrated rover footprint.
HALF_WIDTH_M = 0.46
CELL_M = 0.10

BANDS_M = [
    (0.50, 0.75),
    (0.75, 1.00),
    (1.00, 1.50),
    (1.50, 2.00),
]

LAYERS_M = {
    name: (minimum, maximum)
    for name, minimum, maximum in HEIGHT_BANDS
    if name in ("low", "mid")
}


def query_forward_corridor(npz_path):
    """Measure geometry within a forward rectangular region.

    +X = forward, +Y = left, at scan time.
    All distances are in meters.
    Empty cells are UNKNOWN, not verified free space.
    """

    with np.load(npz_path, allow_pickle=False) as data:
        points = data["points_xyz_intensity"]
        floor_z = float(data["floor_z_sensor_frame"])

    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("Expected an N x 3+ XYZ array")

    valid = np.isfinite(points[:, :3]).all(axis=1)
    points = points[valid]

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    height = z - floor_z

    report = {
        "coordinate_frame": "+X forward, +Y left, scan-time sensor frame",
        "floor_z_estimate_m": round(floor_z, 3),
        "corridor_half_width_m_uncalibrated": HALF_WIDTH_M,
        "xy_cell_size_m_approx": CELL_M,
        "bands": [],
        "grid_front_to_far": {},
        "warning": (
            "Measured returns only. ? means UNKNOWN, not free. "
            "Region inside 0.50 m is not evaluated. "
            "No collision clearance or movement authorization. "
            "Self-returns and floor estimate remain uncalibrated."
        ),
    }

    lanes = [
        ("left", HALF_WIDTH_M / 3, HALF_WIDTH_M),
        ("center", -HALF_WIDTH_M / 3, HALF_WIDTH_M / 3),
        ("right", -HALF_WIDTH_M, -HALF_WIDTH_M / 3),
    ]

    # PART 1: Numerical XYZ measurements by distance,
    # height, and lateral position.
    for x0, x1 in BANDS_M:
        band = {
            "forward_m": [x0, x1],
            "layers": {},
        }

        for name, (h0, h1) in LAYERS_M.items():
            layer = {}

            for side, y0, y1 in lanes:
                mask = (
                    (x >= x0) &
                    (x < x1) &
                    (y >= y0) &
                    (y < y1) &
                    (height >= h0) &
                    (height < h1)
                )

                selected = points[mask]
                count = len(selected)

                if count:
                    xy_cells = np.floor(
                        selected[:, :2] / CELL_M
                    ).astype(np.int32)

                    nearest = selected[
                        np.argmin(selected[:, 0])
                    ]

                    result = {
                        "points": int(count),
                        "observed_xy_cells": int(
                            len(np.unique(xy_cells, axis=0))
                        ),
                        "nearest_xyz_m": [
                            round(float(v), 3)
                            for v in nearest[:3]
                        ],
                    }
                else:
                    result = {
                        "points": 0,
                        "observed_xy_cells": 0,
                        "nearest_xyz_m": None,
                    }

                layer[side] = result

            band["layers"][name] = layer

        report["bands"].append(band)

    # PART 2: Compact spatial evidence grid.
    #
    # X = 10 or more measured returns
    # x = 1 to 9 measured returns
    # ? = no returns (UNKNOWN, not free)
    #
    # Each row represents roughly 10 cm forward.
    # Columns run LEFT to RIGHT.

    x_edges = np.linspace(0.50, 2.00, 16)
    y_edges = np.linspace(
        -HALF_WIDTH_M,
        HALF_WIDTH_M,
        11
    )

    for name, (h0, h1) in LAYERS_M.items():
        mask = (
            (x >= 0.50) &
            (x < 2.00) &
            (np.abs(y) <= HALF_WIDTH_M) &
            (height >= h0) &
            (height < h1)
        )

        hist, _, _ = np.histogram2d(
            x[mask],
            y[mask],
            bins=[x_edges, y_edges]
        )

        rows = []

        for i, row in enumerate(hist):
            symbols = "".join(
                "X" if count >= 10
                else "x" if count > 0
                else "?"
                for count in row[::-1]
            )

            rows.append({
                "forward_m": round(float(x_edges[i]), 2),
                "cells_left_to_right": symbols,
            })

        report["grid_front_to_far"][name] = rows

    return report


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python3 rover/aegis_corridor_geometry.py "
            "/path/to/full_point_geometry.npz"
        )

    result = query_forward_corridor(sys.argv[1])
    print(json.dumps(result, indent=2))
