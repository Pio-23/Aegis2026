
"""AEGIS current-scan LiDAR evidence.

Four tested rover-eye perspectives plus measured
connected surface regions.

Read-only perception processing. No motors or GPIO.

This reproduces the V4-style connected-region approach,
but is not yet the exact original Geometry V4 implementation.
"""

from pathlib import Path
import importlib.util
import json
import math
import time

import numpy as np


CELL_M = 0.035
Z_MIN = -0.49
Z_MAX = 1.50

VIEW_DIRECTIONS = (
    ("right", 1.0, 0.0, "01_right"),
    ("front", 0.0, 1.0, "02_front"),
    ("left", -1.0, 0.0, "03_left"),
    ("back", 0.0, -1.0, "04_back"),
)

DIRECTIONS = (
    "front",
    "front-right",
    "right",
    "back-right",
    "back",
    "back-left",
    "left",
    "front-left",
)



def get_renderer():
    """Load the existing tested 3D LiDAR renderer."""

    import sys
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]

    tests_dir = root / "tests"

    path = tests_dir / "aegis_lidar_front_3d.py"

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing existing renderer: {path}"
        )

    # The renderer imports helper functions from
    # other files inside the tests directory.
    #
    # Make those existing helpers importable.
    tests_path = str(tests_dir)

    if tests_path not in sys.path:
        sys.path.insert(0, tests_path)

    spec = importlib.util.spec_from_file_location(
        "aegis_render_latest_scan",
        path
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Could not load 3D renderer"
        )

    viewer = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(viewer)

    return viewer



def direction_from_xy(x, y):
    """Clockwise bearing from physical rover front."""

    angle = (
        math.degrees(math.atan2(x, y)) + 360.0
    ) % 360.0

    sector = int(
        (angle + 22.5) // 45
    ) % 8

    return DIRECTIONS[sector], round(angle, 1)


def measured_regions(points):
    """Group connected measured 3.5 cm cells."""

    xy = np.floor(
        points[:, :2] / CELL_M
    ).astype(np.int32)

    cells, inverse, hits = np.unique(
        xy,
        axis=0,
        return_inverse=True,
        return_counts=True
    )

    kept_ids = np.flatnonzero(hits >= 2)

    lookup = {
        (
            int(cells[i, 0]),
            int(cells[i, 1])
        ): int(i)
        for i in kept_ids
    }

    radius = np.hypot(
        points[:, 0],
        points[:, 1]
    )

    nearest_per_cell = np.full(
        len(cells),
        np.inf,
        dtype=np.float32
    )

    np.minimum.at(
        nearest_per_cell,
        inverse,
        radius
    )

    visited = set()
    regions = []

    for seed in kept_ids:
        seed = int(seed)

        if seed in visited:
            continue

        visited.add(seed)
        stack = [seed]
        component = []

        while stack:
            index = stack.pop()
            component.append(index)

            cx, cy = (
                int(cells[index, 0]),
                int(cells[index, 1])
            )

            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue

                    neighbor = lookup.get(
                        (cx + dx, cy + dy)
                    )

                    if (
                        neighbor is not None
                        and neighbor not in visited
                    ):
                        visited.add(neighbor)
                        stack.append(neighbor)

        if len(component) < 5:
            continue

        indices = np.asarray(
            component,
            dtype=np.int32
        )

        component_cells = cells[indices]

        low = component_cells.min(axis=0)
        high = component_cells.max(axis=0)

        center = (
            low.astype(float) + high + 1
        ) * CELL_M / 2

        direction, bearing = direction_from_xy(
            float(center[0]),
            float(center[1])
        )

        nearest = float(
            np.min(nearest_per_cell[indices])
        )

        regions.append({
            "direction": direction,
            "bearing_clockwise_from_front_deg": bearing,
            "center_sensor_xy_m": [
                round(float(v), 3)
                for v in center
            ],
            "extent_xy_m": [
                round(float(v), 3)
                for v in (high - low + 1) * CELL_M
            ],
            "nearest_sampled_return_m": round(
                nearest, 3
            ),
            "support_cells": len(component),
            "support_returns": int(
                hits[indices].sum()
            ),
            "possible_rover_self_echo": (
                nearest <= 0.60
            ),
            "type": (
                "connected_measured_surface_region"
            ),
            "not_a_confirmed_object": True,
        })

    regions.sort(
        key=lambda item: item["support_returns"],
        reverse=True
    )

    for i, region in enumerate(regions, 1):
        region["region_id"] = i

    directional = {}

    for name in DIRECTIONS:
        matching = [
            region
            for region in regions
            if region["direction"] == name
        ]

        matching.sort(
            key=lambda region:
            region["nearest_sampled_return_m"]
        )

        directional[name] = {
            "measured_region_centers": len(matching),
            "nearest_measured_region_m": (
                matching[0]["nearest_sampled_return_m"]
                if matching else None
            ),
            "example_regions": matching[:3],
            "zero_regions_does_not_mean_clear": True,
        }

    return regions, directional



def render_dark_rover_map(raw, regions, destination):
    """Render 10 cm scan-local geometry, front up/right right."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Provisional LiDAR -> rover conversion:
    # Rover front = LiDAR Y; rover left = -LiDAR X.
    front = raw[:, 1]
    left = -raw[:, 0]
    height = raw[:, 2]

    # Match NavigationMap's current height limits
    # and provisional rover-body exclusion.
    rover_mask = (
        (front >= -0.6096) & (front <= 0.1524)
        & (left >= -0.4572) & (left <= 0.4572)
    )

    selected = (
        np.isfinite(raw[:, :3]).all(axis=1)
        & (height > -0.27) & (height < 1.00)
        & ~rover_mask
    )

    if not np.any(selected):
        raise ValueError(
            "No mapping-height returns for dark map"
        )

    # Occupied 10 cm cells, same quantization as
    # the existing NavigationMap at heading zero.
    cells = np.column_stack((
        np.rint(front[selected] / 0.10).astype(np.int32),
        np.rint(left[selected] / 0.10).astype(np.int32),
    ))

    cells = np.unique(cells, axis=0)

    right_m = -cells[:, 1] * 0.10
    front_m = cells[:, 0] * 0.10

    fig, ax = plt.subplots(
        figsize=(9, 9),
        dpi=140
    )

    fig.patch.set_facecolor("#0b1120")
    ax.set_facecolor("#0b1120")

    ax.scatter(
        right_m,
        front_m,
        s=2.5,
        c="#54cbd4",
        alpha=0.8,
        linewidths=0,
        label="Measured occupied cells"
    )

    # Highlight the eight largest measured regions.
    for region in regions[:8]:
        x, y = region["center_sensor_xy_m"]

        ax.scatter(
            x, y,
            s=65,
            facecolors="none",
            edgecolors="#ffad53",
            linewidths=1.6,
            zorder=3
        )

        ax.annotate(
            f"R{region['region_id']}",
            (x, y),
            xytext=(5, 5),
            textcoords="offset points",
            color="#ffad53",
            fontsize=8
        )

    ax.scatter(
        0, 0,
        s=120,
        c="#65df98",
        edgecolors="white",
        zorder=4,
        label="LiDAR origin"
    )

    ax.annotate(
        "",
        xy=(0, 0.7),
        xytext=(0, 0),
        arrowprops=dict(
            arrowstyle="->",
            color="#65df98",
            lw=2
        )
    )

    ax.text(
        0.13, 0.62, "FRONT",
        color="#65df98",
        fontsize=10
    )

    extent = max(
        4.0,
        min(
            12.0,
            float(np.percentile(
                np.hypot(right_m, front_m),
                99
            )) + 0.5
        )
    )

    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_aspect("equal")

    ax.set_xlabel(
        "Rover RIGHT direction (m)",
        color="white"
    )

    ax.set_ylabel(
        "Rover FRONT direction (m)",
        color="white"
    )

    ax.set_title(
        "AEGIS | Rover-Oriented LiDAR Geometry\n"
        "10 cm measured cells | Scan-local view",
        color="white"
    )

    ax.grid(True, alpha=0.16, linestyle=":")
    ax.tick_params(colors="white")

    for spine in ax.spines.values():
        spine.set_color("#4b5563")

    ax.legend(
        facecolor="#0b1120",
        edgecolor="#4b5563",
        labelcolor="white",
        loc="upper left"
    )

    destination = Path(destination)

    fig.savefig(
        destination,
        bbox_inches="tight",
        facecolor=fig.get_facecolor()
    )

    plt.close(fig)
    return str(destination)

def prepare_lidar_evidence(
    cloud_filename,
    output_folder
):
    """Process one particular saved scan."""

    start = time.monotonic()

    cloud = Path(cloud_filename).resolve()
    output = Path(output_folder).resolve()

    if not cloud.is_file():
        raise FileNotFoundError(cloud)

    output.mkdir(
        parents=True,
        exist_ok=True
    )

    raw = np.loadtxt(
        cloud,
        dtype=np.float32,
        usecols=(0, 1, 2),
        ndmin=2
    )

    valid = (
        np.isfinite(raw).all(axis=1)
        & (raw[:, 2] >= Z_MIN)
        & (raw[:, 2] <= Z_MAX)
    )

    points = raw[valid]

    if len(points) < 50:
        raise ValueError(
            "Too few valid LiDAR points"
        )

    viewer = get_renderer()

    viewer.OUTPUT = output
    viewer.HORIZONTAL_FOV_DEG = 100.0

    views = {}

    # Same camera geometry as our successful
    # four-direction test.
    for name, dx, dy, suffix in VIEW_DIRECTIONS:
        filename = (
            f"{cloud.stem}_view_{suffix}.png"
        )

        # All four directions use a sensor-origin
        # virtual camera with a consistent 95-degree FOV.
        #
        # This is LiDAR geometry, not an RGB image.
        viewer.HORIZONTAL_FOV_DEG = 105.0

        eye = (0.0, 0.0, 0.0)

        target = (
            4.0 * dx,
            4.0 * dy,
            0.0
        )

        # Distinct filenames preserve older renders.
        filename = (
            f"view_{suffix}.png"
        )

        viewer.render_view(
            points,
            eye=eye,
            target=target,
            filename=filename
        )

        image = output / filename

        if not image.is_file():
            raise RuntimeError(
                f"Missing rendered image: {image}"
            )

        views[name] = str(image)

    # Exclude zero-range echoes from region grouping
    # only. The four images remain unchanged.
    geometry_valid = (
        np.hypot(
            points[:, 0],
            points[:, 1]
        ) > 0.05
    )

    regions, directions = measured_regions(
        points[geometry_valid]
    )

    summary = {
        "schema": (
            "AEGIS_CONNECTED_MEASURED_REGIONS_EXPERIMENTAL_V1"
        ),
        "source_cloud": str(cloud),
        "provisional_lidar_frame": {
            "front": "+Y",
            "right": "+X",
            "left": "-X",
            "back": "-Y",
            "legacy_nav_map_uses_different_axes": True,
            "physical_transform_verified": False,
        },
        "selected_returns": int(len(points)),
        "height_relative_to_lidar_m": [
            Z_MIN,
            Z_MAX
        ],
        "region_count": len(regions),
        "directions": directions,
        "largest_regions": regions[:8],
        "possible_self_echo_regions": [
            region
            for region in regions
            if region["possible_rover_self_echo"]
        ][:8],
        "warnings": [
            "These are measured regions, not verified objects.",
            "Orange is a near-range color, not a rover label.",
            "Near returns can be rover body or external obstacles.",
            "Region centers do not define full boundaries.",
            "Missing returns do not establish free space.",
            "Distances are measured from the LiDAR origin.",
            "This is a historical snapshot once the rover moves.",
            "No movement is authorized by this data.",
            "Grouping is V4-inspired, not the exact original V4."
        ]
    }

    summary_file = (
        output /
        f"{cloud.stem}_measured_regions_v1.json"
    )

    summary_file.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    dark_map_file = output / (
        "view_05_topdown.png"
    )

    render_dark_rover_map(
        raw,
        regions,
        dark_map_file
    )

    views["map_topdown"] = str(dark_map_file)

    return {
        "source_cloud": str(cloud),
        "views": views,
        "summary_file": str(summary_file),
        "region_count": len(regions),
        "render_and_group_seconds": round(
            time.monotonic() - start,
            2
        ),
    }
