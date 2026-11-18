"""Build AEGIS's unified LiDAR output package.

Uses an existing saved scan.
Does not operate hardware.
"""

from pathlib import Path
import json
import shutil

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rover.aegis_lidar_evidence import (
    prepare_lidar_evidence
)
from rover.navigation_map import NavigationMap


def render_roofcut(cloud, destination):
    """All-return top-down raster, roof cut at +1.5 m."""

    raw = np.loadtxt(
        cloud,
        dtype=np.float32,
        usecols=(0, 1, 2),
        ndmin=2
    )

    valid = (
        np.isfinite(raw).all(axis=1)
        & (raw[:, 2] >= -0.49)
        & (raw[:, 2] <= 1.50)
    )

    xyz = raw[valid]

    if len(xyz) == 0:
        raise ValueError("No roof-cut LiDAR returns")

    # Physical orientation:
    # right = original X; front = original Y.
    right = xyz[:, 0]
    front = xyz[:, 1]

    distances = np.hypot(right, front)

    extent = max(
        5.0,
        min(
            16.0,
            float(np.percentile(distances, 99.9))
            + 0.5
        )
    )

    # Fine 2.5 cm grid. All selected returns
    # contribute, including multiple returns
    # in the same XY cell.
    resolution = 0.025

    bins = int(np.ceil(
        (2.0 * extent) / resolution
    ))

    counts, _, _ = np.histogram2d(
        right,
        front,
        bins=(bins, bins),
        range=[
            [-extent, extent],
            [-extent, extent]
        ]
    )

    density = np.ma.masked_equal(
        counts.T,
        0
    )

    density = np.log1p(density)

    cmap = plt.colormaps["turbo"].copy()
    cmap.set_bad((0, 0, 0, 0))

    fig, ax = plt.subplots(
        figsize=(10, 10),
        dpi=150
    )

    fig.patch.set_facecolor("#0b1120")
    ax.set_facecolor("#0b1120")

    ax.imshow(
        density,
        origin="lower",
        extent=[
            -extent, extent,
            -extent, extent
        ],
        cmap=cmap,
        interpolation="nearest"
    )

    ax.scatter(
        [0], [0],
        s=85,
        c="#65df98",
        edgecolors="white",
        zorder=5
    )

    ax.annotate(
        "FRONT",
        xy=(0, 0.8),
        xytext=(0.4, 1.2),
        color="#65df98",
        arrowprops={
            "arrowstyle": "->",
            "color": "#65df98"
        }
    )

    ax.set_aspect("equal")
    ax.set_xlabel("RIGHT (m)", color="white")
    ax.set_ylabel("FRONT (m)", color="white")

    ax.set_title(
        "AEGIS | All-Return Roof-Cut Geometry\n"
        "Sensor-relative Z: -0.49 to +1.50 m",
        color="white"
    )

    ax.tick_params(colors="white")
    ax.grid(alpha=0.12)

    fig.savefig(
        destination,
        bbox_inches="tight",
        facecolor=fig.get_facecolor()
    )

    plt.close(fig)

    return int(len(xyz))


def build_scan_package(cloud_filename):
    cloud = Path(cloud_filename).resolve()

    if not cloud.is_file():
        raise FileNotFoundError(cloud)

    trip = cloud.parent
    output = trip / "lidar_package"
    output.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n=== BUILDING AEGIS LIDAR PACKAGE ===")

    evidence = prepare_lidar_evidence(
        str(cloud),
        str(output)
    )

    final_names = {
        "right": "view_01_right.png",
        "front": "view_02_front.png",
        "left": "view_03_left.png",
        "back": "view_04_back.png",
        "map_topdown": "view_06_navigation_dark.png",
    }

    for key, filename in final_names.items():
        old = Path(evidence["views"][key])
        new = output / filename

        if not old.is_file():
            raise FileNotFoundError(old)

        if old != new:
            old.replace(new)

        evidence["views"][key] = str(new)

    # Rich roof-cut geometry.
    roof_file = output / "view_05_roofcut.png"

    roof_points = render_roofcut(
        str(cloud),
        roof_file
    )

    evidence["views"]["roofcut_topdown"] = str(
        roof_file
    )

    # Existing NavigationMap renderer.
    # Manual scan: this is a single-scan map.
    nav = NavigationMap()

    nav.add_lidar_scan(
        filename=str(cloud),
        scan_frame="lidar_provisional"
    )

    white_file = (
        output / "view_07_navigation_white.png"
    )

    nav.render_map_png(str(white_file))

    evidence["views"]["map_white"] = str(
        white_file
    )

    # Preserve the old perception layer.
    candidates = list(
        trip.glob(
            "perception_*/perception_layers.png"
        )
    )

    if candidates:
        source = max(
            candidates,
            key=lambda p: p.stat().st_mtime
        )

        destination = (
            output / "perception_layers.png"
        )

        shutil.copy2(source, destination)

        evidence["views"]["perception_layers"] = (
            str(destination)
        )

    # Give the region JSON a reusable name.
    original_json = Path(
        evidence["summary_file"]
    )

    final_json = output / "geometry_regions.json"

    if original_json != final_json:
        original_json.replace(final_json)

    evidence["summary_file"] = str(final_json)

    required = [
        "right",
        "front",
        "left",
        "back",
        "roofcut_topdown",
        "map_topdown",
        "map_white",
    ]

    for key in required:
        if not Path(evidence["views"][key]).is_file():
            raise RuntimeError(
                "Missing output: " + key
            )

    # Archive obsolete scanner visualizations.
    # Do not delete source cloud or NPZ geometry.
    archive = trip / "legacy_visualizations"

    old_patterns = (
        "lidar_navigation_xy_*.png",
        "lidar_polar_range_*.png",
        "lidar_left_corridor_xz_*.png",
        "lidar_right_corridor_xz_*.png",
        "lidar_angular_10deg_*.json",
    )

    archived = []

    for pattern in old_patterns:
        for old in trip.glob(pattern):
            archive.mkdir(exist_ok=True)

            destination = archive / old.name

            if destination.exists():
                continue

            shutil.move(
                str(old),
                str(destination)
            )

            archived.append(old.name)

    evidence["output_dir"] = str(output)
    evidence["roofcut_returns"] = roof_points

    manifest = {
        "source_cloud": str(cloud),
        "roofcut_returns": roof_points,
        "region_count": evidence["region_count"],
        "views": evidence["views"],
        "geometry_json": str(final_json),
        "legacy_files_archived": archived,
        "map_note": (
            "White manual-test map contains one "
            "scan, not historical movement poses."
        ),
        "safety_note": (
            "Maps do not establish safe travel."
        ),
    }

    (output / "package_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8"
    )

    print("\n=== FINAL LIDAR FILES ===")

    for image in sorted(output.glob("*.png")):
        print(image.name)

    print("geometry_regions.json")
    print("package_manifest.json")
    print("Roof-cut returns:", roof_points)
    print("Regions:", evidence["region_count"])
    print("Legacy files archived:", len(archived))

    from rover.aegis_directional_coverage import enrich_package_geometry
    enrich_package_geometry(cloud_filename)
    print("[LIDAR] Directional coverage updated.")

    return evidence


if __name__ == "__main__":
    root = Path("stream/static/trips")

    clouds = list(root.glob("*/cloud_*.txt"))

    if not clouds:
        raise SystemExit(
            "No saved LiDAR clouds found."
        )

    latest = max(
        clouds,
        key=lambda p: p.stat().st_mtime
    )

    print("Latest cloud:", latest)

    build_scan_package(latest)
