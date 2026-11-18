"""Full-return LiDAR directional coverage for AEGIS."""

from pathlib import Path
import json
import shutil
import numpy as np

NAMES = (
    "front", "front_right", "right",
    "back_right", "back", "back_left",
    "left", "front_left"
)

def enrich_package_geometry(cloud_filename):
    cloud = Path(cloud_filename)
    output = cloud.parent / "lidar_package" / "geometry_regions.json"

    points = np.loadtxt(
        cloud, usecols=(0, 1, 2), dtype=np.float32
    )

    x, y, z = points.T
    r = np.hypot(x, y)

    keep = (
        np.isfinite(points).all(axis=1)
        & np.isfinite(r)
        & (r > 0.001)
        & (z >= -0.49)
        & (z <= 1.50)
    )

    x, y, z, r = x[keep], y[keep], z[keep], r[keep]

    # Sensor: +Y FRONT, +X RIGHT
    angle = (
        np.degrees(np.arctan2(x, y)) + 360
    ) % 360

    sector = (
        np.floor((angle + 22.5) / 45).astype(int) % 8
    )

    directions = {}
    bins = [0, .35, .7, 1, 2, 3, 5, 10, np.inf]
    labels = [
        "0-0.35", "0.35-0.7", "0.7-1",
        "1-2", "2-3", "3-5", "5-10", "10+"
    ]

    for i, name in enumerate(NAMES):
        mask = sector == i
        d = r[mask]
        far = d[d >= 0.7]

        counts, _ = np.histogram(d, bins=bins)

        directions[name] = {
            "roofcut_return_count": int(len(d)),
            "nearest_measured_xy_m": (
                round(float(d.min()), 3) if len(d) else None
            ),
            "p10_measured_xy_m": (
                round(float(np.percentile(d, 10)), 3)
                if len(d) else None
            ),
            "near_origin_under_0_7m_count":
                int(np.count_nonzero(d < 0.7)),
            "nearest_at_or_beyond_0_7m_xy_m": (
                round(float(far.min()), 3)
                if len(far) else None
            ),
            "range_counts_xy_m": dict(
                zip(labels, [int(v) for v in counts])
            ),
            "navigation_height_return_count": int(
                np.count_nonzero(
                    (z[mask] >= -0.27)
                    & (z[mask] <= 1.0)
                )
            )
        }

    result = {
        "frame": "Provisional: +Y front, +X right",
        "sensor_relative_z_m": [-0.49, 1.50],
        "roofcut_returns": int(len(r)),
        "near_field_note": (
            "Under 0.7m may include rover, obstacles or noise. "
            "All measurements retained."
        ),
        "safety_note": (
            "No returns means UNKNOWN, never verified clear. "
            "Distances are from sensor, not bumper."
        ),
        "directions": directions
    }

    data = json.loads(output.read_text())
    backup = output.parent / ".backup" / "geometry_regions_original.json"
    backup.parent.mkdir(exist_ok=True)

    if not backup.exists():
        shutil.copy2(output, backup)

    data["directional_point_coverage_v1"] = result
    temp = output.with_suffix(".json.tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(output)

    return result


if __name__ == "__main__":
    cloud = Path(
        "stream/static/trips/20261117_153110/"
        "cloud_20261117_153250.txt"
    )

    result = enrich_package_geometry(cloud)

    print("\n=== AEGIS DIRECTIONAL COVERAGE ===")

    total = 0
    for name, d in result["directions"].items():
        count = d["roofcut_return_count"]
        total += count
        print(
            f"{name:13} "
            f"points={count:7} "
            f"nearest={d['nearest_measured_xy_m']}m"
        )

    assert total == result["roofcut_returns"]
    print("\nPASS: All selected returns accounted for.")
    print("Existing geometry regions preserved.")
    print("No hardware or API calls.")
