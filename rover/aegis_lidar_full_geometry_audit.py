
"""AEGIS full LiDAR geometry comparison.

Generates top-down PNGs directly from saved XYZ points.
No ROS, motors, LiDAR hardware, or Arduino required.

These are point projections, NOT navigation occupancy maps.
"""

from pathlib import Path
import struct
import zlib

import numpy as np


CLOUD = Path(
    "stream/static/trips/20261112_175609/"
    "cloud_20261112_175737.txt"
)

OUTPUT = Path(
    "stream/static/ros_slam_tests/nov12_map"
)

IMAGE_SIZE = 1080
VIEW_RADIUS_M = 9.0
LASER_BINS = 720

# Mid-level obstacle band: 25-75 cm above ground.
MIN_Z = -0.29
MAX_Z = 0.21

MIN_RANGE = 0.25
MAX_RANGE = 12.0


def png_chunk(tag, payload):
    return (
        struct.pack(">I", len(payload))
        + tag
        + payload
        + struct.pack(
            ">I",
            zlib.crc32(tag + payload) & 0xFFFFFFFF
        )
    )


def save_png(path, image):
    height, width, channels = image.shape

    assert channels == 3

    raw = b"".join(
        bytes([0]) + row.tobytes()
        for row in image
    )

    png = (
        bytes.fromhex("89504e470d0a1a0a")
        + png_chunk(
            b"IHDR",
            struct.pack(
                ">IIBBBBB",
                width,
                height,
                8,
                2,
                0,
                0,
                0,
            ),
        )
        + png_chunk(
            b"IDAT",
            zlib.compress(raw, level=6),
        )
        + png_chunk(b"IEND", b"")
    )

    path.write_bytes(png)


def render_points(x, y, filename, enlarge=False):
    n = IMAGE_SIZE
    pixels_per_m = n / (2.0 * VIEW_RADIUS_M)
    center = n // 2

    image = np.full(
        (n, n, 3),
        245,
        dtype=np.uint8,
    )

    # Draw reference distance rings.
    yy, xx = np.ogrid[:n, :n]

    radial = np.sqrt(
        (xx - center) ** 2
        + (yy - center) ** 2
    )

    for radius_m in (1, 3, 6):
        ring = (
            np.abs(
                radial - radius_m * pixels_per_m
            ) <= 0.6
        )

        image[ring] = (210, 210, 210)

    # Forward X points upward.
    # Left Y points to the left.
    columns = np.rint(
        center - y * pixels_per_m
    ).astype(np.int32)

    rows = np.rint(
        center - x * pixels_per_m
    ).astype(np.int32)

    inside = (
        (rows >= 0)
        & (rows < n)
        & (columns >= 0)
        & (columns < n)
    )

    rows = rows[inside]
    columns = columns[inside]

    # Count all returns that land on each pixel.
    linear = rows.astype(np.int64) * n + columns

    counts = np.bincount(
        linear,
        minlength=n * n,
    ).reshape(n, n)

    hits = counts > 0

    image[hits] = (35, 54, 80)

    if enlarge:
        # Make individual nearest-hit dots easier to see.
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = rows + dr
                cc = columns + dc

                good = (
                    (rr >= 0) & (rr < n)
                    & (cc >= 0) & (cc < n)
                )

                image[rr[good], cc[good]] = (
                    30, 45, 75
                )

    # Mark LiDAR origin in red.
    image[
        center - 4:center + 5,
        center - 4:center + 5
    ] = (220, 35, 45)

    output_path = OUTPUT / filename

    save_png(output_path, image)

    print(
        f"Saved: {output_path} "
        f"({len(x):,} points)"
    )


def main():
    if not CLOUD.is_file():
        raise SystemExit(
            f"Cloud not found: {CLOUD}"
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)

    print("=== AEGIS FULL GEOMETRY AUDIT ===")

    points = np.loadtxt(
        CLOUD,
        dtype=np.float32,
        usecols=(0, 1, 2),
    )

    valid = np.isfinite(points).all(axis=1)
    points = points[valid]

    x, y, z = points.T

    horizontal_range = np.hypot(x, y)

    print(f"Raw finite points: {len(points):,}")

    # View 1: every XYZ point, projected top-down.
    render_points(
        x, y,
        "nov12_all_xyz_topdown.png",
    )

    # View 2: all valid returns in the chosen band.
    selected = (
        (z >= MIN_Z)
        & (z <= MAX_Z)
        & (horizontal_range >= MIN_RANGE)
        & (horizontal_range <= MAX_RANGE)
    )

    selected_indices = np.flatnonzero(selected)

    render_points(
        x[selected],
        y[selected],
        "nov12_mid_all_returns.png",
    )

    # View 3: choose one nearest hit in each direction.
    angles = np.arctan2(
        y[selected],
        x[selected],
    )

    bins = np.floor(
        (angles + np.pi)
        * LASER_BINS / (2.0 * np.pi)
    ).astype(np.int32)

    bins = np.clip(
        bins,
        0,
        LASER_BINS - 1,
    )

    distances = horizontal_range[selected]

    # Sort by angle bin, then nearest range.
    order = np.lexsort(
        (distances, bins)
    )

    sorted_bins = bins[order]

    first = np.empty(
        len(order),
        dtype=bool,
    )

    first[0] = True
    first[1:] = (
        sorted_bins[1:] != sorted_bins[:-1]
    )

    nearest_indices = selected_indices[
        order[first]
    ]

    render_points(
        x[nearest_indices],
        y[nearest_indices],
        "nov12_mid_nearest_720.png",
        enlarge=True,
    )

    print("\n=== GEOMETRY RESULTS ===")

    print(
        "All XYZ points:",
        len(points)
    )

    print(
        "Points in selected height band:",
        len(selected_indices)
    )

    print(
        "Retained nearest 2D hits:",
        len(nearest_indices)
    )

    print(
        "Farthest raw horizontal hit:",
        f"{np.max(horizontal_range):.2f} m"
    )

    print(
        "Farthest retained 2D hit:",
        f"{np.max(horizontal_range[nearest_indices]):.2f} m"
    )

    print("\nImage orientation:")
    print("  Forward = up")
    print("  Left = left")
    print("  Red square = LiDAR origin")
    print("  Rings = 1m, 3m, 6m")
    print("  Gray background = no plotted return")
    print()
    print(
        "These are projections, not occupancy maps."
    )
    print("No rover hardware accessed.")


if __name__ == "__main__":
    main()
