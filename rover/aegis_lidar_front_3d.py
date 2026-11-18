
"""AEGIS front-facing 3D LiDAR visualization.

Creates two perspective images from a saved XYZ cloud:
- LiDAR-eye front view
- Angled 3D view from behind

Color represents distance from the virtual viewpoint.
Read-only: no motors, ROS, UART or hardware access.
"""

import math
from pathlib import Path

import numpy as np
from rover.aegis_lidar_full_geometry_audit import save_png


CLOUD = Path(
    "stream/static/trips/20261112_175609/"
    "cloud_20261112_175737.txt"
)

OUTPUT = Path(
    "stream/static/ros_slam_tests/nov12_map"
)

WIDTH = 1280
HEIGHT = 800
HORIZONTAL_FOV_DEG = 100.0

# Assumption from our current top-down maps.
# Physical rover-forward alignment is not verified.
FORWARD_AXIS = "+x"

# Preserve the detailed expanded geometry.
Z_MIN = -0.49
Z_MAX = 1.50


def orient_cloud(points):
    """Convert saved sensor axes into assumed
    rover-forward X, left Y, up Z coordinates.
    """

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    if FORWARD_AXIS == "+x":
        return points

    if FORWARD_AXIS == "-x":
        return np.column_stack((-x, -y, z))

    if FORWARD_AXIS == "+y":
        return np.column_stack((y, -x, z))

    if FORWARD_AXIS == "-y":
        return np.column_stack((-y, x, z))

    raise ValueError("Unsupported FORWARD_AXIS")


def render_view(points, eye, target, filename):
    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)

    # Construct virtual 3D camera coordinates.
    forward = target - eye
    forward /= np.linalg.norm(forward)

    world_up = np.array(
        [0.0, 0.0, 1.0],
        dtype=np.float32
    )

    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)

    up = np.cross(right, forward)

    relative = points - eye

    # Depth along camera's forward axis.
    depth = relative @ forward

    valid_depth = (
        (depth > 0.20)
        & (depth < 12.0)
    )

    relative = relative[valid_depth]
    depth = depth[valid_depth]

    focal = (
        (WIDTH / 2.0)
        / math.tan(
            math.radians(HORIZONTAL_FOV_DEG) / 2
        )
    )

    # Perspective projection of 3D XYZ points.
    columns = np.rint(
        WIDTH / 2.0
        + focal * (relative @ right) / depth
    ).astype(np.int32)

    rows = np.rint(
        HEIGHT / 2.0
        - focal * (relative @ up) / depth
    ).astype(np.int32)

    inside = (
        (rows >= 0)
        & (rows < HEIGHT)
        & (columns >= 0)
        & (columns < WIDTH)
    )

    rows = rows[inside]
    columns = columns[inside]
    depth = depth[inside]

    # Z-buffer: nearest point to virtual camera
    # wins at each screen pixel.
    depth_image = np.full(
        WIDTH * HEIGHT,
        np.inf,
        dtype=np.float32
    )

    indices = (
        rows.astype(np.int64) * WIDTH
        + columns
    )

    np.minimum.at(
        depth_image,
        indices,
        depth
    )

    depth_image = depth_image.reshape(
        HEIGHT, WIDTH
    )

    # Slightly enlarge plotted points for display.
    # This changes only the picture, not XYZ data.
    padded = np.pad(
        depth_image,
        1,
        mode="constant",
        constant_values=np.inf
    )

    displayed_depth = np.full_like(
        depth_image,
        np.inf
    )

    for dr in range(3):
        for dc in range(3):
            np.minimum(
                displayed_depth,
                padded[
                    dr:dr + HEIGHT,
                    dc:dc + WIDTH
                ],
                out=displayed_depth
            )

    image = np.full(
        (HEIGHT, WIDTH, 3),
        (17, 23, 35),
        dtype=np.uint8
    )

    observed = np.isfinite(displayed_depth)

    # RGB depth palette:
    # orange/yellow = near,
    # green/cyan = medium,
    # blue/purple = far.
    distance_stops = [
        0.2, 1.0, 2.0, 4.0, 7.0, 12.0
    ]

    depth_colors = [
        (242, 105, 71),
        (239, 182, 64),
        (136, 215, 116),
        (62, 194, 204),
        (81, 123, 223),
        (165, 103, 218),
    ]

    for channel in range(3):
        values = [
            color[channel]
            for color in depth_colors
        ]

        image[:, :, channel][observed] = (
            np.interp(
                displayed_depth[observed],
                distance_stops,
                values
            ).astype(np.uint8)
        )

    path = OUTPUT / filename
    save_png(path, image)

    print("Saved:", path)
    print(
        "Measurements projected into view:",
        f"{len(depth):,}"
    )

    return path


def main():
    if not CLOUD.is_file():
        raise SystemExit(
            f"Saved LiDAR cloud missing: {CLOUD}"
        )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=== AEGIS FRONT-FACING 3D VIEW ===")

    raw = np.loadtxt(
        CLOUD,
        dtype=np.float32,
        usecols=(0, 1, 2),
        ndmin=2
    )

    valid = (
        np.isfinite(raw).all(axis=1)
        & (raw[:, 2] >= Z_MIN)
        & (raw[:, 2] <= Z_MAX)
    )

    points = orient_cloud(raw[valid])

    print("Original points:", f"{len(raw):,}")
    print("Selected 3D points:", f"{len(points):,}")
    print("Assumed forward direction:", FORWARD_AXIS)

    print("\nGenerating front-facing perspective...")

    render_view(
        points,
        eye=(0.0, 0.0, 0.0),
        target=(4.0, 0.0, 0.0),
        filename="nov12_lidar_front_3d.png"
    )

    print("\nGenerating angled 3D perspective...")

    render_view(
        points,
        eye=(-1.5, -2.0, 1.2),
        target=(1.5, 0.0, 0.0),
        filename="nov12_lidar_angled_3d.png"
    )

    print("\n=== COMPLETE ===")
    print("Orange/yellow: closer measurements")
    print("Green/cyan: medium distances")
    print("Blue/purple: farther measurements")
    print("Dark background: no rendered point")
    print()
    print("Both are static views of actual XYZ data.")
    print("Camera direction remains provisional.")
    print("No hardware commands executed.")


if __name__ == "__main__":
    main()
