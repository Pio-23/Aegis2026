#!/usr/bin/env python3
"""AEGIS: offline, read-only geometric LiDAR perception prototype.

This is NOT an obstacle-clearance oracle or a motor/navigation controller.
It retains every valid XYZ/intensity point and makes a distance-preserving,
height-layered, queryable representation from the COMPLETE cloud.

Example:
  python3 aegis_lidar_scene.py cloud_20261104_164038.txt --output /tmp/aegis_scene_164038
  python3 aegis_lidar_scene.py cloud_20261104_164038.txt --output /tmp/aegis_scene_164038 \
      --query-angle 0 --query-half-width 3 --query-layer mid

Dependencies: numpy, matplotlib (already used by AEGIS's visualizer).
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# Tentative bands in METERS ABOVE an automatically estimated floor plane.
# These are perceptual layers, NOT robot-clearance thresholds.
HEIGHT_BANDS = (
    ("floor_candidate", -0.10, 0.10),
    ("low", 0.10, 0.45),
    ("mid", 0.45, 1.10),
    ("high", 1.10, 1.80),
    ("overhead", 1.80, 4.0),
)
NEAR_FIELD_M = 0.25  # potentially rover/sensor self-returns: keep raw, flag uncertain
ANGLE_WIDTH_DEG = 1  # full 360 deg at 1-degree horizontal resolution
GRID_RESOLUTION_M = 0.10


def estimate_floor_z(x, y, z):
    """Empirical Z-mode of likely floor hits; verify against rover hardware."""
    radius = np.hypot(x, y)
    candidate = (radius > 0.4) & (radius < 4.0) & (z > -1.0) & (z < -0.30)
    if np.count_nonzero(candidate) < 200:
        raise ValueError("Insufficient plausible floor points: supply --floor-z manually")
    hist, edges = np.histogram(z[candidate], bins=np.arange(-1.0, -0.29, 0.01))
    peak = int(np.argmax(hist))
    return float((edges[peak] + edges[peak+1]) / 2.0)


def read_cloud(path):
    data = np.loadtxt(path, dtype=np.float32)
    if data.ndim != 2 or data.shape[1] != 4:
        raise ValueError("Expected one XYZ+intensity point per line (four columns)")
    finite = np.isfinite(data).all(axis=1)
    retained = data[finite]
    return retained, int((~finite).sum())


def angular_layers(x, y, height, min_range_m):
    """First return and measured spread by 1-degree direction and height band.

    No-return means UNKNOWN, never FREE. Original points remain queryable.
    """
    radius = np.hypot(x, y)
    angle_idx = (np.floor(np.degrees(np.arctan2(y, x)) + 180.0).astype(np.int32)) % 360
    count = np.zeros((len(HEIGHT_BANDS), 360), dtype=np.int32)
    uncertain_near = np.zeros_like(count)
    near = np.full((len(HEIGHT_BANDS), 360), np.inf, dtype=np.float32)
    # The 10th-percentile range is a robust second descriptor (not clearance).
    p10 = np.full_like(near, np.nan)
    for layer_idx, (_, zlo, zhi) in enumerate(HEIGHT_BANDS):
        height_ok = (height >= zlo) & (height < zhi)
        near_mask = height_ok & (radius < min_range_m)
        uncertain_near[layer_idx] = np.bincount(angle_idx[near_mask], minlength=360)
        mask = height_ok & (radius >= min_range_m)
        ids, dist = angle_idx[mask], radius[mask]
        count[layer_idx] = np.bincount(ids, minlength=360)
        np.minimum.at(near[layer_idx], ids, dist)
        # Group all points by angle using a single sort, rather than filtering 360x.
        order = np.argsort(ids, kind="stable")
        idx_sorted, dist_sorted = ids[order], dist[order]
        cut = np.searchsorted(idx_sorted, np.arange(361))
        for a in range(360):
            if cut[a+1] > cut[a]:
                p10[layer_idx, a] = float(np.quantile(dist_sorted[cut[a]:cut[a+1]], 0.1))
    near[~np.isfinite(near)] = np.nan
    return count, near, p10, uncertain_near


def grid_layers(x, y, height, radius, floor_z, res=GRID_RESOLUTION_M):
    """Visual spatial raster: 0 unknown, 1 floor returns, 2 potential obstacles,
    3 overhead returns, 4 near-field ambiguity. No free-space claim.
    """
    # Trim extreme spatial outliers from display only. All points remain in .npz.
    valid_plot = (np.abs(x) <= 12) & (np.abs(y) <= 12) & (height > -0.4) & (height < 4.0)
    x2 = x[valid_plot]; y2 = y[valid_plot]
    h2 = height[valid_plot]; r2 = radius[valid_plot]
    max_span = max(3.0, float(np.max(np.abs(np.concatenate((x2,y2))))))
    # Use origin-centered bounds; cell row is forward X, col is lateral Y.
    limit = min(12.5, max(4.0, math.ceil(max_span+0.3)))
    bins = np.arange(-limit, limit+res*1.01, res)
    dim = len(bins) - 1
    grid = np.zeros((dim,dim), dtype=np.uint8)
    col = np.floor((y2 + limit) / res).astype(np.int32)
    row = np.floor((x2 + limit) / res).astype(np.int32)
    inside = (row>=0)&(row<dim)&(col>=0)&(col<dim)
    row=row[inside];col=col[inside]; h2=h2[inside];r2=r2[inside]
    floor_mask = (h2 >= -.08) & (h2 <= .10) & (r2 >= NEAR_FIELD_M)
    over_mask = (h2 >= 1.80) & (r2 >= NEAR_FIELD_M)
    obstacle_mask = (h2 > .10) & (h2 < 1.80) & (r2 >= NEAR_FIELD_M)
    uncertain_mask = r2 < NEAR_FIELD_M
    # display priority: floor (1), overhead (3), obstacle (2), near field (4)
    # Avoid visually filling obstacles with distant ceiling returns.
    grid[row[over_mask], col[over_mask]]=3
    grid[row[floor_mask], col[floor_mask]]=1
    grid[row[obstacle_mask], col[obstacle_mask]]=2
    grid[row[uncertain_mask], col[uncertain_mask]]=4
    return grid, limit


def render_perception(outpath, count, near, grid, limit, source_name, floor_z,
                      map_radius_m=9.0):
    """Large scene-first visualization with a small distance/height guide.

    Display-only crop never changes point cloud, angle data or exact queries.
    """
    from matplotlib.patches import Patch

    fig = plt.figure(figsize=(16, 9), facecolor="white")
    # Most of the canvas is the spatial map; the guide is deliberately compact.
    map_ax = fig.add_axes([0.065, 0.095, 0.68, 0.79])
    guide_ax = fig.add_axes([0.79, 0.48, 0.185, 0.29])
    color_ax = fig.add_axes([0.795, 0.415, 0.175, 0.018])

    cmap = ListedColormap(["#f8fafc", "#d2d9df", "#104858", "#e0dcf9", "#e99a19"])
    norm = BoundaryNorm(np.arange(-0.5, 5.5, 1), cmap.N)

    # +Y is rover LEFT, drawn to the left; +X is FRONT, drawn upward.
    map_ax.imshow(grid, origin="lower", extent=[limit, -limit, -limit, limit],
                  interpolation="nearest", cmap=cmap, norm=norm)
    map_ax.plot(0, 0, marker="^", markersize=13, markeredgecolor="black",
                markeredgewidth=1.5, color="#f9fafb", zorder=20)
    map_ax.annotate("AEGIS  |  FRONT +X", (0, 0.18), xytext=(0, 17),
                    textcoords="offset points", ha="center", fontsize=10,
                    bbox={"boxstyle": "round,pad=0.25", "fc": "white", "alpha": 0.90})
    shown_radius = min(float(limit), float(map_radius_m))
    map_ax.set_xlim(shown_radius, -shown_radius)
    map_ax.set_ylim(-shown_radius, shown_radius)
    map_ax.set_aspect("equal")
    map_ax.set_xlabel("Lateral Y (m)  |  LEFT +  /  RIGHT -", fontsize=11)
    map_ax.set_ylabel("Forward X (m)  |  FRONT +  /  BACK -", fontsize=11)
    map_ax.set_title("Observed LiDAR geometry around AEGIS", fontsize=16,
                     weight="semibold", loc="left", pad=11)
    map_ax.tick_params(labelsize=9)
    map_ax.grid(alpha=0.14, linewidth=0.7)

    # Thin, concise supporting guide: distance is represented by color,
    # but NO RETURN is an unknown area, never evidence that it is clear.
    masked = np.ma.masked_invalid(near)
    depth_cmap = plt.get_cmap("viridis").copy()
    depth_cmap.set_bad("#e5e7eb")
    im = guide_ax.imshow(masked, origin="lower", aspect="auto",
                         extent=[-180, 180, -0.5, len(HEIGHT_BANDS)-0.5],
                         cmap=depth_cmap, vmin=0, vmax=6,
                         interpolation="nearest")
    guide_ax.set_yticks(np.arange(len(HEIGHT_BANDS)),
                        [n.replace("_candidate", "?") for n, _, _ in HEIGHT_BANDS])
    guide_ax.set_xticks([-180, -90, 0, 90, 180])
    guide_ax.set_xticklabels(["BACK", "RIGHT", "FRONT", "LEFT", "BACK"],
                             fontsize=7, rotation=30)
    guide_ax.tick_params(axis="y", labelsize=9)
    guide_ax.set_title("Distance by direction / height", fontsize=11, pad=10)
    cb = fig.colorbar(im, cax=color_ax, orientation="horizontal")
    cb.ax.tick_params(labelsize=8)
    cb.set_label("Nearest measured horizontal distance (m)  |  6+ clipped",
                 fontsize=8)

    fig.legend(handles=[Patch(color="#d2d9df", label="Possible floor"),
                        Patch(color="#104858", label="Potential obstacle"),
                        Patch(color="#e0dcf9", label="Overhead"),
                        Patch(color="#e99a19", label="Near sensor: uncertain")],
               loc="center left", bbox_to_anchor=(0.793, 0.265),
               fontsize=9, frameon=False, ncol=1)
    fig.text(0.795, 0.13,
             "Gray in distance guide = no return (UNKNOWN)\n"
             "Map is observations, NOT confirmed free space.\n"
             "Map zoom affects display only; all XYZ points retained.",
             fontsize=8.5, color="#475569", va="top")

    fig.suptitle(f"AEGIS  /  3D LiDAR perception replay   |   {source_name}",
                 fontsize=17, weight="bold", y=0.965)
    fig.text(0.50, 0.925,
             f"Approximate floor Z {floor_z:+.2f} m (unverified)  |  "
             f"1-degree ranges  |  5 height layers  |  full cloud retained",
             fontsize=10, color="#475569", ha="center")
    fig.savefig(outpath, dpi=150, facecolor="white")
    plt.close(fig)

def query_sector(points, floor_z, angle_deg, half_width_deg, layer_name):
    """Exact individual measured XYZ returns in any requested angular slice."""
    if layer_name not in {l[0] for l in HEIGHT_BANDS}:
        raise ValueError("Unknown height layer")
    x,y,z=points[:,0],points[:,1],points[:,2]
    h=z-floor_z
    r=np.hypot(x,y)
    ang=np.degrees(np.arctan2(y,x))
    delta=((ang-angle_deg+180)%360)-180
    lo,hi=next((lo,hi) for n,lo,hi in HEIGHT_BANDS if n==layer_name)
    sector=(np.abs(delta)<=half_width_deg)&(h>=lo)&(h<hi)
    ambiguous=sector & (r<NEAR_FIELD_M)
    mask=sector & (r>=NEAR_FIELD_M)
    if not np.any(mask):
        return {"count":0,"near_field_uncertain_count":int(ambiguous.sum()),
                "status":"NO MEASURED RETURN BEYOND UNCERTAIN NEAR FIELD - NOT CLEAR"}
    matches=points[mask]
    dist=r[mask]
    ii=int(np.argmin(dist))
    rec=matches[ii]
    return {"count":int(len(dist)),"near_field_uncertain_count":int(ambiguous.sum()),
            "nearest_horizontal_distance_m":float(dist[ii]),
            "nearest_point_xyz_intensity":[float(t) for t in rec],
            "p10_horizontal_distance_m":float(np.quantile(dist,.1)),
            "status":"RETURNS OBSERVED; NOT A COLLISION CLEARANCE"}


def create_perception_outputs(points, output_dir, *, source_name="AEGIS scan",
                              floor_z=None, map_radius_m=9.0,
                              save_full_geometry=True):
    """Process an in-memory full-resolution Nx4 XYZ/intensity scan (NO motor control).

    Usable by the existing LiDAR visualizer without re-reading a 42 MB text file.
    Never infer navigable free space or change any rover motion constraints.
    Returns paths to outputs so the caller can attach or inspect them later.
    """
    t0 = time.monotonic()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source = np.asarray(points, dtype=np.float32)
    if source.ndim != 2 or source.shape[1] != 4:
        raise ValueError("Expected Nx4 full XYZ/intensity cloud, not the reduced Nx3 visual points")
    finite = np.isfinite(source).all(axis=1)
    cloud = source[finite]
    if cloud.shape[0] == 0:
        raise ValueError("No valid LiDAR points")
    x, y, z = cloud[:, 0], cloud[:, 1], cloud[:, 2]
    calculated_floor_z = floor_z if floor_z is not None else estimate_floor_z(x, y, z)
    height = z - calculated_floor_z
    radius = np.hypot(x, y)
    counts, nearest, p10, uncertain_near = angular_layers(x, y, height, NEAR_FIELD_M)
    grid, limit = grid_layers(x, y, height, radius, calculated_floor_z)

    saved = {}
    if save_full_geometry:
        geom = output / "full_point_geometry.npz"
        np.savez_compressed(geom, points_xyz_intensity=cloud,
                            floor_z_sensor_frame=np.float32(calculated_floor_z),
                            angle_count=counts,
                            angle_nearfield_uncertain_count=uncertain_near,
                            angle_nearest_m=nearest, angle_p10_m=p10,
                            grid_returns=grid,
                            angle_centers_deg=np.arange(-179.5, 180.5, 1, dtype=np.float32))
        saved["full_point_geometry"] = str(geom)

    layers = []
    for idx, (name, lo, hi) in enumerate(HEIGHT_BANDS):
        layers.append({"name": name, "height_above_estimated_floor_m": [lo, hi],
                       "angles_deg_centers": [i + .5 for i in range(-180, 180)],
                       "point_counts": counts[idx].tolist(),
                       "near_field_uncertain_counts": uncertain_near[idx].tolist(),
                       "nearest_horizontal_m": [None if np.isnan(v) else round(float(v), 3)
                                                for v in nearest[idx]],
                       "p10_horizontal_m": [None if np.isnan(v) else round(float(v), 3)
                                            for v in p10[idx]]})
    info = {"source": source_name,
            "total_valid_points": int(len(cloud)),
            "invalid_points_removed": int((~finite).sum()),
            "floor_z_sensor_m_estimated": calculated_floor_z,
            "floor_estimate_warning": "Estimate only; calibrate actual sensor mounting and floor plane.",
            "near_field_flagged_count": int((radius < NEAR_FIELD_M).sum()),
            "angle_resolution_deg": 1,
            "coordinate_system": "+X FRONT, +Y LEFT, +Z UP; meters; angles +LEFT",
            "output_semantics": "Measured returns only; UNKNOWN is not FREE; no safety clearance",
            "layers": layers}
    angle_file = output / "angular_3d_ranges.json"
    angle_file.write_text(json.dumps(info, separators=(",", ":")), encoding="utf8")
    saved["angular_3d_ranges"] = str(angle_file)

    png = output / "perception_layers.png"
    render_perception(png, counts, nearest, grid, limit, source_name,
                      calculated_floor_z, map_radius_m=map_radius_m)
    saved["perception_layers"] = str(png)
    print(f"[LIDAR PERCEPTION] {len(cloud):,} full points, "
          f"{counts.size} directional/height cells, "
          f"{time.monotonic()-t0:.2f}s, outputs={output}")
    return saved


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cloud", help="Full LiDAR cloud text file, four columns XYZ+intensity")
    ap.add_argument("--output", required=True, help="Directory for isolated offline outputs")
    ap.add_argument("--floor-z", type=float, default=None,
                    help="Override estimated floor Z in sensor frame")
    ap.add_argument("--map-radius", type=float, default=9.0,
                    help="Visualization radius in meters (default 9); data always full")
    ap.add_argument("--query-angle", type=float, default=None)
    ap.add_argument("--query-half-width", type=float, default=2.0)
    ap.add_argument("--query-layer", choices=[i[0] for i in HEIGHT_BANDS], default="mid")
    args = ap.parse_args()
    if args.map_radius <= 0:
        ap.error("--map-radius must be positive")
    if not 0 < args.query_half_width <= 180:
        ap.error("--query-half-width must be within (0,180]")
    t0 = time.monotonic()
    points, invalid_count = read_cloud(args.cloud)
    saved = create_perception_outputs(
        points, args.output, source_name=Path(args.cloud).name,
        floor_z=args.floor_z, map_radius_m=args.map_radius)
    if args.query_angle is not None:
        floor_z = float(np.load(saved["full_point_geometry"])["floor_z_sensor_frame"])
        result = query_sector(points, floor_z, args.query_angle,
                              args.query_half_width, args.query_layer)
        print("EXACT RAW-POINT QUERY", json.dumps(result))
    print(json.dumps({"source": args.cloud,
                      "valid_points": len(points),
                      "invalid_points_removed": invalid_count,
                      "files": saved,
                      "seconds_with_read": round(time.monotonic()-t0, 2)}, indent=2))


if __name__ == "__main__":
    main()
