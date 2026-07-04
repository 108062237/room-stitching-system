#!/usr/bin/env python3
"""
Step 4.5 (fixed): Draw floorplan overlay using the SAME layout projection
convention as tool_generate_gtsam_edges.py.

Key fixes:
1. Keeps visualization consistent with Step 1 edge generation.
2. Adds scale diagnostics.
3. Exposes --layout_z so you can deliberately keep it identical across steps.
"""

from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = [
    "Noto Sans CJK JP",
    "Arial Unicode MS",
    "Microsoft YaHei",
    "SimHei",
    "sans-serif",
]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np

try:
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.utils.labels import get_room_labels, get_display_label
from src.utils.axis_align import align_polys_to_xy, rotate_points

_LAYOUTHUB = Path(__file__).resolve().parent.parent.parent / "LayoutHub"
if _LAYOUTHUB.exists():
    sys.path.insert(0, str(_LAYOUTHUB))

try:
    from utils.geom import rectify_polygon
except ImportError:
    rectify_polygon = None

try:
    from utils.post_proc import np_coor2xy

    _HAS_COOR2XY = True
except Exception:
    np_coor2xy = None
    _HAS_COOR2XY = False


def load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text())


def ensure_parent_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def se2_apply(pose: Tuple[float, float, float], pts: np.ndarray) -> np.ndarray:
    x, y, th = pose
    c, s = math.cos(th), math.sin(th)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    return (pts @ R.T) + np.array([x, y], dtype=np.float64)


def load_layout_gt_txt_as_local_xy(
    txt_path: Path, pano_w: int, pano_h: int, layout_z: float
) -> Optional[np.ndarray]:
    if not txt_path.exists() or not _HAS_COOR2XY:
        return None

    pts = []
    for line in txt_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) >= 2:
            pts.append([float(parts[0]), float(parts[1])])

    if len(pts) < 2:
        return None

    if len(pts) % 2 == 0:
        floor_pixel = np.array(
            [
                pts[i] if pts[i][1] > pts[i + 1][1] else pts[i + 1]
                for i in range(0, len(pts), 2)
            ],
            dtype=np.float64,
        )
    else:
        floor_pixel = np.array(pts, dtype=np.float64)

    if len(floor_pixel) < 3:
        return None

    floor_xy = np_coor2xy(
        floor_pixel,
        z=layout_z,
        coorW=pano_w,
        coorH=pano_h,
        floorW=pano_w,
        floorH=pano_w,
    )
    center = pano_w / 2 - 0.5
    floor_xy[:, 0] -= center
    floor_xy[:, 1] -= center
    floor_xy[:, 1] = -floor_xy[:, 1]

    if rectify_polygon is not None:
        try:
            floor_xy = rectify_polygon(floor_xy)
        except Exception:
            pass

    return floor_xy.astype(np.float64)


def print_scale_diagnostics(
    poses: Dict[str, Dict[str, float]], polys_world: list[np.ndarray]
) -> None:
    if len(poses) >= 2:
        ids = sorted(poses.keys())
        pose_xy = np.array(
            [[poses[i]["x"], poses[i]["y"]] for i in ids], dtype=np.float64
        )
        pose_pair_dists = []
        for i in range(len(pose_xy)):
            for j in range(i + 1, len(pose_xy)):
                pose_pair_dists.append(float(np.linalg.norm(pose_xy[i] - pose_xy[j])))
        med_pose_dist = float(np.median(pose_pair_dists)) if pose_pair_dists else 0.0
    else:
        med_pose_dist = 0.0

    room_diags = []
    for poly in polys_world:
        mins = poly.min(axis=0)
        maxs = poly.max(axis=0)
        room_diags.append(float(np.linalg.norm(maxs - mins)))

    med_room_diag = float(np.median(room_diags)) if room_diags else 0.0
    ratio = (
        med_room_diag / max(med_pose_dist, 1e-9) if med_pose_dist > 0 else float("inf")
    )
    print(
        f"[SCALE] median room bbox diag={med_room_diag:.4f}, median pose-pair dist={med_pose_dist:.4f}, ratio={ratio:.4f}"
    )
    if ratio > 10:
        print(
            "[WARNING] Layout polygons are much larger than pose distances. Check that Step 1/4/5 use the SAME layout_z and SAME projection convention."
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", type=str, required=True)
    ap.add_argument("--poses", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--draw_camera_points", action="store_true")
    ap.add_argument("--limit_nodes", type=int, default=0)
    ap.add_argument("--pano_w", type=int, default=1024)
    ap.add_argument("--pano_h", type=int, default=512)
    ap.add_argument(
        "--layout_z",
        type=float,
        default=50.0,
        help="must match Step 1 / edge generation",
    )
    ap.add_argument(
        "--no_axis_align",
        action="store_true",
        help="disable global visualization-only rotation to align dominant walls to XY axes",
    )
    args = ap.parse_args()

    scene_dir = Path(args.scene_dir)
    manifest = load_json(scene_dir / "manifest.json")
    nodes = manifest.get("nodes", [])
    label_map = get_room_labels(scene_dir)

    poses_data = load_json(Path(args.poses))
    poses = poses_data.get("poses", {})
    if not poses:
        raise RuntimeError("poses json contains no poses")

    polys_world = []
    cam_xy = []
    pids = []
    doors = []
    drawn = 0
    skipped_no_layout = 0
    skipped_parse_fail = 0
    skipped_no_pose = 0

    iter_nodes = nodes[: args.limit_nodes] if args.limit_nodes > 0 else nodes
    for n in iter_nodes:
        pid = n.get("pano_id", "")
        if not pid:
            continue
        if pid not in poses:
            skipped_no_pose += 1
            continue

        lp_path = scene_dir / "layout_gt" / f"{pid}.txt"
        if not lp_path.exists():
            skipped_no_layout += 1
            continue

        poly_local = load_layout_gt_txt_as_local_xy(
            lp_path, pano_w=args.pano_w, pano_h=args.pano_h, layout_z=args.layout_z
        )
        if poly_local is None:
            skipped_parse_fail += 1
            continue

        x = float(poses[pid]["x"])
        y = float(poses[pid]["y"])
        th = float(poses[pid]["theta"])
        poly_world = se2_apply((x, y, th), poly_local)

        for conn in n.get("connections", []):
            if "hotspot_xy" not in conn:
                continue
            hx, hy = conn["hotspot_xy"]
            gx = x + math.cos(th) * hx - math.sin(th) * hy
            gy = y + math.sin(th) * hx + math.cos(th) * hy
            doors.append((gx, gy))

        polys_world.append(poly_world)
        cam_xy.append([x, y])
        pids.append(pid)
        drawn += 1

    if drawn == 0:
        raise RuntimeError("No layout polygons drawn.")

    cam_xy = np.array(cam_xy, dtype=np.float64) if cam_xy else np.zeros((0, 2))
    print_scale_diagnostics(poses, polys_world)

    if not args.no_axis_align:
        polys_world, align_angle, align_origin = align_polys_to_xy(polys_world)
        if cam_xy.shape[0] > 0:
            cam_xy = rotate_points(cam_xy, align_angle, align_origin)
        if doors:
            doors_xy = np.array(doors, dtype=np.float64)
            doors = [tuple(p) for p in rotate_points(doors_xy, align_angle, align_origin)]
        print(
            f"[AXIS_ALIGN] rotated visualization by {math.degrees(align_angle):.3f} deg to align dominant walls to XY axes"
        )

    out_path = Path(args.out)
    ensure_parent_dir(out_path)

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)

    if HAS_SHAPELY and len(polys_world) > 1:
        shapely_polys = [Polygon(p).buffer(0) for p in polys_world]
        union_poly = unary_union(shapely_polys)
        if union_poly.geom_type == "MultiPolygon":
            for geom in union_poly.geoms:
                ex_x, ex_y = geom.exterior.xy
                ax.plot(ex_x, ex_y, color="black", linewidth=3)
        else:
            ex_x, ex_y = union_poly.exterior.xy
            ax.plot(ex_x, ex_y, color="black", linewidth=3)

    for i, poly in enumerate(polys_world):
        closed = np.vstack([poly, poly[0:1]])
        ax.plot(
            closed[:, 0],
            closed[:, 1],
            color="black",
            linewidth=1.5,
            alpha=0.8,
            zorder=3,
        )
        ax.fill(poly[:, 0], poly[:, 1], alpha=args.alpha, zorder=2)
        cx, cy = cam_xy[i]
        label_text = get_display_label(pids[i], label_map)
        ax.text(
            cx,
            cy - 0.1,
            label_text,
            fontsize=9,
            ha="center",
            va="top",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1),
            zorder=10,
        )

    if doors:
        dx = [d[0] for d in doors]
        dy = [d[1] for d in doors]
        ax.scatter(
            dx,
            dy,
            marker="s",
            color="#FF4500",
            edgecolor="black",
            s=45,
            zorder=6,
            label="Doors",
        )

    if args.draw_camera_points and cam_xy.shape[0] > 0:
        ax.scatter(cam_xy[:, 0], cam_xy[:, 1], s=25, c="black", zorder=5)

    ax.set_title(
        f"{scene_dir.name} | polygons drawn={drawn}, "
        f"skipped(no_pose={skipped_no_pose}, no_layout={skipped_no_layout}, parse_fail={skipped_parse_fail})"
    )
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.5)

    all_pts = np.vstack(polys_world)
    xmin, ymin = np.min(all_pts, axis=0)
    xmax, ymax = np.max(all_pts, axis=0)
    pad = 0.1 * max(xmax - xmin, ymax - ymin, 1e-6)
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)

    fig.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)

    print(f"[OK] wrote floorplan overlay -> {out_path}")
    print(f"     polygons drawn={drawn}")


if __name__ == "__main__":
    main()
