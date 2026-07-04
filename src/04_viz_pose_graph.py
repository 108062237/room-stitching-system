#!/usr/bin/env python3
"""
Step 4 (fixed): Visualize pose graph before/after optimization using the SAME
layout projection convention as tool_generate_gtsam_edges.py / 05_draw_floorplan_overlay.py.

Key fixes:
1. No longer relies on manifest['layout_gt_path'].
2. Reads scene_dir/layout_gt/<pano_id>.txt directly.
3. Uses LayoutHub np_coor2xy + center shift + Y flip + rectify_polygon,
   matching edge generation.
4. Adds simple scale diagnostics.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

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

import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.utils.labels import get_room_labels, get_display_label
from src.utils.axis_align import align_polys_to_xy, rotate_pose_dict

# LayoutHub loader (same convention as tool_generate_gtsam_edges.py / 05_draw_floorplan_overlay.py)
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


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def se2_apply(pose: Tuple[float, float, float], pts: np.ndarray) -> np.ndarray:
    x, y, th = pose
    c, s = math.cos(th), math.sin(th)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    return (pts @ R.T) + np.array([x, y], dtype=np.float64)


def align_polygon_to_xy(pts: np.ndarray) -> np.ndarray:
    if len(pts) < 2:
        return pts

    closed = np.vstack([pts, pts[0:1]])
    segs = closed[1:] - closed[:-1]
    lengths = np.linalg.norm(segs, axis=1)
    if not np.any(lengths > 1e-9):
        return pts

    longest = segs[int(np.argmax(lengths))]
    angle = math.atan2(float(longest[1]), float(longest[0]))
    target = round(angle / (math.pi / 2.0)) * (math.pi / 2.0)
    correction = target - angle

    c, s = math.cos(correction), math.sin(correction)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    return pts @ R.T


def load_layout_gt_txt_as_local_xy(
    txt_path: Path,
    pano_w: int = 1024,
    pano_h: int = 512,
    layout_z: float = 50.0,
) -> Optional[np.ndarray]:
    """Same projection convention as tool_generate_gtsam_edges.py."""
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


def plot_graph(
    ax,
    poses: Dict[str, Dict[str, float]],
    edges: List[Dict[str, Any]],
    title: str,
    draw_edges: bool = True,
    draw_labels: bool = True,
    manifest_nodes: Optional[List[Dict[str, Any]]] = None,
    label_map: Optional[Dict[str, str]] = None,
    label_font_size: int = 8,
):
    ids = sorted(poses.keys())
    xy = np.array([[poses[i]["x"], poses[i]["y"]] for i in ids], dtype=np.float64)
    ax.scatter(xy[:, 0], xy[:, 1], s=40, zorder=5)

    for i in ids:
        x, y, th = poses[i]["x"], poses[i]["y"], poses[i]["theta"]
        ax.arrow(
            x,
            y,
            0.25 * math.cos(th),
            0.25 * math.sin(th),
            head_width=0.06,
            length_includes_head=True,
            zorder=6,
            color="black",
        )

    if draw_edges:
        node_map = {n["pano_id"]: n for n in manifest_nodes} if manifest_nodes else {}
        for e in edges:
            i, j = e["i"], e["j"]
            if i not in poses or j not in poses:
                continue

            xi, yi, thi = poses[i]["x"], poses[i]["y"], poses[i]["theta"]
            xj, yj = poses[j]["x"], poses[j]["y"]

            hotspot_coords = None
            if node_map and i in node_map:
                conn_i = next(
                    (
                        c
                        for c in node_map[i].get("connections", [])
                        if c.get("neighbor") == j
                    ),
                    None,
                )
                if conn_i and "hotspot_xy" in conn_i:
                    hx, hy = conn_i["hotspot_xy"]
                    c_i, s_i = math.cos(thi), math.sin(thi)
                    gx = xi + c_i * hx - s_i * hy
                    gy = yi + s_i * hx + c_i * hy
                    hotspot_coords = (gx, gy)

            if hotspot_coords is not None:
                gx, gy = hotspot_coords
                ax.plot(
                    [xi, gx], [yi, gy], color="blue", linewidth=1, alpha=0.6, zorder=3
                )
                ax.plot(
                    [gx, xj], [gy, yj], color="blue", linewidth=1, alpha=0.6, zorder=3
                )
                ax.scatter([gx], [gy], marker="s", color="#FFA500", s=15, zorder=4)
            else:
                ax.plot(
                    [xi, xj], [yi, yj], color="blue", linewidth=1, alpha=0.6, zorder=3
                )

    if draw_labels:
        for i in ids:
            label_text = get_display_label(i, label_map) if label_map else i[-6:]
            ax.text(
                poses[i]["x"] + 0.1,
                poses[i]["y"] + 0.1,
                label_text,
                fontsize=label_font_size,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1),
                zorder=10,
            )

    if title:
        ax.set_title(title)
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.5)


def plot_extra_edges(ax, extra_points, poses, scene_dir, pano_w, pano_h, layout_z):
    for item in extra_points:
        src = item["src"].replace(".txt", "")
        dst = item["dst"].replace(".txt", "")
        if src not in poses or dst not in poses: continue
        
        # Connect room centers
        c_xi, c_yi = poses[src]["x"], poses[src]["y"]
        c_xj, c_yj = poses[dst]["x"], poses[dst]["y"]
        ax.plot([c_xi, c_xj], [c_yi, c_yj], color="red", linestyle=":", linewidth=1.5, alpha=0.6, zorder=8)
        
        src_txt = scene_dir / "layout_gt" / f"{src}.txt"
        dst_txt = scene_dir / "layout_gt" / f"{dst}.txt"
        src_poly = load_layout_gt_txt_as_local_xy(src_txt, pano_w, pano_h, layout_z)
        dst_poly = load_layout_gt_txt_as_local_xy(dst_txt, pano_w, pano_h, layout_z)
        if src_poly is None or dst_poly is None: continue
        for p in item.get("pairs", []):
            try:
                idx_s, idx_d = p[0]-1, p[1]-1
                if idx_s >= len(src_poly) or idx_d >= len(dst_poly): continue
                ps = src_poly[idx_s]
                pd = dst_poly[idx_d]
                gs = se2_apply((poses[src]["x"], poses[src]["y"], poses[src]["theta"]), ps)
                gd = se2_apply((poses[dst]["x"], poses[dst]["y"], poses[dst]["theta"]), pd)
                ax.plot([gs[0], gd[0]], [gs[1], gd[1]], color="red", linestyle="--", linewidth=2, zorder=10)
                ax.scatter([gs[0], gd[0]], [gs[1], gd[1]], color="red", s=30, zorder=11)
            except Exception:
                pass

def plot_layouts(
    ax,
    scene_dir: Path,
    manifest_nodes: List[Dict[str, Any]],
    poses: Dict[str, Dict[str, float]],
    pano_w: int,
    pano_h: int,
    layout_z: float,
    alpha: float = 0.25,
    axis_align_layouts: bool = False,
) -> Tuple[int, int, int]:
    drawn = 0
    skipped_no_layout = 0
    skipped_parse_fail = 0

    for n in manifest_nodes:
        pid = n.get("pano_id", "")
        if not pid or pid not in poses:
            continue

        lp_path = scene_dir / "layout_gt" / f"{pid}.txt"
        if not lp_path.exists():
            skipped_no_layout += 1
            continue

        poly_local = load_layout_gt_txt_as_local_xy(
            lp_path, pano_w=pano_w, pano_h=pano_h, layout_z=layout_z
        )
        if poly_local is None:
            skipped_parse_fail += 1
            continue

        theta = poses[pid]["theta"]
        if axis_align_layouts:
            poly_local = align_polygon_to_xy(poly_local)
            theta = round(theta / (math.pi / 2.0)) * (math.pi / 2.0)
        pose = (poses[pid]["x"], poses[pid]["y"], theta)
        poly_world = se2_apply(pose, poly_local)
        closed = np.vstack([poly_world, poly_world[0:1]])
        ax.plot(
            closed[:, 0],
            closed[:, 1],
            color="black",
            linewidth=1.5,
            alpha=alpha,
            zorder=2,
        )
        ax.fill(
            poly_world[:, 0], poly_world[:, 1], alpha=max(0.05, alpha * 0.5), zorder=1
        )
        drawn += 1

    return drawn, skipped_no_layout, skipped_parse_fail


def collect_world_layout_polys(
    scene_dir: Path,
    manifest_nodes: List[Dict[str, Any]],
    poses: Dict[str, Dict[str, float]],
    pano_w: int,
    pano_h: int,
    layout_z: float,
) -> List[np.ndarray]:
    polys_world = []
    for n in manifest_nodes:
        pid = n.get("pano_id", "")
        if not pid or pid not in poses:
            continue
        lp_path = scene_dir / "layout_gt" / f"{pid}.txt"
        poly_local = load_layout_gt_txt_as_local_xy(
            lp_path, pano_w=pano_w, pano_h=pano_h, layout_z=layout_z
        )
        if poly_local is None:
            continue
        pose = (poses[pid]["x"], poses[pid]["y"], poses[pid]["theta"])
        polys_world.append(se2_apply(pose, poly_local))
    return polys_world


def plot_error_quiver(
    ax,
    before_poses: Dict[str, Dict[str, float]],
    after_poses: Dict[str, Dict[str, float]],
):
    for i in before_poses.keys():
        if i in after_poses:
            bx, by = before_poses[i]["x"], before_poses[i]["y"]
            ax.annotate(
                "",
                xy=(after_poses[i]["x"], after_poses[i]["y"]),
                xytext=(bx, by),
                arrowprops=dict(
                    arrowstyle="->",
                    color="gray",
                    linewidth=1,
                    alpha=0.7,
                    linestyle="dashed",
                ),
            )


def print_scale_diagnostics(
    tag: str,
    poses: Dict[str, Dict[str, float]],
    scene_dir: Path,
    pano_w: int,
    pano_h: int,
    layout_z: float,
):
    ids = sorted(poses.keys())
    if len(ids) < 2:
        return

    pose_xy = np.array([[poses[i]["x"], poses[i]["y"]] for i in ids], dtype=np.float64)
    pair_dists = []
    for i in range(len(pose_xy)):
        for j in range(i + 1, len(pose_xy)):
            pair_dists.append(float(np.linalg.norm(pose_xy[i] - pose_xy[j])))

    room_diags = []
    for pid in ids:
        lp_path = scene_dir / "layout_gt" / f"{pid}.txt"
        poly_local = load_layout_gt_txt_as_local_xy(
            lp_path, pano_w=pano_w, pano_h=pano_h, layout_z=layout_z
        )
        if poly_local is None:
            continue
        mins = poly_local.min(axis=0)
        maxs = poly_local.max(axis=0)
        room_diags.append(float(np.linalg.norm(maxs - mins)))

    if room_diags and pair_dists:
        med_room_diag = float(np.median(room_diags))
        med_pose_dist = float(np.median(pair_dists))
        ratio = med_room_diag / max(med_pose_dist, 1e-9)
        print(
            f"[SCALE] {tag}: median room bbox diag={med_room_diag:.4f}, median pose-pair dist={med_pose_dist:.4f}, ratio={ratio:.4f}"
        )
        if ratio > 10:
            print(
                "[WARNING] Layout polygons are much larger than pose distances. Check that Step 1/4/5 use the SAME layout_z and SAME projection convention."
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", type=str, required=True, help="data/group/<scene>")
    ap.add_argument("--before", type=str, required=True, help="initial_poses.json")
    ap.add_argument("--after", type=str, required=True, help="optimized_poses.json")
    ap.add_argument("--edges", type=str, required=True, help="edges_measurements.json")
    ap.add_argument(
        "--out_dir", type=str, required=True, help="output directory for images"
    )
    ap.add_argument("--no_labels", action="store_true", help="hide node labels")
    ap.add_argument("--no_edges", action="store_true", help="hide edges")
    ap.add_argument("--extra_points", type=str, default="", help="json for extra point matches")
    ap.add_argument(
        "--relation_json",
        type=str,
        default="",
        help="relation.json containing pano id to room name labels",
    )
    ap.add_argument(
        "--axis_align_layouts",
        action="store_true",
        help="snap each room layout rotation to the nearest 90 degrees for display",
    )
    ap.add_argument(
        "--no_axis_align",
        action="store_true",
        help="disable global visualization-only rotation to align dominant walls to XY axes",
    )
    ap.add_argument(
        "--label_font_size",
        type=int,
        default=8,
        help="font size for room labels",
    )
    ap.add_argument("--no_title", action="store_true", help="hide plot titles")
    ap.add_argument(
        "--draw_layouts",
        action="store_true",
        help="draw layout_gt polygons using the SAME loader as edge generation",
    )
    ap.add_argument("--pano_w", type=int, default=1024)
    ap.add_argument("--pano_h", type=int, default=512)
    ap.add_argument(
        "--layout_z",
        type=float,
        default=50.0,
        help="must match Step 1 / edge generation",
    )
    args = ap.parse_args()

    scene_dir = Path(args.scene_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    manifest = load_json(scene_dir / "manifest.json")
    nodes = manifest.get("nodes", [])
    before = load_json(Path(args.before))
    after = load_json(Path(args.after))
    edges_data = load_json(Path(args.edges))

    poses_before = before.get("poses", {})
    poses_after = after.get("poses", {})
    edges = edges_data.get("edges", [])
    relation_json = Path(args.relation_json) if args.relation_json else None
    label_map = get_room_labels(scene_dir, relation_json=relation_json)
    extra_pts = []
    if args.extra_points and Path(args.extra_points).exists():
        extra_pts = load_json(Path(args.extra_points))
        if isinstance(extra_pts, dict) and "constraints" in extra_pts:
            extra_pts = extra_pts["constraints"]

    print_scale_diagnostics(
        "before", poses_before, scene_dir, args.pano_w, args.pano_h, args.layout_z
    )
    print_scale_diagnostics(
        "after", poses_after, scene_dir, args.pano_w, args.pano_h, args.layout_z
    )

    if args.draw_layouts and not args.no_axis_align:
        align_source_polys = collect_world_layout_polys(
            scene_dir, nodes, poses_after, args.pano_w, args.pano_h, args.layout_z
        )
        if align_source_polys:
            _, align_angle, align_origin = align_polys_to_xy(align_source_polys)
            poses_before = rotate_pose_dict(poses_before, align_angle, align_origin)
            poses_after = rotate_pose_dict(poses_after, align_angle, align_origin)
            print(
                f"[AXIS_ALIGN] rotated visualization by {math.degrees(align_angle):.3f} deg to align dominant walls to XY axes"
            )

    # before
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    plot_graph(
        ax,
        poses_before,
        edges,
        title="" if args.no_title else f"{scene_dir.name} - BEFORE (initial)",
        draw_edges=not args.no_edges,
        draw_labels=not args.no_labels,
        manifest_nodes=nodes,
        label_map=label_map,
        label_font_size=args.label_font_size,
    )
    if args.draw_layouts:
        drawn, skipped_no_layout, skipped_parse_fail = plot_layouts(
            ax,
            scene_dir,
            nodes,
            poses_before,
            pano_w=args.pano_w,
            pano_h=args.pano_h,
            layout_z=args.layout_z,
            alpha=0.35,
            axis_align_layouts=args.axis_align_layouts,
        )
        if not args.no_title:
            ax.set_title(
                ax.get_title()
                + f" | layouts drawn: {drawn}, no_layout: {skipped_no_layout}, parse_fail: {skipped_parse_fail}"
            )
        print(
            f"[INFO] before layouts drawn={drawn}, no_layout={skipped_no_layout}, parse_fail={skipped_parse_fail}"
        )
    fig.tight_layout()
    fig.savefig(out_dir / "before.png", dpi=200)
    plt.close(fig)

    # after
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    plot_graph(
        ax,
        poses_after,
        edges,
        title="" if args.no_title else f"{scene_dir.name} - AFTER (optimized)",
        draw_edges=not args.no_edges,
        draw_labels=not args.no_labels,
        manifest_nodes=nodes,
        label_map=label_map,
        label_font_size=args.label_font_size,
    )
    if args.draw_layouts:
        drawn, skipped_no_layout, skipped_parse_fail = plot_layouts(
            ax,
            scene_dir,
            nodes,
            poses_after,
            pano_w=args.pano_w,
            pano_h=args.pano_h,
            layout_z=args.layout_z,
            alpha=0.35,
            axis_align_layouts=args.axis_align_layouts,
        )
        if not args.no_title:
            ax.set_title(
                ax.get_title()
                + f" | layouts drawn: {drawn}, no_layout: {skipped_no_layout}, parse_fail: {skipped_parse_fail}"
            )
        print(
            f"[INFO] after layouts drawn={drawn}, no_layout={skipped_no_layout}, parse_fail={skipped_parse_fail}"
        )
    if extra_pts: plot_extra_edges(ax, extra_pts, poses_after, scene_dir, args.pano_w, args.pano_h, args.layout_z)
    fig.tight_layout()
    fig.savefig(out_dir / "after.png", dpi=200)
    plt.close(fig)

    # overlay
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    plot_error_quiver(ax, poses_before, poses_after)
    plot_graph(
        ax,
        poses_before,
        edges,
        title="",
        draw_edges=not args.no_edges,
        draw_labels=False,
        manifest_nodes=nodes,
        label_map=label_map,
    )
    for collection in ax.collections:
        collection.set_alpha(0.15)
    for line in ax.lines:
        line.set_alpha(0.15)
    if args.draw_layouts:
        plot_layouts(
            ax,
            scene_dir,
            nodes,
            poses_before,
            pano_w=args.pano_w,
            pano_h=args.pano_h,
            layout_z=args.layout_z,
            alpha=0.10,
            axis_align_layouts=args.axis_align_layouts,
        )

    plot_graph(
        ax,
        poses_after,
        edges,
        title="" if args.no_title else f"{scene_dir.name} - OVERLAY",
        draw_edges=not args.no_edges,
        draw_labels=not args.no_labels,
        manifest_nodes=nodes,
        label_map=label_map,
        label_font_size=args.label_font_size,
    )
    if args.draw_layouts:
        plot_layouts(
            ax,
            scene_dir,
            nodes,
            poses_after,
            pano_w=args.pano_w,
            pano_h=args.pano_h,
            layout_z=args.layout_z,
            alpha=0.40,
            axis_align_layouts=args.axis_align_layouts,
        )
    fig.tight_layout()
    fig.savefig(out_dir / "overlay.png", dpi=200)
    plt.close(fig)

    print(f"[OK] Wrote viz images -> {out_dir}")


if __name__ == "__main__":
    main()
