#!/usr/bin/env python3
"""
Visualize candidate structural constraints on a stitched floorplan.

Supports both annotation formats produced by floorplan_constraint_tool.py:
  - vertex pairs: {"pairs": [[src_idx, dst_idx], ...]}
  - free point pairs: {"point_pairs": [{"src_xy": [...], "dst_xy": [...]}, ...]}
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

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

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.utils.geom import load_layout_gt_txt_as_local_xy, se2_apply
from src.utils.labels import get_display_label, get_room_labels


TYPE_STYLE = {
    "wall_alignment": {"color": "#d62728", "linestyle": "-", "marker": "o"},
    "structural_adjacency": {"color": "#9467bd", "linestyle": "--", "marker": "s"},
    "single_point_adjacency": {"color": "#ff7f0e", "linestyle": ":", "marker": "^"},
    "connectivity": {"color": "#2ca02c", "linestyle": "-", "marker": "D"},
    "candidate": {"color": "#8c564b", "linestyle": "-.", "marker": "o"},
}

ROOM_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_room_id(room_id: str) -> str:
    return room_id.replace(".txt", "")


def load_pose_map(path: Path) -> Dict[str, Tuple[float, float, float]]:
    data = load_json(path)
    poses_raw = data.get("poses", data)
    return {
        normalize_room_id(room_id): (
            float(pose["x"]),
            float(pose["y"]),
            float(pose["theta"]),
        )
        for room_id, pose in poses_raw.items()
    }


def normalize_entries(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, dict):
        if "constraints" in raw:
            raw = raw["constraints"]
        else:
            raw = [raw]
    if not isinstance(raw, list):
        raise ValueError("constraints JSON must be a list or a dict with key 'constraints'")
    return [item for item in raw if isinstance(item, dict)]


def normalize_edges(raw: Any) -> List[Tuple[str, str]]:
    if isinstance(raw, dict):
        raw = raw.get("edges", [])
    if not isinstance(raw, list):
        return []

    out: List[Tuple[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        src = item.get("src", item.get("i"))
        dst = item.get("dst", item.get("j"))
        if src and dst:
            out.append((normalize_room_id(str(src)), normalize_room_id(str(dst))))
    return out


def local_to_world(poses: Dict[str, Tuple[float, float, float]], room_id: str, xy: Iterable[float]) -> np.ndarray:
    return se2_apply(poses[room_id], np.array([list(xy)], dtype=np.float64))[0]


def load_world_polys(scene_dir: Path, poses: Dict[str, Tuple[float, float, float]]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    layout_dir = scene_dir / "layout_gt"
    for room_id, pose in poses.items():
        txt_path = layout_dir / f"{room_id}.txt"
        if not txt_path.exists():
            continue
        local_poly = load_layout_gt_txt_as_local_xy(txt_path)
        if local_poly is None:
            continue
        out[room_id] = se2_apply(pose, local_poly)
    return out


def vertex_point(polys_world: Dict[str, np.ndarray], room_id: str, idx_1based: int) -> np.ndarray:
    poly = polys_world[room_id]
    if idx_1based < 1 or idx_1based > len(poly):
        raise IndexError(f"vertex index {idx_1based} out of range for {room_id} 1..{len(poly)}")
    return poly[idx_1based - 1]


def draw_room_layouts(
    ax,
    polys_world: Dict[str, np.ndarray],
    poses: Dict[str, Tuple[float, float, float]],
    label_map: Dict[str, str],
    highlight_rooms: set[str],
) -> None:
    sorted_rooms = sorted(polys_world)
    room_colors = {
        room_id: ROOM_COLORS[idx % len(ROOM_COLORS)]
        for idx, room_id in enumerate(sorted_rooms)
    }
    for room_id in sorted_rooms:
        poly = polys_world[room_id]
        closed = np.vstack([poly, poly[:1]])
        is_highlight = room_id in highlight_rooms
        color = room_colors[room_id]
        alpha = 0.18 if is_highlight else 0.08
        lw = 1.7 if is_highlight else 0.8
        ax.plot(closed[:, 0], closed[:, 1], color=color, linewidth=lw, alpha=0.8, zorder=1)
        ax.fill(poly[:, 0], poly[:, 1], color=color, alpha=alpha, zorder=0)
        x, y, _ = poses[room_id]
        ax.text(
            x,
            y,
            get_display_label(room_id, label_map),
            fontsize=10 if is_highlight else 8,
            ha="center",
            va="center",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1),
            zorder=20,
        )


def draw_tree_edges(
    ax,
    edges: List[Tuple[str, str]],
    poses: Dict[str, Tuple[float, float, float]],
) -> int:
    drawn = 0
    for src, dst in edges:
        if src not in poses or dst not in poses:
            continue
        x0, y0, _ = poses[src]
        x1, y1, _ = poses[dst]
        ax.plot(
            [x0, x1],
            [y0, y1],
            color="#0057b8",
            linestyle="-",
            linewidth=3.2,
            alpha=0.82,
            zorder=4,
        )
        drawn += 1
    return drawn


def draw_candidates(
    ax,
    entries: List[Dict[str, Any]],
    poses: Dict[str, Tuple[float, float, float]],
    polys_world: Dict[str, np.ndarray],
) -> Tuple[int, List[str]]:
    warnings: List[str] = []
    drawn_pairs = 0

    for idx, entry in enumerate(entries):
        src = normalize_room_id(str(entry.get("src", "")))
        dst = normalize_room_id(str(entry.get("dst", "")))
        ctype = str(entry.get("constraint_type", "candidate"))
        style = TYPE_STYLE.get(ctype, TYPE_STYLE["candidate"])
        color = style["color"]
        linestyle = style["linestyle"]
        marker = style["marker"]

        if src not in poses or dst not in poses:
            warnings.append(f"entry {idx}: missing pose for {src[-6:]} -> {dst[-6:]}")
            continue
        if src not in polys_world or dst not in polys_world:
            warnings.append(f"entry {idx}: missing layout for {src[-6:]} -> {dst[-6:]}")
            continue

        cx0, cy0, _ = poses[src]
        cx1, cy1, _ = poses[dst]
        if ctype != "single_point_adjacency":
            ax.plot(
                [cx0, cx1],
                [cy0, cy1],
                color=color,
                linestyle=":",
                linewidth=2.6,
                alpha=0.85,
                zorder=7,
            )

        src_points: List[np.ndarray] = []
        dst_points: List[np.ndarray] = []
        if "point_pairs" in entry:
            for pair in entry.get("point_pairs", []):
                try:
                    src_points.append(local_to_world(poses, src, pair["src_xy"]))
                    dst_points.append(local_to_world(poses, dst, pair["dst_xy"]))
                except Exception as exc:
                    warnings.append(f"entry {idx}: invalid free point pair: {exc}")
        else:
            for pair in entry.get("pairs", []):
                try:
                    src_idx, dst_idx = int(pair[0]), int(pair[1])
                    src_points.append(vertex_point(polys_world, src, src_idx))
                    dst_points.append(vertex_point(polys_world, dst, dst_idx))
                except Exception as exc:
                    warnings.append(f"entry {idx}: invalid vertex pair: {exc}")

        for p_src, p_dst in zip(src_points, dst_points):
            if ctype != "single_point_adjacency":
                ax.plot([p_src[0], p_dst[0]], [p_src[1], p_dst[1]], color=color, linestyle=linestyle, linewidth=2.1, alpha=0.95, zorder=12)
            ax.scatter([p_src[0]], [p_src[1]], color=color, edgecolor="black", marker=marker, s=70, zorder=13)
            ax.scatter([p_dst[0]], [p_dst[1]], color="white", edgecolor=color, marker=marker, linewidth=2.0, s=78, zorder=13)
            drawn_pairs += 1

        if len(src_points) >= 2 and ctype in {"wall_alignment", "structural_adjacency"}:
            ax.plot(
                [src_points[0][0], src_points[1][0]],
                [src_points[0][1], src_points[1][1]],
                color=color,
                linewidth=3.0,
                alpha=0.45,
                zorder=10,
            )
            ax.plot(
                [dst_points[0][0], dst_points[1][0]],
                [dst_points[0][1], dst_points[1][1]],
                color=color,
                linewidth=3.0,
                alpha=0.45,
                zorder=10,
            )

    return drawn_pairs, warnings


def add_legend(ax) -> None:
    handles = []
    labels = []
    for ctype, style in TYPE_STYLE.items():
        handles.append(
            plt.Line2D(
                [0],
                [0],
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=2,
                markersize=6,
            )
        )
        labels.append(ctype)
    handles.append(plt.Line2D([0], [0], color="#555555", linestyle=":", linewidth=2.6))
    labels.append("room-center link")
    handles.append(plt.Line2D([0], [0], color="#0057b8", linestyle="-", linewidth=3.2))
    labels.append("tree link")
    ax.legend(handles, labels, loc="upper left", fontsize=8, framealpha=0.88)


def write_summary(path: Path, entries: List[Dict[str, Any]], drawn_pairs: int, warnings: List[str]) -> None:
    by_type: Dict[str, int] = {}
    for entry in entries:
        ctype = str(entry.get("constraint_type", "candidate"))
        by_type[ctype] = by_type.get(ctype, 0) + 1
    summary = {
        "num_constraints": len(entries),
        "num_drawn_point_pairs": drawn_pairs,
        "by_constraint_type": by_type,
        "warnings": warnings,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", required=True)
    parser.add_argument("--poses", required=True)
    parser.add_argument("--constraints", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary", default="")
    parser.add_argument("--tree_edges", default="", help="optional pose graph tree/perfect edges JSON")
    parser.add_argument("--no_tree_edges", action="store_true")
    parser.add_argument("--title", default="Candidate structural constraints")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene_dir = Path(args.scene_dir)
    poses = load_pose_map(Path(args.poses))
    entries = normalize_entries(load_json(Path(args.constraints)))
    polys_world = load_world_polys(scene_dir, poses)
    label_map = get_room_labels(scene_dir)
    highlight_rooms = {
        normalize_room_id(str(entry.get("src", ""))) for entry in entries
    } | {
        normalize_room_id(str(entry.get("dst", ""))) for entry in entries
    }

    fig, ax = plt.subplots(figsize=(15, 12))
    draw_room_layouts(ax, polys_world, poses, label_map, highlight_rooms)
    tree_edges_path = Path(args.tree_edges) if args.tree_edges else scene_dir / "edges" / "perfect_edges.json"
    if not args.no_tree_edges and tree_edges_path.exists():
        draw_tree_edges(ax, normalize_edges(load_json(tree_edges_path)), poses)
    drawn_pairs, warnings = draw_candidates(ax, entries, poses, polys_world)
    add_legend(ax)

    ax.set_title(args.title, fontsize=15)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.22)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    summary_path = Path(args.summary) if args.summary else out_path.with_suffix(".summary.json")
    write_summary(summary_path, entries, drawn_pairs, warnings)
    print(f"[OK] wrote {out_path}")
    print(f"[OK] wrote {summary_path}")
    if warnings:
        for warning in warnings:
            print(f"[WARNING] {warning}")


if __name__ == "__main__":
    main()
