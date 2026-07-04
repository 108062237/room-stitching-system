#!/usr/bin/env python3
"""CLI for floor-plan-level layout objective evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.layout_objective import (
    evaluate_layout_objective,
    load_poses,
    load_rooms,
    polygon_to_shapely,
    apply_pose_to_points,
    transformed_polygons,
)


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing input file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def plot_polygon(ax, geom, **kwargs) -> None:
    if geom.is_empty:
        return
    if geom.geom_type == "Polygon":
        xs, ys = geom.exterior.xy
        ax.fill(xs, ys, **kwargs)
    elif geom.geom_type == "MultiPolygon":
        for part in geom.geoms:
            plot_polygon(ax, part, **kwargs)


def draw_visualization(data: Dict[str, Any], report: Dict[str, Any], out_path: Path) -> None:
    rooms = load_rooms(data)
    poses = load_poses(data)
    polygons = transformed_polygons(rooms, poses)

    fig, ax = plt.subplots(figsize=(12, 10))
    cmap = plt.get_cmap("tab10")

    for idx, room_id in enumerate(sorted(polygons)):
        poly = polygons[room_id]
        xs, ys = poly.exterior.xy
        color = cmap(idx % 10)
        ax.plot(xs, ys, color=color, linewidth=2.0, alpha=0.9)
        ax.fill(xs, ys, color=color, alpha=0.12)
        cx, cy = poly.centroid.x, poly.centroid.y
        ax.text(
            cx,
            cy,
            room_id[-6:],
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1),
            zorder=10,
        )

    # Overlap regions.
    for item in report["terms"]["overlap"]["details"]:
        inter = polygons[item["room_a"]].intersection(polygons[item["room_b"]])
        plot_polygon(ax, inter, color="#ef4444", alpha=0.35, zorder=20)

    # Door/opening correspondences.
    for item in report["terms"]["door"]["details"]:
        src_seg = item["src_world_segment"]
        dst_seg = item["dst_world_segment"]
        ax.plot(
            [src_seg[0][0], src_seg[1][0]],
            [src_seg[0][1], src_seg[1][1]],
            color="#16a34a",
            linewidth=3.0,
            zorder=30,
        )
        ax.plot(
            [dst_seg[0][0], dst_seg[1][0]],
            [dst_seg[0][1], dst_seg[1][1]],
            color="#15803d",
            linewidth=3.0,
            linestyle="--",
            zorder=30,
        )
        src_c = item["src_world_segment"][0]
        dst_c = item["dst_world_segment"][0]
        ax.plot([src_c[0], dst_c[0]], [src_c[1], dst_c[1]], color="#22c55e", linestyle=":", linewidth=1.5)

    # Misaligned wall pairs.
    for item in report["terms"]["wall_align"]["details"][:50]:
        wall_a = item["wall_a"]
        wall_b = item["wall_b"]
        ax.plot([wall_a[0][0], wall_a[1][0]], [wall_a[0][1], wall_a[1][1]], color="#f97316", linewidth=2.5, zorder=25)
        ax.plot([wall_b[0][0], wall_b[1][0]], [wall_b[0][1], wall_b[1][1]], color="#fb923c", linewidth=2.5, linestyle="--", zorder=25)

    ax.set_title(f"Layout objective total score = {report['total_score']:.4f}")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="layout objective input JSON")
    parser.add_argument("--out", required=True, help="output report JSON")
    parser.add_argument("--viz", default="", help="optional output visualization PNG")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_json(Path(args.input))
    report = evaluate_layout_objective(data)
    save_json(Path(args.out), report)
    print(f"[OK] wrote layout objective report -> {args.out}")
    if args.viz:
        draw_visualization(data, report, Path(args.viz))
        print(f"[OK] wrote visualization -> {args.viz}")


if __name__ == "__main__":
    main()
