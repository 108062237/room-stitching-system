import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.system.room_loader import (
    load_room_pair,
    load_scene_rooms,
    polygon_bbox,
    polygon_centroid,
)


def draw_polygon_with_indices(ax, poly: np.ndarray, title: str):
    closed = np.vstack([poly, poly[0:1]])
    ax.plot(closed[:, 0], closed[:, 1], color="black", linewidth=1.5)
    ax.fill(poly[:, 0], poly[:, 1], alpha=0.25)

    for idx, (x, y) in enumerate(poly, start=1):
        ax.scatter([x], [y], s=30)
        ax.text(
            x,
            y,
            str(idx),
            fontsize=9,
            ha="center",
            va="center",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=0.5),
        )

    c = polygon_centroid(poly)
    ax.scatter([c[0]], [c[1]], s=60, marker="x")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.4)


def main():
    parser = argparse.ArgumentParser(description="Manual test for room_loader.py")
    parser.add_argument("--scene_dir", required=True, help="Path to scene folder")
    parser.add_argument("--src_room", default=None, help="Optional src room id")
    parser.add_argument("--dst_room", default=None, help="Optional dst room id")
    parser.add_argument("--layout_z", type=float, default=50.0, help="Projection z")
    parser.add_argument("--pano_w", type=int, default=1024)
    parser.add_argument("--pano_h", type=int, default=512)
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir)

    print("=" * 80)
    print("[1] Loading all rooms")
    print("=" * 80)

    rooms = load_scene_rooms(
        scene_dir=scene_dir,
        pano_w=args.pano_w,
        pano_h=args.pano_h,
        layout_z=args.layout_z,
        require_layout=True,
    )

    print(f"Scene: {scene_dir}")
    print(f"Loaded rooms: {len(rooms)}")

    room_ids = sorted(rooms.keys())
    for i, rid in enumerate(room_ids, start=1):
        room = rooms[rid]
        poly = room.polygon_local
        bbox = polygon_bbox(poly)
        centroid = polygon_centroid(poly)
        print(
            f"[{i:02d}] {rid} | points={len(poly)} | "
            f"centroid=({centroid[0]:.3f}, {centroid[1]:.3f}) | "
            f"bbox=({bbox['xmin']:.3f}, {bbox['ymin']:.3f}) ~ ({bbox['xmax']:.3f}, {bbox['ymax']:.3f})"
        )

    if not room_ids:
        print("No rooms loaded.")
        return

    first_room = rooms[room_ids[0]]
    print("\n" + "=" * 80)
    print("[2] First room detail")
    print("=" * 80)
    print(f"Room ID: {first_room.pano_id}")
    print(f"TXT path: {first_room.txt_path}")
    print(f"Display label: {first_room.display_label}")
    print(f"Connections: {len(first_room.connections)}")
    print("Polygon local shape:", first_room.polygon_local.shape)
    print("Polygon local points:")
    print(first_room.polygon_local)

    if args.src_room and args.dst_room:
        print("\n" + "=" * 80)
        print("[3] Loading specified room pair")
        print("=" * 80)

        pair = load_room_pair(
            scene_dir=scene_dir,
            src_room_id=args.src_room,
            dst_room_id=args.dst_room,
            pano_w=args.pano_w,
            pano_h=args.pano_h,
            layout_z=args.layout_z,
        )

        src_room = pair[args.src_room]
        dst_room = pair[args.dst_room]

        print(f"src_room: {src_room.pano_id}, points={len(src_room.polygon_local)}")
        print(f"dst_room: {dst_room.pano_id}, points={len(dst_room.polygon_local)}")

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        draw_polygon_with_indices(
            axes[0],
            src_room.polygon_local,
            f"SRC: {src_room.display_label}",
        )
        draw_polygon_with_indices(
            axes[1],
            dst_room.polygon_local,
            f"DST: {dst_room.display_label}",
        )
        fig.tight_layout()
        plt.show()
    else:
        print("\nNo --src_room / --dst_room given, skip pair visualization.")


if __name__ == "__main__":
    main()