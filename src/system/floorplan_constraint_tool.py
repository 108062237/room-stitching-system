#!/usr/bin/env python3
"""
Top-down floorplan constraint annotation tool.

This tool lets you mark corresponding layout points directly on the stitched
floorplan. By default clicks snap to layout vertices. With --point_mode free,
clicks are stored as arbitrary local room coordinates, which is useful when a
wall overlap point is not an existing polygon corner.

Example:
  python -m src.system.floorplan_constraint_tool \
    --scene_dir data/group/58472_Floor1 \
    --poses data/group/58472_Floor1/poses/initial_poses.json \
    --src_room 08ddb586-be87-4f65-8199-4f9e21ff613b \
    --dst_room 08ddb587-12ac-4aba-899e-552644b5805f \
    --constraint_type structural_adjacency \
    --out data/group/58472_Floor1/matches/candidate_structural_matches.json

Controls:
  vertex mode: left click near a source vertex, then near a destination vertex
  free mode: left click a source point, then a destination point
  u: undo last pair
  c: clear current selection
  s: save
  q: quit
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils.geom import load_layout_gt_txt_as_local_xy, se2_apply
from src.utils.labels import get_display_label, get_room_labels


DEFAULT_CONSTRAINT_TYPES = [
    "structural_adjacency",
    "wall_alignment",
    "single_point_adjacency",
    "connectivity",
    "candidate",
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


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


def normalize_match_entries(raw: Any) -> List[Dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        if "constraints" in raw and isinstance(raw["constraints"], list):
            raw = raw["constraints"]
        else:
            raw = [raw]
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def load_existing_entries(path: Optional[Path]) -> List[Dict[str, Any]]:
    if path is None or not path.exists():
        return []
    return normalize_match_entries(load_json(path))


def upsert_entry(entries: List[Dict[str, Any]], new_entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    src = new_entry["src"]
    dst = new_entry["dst"]
    out = []
    replaced = False
    for entry in entries:
        if entry.get("src") == src and entry.get("dst") == dst:
            out.append(new_entry)
            replaced = True
        else:
            out.append(entry)
    if not replaced:
        out.append(new_entry)
    return out


def as_float_pair(value: Any, field_name: str) -> Tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{field_name} must be a 2-number list")
    return float(value[0]), float(value[1])


class FloorplanConstraintTool:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.scene_dir = Path(args.scene_dir)
        self.layout_dir = self.scene_dir / "layout_gt"
        self.src_room = normalize_room_id(args.src_room)
        self.dst_room = normalize_room_id(args.dst_room)
        self.src_name = f"{self.src_room}.txt"
        self.dst_name = f"{self.dst_room}.txt"
        self.constraint_type = args.constraint_type
        self.connected_by = args.connected_by
        self.poses = load_pose_map(Path(args.poses))
        self.label_map = get_room_labels(self.scene_dir)

        if self.src_room not in self.poses:
            raise KeyError(f"src_room missing from poses: {self.src_room}")
        if self.dst_room not in self.poses:
            raise KeyError(f"dst_room missing from poses: {self.dst_room}")

        self.polys_local: Dict[str, np.ndarray] = {}
        self.polys_world: Dict[str, np.ndarray] = {}
        for room_id in self.poses:
            txt_path = self.layout_dir / f"{room_id}.txt"
            if not txt_path.exists():
                continue
            poly_local = load_layout_gt_txt_as_local_xy(txt_path)
            if poly_local is None:
                continue
            self.polys_local[room_id] = poly_local
            self.polys_world[room_id] = se2_apply(self.poses[room_id], poly_local)

        for room_id in [self.src_room, self.dst_room]:
            if room_id not in self.polys_world:
                raise RuntimeError(f"Cannot load layout polygon for room: {room_id}")

        self.pairs: List[Tuple[int, int]] = []
        self.point_pairs: List[Dict[str, List[float]]] = []
        self.pending_src_idx: Optional[int] = None
        self.pending_src_point: Optional[Dict[str, List[float]]] = None
        self.status_text = None
        self.selected_artist = None
        self.pair_artists = []

        self.load_existing_pairs()

    def load_existing_pairs(self) -> None:
        entries = load_existing_entries(Path(self.args.load_json) if self.args.load_json else Path(self.args.out))
        for entry in entries:
            if entry.get("src") == self.src_name and entry.get("dst") == self.dst_name:
                self.pairs = [(int(a), int(b)) for a, b in entry.get("pairs", [])]
                self.point_pairs = []
                for pair in entry.get("point_pairs", []):
                    src_xy = as_float_pair(pair.get("src_xy"), "src_xy")
                    dst_xy = as_float_pair(pair.get("dst_xy"), "dst_xy")
                    loaded = {
                        "src_xy": [src_xy[0], src_xy[1]],
                        "dst_xy": [dst_xy[0], dst_xy[1]],
                    }
                    if "src_world_xy" in pair:
                        src_world_xy = as_float_pair(pair.get("src_world_xy"), "src_world_xy")
                        loaded["src_world_xy"] = [src_world_xy[0], src_world_xy[1]]
                    if "dst_world_xy" in pair:
                        dst_world_xy = as_float_pair(pair.get("dst_world_xy"), "dst_world_xy")
                        loaded["dst_world_xy"] = [dst_world_xy[0], dst_world_xy[1]]
                    self.point_pairs.append(loaded)
                self.constraint_type = entry.get("constraint_type", self.constraint_type)
                self.connected_by = entry.get("connected_by", self.connected_by)
                break

    def nearest_vertex(self, room_id: str, x: float, y: float) -> Tuple[int, float]:
        poly = self.polys_world[room_id]
        point = np.array([x, y], dtype=np.float64)
        dists = np.linalg.norm(poly - point, axis=1)
        idx = int(np.argmin(dists))
        return idx + 1, float(dists[idx])

    def world_vertex(self, room_id: str, idx_1based: int) -> np.ndarray:
        return self.polys_world[room_id][idx_1based - 1]

    def local_to_world_point(self, room_id: str, xy: Tuple[float, float]) -> np.ndarray:
        point = np.array([[float(xy[0]), float(xy[1])]], dtype=np.float64)
        return se2_apply(self.poses[room_id], point)[0]

    def world_to_local_point(self, room_id: str, x: float, y: float) -> Tuple[float, float]:
        px, py, theta = self.poses[room_id]
        dx = x - px
        dy = y - py
        c = math.cos(theta)
        s = math.sin(theta)
        return c * dx + s * dy, -s * dx + c * dy

    def make_free_point(self, room_id: str, x: float, y: float) -> Dict[str, List[float]]:
        local_x, local_y = self.world_to_local_point(room_id, x, y)
        return {
            "xy": [float(local_x), float(local_y)],
            "world_xy": [float(x), float(y)],
        }

    def setup_plot(self) -> None:
        self.fig, self.ax = plt.subplots(figsize=(14, 12))
        self.fig.subplots_adjust(right=0.82)
        self.ax_radio = self.fig.add_axes([0.84, 0.55, 0.14, 0.30])
        labels = self.args.constraint_types.split(",") if self.args.constraint_types else DEFAULT_CONSTRAINT_TYPES
        labels = [label.strip() for label in labels if label.strip()]
        if self.constraint_type not in labels:
            labels.insert(0, self.constraint_type)
        self.radio = RadioButtons(self.ax_radio, labels, active=labels.index(self.constraint_type))
        self.radio.on_clicked(self.on_constraint_type_change)

        for room_id, poly in self.polys_world.items():
            is_src = room_id == self.src_room
            is_dst = room_id == self.dst_room
            if is_src:
                color, alpha, lw = "#1f77b4", 0.18, 2.2
            elif is_dst:
                color, alpha, lw = "#ff7f0e", 0.18, 2.2
            else:
                color, alpha, lw = "black", 0.05, 0.9
            closed = np.vstack([poly, poly[:1]])
            self.ax.plot(closed[:, 0], closed[:, 1], color=color, linewidth=lw, alpha=0.75, zorder=1)
            self.ax.fill(poly[:, 0], poly[:, 1], color=color, alpha=alpha, zorder=0)
            pose = self.poses[room_id]
            label = get_display_label(room_id, self.label_map)
            self.ax.text(
                pose[0],
                pose[1],
                label,
                fontsize=9,
                ha="center",
                va="center",
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1),
                zorder=5,
            )

        if self.args.point_mode == "vertex":
            self.draw_vertices(self.src_room, "#1f77b4", "o")
            self.draw_vertices(self.dst_room, "#ff7f0e", "s")
        self.redraw_pairs()

        self.status_text = self.ax.text(
            0.01,
            0.01,
            "",
            transform=self.ax.transAxes,
            fontsize=10,
            va="bottom",
            ha="left",
            bbox=dict(facecolor="white", alpha=0.88, edgecolor="black", linewidth=0.5),
            zorder=30,
        )
        self.update_status()
        self.ax.set_title(f"{self.src_room[-6:]} -> {self.dst_room[-6:]} constraint annotation")
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.grid(True, linestyle="--", alpha=0.25)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    def draw_vertices(self, room_id: str, color: str, marker: str) -> None:
        poly = self.polys_world[room_id]
        self.ax.scatter(poly[:, 0], poly[:, 1], color=color, edgecolor="black", marker=marker, s=75, zorder=10)
        for idx, point in enumerate(poly, start=1):
            self.ax.text(
                point[0],
                point[1],
                str(idx),
                fontsize=9,
                color=color,
                fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=0.6),
                zorder=11,
            )

    def update_status(self) -> None:
        if self.args.point_mode == "free":
            pending = "pending src point" if self.pending_src_point else "click source point"
            pair_count = len(self.point_pairs)
        else:
            pending = f"pending src={self.pending_src_idx}" if self.pending_src_idx else "click source vertex"
            pair_count = len(self.pairs)
        text = (
            f"type: {self.constraint_type}\n"
            f"mode: {self.args.point_mode}\n"
            f"pairs: {pair_count}\n"
            f"{pending}\n"
            "keys: u undo, c clear, s save, q quit"
        )
        if self.status_text is not None:
            self.status_text.set_text(text)
        self.fig.canvas.draw_idle()

    def clear_pair_artists(self) -> None:
        for artist in self.pair_artists:
            artist.remove()
        self.pair_artists = []

    def redraw_pairs(self) -> None:
        if not hasattr(self, "ax"):
            return
        self.clear_pair_artists()
        if self.args.point_mode == "free":
            for pair in self.point_pairs:
                ps = self.local_to_world_point(self.src_room, tuple(pair["src_xy"]))
                pd = self.local_to_world_point(self.dst_room, tuple(pair["dst_xy"]))
                line = self.ax.plot([ps[0], pd[0]], [ps[1], pd[1]], color="#2ca02c", linestyle="--", linewidth=2.2, zorder=20)[0]
                s1 = self.ax.scatter([ps[0]], [ps[1]], color="#2ca02c", edgecolor="black", s=110, marker="o", zorder=21)
                s2 = self.ax.scatter([pd[0]], [pd[1]], color="#2ca02c", edgecolor="black", s=110, marker="s", zorder=21)
                self.pair_artists.extend([line, s1, s2])
            self.fig.canvas.draw_idle()
            return
        for src_idx, dst_idx in self.pairs:
            ps = self.world_vertex(self.src_room, src_idx)
            pd = self.world_vertex(self.dst_room, dst_idx)
            line = self.ax.plot([ps[0], pd[0]], [ps[1], pd[1]], color="#2ca02c", linestyle="--", linewidth=2.2, zorder=20)[0]
            s1 = self.ax.scatter([ps[0]], [ps[1]], color="#2ca02c", edgecolor="black", s=110, marker="o", zorder=21)
            s2 = self.ax.scatter([pd[0]], [pd[1]], color="#2ca02c", edgecolor="black", s=110, marker="s", zorder=21)
            self.pair_artists.extend([line, s1, s2])
        self.fig.canvas.draw_idle()

    def on_constraint_type_change(self, label: str) -> None:
        self.constraint_type = label
        self.update_status()

    def on_click(self, event) -> None:
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        x, y = float(event.xdata), float(event.ydata)
        if self.args.point_mode == "free":
            self.on_free_click(x, y)
            return
        if self.pending_src_idx is None:
            idx, dist = self.nearest_vertex(self.src_room, x, y)
            self.pending_src_idx = idx
            if self.selected_artist is not None:
                self.selected_artist.remove()
            point = self.world_vertex(self.src_room, idx)
            self.selected_artist = self.ax.scatter([point[0]], [point[1]], color="yellow", edgecolor="black", s=150, marker="o", zorder=25)
            print(f"[SELECT] src vertex {idx} (distance={dist:.3f})")
        else:
            idx, dist = self.nearest_vertex(self.dst_room, x, y)
            self.pairs.append((self.pending_src_idx, idx))
            print(f"[PAIR] src {self.pending_src_idx} -> dst {idx} (distance={dist:.3f})")
            self.pending_src_idx = None
            if self.selected_artist is not None:
                self.selected_artist.remove()
                self.selected_artist = None
            self.redraw_pairs()
        self.update_status()

    def on_free_click(self, x: float, y: float) -> None:
        if self.pending_src_point is None:
            self.pending_src_point = self.make_free_point(self.src_room, x, y)
            if self.selected_artist is not None:
                self.selected_artist.remove()
            self.selected_artist = self.ax.scatter([x], [y], color="yellow", edgecolor="black", s=150, marker="o", zorder=25)
            print(f"[SELECT] src free point world=({x:.3f}, {y:.3f}) local=({self.pending_src_point['xy'][0]:.3f}, {self.pending_src_point['xy'][1]:.3f})")
        else:
            dst_point = self.make_free_point(self.dst_room, x, y)
            self.point_pairs.append(
                {
                    "src_xy": self.pending_src_point["xy"],
                    "dst_xy": dst_point["xy"],
                    "src_world_xy": self.pending_src_point["world_xy"],
                    "dst_world_xy": dst_point["world_xy"],
                }
            )
            print(
                "[PAIR] free point "
                f"src local=({self.pending_src_point['xy'][0]:.3f}, {self.pending_src_point['xy'][1]:.3f}) "
                f"-> dst local=({dst_point['xy'][0]:.3f}, {dst_point['xy'][1]:.3f})"
            )
            self.pending_src_point = None
            if self.selected_artist is not None:
                self.selected_artist.remove()
                self.selected_artist = None
            self.redraw_pairs()
        self.update_status()

    def on_key(self, event) -> None:
        if event.key == "u":
            if self.args.point_mode == "free" and self.point_pairs:
                removed = self.point_pairs.pop()
                print(f"[UNDO] removed free point pair {removed}")
                self.redraw_pairs()
            elif self.args.point_mode == "vertex" and self.pairs:
                removed = self.pairs.pop()
                print(f"[UNDO] removed pair {removed}")
                self.redraw_pairs()
            self.update_status()
        elif event.key == "c":
            self.pending_src_idx = None
            self.pending_src_point = None
            if self.selected_artist is not None:
                self.selected_artist.remove()
                self.selected_artist = None
            self.update_status()
        elif event.key == "s":
            self.save()
        elif event.key == "q":
            plt.close(self.fig)

    def save(self) -> None:
        out_path = Path(self.args.out)
        entries = load_existing_entries(out_path)
        entry: Dict[str, Any] = {
            "src": self.src_name,
            "dst": self.dst_name,
            "point_mode": self.args.point_mode,
            "constraint_type": self.constraint_type,
            "edge_type": self.args.edge_type,
            "connected_by": self.connected_by,
            "confidence": float(self.args.confidence),
            "recommended_sigma_xy": float(self.args.recommended_sigma_xy),
            "recommended_sigma_theta": float(self.args.recommended_sigma_theta),
        }
        if self.args.point_mode == "free":
            entry["point_pairs"] = self.point_pairs
        else:
            entry["pairs"] = [[int(a), int(b)] for a, b in self.pairs]
        if self.args.note:
            entry["note"] = self.args.note
        entries = upsert_entry(entries, entry)
        save_json(out_path, entries)
        pair_count = len(self.point_pairs) if self.args.point_mode == "free" else len(self.pairs)
        print(f"[OK] saved {pair_count} {self.args.point_mode} pairs -> {out_path}")

    def run(self) -> None:
        self.setup_plot()
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", required=True)
    parser.add_argument("--poses", required=True)
    parser.add_argument("--src_room", required=True)
    parser.add_argument("--dst_room", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--load_json", default="")
    parser.add_argument("--constraint_type", default="structural_adjacency")
    parser.add_argument("--constraint_types", default=",".join(DEFAULT_CONSTRAINT_TYPES))
    parser.add_argument("--edge_type", default="candidate")
    parser.add_argument("--point_mode", choices=["vertex", "free"], default="vertex")
    parser.add_argument("--connected_by", default="none")
    parser.add_argument("--confidence", type=float, default=0.3)
    parser.add_argument("--recommended_sigma_xy", type=float, default=5.0)
    parser.add_argument("--recommended_sigma_theta", type=float, default=5.0)
    parser.add_argument("--note", default="")
    return parser.parse_args()


def main() -> None:
    FloorplanConstraintTool(parse_args()).run()


if __name__ == "__main__":
    main()
