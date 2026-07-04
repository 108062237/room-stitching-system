import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

from src.system.room_loader import RoomData, load_scene_rooms
from src.utils.post_proc import np_coor2xy

try:
    from src.utils.panostretch import pano_connect_points
except ImportError:
    pano_connect_points = None


def draw_curved_line(ax, p1, p2, z, color, w, h):
    if pano_connect_points is not None:
        pts = pano_connect_points(p1, p2, z=z, w=w, h=h)
        pts_x = pts[:, 0] % w
        pts_y = pts[:, 1]

        diffs = np.abs(np.diff(pts_x))
        split_indices = np.where(diffs > w / 2)[0]

        if len(split_indices) > 0:
            splits = np.split(np.stack([pts_x, pts_y], axis=1), split_indices + 1)
            for seg in splits:
                ax.plot(seg[:, 0], seg[:, 1], color=color, linewidth=2.0, alpha=0.95)
        else:
            ax.plot(pts_x, pts_y, color=color, linewidth=2.0, alpha=0.95)
            return True, np.stack([pts_x, pts_y], axis=1)
    else:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=2.0, alpha=0.95)
        return True, np.array([p1, p2])
    return False, None


def load_paired_layout_pixels(txt_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load paired-mode layout txt.

    Expected storage:
      ceiling_1
      floor_1
      ceiling_2
      floor_2
      ...

    If the y-order is flipped in a pair, we still normalize to:
      smaller y -> ceiling
      larger y  -> floor
    """
    pts: List[List[float]] = []
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) >= 2:
            pts.append([float(parts[0]), float(parts[1])])

    if len(pts) < 4 or len(pts) % 2 != 0:
        raise ValueError(f"Paired-mode txt must contain an even number of points >= 4: {txt_path}")

    ceiling: List[List[float]] = []
    floor: List[List[float]] = []
    for i in range(0, len(pts), 2):
        p1 = pts[i]
        p2 = pts[i + 1]
        if p1[1] <= p2[1]:
            ceiling.append(p1)
            floor.append(p2)
        else:
            ceiling.append(p2)
            floor.append(p1)

    return np.array(ceiling, dtype=np.float64), np.array(floor, dtype=np.float64)


def save_paired_layout_pixels(txt_path: Path, ceiling: np.ndarray, floor: np.ndarray) -> None:
    if len(ceiling) != len(floor):
        raise ValueError("ceiling and floor must have the same number of points")

    txt_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    for c, f in zip(ceiling, floor):
        lines.append(f"{c[0]:.6f} {c[1]:.6f}")
        lines.append(f"{f[0]:.6f} {f[1]:.6f}")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolve_pano_image(scene_dir: Path, room: RoomData) -> Optional[Path]:
    raw = room.raw_node
    candidate_keys = [
        "image_path",
        "rgb_path",
        "pano_path",
        "pano_image",
        "image",
        "path",
    ]

    for key in candidate_keys:
        value = raw.get(key)
        if isinstance(value, str) and value:
            p = Path(value)
            if not p.is_absolute():
                p = scene_dir / p
            if p.exists():
                return p

    candidate_dirs = [
        scene_dir / "rgb",
        scene_dir / "panos",
        scene_dir / "panorama",
        scene_dir / "images",
        scene_dir / "pano",
    ]
    exts = [".jpg", ".jpeg", ".png", ".webp"]

    for d in candidate_dirs:
        for ext in exts:
            p = d / f"{room.pano_id}{ext}"
            if p.exists():
                return p

    return None


def get_segment_colors(n_segments: int) -> List[Any]:
    from matplotlib import cm
    return [cm.hsv(i / max(1, n_segments)) for i in range(max(n_segments, 1))]


def scale_points_to_image(points: np.ndarray, image_shape: Tuple[int, int], ref_w: int, ref_h: int) -> np.ndarray:
    img_h, img_w = image_shape[:2]
    out = points.copy().astype(np.float64)
    out[:, 0] *= img_w / float(ref_w)
    out[:, 1] *= img_h / float(ref_h)
    return out


def image_to_ref_point(x: float, y: float, image_shape: Tuple[int, int], ref_w: int, ref_h: int) -> np.ndarray:
    img_h, img_w = image_shape[:2]
    return np.array([
        x * float(ref_w) / img_w,
        y * float(ref_h) / img_h,
    ], dtype=np.float64)


def point_to_segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-12:
        return float(np.linalg.norm(point - a))
    t = float(np.dot(point - a, ab) / denom)
    t = max(0.0, min(1.0, t))
    proj = a + t * ab
    return float(np.linalg.norm(point - proj))


def nearest_floor_segment_insert_index(floor_ref: np.ndarray, point_ref: np.ndarray, ref_w: int, ref_h: int) -> int:
    n = len(floor_ref)
    if n < 2:
        return n

    floor_xy = np_coor2xy(floor_ref, z=50.0, coorW=ref_w, coorH=ref_h, floorW=ref_w, floorH=ref_w)
    point_arr = np.array([point_ref])
    point_xy = np_coor2xy(point_arr, z=50.0, coorW=ref_w, coorH=ref_h, floorW=ref_w, floorH=ref_w)[0]

    best_i = 0
    best_d = float("inf")
    for i in range(n):
        j = (i + 1) % n
        d = point_to_segment_distance(point_xy, floor_xy[i], floor_xy[j])
        if d < best_d:
            best_d = d
            best_i = i

    return best_i + 1


@dataclass
class SelectedPoint:
    kind: str  # "ceiling" or "floor"
    idx: int   # 0-based


class CornerEditor:
    def __init__(
        self,
        scene_dir: Path,
        room: RoomData,
        pano_image: Optional[np.ndarray],
        txt_path: Path,
        ceiling_ref: np.ndarray,
        floor_ref: np.ndarray,
        ref_w: int = 1024,
        ref_h: int = 512,
    ) -> None:
        self.scene_dir = scene_dir
        self.room = room
        self.pano_image = pano_image
        self.txt_path = txt_path
        self.ceiling_ref = ceiling_ref.copy()
        self.floor_ref = floor_ref.copy()
        self.ref_w = ref_w
        self.ref_h = ref_h

        self.mode = "move"
        self.selected: Optional[SelectedPoint] = None
        self.dragging = False
        self.pending_add_ceiling: Optional[np.ndarray] = None
        self.pending_add_floor: Optional[np.ndarray] = None
        self.history: List[Tuple[np.ndarray, np.ndarray]] = []

        self.fig = plt.figure(figsize=(16, 8))
        gs = GridSpec(2, 2, height_ratios=[4, 1.5], width_ratios=[2.5, 1], figure=self.fig)
        self.ax_pano = self.fig.add_subplot(gs[0, 0])
        self.ax_plan = self.fig.add_subplot(gs[0, 1])
        self.ax_info = self.fig.add_subplot(gs[1, :])
        self.ax_info.axis("off")

        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self.refresh()

    def _push_history(self) -> None:
        self.history.append((self.ceiling_ref.copy(), self.floor_ref.copy()))

    def _update_polygon_local(self) -> None:
        if len(self.floor_ref) < 3:
            return
        floor_xy = np_coor2xy(
            self.floor_ref,
            z=50.0,
            coorW=self.ref_w,
            coorH=self.ref_h,
            floorW=self.ref_w,
            floorH=self.ref_w,
        )
        center = self.ref_w / 2 - 0.5
        floor_xy[:, 0] -= center
        floor_xy[:, 1] -= center
        floor_xy[:, 1] = -floor_xy[:, 1]
        
        try:
            from src.system.annotation_tool import manhattan_align
            floor_xy, _ = manhattan_align(floor_xy)
        except Exception:
            pass
            
        self.room.polygon_local = floor_xy

    def _segment_colors(self) -> List[Any]:
        return get_segment_colors(len(self.floor_ref))

    def _scaled_layout(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.pano_image is None:
            return self.ceiling_ref.copy(), self.floor_ref.copy()
        ceiling_img = scale_points_to_image(self.ceiling_ref, self.pano_image.shape, self.ref_w, self.ref_h)
        floor_img = scale_points_to_image(self.floor_ref, self.pano_image.shape, self.ref_w, self.ref_h)
        return ceiling_img, floor_img

    def _nearest_point_in_image(self, x: float, y: float, max_dist_px: float = 25.0) -> Optional[SelectedPoint]:
        if self.pano_image is None:
            return None

        ceiling_img, floor_img = self._scaled_layout()
        query = np.array([x, y], dtype=np.float64)

        best_kind = None
        best_idx = -1
        best_dist = float("inf")

        for idx, p in enumerate(ceiling_img):
            d = float(np.linalg.norm(query - p))
            if d < best_dist:
                best_dist = d
                best_kind = "ceiling"
                best_idx = idx

        for idx, p in enumerate(floor_img):
            d = float(np.linalg.norm(query - p))
            if d < best_dist:
                best_dist = d
                best_kind = "floor"
                best_idx = idx

        if best_kind is None or best_dist > max_dist_px:
            return None

        return SelectedPoint(best_kind, best_idx)

    def refresh(self) -> None:
        self.draw_panorama()
        self.draw_plan()
        self.draw_info()
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

    def draw_panorama(self) -> None:
        ax = self.ax_pano
        ax.clear()

        colors = self._segment_colors()
        if self.pano_image is not None:
            ax.imshow(self.pano_image)
            img_h, img_w = self.pano_image.shape[:2]
            ax.set_xlim(0, img_w)
            ax.set_ylim(img_h, 0)
        else:
            ax.text(0.5, 0.5, "Panorama image not found", ha="center", va="center", transform=ax.transAxes)

        if len(self.floor_ref) == 0:
            ax.set_title(f"Panorama: {self.room.display_label}")
            ax.axis("off")
            return

        ceiling_img, floor_img = self._scaled_layout()
        n = len(self.floor_ref)
        img_h, img_w = self.ref_h, self.ref_w
        if self.pano_image is not None:
            img_h, img_w = self.pano_image.shape[:2]

        for i in range(n):
            j = (i + 1) % n
            color = colors[i % len(colors)]
            
            ax.plot([ceiling_img[i, 0], floor_img[i, 0]], [ceiling_img[i, 1], floor_img[i, 1]], 
                    color=color, linewidth=2.0, linestyle="--", alpha=0.95)

            no_wrap_c, pts_c = draw_curved_line(ax, ceiling_img[i], ceiling_img[j], z=-50, color=color, w=img_w, h=img_h)
            no_wrap_f, pts_f = draw_curved_line(ax, floor_img[i], floor_img[j], z=50, color=color, w=img_w, h=img_h)

            if no_wrap_c and no_wrap_f and pts_c is not None and pts_f is not None:
                poly_pts = np.vstack([pts_c, pts_f[::-1]])
                ax.fill(poly_pts[:, 0], poly_pts[:, 1], color=color, alpha=0.07)

        for idx, (c, f) in enumerate(zip(ceiling_img, floor_img), start=1):
            ax.scatter([c[0]], [c[1]], s=50, color="tab:purple", edgecolor="black", zorder=5)
            ax.scatter([f[0]], [f[1]], s=60, color="tab:blue", edgecolor="black", zorder=5)
            ax.text(c[0], c[1], f"C{idx}", fontsize=8, ha="center", va="center",
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=0.4), zorder=6)
            ax.text(f[0], f[1], f"F{idx}", fontsize=8, ha="center", va="center",
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=0.4), zorder=6)

        if self.pending_add_ceiling is not None and self.pano_image is not None:
            p = scale_points_to_image(self.pending_add_ceiling[None, :], self.pano_image.shape, self.ref_w, self.ref_h)[0]
            ax.scatter([p[0]], [p[1]], s=140, marker="x", color="tab:red", linewidths=2.5, zorder=7)

        if self.selected is not None and self.pano_image is not None:
            target = ceiling_img[self.selected.idx] if self.selected.kind == "ceiling" else floor_img[self.selected.idx]
            ax.scatter([target[0]], [target[1]], s=170, facecolors="none", edgecolors="red", linewidths=2.5, zorder=7)

        ax.set_title(f"Panorama Editor: {self.room.display_label}")
        ax.axis("off")

    def draw_plan(self) -> None:
        ax = self.ax_plan
        ax.clear()

        poly = self.room.polygon_local
        n = len(poly)
        colors = self._segment_colors()

        for i in range(n):
            j = (i + 1) % n
            ax.plot(
                [poly[i, 0], poly[j, 0]],
                [poly[i, 1], poly[j, 1]],
                color=colors[i % len(colors)],
                linewidth=2.5,
            )

        ax.fill(poly[:, 0], poly[:, 1], alpha=0.08, color="gray")

        for idx, (x, y) in enumerate(poly, start=1):
            ax.scatter([x], [y], s=45, color="tab:blue", edgecolor="white", zorder=5)
            ax.text(
                x,
                y,
                str(idx),
                fontsize=9,
                ha="center",
                va="center",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=0.5),
                zorder=6,
            )

        ax.set_title("Floorplan Preview (local XY)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle="--", alpha=0.4)

    def draw_info(self) -> None:
        self.ax_info.clear()
        self.ax_info.axis("off")

        help_text = (
            f"Mode: {self.mode}\n"
            "Keys:\n"
            "  m = move mode\n"
            "  a = add pair mode (click ceiling, then click floor)\n"
            "  d = delete pair mode (click any ceiling/floor point in panorama)\n"
            "  u = undo\n"
            "  s = save\n"
            "  q = quit\n"
            "\n"
            "Move mode:\n"
            "  click and drag one point in panorama\n"
            "\n"
            "Add mode:\n"
            "  first click a ceiling point position\n"
            "  then click its matching floor point position\n"
            "  new pair is inserted after nearest floor segment\n"
        )

        status_lines = [
            f"Room ID: {self.room.pano_id}",
            f"TXT: {self.txt_path}",
            f"Pairs: {len(self.floor_ref)}",
        ]

        if self.pending_add_ceiling is not None:
            status_lines.append("Pending add: ceiling selected, waiting for floor click")
        else:
            status_lines.append("Pending add: none")

        if self.selected is not None:
            status_lines.append(f"Selected: {self.selected.kind}[{self.selected.idx + 1}]")
        else:
            status_lines.append("Selected: none")

        text = help_text + "\n" + "\n".join(status_lines)
        self.ax_info.text(0.01, 0.98, text, ha="left", va="top", fontsize=11, family="monospace")

    def on_press(self, event) -> None:
        if event.inaxes != self.ax_pano or event.xdata is None or event.ydata is None:
            return

        if self.mode == "move":
            selected = self._nearest_point_in_image(event.xdata, event.ydata)
            if selected is not None:
                self._push_history()
                self.selected = selected
                self.dragging = True
                self.refresh()

        elif self.mode == "delete":
            selected = self._nearest_point_in_image(event.xdata, event.ydata)
            if selected is None:
                return

            self._push_history()
            idx = selected.idx
            self.ceiling_ref = np.delete(self.ceiling_ref, idx, axis=0)
            self.floor_ref = np.delete(self.floor_ref, idx, axis=0)
            print(f"[INFO] Deleted pair index {idx + 1}")
            self.selected = None
            self.pending_add_ceiling = None
            self._update_polygon_local()
            self.refresh()

        elif self.mode == "add":
            if self.pano_image is None:
                return
            point_ref = image_to_ref_point(event.xdata, event.ydata, self.pano_image.shape, self.ref_w, self.ref_h)

            if self.pending_add_ceiling is None:
                self.pending_add_ceiling = point_ref
                print("[INFO] Ceiling point added. Now click floor point.")
            else:
                self._push_history()
                floor_ref = point_ref
                insert_idx = nearest_floor_segment_insert_index(self.floor_ref, floor_ref, self.ref_w, self.ref_h)
                self.ceiling_ref = np.insert(self.ceiling_ref, insert_idx, self.pending_add_ceiling, axis=0)
                self.floor_ref = np.insert(self.floor_ref, insert_idx, floor_ref, axis=0)
                print(f"[INFO] Inserted new pair at position {insert_idx + 1}.")
                self.pending_add_ceiling = None
                self._update_polygon_local()
            self.refresh()

    def on_motion(self, event) -> None:
        if not self.dragging or self.selected is None:
            return
        if event.inaxes != self.ax_pano or event.xdata is None or event.ydata is None:
            return
        if self.pano_image is None:
            return

        new_ref = image_to_ref_point(event.xdata, event.ydata, self.pano_image.shape, self.ref_w, self.ref_h)
        if self.selected.kind == "ceiling":
            self.ceiling_ref[self.selected.idx] = new_ref
        else:
            self.floor_ref[self.selected.idx] = new_ref
        
        self._update_polygon_local()
        self.refresh()

    def on_release(self, event) -> None:
        if self.dragging:
            self.dragging = False
            self.selected = None
            self._update_polygon_local()
            self.refresh()

    def on_key(self, event) -> None:
        if event.key == "m":
            self.mode = "move"
            self.pending_add_ceiling = None
            print("[INFO] Switched to move mode")
            self.refresh()
        elif event.key == "a":
            self.mode = "add"
            self.selected = None
            print("[INFO] Switched to add mode")
            self.refresh()
        elif event.key == "d":
            self.mode = "delete"
            self.pending_add_ceiling = None
            self.selected = None
            print("[INFO] Switched to delete mode")
            self.refresh()
        elif event.key == "u":
            if self.history:
                last_ceiling, last_floor = self.history.pop()
                self.ceiling_ref = last_ceiling
                self.floor_ref = last_floor
                self.pending_add_ceiling = None
                self.selected = None
                self.dragging = False
                self._update_polygon_local()
                print("[INFO] Undo successful.")
                self.refresh()
            else:
                print("[INFO] No history to undo.")
        elif event.key == "s":
            backup = self.txt_path.with_suffix(self.txt_path.suffix + ".bak")
            if not backup.exists():
                backup.write_text(self.txt_path.read_text(encoding="utf-8"), encoding="utf-8")
            save_paired_layout_pixels(self.txt_path, self.ceiling_ref, self.floor_ref)
            print(f"[OK] Saved paired layout to: {self.txt_path}")
            print(f"[OK] Backup kept at: {backup}")
        elif event.key == "q":
            print("[INFO] Quit corner editor.")
            plt.close(self.fig)

    def run(self) -> None:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Edit panorama layout corners in paired mode.")
    parser.add_argument("--scene_dir", required=True, help="Path to scene folder")
    parser.add_argument("--room", required=True, help="Room pano_id (without .txt)")
    parser.add_argument("--pano_w", type=int, default=1024, help="Reference panorama width used by layout txt")
    parser.add_argument("--pano_h", type=int, default=512, help="Reference panorama height used by layout txt")
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir)
    rooms = load_scene_rooms(scene_dir, pano_w=args.pano_w, pano_h=args.pano_h, require_layout=True)
    if args.room not in rooms:
        raise KeyError(f"Room not found: {args.room}")

    room = rooms[args.room]
    txt_path = scene_dir / "layout_gt" / f"{room.pano_id}.txt"
    ceiling_ref, floor_ref = load_paired_layout_pixels(txt_path)

    pano_path = resolve_pano_image(scene_dir, room)
    pano_image = plt.imread(pano_path) if pano_path is not None else None

    print("=" * 80)
    print("Corner Editor")
    print("=" * 80)
    print("Scene      :", scene_dir)
    print("Room       :", room.pano_id)
    print("TXT        :", txt_path)
    print("Panorama   :", pano_path if pano_path else "not found")
    print("Pairs      :", len(floor_ref))
    print("Reference  :", f"{args.pano_w}x{args.pano_h}")

    editor = CornerEditor(
        scene_dir=scene_dir,
        room=room,
        pano_image=pano_image,
        txt_path=txt_path,
        ceiling_ref=ceiling_ref,
        floor_ref=floor_ref,
        ref_w=args.pano_w,
        ref_h=args.pano_h,
    )
    editor.run()


if __name__ == "__main__":
    main()
