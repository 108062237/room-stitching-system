import argparse
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

try:
    from src.utils.panostretch import pano_connect_points
except ImportError:
    pano_connect_points = None

from src.utils.geom import align_to_manhattan

from src.system.preview_pipeline import (
    apply_hmat,
    choose_root_room,
    compose_hmat,
    invert_hmat,
    transform_from_pairs,
)
from src.system.room_loader import (
    RoomData,
    load_room_pair,
    load_scene_rooms,
    polygon_bbox,
)


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


def rigid_align_points(src_pts: np.ndarray, dst_pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute rigid transform that maps dst_pts -> src_pts."""
    if src_pts.shape != dst_pts.shape:
        raise ValueError("src_pts and dst_pts must have the same shape")
    if src_pts.shape[0] < 2:
        raise ValueError("Need at least 2 points for rigid alignment")

    src_centroid = np.mean(src_pts, axis=0)
    dst_centroid = np.mean(dst_pts, axis=0)

    src_centered = src_pts - src_centroid
    dst_centered = dst_pts - dst_centroid

    h = dst_centered.T @ src_centered
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T

    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T

    t = src_centroid - dst_centroid @ r.T
    return r, t


def align_polygon_from_pairs(
    src_poly: np.ndarray,
    dst_poly: np.ndarray,
    pairs: List[Tuple[int, int]],
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Build a simple preview alignment of dst onto src using current pairs."""
    if not pairs:
        return dst_poly.copy(), None, None

    src_pts = np.array([src_poly[s_idx - 1] for s_idx, _ in pairs], dtype=np.float64)
    dst_pts = np.array([dst_poly[d_idx - 1] for _, d_idx in pairs], dtype=np.float64)

    if len(pairs) == 1:
        t = src_pts[0] - dst_pts[0]
        return dst_poly + t, np.eye(2), t

    r, t = rigid_align_points(src_pts, dst_pts)
    aligned_dst = dst_poly @ r.T + t
    return aligned_dst, r, t


def normalize_loaded_entry(item: Dict[str, Any]) -> List[Tuple[int, int]]:
    if "pairs" in item:
        return [(int(a), int(b)) for a, b in item["pairs"]]

    idx_src = item.get("idx_src", [])
    idx_dst = item.get("idx_dst", [])
    return [(int(a), int(b)) for a, b in zip(idx_src, idx_dst)]


def load_existing_pairs(path: Optional[Path], src_name: str, dst_name: str) -> List[Tuple[int, int]]:
    if path is None or not path.exists():
        return []

    data = json.loads(path.read_text(encoding="utf-8"))
    items: List[Dict[str, Any]]

    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        if "constraints" in data and isinstance(data["constraints"], list):
            items = data["constraints"]
        else:
            items = [data]
    else:
        return []

    for item in items:
        if item.get("src") == src_name and item.get("dst") == dst_name:
            return normalize_loaded_entry(item)

    return []


def load_all_match_entries(path: Optional[Path]) -> List[Dict[str, Any]]:
    if path is None or not path.exists():
        return []

    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        if "constraints" in raw and isinstance(raw["constraints"], list):
            raw = raw["constraints"]
        else:
            raw = [raw]

    out: List[Dict[str, Any]] = []
    for item in raw:
        src_name = str(item["src"])
        dst_name = str(item["dst"])
        src = src_name.replace(".txt", "")
        dst = dst_name.replace(".txt", "")

        if "pairs" in item:
            pairs = [(int(a), int(b)) for a, b in item["pairs"]]
        else:
            idx_src = item.get("idx_src", [])
            idx_dst = item.get("idx_dst", [])
            pairs = [(int(a), int(b)) for a, b in zip(idx_src, idx_dst)]

        out.append(
            {
                "src": src,
                "dst": dst,
                "src_name": src_name,
                "dst_name": dst_name,
                "pairs": pairs,
            }
        )
    return out


def upsert_match_entries(
    entries: List[Dict[str, Any]],
    src_room_id: str,
    dst_room_id: str,
    src_name: str,
    dst_name: str,
    pairs: List[Tuple[int, int]],
) -> List[Dict[str, Any]]:
    updated: List[Dict[str, Any]] = []
    replaced = False

    for item in entries:
        if item["src"] == src_room_id and item["dst"] == dst_room_id:
            if pairs:
                updated.append(
                    {
                        "src": src_room_id,
                        "dst": dst_room_id,
                        "src_name": src_name,
                        "dst_name": dst_name,
                        "pairs": list(pairs),
                    }
                )
            replaced = True
        else:
            updated.append(item)

    if not replaced and pairs:
        updated.append(
            {
                "src": src_room_id,
                "dst": dst_room_id,
                "src_name": src_name,
                "dst_name": dst_name,
                "pairs": list(pairs),
            }
        )

    return updated


def upsert_annotation_file(
    out_path: Path,
    src_name: str,
    dst_name: str,
    pairs: List[Tuple[int, int]],
) -> None:
    entry = {
        "src": src_name,
        "dst": dst_name,
        "pairs": [[int(a), int(b)] for a, b in pairs],
    }

    if out_path.exists():
        try:
            data = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            data = []
    else:
        data = []

    if isinstance(data, list):
        updated = False
        for idx, item in enumerate(data):
            if item.get("src") == src_name and item.get("dst") == dst_name:
                data[idx] = entry
                updated = True
                break
        if not updated:
            data.append(entry)
        payload = data
    elif isinstance(data, dict) and "constraints" in data and isinstance(data["constraints"], list):
        updated = False
        for idx, item in enumerate(data["constraints"]):
            if item.get("src") == src_name and item.get("dst") == dst_name:
                data["constraints"][idx] = entry
                updated = True
                break
        if not updated:
            data["constraints"].append(entry)
        payload = data
    else:
        payload = [entry]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_layout_pixels(txt_path: Path) -> Dict[str, Optional[np.ndarray]]:
    pts: List[List[float]] = []
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) >= 2:
            pts.append([float(parts[0]), float(parts[1])])

    if len(pts) < 2:
        raise ValueError("TXT point count too small: {}".format(txt_path))

    if len(pts) % 2 == 0:
        ceiling = []
        floor = []
        for i in range(0, len(pts), 2):
            p1 = pts[i]
            p2 = pts[i + 1]
            if p1[1] <= p2[1]:
                ceiling.append(p1)
                floor.append(p2)
            else:
                ceiling.append(p2)
                floor.append(p1)
        return {
            "ceiling": np.array(ceiling, dtype=np.float64),
            "floor": np.array(floor, dtype=np.float64),
        }

    floor = np.array(pts, dtype=np.float64)
    return {"ceiling": None, "floor": floor}


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


def draw_plan_polygon(
    ax,
    poly: np.ndarray,
    title: str,
    segment_colors: List[Any],
    highlight_idx: Optional[int] = None,
    pair_indices: Optional[List[int]] = None,
) -> None:
    ax.clear()
    n = len(poly)
    pair_indices = pair_indices or []

    for i in range(n):
        j = (i + 1) % n
        color = segment_colors[i % len(segment_colors)]
        ax.plot(
            [poly[i, 0], poly[j, 0]],
            [poly[i, 1], poly[j, 1]],
            color=color,
            linewidth=2.5,
        )

    ax.fill(poly[:, 0], poly[:, 1], alpha=0.08, color="gray")

    for idx, (x, y) in enumerate(poly, start=1):
        marker_size = 40
        color = "tab:blue"
        edge = "white"

        if idx == highlight_idx:
            color = "tab:red"
            marker_size = 120
            edge = "black"
        elif idx in pair_indices:
            color = "tab:green"
            marker_size = 90
            edge = "black"

        ax.scatter([x], [y], s=marker_size, color=color, edgecolor=edge, zorder=5)
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

    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.4)


def draw_preview(ax, src_room: RoomData, dst_room: RoomData, pairs: List[Tuple[int, int]]) -> None:
    ax.clear()
    src_poly = src_room.polygon_local
    dst_poly = dst_room.polygon_local
    aligned_dst, _, _ = align_polygon_from_pairs(src_poly, dst_poly, pairs)

    src_closed = np.vstack([src_poly, src_poly[0:1]])
    dst_closed = np.vstack([aligned_dst, aligned_dst[0:1]])

    ax.plot(src_closed[:, 0], src_closed[:, 1], color="black", linewidth=1.8, label="SRC")
    ax.fill(src_poly[:, 0], src_poly[:, 1], alpha=0.12)

    ax.plot(dst_closed[:, 0], dst_closed[:, 1], color="tab:orange", linewidth=1.8, label="DST aligned")
    ax.fill(aligned_dst[:, 0], aligned_dst[:, 1], alpha=0.12, color="tab:orange")

    for src_idx, dst_idx in pairs:
        src_pt = src_poly[src_idx - 1]
        dst_pt = aligned_dst[dst_idx - 1]
        ax.scatter([src_pt[0]], [src_pt[1]], s=70, color="tab:green", edgecolor="black", zorder=6)
        ax.scatter([dst_pt[0]], [dst_pt[1]], s=70, color="tab:red", edgecolor="black", zorder=6)
        ax.plot([src_pt[0], dst_pt[0]], [src_pt[1], dst_pt[1]], linestyle="--", alpha=0.7)

    ax.set_title("Pairwise Preview")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best")


def draw_pano_overlay(
    ax,
    image: Optional[np.ndarray],
    layout_pixels: Dict[str, Optional[np.ndarray]],
    title: str,
    segment_colors: List[Any],
    pair_indices: Optional[List[int]] = None,
    ref_w: int = 1024,
    ref_h: int = 512,
) -> None:
    ax.clear()
    pair_indices = pair_indices or []

    scale_x, scale_y = 1.0, 1.0
    img_w, img_h = ref_w, ref_h

    if image is not None:
        ax.imshow(image)
        height, width = image.shape[:2]
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        img_w, img_h = width, height
        scale_x = width / ref_w
        scale_y = height / ref_h
    else:
        ax.text(0.5, 0.5, "Panorama image not found", ha="center", va="center", transform=ax.transAxes)

    floor = layout_pixels.get("floor")
    ceiling = layout_pixels.get("ceiling")

    if floor is not None:
        floor = floor.copy()
        floor[:, 0] *= scale_x
        floor[:, 1] *= scale_y

    if ceiling is not None:
        ceiling = ceiling.copy()
        ceiling[:, 0] *= scale_x
        ceiling[:, 1] *= scale_y

    if floor is not None and len(floor) >= 2:
        n = len(floor)
        for i in range(n):
            j = (i + 1) % n
            color = segment_colors[i % len(segment_colors)]

            if ceiling is not None and len(ceiling) == n:
                ax.plot(
                    [ceiling[i, 0], floor[i, 0]],
                    [ceiling[i, 1], floor[i, 1]],
                    color=color,
                    linewidth=2.0,
                    linestyle="--",
                    alpha=0.95,
                )

                no_wrap_c, pts_c = draw_curved_line(
                    ax,
                    ceiling[i],
                    ceiling[j],
                    z=-50,
                    color=color,
                    w=img_w,
                    h=img_h,
                )
                no_wrap_f, pts_f = draw_curved_line(
                    ax,
                    floor[i],
                    floor[j],
                    z=50,
                    color=color,
                    w=img_w,
                    h=img_h,
                )

                if no_wrap_c and no_wrap_f and pts_c is not None and pts_f is not None:
                    poly_pts = np.vstack([pts_c, pts_f[::-1]])
                    ax.fill(poly_pts[:, 0], poly_pts[:, 1], color=color, alpha=0.08)
            else:
                draw_curved_line(ax, floor[i], floor[j], z=50, color=color, w=img_w, h=img_h)

        for idx, (x, y) in enumerate(floor, start=1):
            marker_size = 40
            color = "tab:blue"
            edge = "white"
            if idx in pair_indices:
                color = "tab:green"
                marker_size = 90
                edge = "black"

            ax.scatter([x], [y], s=marker_size, color=color, edgecolor=edge, zorder=5)
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

    ax.set_title(title)
    ax.axis("off")


def nearest_vertex_index(poly: np.ndarray, x: float, y: float) -> int:
    pts = poly[:, :2]
    d = np.sum((pts - np.array([x, y], dtype=np.float64)) ** 2, axis=1)
    return int(np.argmin(d)) + 1


class AnnotationTool:
    def __init__(
        self,
        scene_dir: Path,
        src_room: RoomData,
        dst_room: RoomData,
        out_path: Path,
        preload_pairs: Optional[List[Tuple[int, int]]] = None,
        pano_w: int = 1024,
        pano_h: int = 512,
        match_store_path: Optional[Path] = None,
    ) -> None:
        self.scene_dir = scene_dir
        self.src_room = src_room
        self.dst_room = dst_room
        self.out_path = out_path
        self.pano_w = pano_w
        self.pano_h = pano_h
        self.match_store_path = match_store_path

        self.pairs: List[Tuple[int, int]] = preload_pairs[:] if preload_pairs else []
        self.pending_src_idx: Optional[int] = None

        self.src_seg_colors = get_segment_colors(len(self.src_room.polygon_local))
        self.dst_seg_colors = get_segment_colors(len(self.dst_room.polygon_local))

        self.src_layout_pixels = load_layout_pixels(self.src_room.txt_path)
        self.dst_layout_pixels = load_layout_pixels(self.dst_room.txt_path)
        self.src_pano_path = resolve_pano_image(scene_dir, src_room)
        self.dst_pano_path = resolve_pano_image(scene_dir, dst_room)
        self.src_pano_img = plt.imread(self.src_pano_path) if self.src_pano_path else None
        self.dst_pano_img = plt.imread(self.dst_pano_path) if self.dst_pano_path else None

        self.scene_rooms = load_scene_rooms(
            scene_dir=scene_dir,
            pano_w=pano_w,
            pano_h=pano_h,
            layout_z=50.0,
            require_layout=True,
        )

        self.src_room.polygon_local, _ = align_to_manhattan(self.src_room.polygon_local)
        self.dst_room.polygon_local, _ = align_to_manhattan(self.dst_room.polygon_local)
        for r in self.scene_rooms.values():
            r.polygon_local, _ = align_to_manhattan(r.polygon_local)

        self.all_match_entries = load_all_match_entries(match_store_path)

        if self.src_room.pano_id in self.scene_rooms:
            self.preview_root = self.src_room.pano_id
        else:
            chosen = choose_root_room(self.all_match_entries)
            self.preview_root = chosen if chosen is not None else self.src_room.pano_id

        self.base_world_h: Dict[str, np.ndarray] = {}
        self.tree_parent: Dict[str, Optional[str]] = {}
        self.tree_children: Dict[str, List[str]] = defaultdict(list)
        self.tree_edge_to_parent: Dict[str, np.ndarray] = {}
        self.scene_preview_status = "init"

        self.fig = plt.figure(figsize=(15, 8))
        gs = GridSpec(2, 3, height_ratios=[4, 1.4], width_ratios=[1, 1, 1.1], figure=self.fig)
        self.ax_src = self.fig.add_subplot(gs[0, 0])
        self.ax_dst = self.fig.add_subplot(gs[0, 1])
        self.ax_preview = self.fig.add_subplot(gs[0, 2])
        self.ax_info = self.fig.add_subplot(gs[1, :])
        self.ax_info.axis("off")

        self.fig_pano, (self.ax_pano_src, self.ax_pano_dst) = plt.subplots(1, 2, figsize=(16, 5))
        try:
            self.fig_pano.canvas.manager.set_window_title("Panorama View")
        except Exception:
            pass

        self.fig_scene_preview, self.ax_scene_preview = plt.subplots(figsize=(8, 8))
        try:
            self.fig_scene_preview.canvas.manager.set_window_title("Scene Direct Preview")
        except Exception:
            pass

        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig_pano.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig_scene_preview.canvas.mpl_connect("key_press_event", self.on_key)

        self.rebuild_scene_preview_baseline()
        self.refresh()

    def _src_pair_indices(self) -> List[int]:
        return [a for a, _ in self.pairs]

    def _dst_pair_indices(self) -> List[int]:
        return [b for _, b in self.pairs]

    def rebuild_scene_preview_baseline(self) -> None:
        """
        Build a stable base tree once from the saved/global matches.
        This is not rebuilt on every click.
        """
        entries = self.all_match_entries
        adjacency: Dict[str, List[Tuple[str, np.ndarray]]] = defaultdict(list)

        for item in entries:
            src = item["src"]
            dst = item["dst"]
            pairs = item["pairs"]

            if not pairs:
                continue
            if src not in self.scene_rooms or dst not in self.scene_rooms:
                continue

            src_poly = self.scene_rooms[src].polygon_local
            dst_poly = self.scene_rooms[dst].polygon_local

            # maps dst -> src
            h_dst_to_src = transform_from_pairs(src_poly, dst_poly, pairs)
            h_src_to_dst = invert_hmat(h_dst_to_src)

            # store as neighbor_local -> current_local
            adjacency[src].append((dst, h_dst_to_src))
            adjacency[dst].append((src, h_src_to_dst))

        self.base_world_h = {self.preview_root: np.eye(3, dtype=np.float64)}
        self.tree_parent = {self.preview_root: None}
        self.tree_children = defaultdict(list)
        self.tree_edge_to_parent = {}

        visited = set([self.preview_root])
        q = deque([self.preview_root])

        while q:
            cur = q.popleft()
            h_world_cur = self.base_world_h[cur]

            for nxt, h_nxt_to_cur in adjacency[cur]:
                if nxt in visited:
                    continue

                self.base_world_h[nxt] = compose_hmat(h_world_cur, h_nxt_to_cur)
                self.tree_parent[nxt] = cur
                self.tree_children[cur].append(nxt)
                self.tree_edge_to_parent[nxt] = h_nxt_to_cur

                visited.add(nxt)
                q.append(nxt)

    def propagate_subtree_from(self, root_child: str, world_h: Dict[str, np.ndarray]) -> None:
        for child in self.tree_children.get(root_child, []):
            h_child_to_parent = self.tree_edge_to_parent[child]
            parent = self.tree_parent[child]
            if parent is None:
                continue
            world_h[child] = compose_hmat(world_h[parent], h_child_to_parent)
            self.propagate_subtree_from(child, world_h)

    def compute_incremental_scene_preview(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Incrementally update scene preview based only on the current edited edge.
        Returns:
          world_h: actual placed room transforms
          ghost_polys: optional ghost overlay for loop/cross edge
        """
        world_h = {rid: h.copy() for rid, h in self.base_world_h.items()}
        ghost_polys: Dict[str, np.ndarray] = {}

        src_id = self.src_room.pano_id
        dst_id = self.dst_room.pano_id

        if not self.pairs:
            self.scene_preview_status = "baseline only"
            return world_h, ghost_polys

        src_poly = self.src_room.polygon_local
        dst_poly = self.dst_room.polygon_local

        # maps dst -> src
        h_dst_to_src = transform_from_pairs(src_poly, dst_poly, self.pairs)
        h_src_to_dst = invert_hmat(h_dst_to_src)

        src_in_world = src_id in world_h
        dst_in_world = dst_id in world_h

        if self.tree_parent.get(dst_id) == src_id and src_in_world:
            world_h[dst_id] = compose_hmat(world_h[src_id], h_dst_to_src)
            self.propagate_subtree_from(dst_id, world_h)
            self.scene_preview_status = "updated dst subtree from src"
            return world_h, ghost_polys

        if self.tree_parent.get(src_id) == dst_id and dst_in_world:
            world_h[src_id] = compose_hmat(world_h[dst_id], h_src_to_dst)
            self.propagate_subtree_from(src_id, world_h)
            self.scene_preview_status = "updated src subtree from dst"
            return world_h, ghost_polys

        if src_in_world and not dst_in_world:
            world_h[dst_id] = compose_hmat(world_h[src_id], h_dst_to_src)
            self.scene_preview_status = "attached dst to placed src"
            return world_h, ghost_polys

        if dst_in_world and not src_in_world:
            world_h[src_id] = compose_hmat(world_h[dst_id], h_src_to_dst)
            self.scene_preview_status = "attached src to placed dst"
            return world_h, ghost_polys

        if src_in_world and dst_in_world:
            ghost_h = compose_hmat(world_h[src_id], h_dst_to_src)
            ghost_polys[dst_id] = apply_hmat(dst_poly, ghost_h)
            self.scene_preview_status = "loop/cross edge: ghost dst overlay only"
            return world_h, ghost_polys

        if not src_in_world and not dst_in_world:
            world_h[src_id] = np.eye(3, dtype=np.float64)
            world_h[dst_id] = compose_hmat(world_h[src_id], h_dst_to_src)
            self.scene_preview_status = "isolated current pair"
            return world_h, ghost_polys

        self.scene_preview_status = "no update"
        return world_h, ghost_polys

    def draw_scene_preview(self) -> None:
        self.ax_scene_preview.clear()
        cmap = plt.get_cmap("tab20")

        world_h, ghost_polys = self.compute_incremental_scene_preview()

        room_ids = sorted(world_h.keys())
        for idx, rid in enumerate(room_ids):
            color = cmap(idx % 20)
            poly = apply_hmat(self.scene_rooms[rid].polygon_local, world_h[rid])
            closed = np.vstack([poly, poly[0:1]])
            self.ax_scene_preview.plot(closed[:, 0], closed[:, 1], color=color, linewidth=2.0)
            self.ax_scene_preview.fill(poly[:, 0], poly[:, 1], color=color, alpha=0.12)

            center = np.mean(poly, axis=0)
            label = self.scene_rooms[rid].display_label if self.scene_rooms[rid].display_label else rid[-6:]
            if rid == self.preview_root:
                label = f"{label} (root)"
            self.ax_scene_preview.text(
                center[0],
                center[1],
                label,
                fontsize=9,
                ha="center",
                va="center",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=0.5),
            )

        for rid, poly in ghost_polys.items():
            closed = np.vstack([poly, poly[0:1]])
            self.ax_scene_preview.plot(
                closed[:, 0],
                closed[:, 1],
                color="red",
                linewidth=2.0,
                linestyle="--",
            )
            self.ax_scene_preview.fill(poly[:, 0], poly[:, 1], color="red", alpha=0.06)

            center = np.mean(poly, axis=0)
            self.ax_scene_preview.text(
                center[0],
                center[1],
                f"{rid[-6:]} ghost",
                fontsize=9,
                ha="center",
                va="center",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=0.5),
            )

        self.ax_scene_preview.set_title("Scene Direct Preview (Incremental, No GTSAM)")
        self.ax_scene_preview.set_aspect("equal", adjustable="box")
        self.ax_scene_preview.grid(True, linestyle="--", alpha=0.35)
        self.fig_scene_preview.tight_layout()
        self.fig_scene_preview.canvas.draw_idle()

    def refresh(self) -> None:
        draw_plan_polygon(
            self.ax_src,
            self.src_room.polygon_local,
            f"SRC Plan: {self.src_room.display_label}",
            self.src_seg_colors,
            highlight_idx=self.pending_src_idx,
            pair_indices=self._src_pair_indices(),
        )
        draw_plan_polygon(
            self.ax_dst,
            self.dst_room.polygon_local,
            f"DST Plan: {self.dst_room.display_label}",
            self.dst_seg_colors,
            highlight_idx=None,
            pair_indices=self._dst_pair_indices(),
        )
        draw_preview(self.ax_preview, self.src_room, self.dst_room, self.pairs)
        self.draw_info()
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

        draw_pano_overlay(
            self.ax_pano_src,
            self.src_pano_img,
            self.src_layout_pixels,
            f"SRC Panorama: {self.src_room.display_label}",
            self.src_seg_colors,
            pair_indices=self._src_pair_indices(),
            ref_w=self.pano_w,
            ref_h=self.pano_h,
        )
        draw_pano_overlay(
            self.ax_pano_dst,
            self.dst_pano_img,
            self.dst_layout_pixels,
            f"DST Panorama: {self.dst_room.display_label}",
            self.dst_seg_colors,
            pair_indices=self._dst_pair_indices(),
            ref_w=self.pano_w,
            ref_h=self.pano_h,
        )
        self.fig_pano.tight_layout()
        self.fig_pano.canvas.draw_idle()

        self.draw_scene_preview()

    def draw_info(self) -> None:
        self.ax_info.clear()
        self.ax_info.axis("off")

        instructions = (
            "Instructions:\n"
            "1) Click a point in the SRC plan plot (left)\n"
            "2) Click the corresponding point in the DST plan plot (middle)\n"
            "3) Keyboard: [u] undo  [c] clear  [s] save  [q] quit\n"
            "4) Panorama window shows the same room segments with matching colors\n"
            "5) Scene preview updates incrementally without rebuilding the whole tree\n"
        )

        pair_lines = ["Current pairs:"]
        if self.pairs:
            for idx, (a, b) in enumerate(self.pairs, start=1):
                pair_lines.append(f"  {idx}. SRC {a}  ↔  DST {b}")
        else:
            pair_lines.append("  (none)")

        pending_line = (
            f"Pending SRC index: {self.pending_src_idx}"
            if self.pending_src_idx is not None
            else "Pending SRC index: None"
        )

        pano_info = (
            f"SRC pano: {self.src_pano_path if self.src_pano_path else 'not found'}\n"
            f"DST pano: {self.dst_pano_path if self.dst_pano_path else 'not found'}"
        )

        scene_preview_info = (
            f"Scene preview root: {self.preview_root}\n"
            f"Scene preview status: {self.scene_preview_status}\n"
            f"Baseline placed rooms: {len(self.base_world_h)}"
        )

        text = (
            instructions
            + "\n"
            + pending_line
            + "\n\n"
            + "\n".join(pair_lines)
            + "\n\n"
            + pano_info
            + "\n\n"
            + scene_preview_info
        )
        self.ax_info.text(0.01, 0.98, text, ha="left", va="top", fontsize=11, family="monospace")

    def on_click(self, event) -> None:
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return

        if event.inaxes == self.ax_src:
            idx = nearest_vertex_index(self.src_room.polygon_local, event.xdata, event.ydata)
            self.pending_src_idx = idx
            self.refresh()
            return

        if event.inaxes == self.ax_dst:
            if self.pending_src_idx is None:
                print("[INFO] Please click a SRC point first.")
                return

            dst_idx = nearest_vertex_index(self.dst_room.polygon_local, event.xdata, event.ydata)
            pair = (self.pending_src_idx, dst_idx)

            if pair not in self.pairs:
                self.pairs.append(pair)
                print(f"[INFO] Added pair: SRC {pair[0]} ↔ DST {pair[1]}")
            else:
                print(f"[INFO] Pair already exists: SRC {pair[0]} ↔ DST {pair[1]}")

            self.pending_src_idx = None
            self.refresh()

    def on_key(self, event) -> None:
        if event.key == "u":
            if self.pairs:
                removed = self.pairs.pop()
                print(f"[INFO] Undo pair: SRC {removed[0]} ↔ DST {removed[1]}")
            else:
                print("[INFO] No pairs to undo.")
            self.pending_src_idx = None
            self.refresh()

        elif event.key == "c":
            self.pairs.clear()
            self.pending_src_idx = None
            print("[INFO] Cleared all pairs.")
            self.refresh()

        elif event.key == "s":
            upsert_annotation_file(
                self.out_path,
                self.src_room.txt_path.name,
                self.dst_room.txt_path.name,
                self.pairs,
            )
            print(f"[OK] Saved annotation to: {self.out_path}")

            self.all_match_entries = upsert_match_entries(
                self.all_match_entries,
                self.src_room.pano_id,
                self.dst_room.pano_id,
                self.src_room.txt_path.name,
                self.dst_room.txt_path.name,
                self.pairs,
            )

            self.rebuild_scene_preview_baseline()
            self.refresh()

        elif event.key == "q":
            print("[INFO] Quit annotation tool.")
            plt.close(self.fig)
            plt.close(self.fig_pano)
            plt.close(self.fig_scene_preview)

    def run(self) -> None:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive annotation tool for room correspondences.")
    parser.add_argument("--scene_dir", required=True, help="Path to scene folder")
    parser.add_argument("--src_room", required=True, help="Source room pano_id (without .txt)")
    parser.add_argument("--dst_room", required=True, help="Destination room pano_id (without .txt)")
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--load_json", default=None, help="Optional existing JSON to preload")
    parser.add_argument("--layout_z", type=float, default=50.0)
    parser.add_argument("--pano_w", type=int, default=1024)
    parser.add_argument("--pano_h", type=int, default=512)
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir)
    out_path = Path(args.out)
    load_path = Path(args.load_json) if args.load_json else out_path

    pair_dict = load_room_pair(
        scene_dir=scene_dir,
        src_room_id=args.src_room,
        dst_room_id=args.dst_room,
        pano_w=args.pano_w,
        pano_h=args.pano_h,
        layout_z=args.layout_z,
    )

    src_room = pair_dict[args.src_room]
    dst_room = pair_dict[args.dst_room]

    preload_pairs = load_existing_pairs(load_path, src_room.txt_path.name, dst_room.txt_path.name)

    print("=" * 80)
    print("Annotation Tool")
    print("=" * 80)
    print("Scene      :", scene_dir)
    print("SRC room   :", src_room.pano_id)
    print("DST room   :", dst_room.pano_id)
    print("Output JSON:", out_path)
    print("Loaded pairs:", preload_pairs)

    src_bbox = polygon_bbox(src_room.polygon_local)
    dst_bbox = polygon_bbox(dst_room.polygon_local)
    print("SRC bbox:", src_bbox)
    print("DST bbox:", dst_bbox)

    tool = AnnotationTool(
        scene_dir=scene_dir,
        src_room=src_room,
        dst_room=dst_room,
        out_path=out_path,
        preload_pairs=preload_pairs,
        pano_w=args.pano_w,
        pano_h=args.pano_h,
        match_store_path=load_path,
    )
    tool.run()


if __name__ == "__main__":
    main()