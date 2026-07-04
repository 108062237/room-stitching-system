import argparse
import json
from collections import deque, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.system.room_loader import RoomData, load_scene_rooms, polygon_bbox


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------

def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# -----------------------------------------------------------------------------
# Match parsing
# -----------------------------------------------------------------------------

def normalize_match_entries(raw: Any) -> List[Dict[str, Any]]:
    """
    Supports either:
      [{"src": "a.txt", "dst": "b.txt", "pairs": [[1,3],[2,4]]}, ...]
    or old format:
      [{"src": "a.txt", "dst": "b.txt", "idx_src": [1,2], "idx_dst": [3,4]}, ...]
    """
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

        out.append({
            "src": src,
            "dst": dst,
            "pairs": pairs,
            "src_name": src_name,
            "dst_name": dst_name,
        })
    return out


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------

def to_hmat(r: np.ndarray, t: np.ndarray) -> np.ndarray:
    h = np.eye(3, dtype=np.float64)
    h[:2, :2] = r
    h[:2, 2] = t
    return h


def invert_hmat(h: np.ndarray) -> np.ndarray:
    r = h[:2, :2]
    t = h[:2, 2]
    h_inv = np.eye(3, dtype=np.float64)
    h_inv[:2, :2] = r.T
    h_inv[:2, 2] = -r.T @ t
    return h_inv


def compose_hmat(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b


def apply_hmat(poly: np.ndarray, h: np.ndarray) -> np.ndarray:
    pts_h = np.hstack([poly, np.ones((len(poly), 1), dtype=np.float64)])
    out = (h @ pts_h.T).T
    return out[:, :2]


def rigid_align_points(src_pts: np.ndarray, dst_pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute rigid transform that maps dst_pts -> src_pts.
    aligned = dst_pts @ R.T + t
    """
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


def transform_from_pairs(
    src_poly: np.ndarray,
    dst_poly: np.ndarray,
    pairs: List[Tuple[int, int]],
) -> np.ndarray:
    """
    Return H such that dst_local transformed by H aligns to src_local.

    - 0 pairs: identity
    - 1 pair : translation only
    - >=2    : rigid alignment
    """
    if not pairs:
        return np.eye(3, dtype=np.float64)

    src_pts = np.array([src_poly[s_idx - 1] for s_idx, _ in pairs], dtype=np.float64)
    dst_pts = np.array([dst_poly[d_idx - 1] for _, d_idx in pairs], dtype=np.float64)

    if len(pairs) == 1:
        t = src_pts[0] - dst_pts[0]
        return to_hmat(np.eye(2, dtype=np.float64), t)

    r, t = rigid_align_points(src_pts, dst_pts)
    return to_hmat(r, t)


# -----------------------------------------------------------------------------
# Direct paste preview (NO GTSAM)
# -----------------------------------------------------------------------------

def choose_root_room(matches: List[Dict[str, Any]]) -> Optional[str]:
    if not matches:
        return None
    degree = defaultdict(int)
    for m in matches:
        degree[m["src"]] += 1
        degree[m["dst"]] += 1
    return max(degree.items(), key=lambda kv: kv[1])[0]


def build_direct_preview_world(
    rooms: Dict[str, RoomData],
    matches: List[Dict[str, Any]],
    root_room: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a direct-paste scene preview using only pairwise rigid alignment.

    Important properties:
    - no GTSAM
    - no non-linear optimization
    - loop edges are NOT globally reconciled
    - first-visit BFS assignment wins (tree-like expansion)

    This is exactly meant for fast temporary preview during annotation.
    """
    if not matches:
        return {
            "root": None,
            "world_polys": {},
            "world_transforms": {},
            "ignored_loop_edges": [],
            "used_edges": [],
        }

    if root_room is None:
        root_room = choose_root_room(matches)

    adjacency: Dict[str, List[Tuple[str, np.ndarray, Dict[str, Any]]]] = defaultdict(list)

    for m in matches:
        src = m["src"]
        dst = m["dst"]
        if src not in rooms or dst not in rooms:
            continue

        src_poly = rooms[src].polygon_local
        dst_poly = rooms[dst].polygon_local
        h_dst_to_src = transform_from_pairs(src_poly, dst_poly, m["pairs"])
        h_src_to_dst = invert_hmat(h_dst_to_src)

        # store transform as: neighbor_local -> current_local
        adjacency[src].append((dst, h_dst_to_src, m))
        adjacency[dst].append((src, h_src_to_dst, m))

    if root_room not in rooms:
        raise KeyError(f"Root room not found in loaded rooms: {root_room}")

    visited = set([root_room])
    q = deque([root_room])
    world_transforms: Dict[str, np.ndarray] = {root_room: np.eye(3, dtype=np.float64)}
    used_edges: List[Dict[str, Any]] = []
    ignored_loop_edges: List[Dict[str, Any]] = []

    while q:
        cur = q.popleft()
        h_world_cur = world_transforms[cur]

        for nxt, h_cur_to_nxt, meta in adjacency[cur]:
            if nxt in visited:
                ignored_loop_edges.append({
                    "from": cur,
                    "to": nxt,
                    "reason": "already_assigned_in_direct_preview",
                })
                continue

            world_transforms[nxt] = compose_hmat(h_world_cur, h_cur_to_nxt)
            visited.add(nxt)
            q.append(nxt)
            used_edges.append({"from": cur, "to": nxt})

    world_polys = {
        rid: apply_hmat(room.polygon_local, world_transforms[rid])
        for rid, room in rooms.items()
        if rid in world_transforms
    }

    return {
        "root": root_room,
        "world_polys": world_polys,
        "world_transforms": world_transforms,
        "ignored_loop_edges": ignored_loop_edges,
        "used_edges": used_edges,
    }


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------

def draw_direct_preview(
    world_polys: Dict[str, np.ndarray],
    out_path: Path,
    title: str = "Direct Paste Preview (No GTSAM)",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = plt.get_cmap("tab20")

    room_ids = sorted(world_polys.keys())
    for idx, rid in enumerate(room_ids):
        poly = world_polys[rid]
        color = cmap(idx % 20)
        closed = np.vstack([poly, poly[0:1]])
        ax.plot(closed[:, 0], closed[:, 1], color=color, linewidth=2.0)
        ax.fill(poly[:, 0], poly[:, 1], color=color, alpha=0.12)

        center = np.mean(poly, axis=0)
        ax.text(
            center[0],
            center[1],
            rid[:8],
            fontsize=9,
            ha="center",
            va="center",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=0.5),
        )

    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Temp scene update API
# -----------------------------------------------------------------------------

def run_direct_preview_for_scene(
    scene_dir: Path,
    matches_path: Path,
    out_png: Optional[Path] = None,
    out_json: Optional[Path] = None,
    root_room: Optional[str] = None,
) -> Dict[str, Any]:
    scene_dir = Path(scene_dir)
    matches_path = Path(matches_path)

    rooms = load_scene_rooms(scene_dir, require_layout=True)
    matches = normalize_match_entries(load_json(matches_path))
    result = build_direct_preview_world(rooms, matches, root_room=root_room)

    if out_png is None:
        out_png = scene_dir / "preview_tmp" / "direct_preview.png"
    if out_json is None:
        out_json = scene_dir / "preview_tmp" / "direct_preview.json"

    serializable = {
        "root": result["root"],
        "used_edges": result["used_edges"],
        "ignored_loop_edges": result["ignored_loop_edges"],
        "world_polys": {
            rid: result["world_polys"][rid].tolist() for rid in result["world_polys"]
        },
        "world_transforms": {
            rid: result["world_transforms"][rid].tolist() for rid in result["world_transforms"]
        },
    }
    save_json(out_json, serializable)
    draw_direct_preview(result["world_polys"], out_png)

    return {
        **result,
        "out_png": out_png,
        "out_json": out_json,
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fast direct-paste scene preview using current matches only (no GTSAM)."
    )
    parser.add_argument("--scene_dir", required=True, help="Path to scene folder")
    parser.add_argument("--matches", required=True, help="Path to matches json")
    parser.add_argument("--out_png", default=None, help="Optional output preview png")
    parser.add_argument("--out_json", default=None, help="Optional output preview json")
    parser.add_argument("--root_room", default=None, help="Optional root room pano_id")
    args = parser.parse_args()

    result = run_direct_preview_for_scene(
        scene_dir=Path(args.scene_dir),
        matches_path=Path(args.matches),
        out_png=Path(args.out_png) if args.out_png else None,
        out_json=Path(args.out_json) if args.out_json else None,
        root_room=args.root_room,
    )

    print("=" * 80)
    print("Direct Preview Pipeline")
    print("=" * 80)
    print("Root room:", result["root"])
    print("Placed rooms:", len(result["world_polys"]))
    print("Used edges:", len(result["used_edges"]))
    print("Ignored loop edges:", len(result["ignored_loop_edges"]))
    print("PNG:", result["out_png"])
    print("JSON:", result["out_json"])


if __name__ == "__main__":
    main()