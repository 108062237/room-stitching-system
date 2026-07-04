from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class RoomData:
    pano_id: str
    txt_path: Path
    polygon_local: np.ndarray
    connections: List[Dict[str, Any]]
    raw_node: Dict[str, Any]
    display_label: str


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _try_load_labels_utils():
    """
    Optional import:
    If your repo already has src.utils.labels, use it.
    Otherwise fall back to pano_id as display label.
    """
    try:
        from src.utils.labels import get_display_label, get_room_labels

        return get_room_labels, get_display_label
    except Exception:
        return None, None


def _load_rectify_polygon():
    """
    Try to import rectify_polygon from your local project first.
    If unavailable, try LayoutHub/utils/geom.py style import path.
    If still unavailable, return None and skip rectification.
    """
    try:
        from src.utils.geom import rectify_polygon

        return rectify_polygon
    except Exception:
        pass

    return None


def _load_np_coor2xy(scene_dir: Path):
    try:
        from src.utils.post_proc import np_coor2xy
        return np_coor2xy
    except ImportError:
        raise RuntimeError("Cannot import np_coor2xy from src.utils.post_proc")


def load_layout_txt_as_local_xy(
    txt_path: Path,
    np_coor2xy_func,
    rectify_polygon_func=None,
    pano_w: int = 1024,
    pano_h: int = 512,
    layout_z: float = 50.0,
) -> np.ndarray:
    """
    Load layout_gt/*.txt and convert to local floor XY.

    This follows the same convention used in your edge generation / overlay scripts:
    - parse txt
    - if even number of points, keep the floor points (larger y)
    - np_coor2xy(z=layout_z)
    - center shift
    - Y flip
    - optional rectify_polygon
    """
    if not txt_path.exists():
        raise FileNotFoundError("Layout txt not found: {}".format(txt_path))

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
        floor_pixel = [
            pts[i] if pts[i][1] > pts[i + 1][1] else pts[i + 1]
            for i in range(0, len(pts), 2)
        ]
    else:
        floor_pixel = pts

    floor_pixel_arr = np.array(floor_pixel, dtype=np.float64)
    if floor_pixel_arr.shape[0] < 3:
        raise ValueError("Need at least 3 floor points: {}".format(txt_path))

    floor_xy = np_coor2xy_func(
        floor_pixel_arr,
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

    if rectify_polygon_func is not None:
        try:
            floor_xy = rectify_polygon_func(floor_xy)
        except Exception:
            pass

    return floor_xy.astype(np.float64)


def load_scene_manifest(scene_dir: Path) -> Dict[str, Any]:
    manifest_path = scene_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("manifest.json not found: {}".format(manifest_path))
    return _load_json(manifest_path)


def list_room_ids(scene_dir: Path) -> List[str]:
    manifest = load_scene_manifest(scene_dir)
    nodes = manifest.get("nodes", [])
    return [node["pano_id"] for node in nodes if node.get("pano_id")]


def load_scene_rooms(
    scene_dir: Path,
    pano_w: int = 1024,
    pano_h: int = 512,
    layout_z: float = 50.0,
    require_layout: bool = True,
) -> Dict[str, RoomData]:
    """
    Load all rooms in a scene and return:
        {
          pano_id: RoomData(...)
        }

    Each room contains:
    - pano_id
    - txt_path
    - polygon_local (Nx2)
    - connections
    - raw_node
    - display_label
    """
    scene_dir = Path(scene_dir)
    manifest = load_scene_manifest(scene_dir)
    nodes = manifest.get("nodes", [])

    np_coor2xy_func = _load_np_coor2xy(scene_dir)
    rectify_polygon_func = _load_rectify_polygon()
    get_room_labels, get_display_label = _try_load_labels_utils()

    label_map: Dict[str, str] = {}
    if get_room_labels is not None:
        try:
            label_map = get_room_labels(scene_dir)
        except Exception:
            label_map = {}

    rooms: Dict[str, RoomData] = {}

    for node in nodes:
        pano_id = node.get("pano_id", "")
        if not pano_id:
            continue

        txt_path = scene_dir / "layout_gt" / "{}.txt".format(pano_id)
        if require_layout and not txt_path.exists():
            continue

        polygon_local = load_layout_txt_as_local_xy(
            txt_path=txt_path,
            np_coor2xy_func=np_coor2xy_func,
            rectify_polygon_func=rectify_polygon_func,
            pano_w=pano_w,
            pano_h=pano_h,
            layout_z=layout_z,
        )

        if get_display_label is not None:
            try:
                display_label = get_display_label(pano_id, label_map)
            except Exception:
                display_label = pano_id
        else:
            display_label = pano_id

        rooms[pano_id] = RoomData(
            pano_id=pano_id,
            txt_path=txt_path,
            polygon_local=polygon_local,
            connections=node.get("connections", []),
            raw_node=node,
            display_label=display_label,
        )

    return rooms


def load_room_pair(
    scene_dir: Path,
    src_room_id: str,
    dst_room_id: str,
    pano_w: int = 1024,
    pano_h: int = 512,
    layout_z: float = 50.0,
) -> Dict[str, RoomData]:
    """
    Convenience helper for annotation UI.
    Returns only the two selected rooms.
    """
    rooms = load_scene_rooms(
        scene_dir=scene_dir,
        pano_w=pano_w,
        pano_h=pano_h,
        layout_z=layout_z,
        require_layout=True,
    )

    if src_room_id not in rooms:
        raise KeyError("src room not found: {}".format(src_room_id))
    if dst_room_id not in rooms:
        raise KeyError("dst room not found: {}".format(dst_room_id))

    return {
        src_room_id: rooms[src_room_id],
        dst_room_id: rooms[dst_room_id],
    }


def polygon_centroid(poly: np.ndarray) -> np.ndarray:
    return np.mean(poly, axis=0)


def polygon_bbox(poly: np.ndarray) -> Dict[str, float]:
    xmin = float(np.min(poly[:, 0]))
    xmax = float(np.max(poly[:, 0]))
    ymin = float(np.min(poly[:, 1]))
    ymax = float(np.max(poly[:, 1]))
    return {
        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
        "width": xmax - xmin,
        "height": ymax - ymin,
    }