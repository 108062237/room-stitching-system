"""Room assembly calculations."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
PI = float(np.pi)

def np_coorx2u(coorx, coorW=1024):
    return ((coorx + 0.5) / coorW - 0.5) * 2 * PI


def np_coory2v(coory, coorH=512):
    return -((coory + 0.5) / coorH - 0.5) * PI


def np_coor2xy(coor, z=50, coorW=1024, coorH=512, floorW=1024, floorH=512):
    """
    coor: N x 2, index of array in (col, row) format
    """
    coor = np.array(coor)
    u = np_coorx2u(coor[:, 0], coorW)
    v = np_coory2v(coor[:, 1], coorH)
    c = z / np.tan(v)
    x = c * np.sin(u) + floorW / 2 - 0.5
    y = -c * np.cos(u) + floorH / 2 - 0.5
    return np.hstack([x[:, None], y[:, None]])


_np_coor2xy = np_coor2xy


def align_to_manhattan(xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    利用連續向量平均法，自動找出房間的主軸角度，並旋轉至正南正北。
    回傳 (旋轉對齊後的座標, 旋轉矩陣 R)
    """
    if len(xy) < 3:
        return xy, np.eye(2)

    shifted_xy = np.roll(xy, -1, axis=0)
    dx = shifted_xy[:, 0] - xy[:, 0]
    dy = shifted_xy[:, 1] - xy[:, 1]

    lengths = np.hypot(dx, dy)
    angles = np.arctan2(dy, dx)

    Sx = np.sum(lengths * np.cos(4 * angles))
    Sy = np.sum(lengths * np.sin(4 * angles))

    theta_dom = np.arctan2(Sy, Sx) / 4.0

    cos_t = np.cos(-theta_dom)
    sin_t = np.sin(-theta_dom)

    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

    # 套用旋轉 R.T
    return xy @ R.T, R


def rectify_polygon(xy: np.ndarray, rotate_back: bool = True) -> np.ndarray:
    """
    強制將偏離些許的角度修正為完全 90 度與 270 度的完美曼哈頓形狀。
    如果 rotate_back=True，則在修正完長寬比例後轉回原來的朝向。
    如果為 False，則會直接回傳完美平行於 Global X 與 Y 軸的方形格局！
    """
    if len(xy) < 3:
        return xy

    xy_aligned, R = align_to_manhattan(xy)
    N = len(xy_aligned)

    # 決定各個相鄰邊是水平 H 或垂直 V
    edge_types = []
    for i in range(N):
        p1 = xy_aligned[i]
        p2 = xy_aligned[(i + 1) % N]
        if abs(p2[0] - p1[0]) > abs(p2[1] - p1[1]):
            edge_types.append("H")  # y 一致
        else:
            edge_types.append("V")  # x 一致

    x_labels = list(range(N))
    y_labels = list(range(N))

    def merge(labels, a, b):
        target = labels[b]
        source = labels[a]
        for i in range(N):
            if labels[i] == source:
                labels[i] = target

    for i in range(N):
        next_i = (i + 1) % N
        if edge_types[i] == "H":
            merge(y_labels, i, next_i)  # 水平線相連的兩個點，Y是相同的
        else:
            merge(x_labels, i, next_i)  # 垂直線相連的兩個點，X是相同的

    rectified = np.copy(xy_aligned)

    for label in set(x_labels):
        indices = [i for i, l in enumerate(x_labels) if l == label]
        rectified[indices, 0] = np.mean(xy_aligned[indices, 0])

    for label in set(y_labels):
        indices = [i for i, l in enumerate(y_labels) if l == label]
        rectified[indices, 1] = np.mean(xy_aligned[indices, 1])

    # 因為 xy_aligned = xy @ R.T，所以旋轉回去就是乘以 R (因為 R.T @ R = I)
    if rotate_back:
        return rectified @ R
    else:
        return rectified


def load_layout_gt_txt_as_local_xy(
    txt_path: Path,
    pano_w: int = 1024,
    pano_h: int = 512,
    z: float = 50.0,
) -> Optional[np.ndarray]:
    """
    Load a layout .txt file and project the floor pixels to local XY space.

    Returns an Nx2 float64 array in the camera-local frame, or None on failure.
    All scripts (04, 05, 08, tool_generate_gtsam_edges) should use ONLY this
    function to guarantee a consistent coordinate frame.
    """
    if _np_coor2xy is None:
        raise RuntimeError(
            "Cannot import src.utils.post_proc.np_coor2xy. Cannot load polygon."
        )

    txt_path = Path(txt_path)
    if not txt_path.exists():
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

    # Interleaved (ceiling, floor) pairs — pick the floor point (larger y).
    if len(pts) % 2 == 0:
        floor_pixel = np.array(
            [
                pts[i] if pts[i][1] > pts[i + 1][1] else pts[i + 1]
                for i in range(0, len(pts), 2)
            ]
        )
    else:
        floor_pixel = np.array(pts)

    if len(floor_pixel) < 3:
        return None

    floor_xy = _np_coor2xy(
        floor_pixel, z=z, coorW=pano_w, coorH=pano_h, floorW=pano_w, floorH=pano_w
    )
    center = pano_w / 2 - 0.5
    floor_xy[:, 0] -= center
    floor_xy[:, 1] -= center
    floor_xy[:, 1] = -floor_xy[:, 1]  # Y-axis flip (standard for this project)

    return rectify_polygon(floor_xy).astype(np.float64)
