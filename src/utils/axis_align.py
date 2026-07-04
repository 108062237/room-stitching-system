"""Helpers for globally axis-aligning stitched floorplan visualizations."""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Any

import numpy as np


def dominant_manhattan_angle(polys: Iterable[np.ndarray]) -> float:
    """Return dominant wall angle modulo 90 degrees, weighted by segment length."""
    sx = 0.0
    sy = 0.0

    for poly in polys:
        if poly is None or len(poly) < 2:
            continue
        closed = np.vstack([poly, poly[0:1]])
        segs = closed[1:] - closed[:-1]
        lengths = np.linalg.norm(segs, axis=1)
        angles = np.arctan2(segs[:, 1], segs[:, 0])
        valid = lengths > 1e-9
        if not np.any(valid):
            continue
        sx += float(np.sum(lengths[valid] * np.cos(4.0 * angles[valid])))
        sy += float(np.sum(lengths[valid] * np.sin(4.0 * angles[valid])))

    if abs(sx) < 1e-12 and abs(sy) < 1e-12:
        return 0.0
    return math.atan2(sy, sx) / 4.0


def rotation_matrix(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def rotate_points(
    pts: np.ndarray,
    theta: float,
    origin: np.ndarray | None = None,
) -> np.ndarray:
    """Rotate Nx2 row-vector points by theta around origin."""
    if pts.size == 0:
        return pts
    if origin is None:
        origin = np.zeros(2, dtype=np.float64)
    r = rotation_matrix(theta)
    return (pts - origin) @ r.T + origin


def align_polys_to_xy(
    polys: list[np.ndarray],
    *,
    angle: float | None = None,
    origin: np.ndarray | None = None,
) -> tuple[list[np.ndarray], float, np.ndarray]:
    """Rotate all polygons so their dominant Manhattan direction aligns to XY axes."""
    if not polys:
        origin_out = np.zeros(2, dtype=np.float64) if origin is None else origin
        return polys, 0.0, origin_out

    if origin is None:
        origin = np.mean(np.vstack(polys), axis=0)
    if angle is None:
        angle = -dominant_manhattan_angle(polys)
    return [rotate_points(poly, angle, origin) for poly in polys], angle, origin


def rotate_pose_dict(
    poses: Mapping[str, Mapping[str, Any]],
    theta: float,
    origin: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Rotate pose coordinates and headings for visualization only."""
    out: dict[str, dict[str, float]] = {}
    for pid, pose in poses.items():
        xy = np.array([[float(pose["x"]), float(pose["y"])]], dtype=np.float64)
        xy_rot = rotate_points(xy, theta, origin)[0]
        out[pid] = {
            "x": float(xy_rot[0]),
            "y": float(xy_rot[1]),
            "theta": float(pose.get("theta", 0.0)) + theta,
        }
        for key, value in pose.items():
            if key not in out[pid]:
                out[pid][key] = value
    return out
