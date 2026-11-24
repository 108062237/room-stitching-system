"""Room assembly calculations."""
from __future__ import annotations

import math
import numpy as np
Pose = tuple[float, float, float]

def wrap_pi(theta: float) -> float:
    return (float(theta) + math.pi) % (2.0 * math.pi) - math.pi


def compose(a: Pose, b: Pose) -> Pose:
    ax, ay, at = a
    bx, by, bt = b
    c, s = math.cos(at), math.sin(at)
    return ax + c * bx - s * by, ay + s * bx + c * by, wrap_pi(at + bt)


def invert(pose: Pose) -> Pose:
    x, y, theta = pose
    c, s = math.cos(theta), math.sin(theta)
    return -(c * x + s * y), -(-s * x + c * y), wrap_pi(-theta)


def apply_pose(points: np.ndarray, pose: Pose) -> np.ndarray:
    x, y, theta = pose
    c, s = math.cos(theta), math.sin(theta)
    rotation = np.asarray([[c, -s], [s, c]], dtype=float)
    return np.asarray(points, dtype=float) @ rotation.T + np.asarray([x, y])
