"""SE(2) outline previews only; no placement evaluation or snapping."""
import numpy as np
from .geometry import apply_pose


def transform_preview_geometry(polygons, poses, preview_room):
    rooms = [{"room_id": room, "preview": room == preview_room,
              "world_pose": pose.to_dict(),
              "polygon": apply_pose(polygons[room], pose.radians()).tolist()}
             for room, pose in sorted(poses.items())]
    points = np.vstack([item["polygon"] for item in rooms])
    low, high = points.min(axis=0), points.max(axis=0)
    bounds = {"min_x": float(low[0]), "min_y": float(low[1]), "max_x": float(high[0]), "max_y": float(high[1])}
    return rooms, bounds
