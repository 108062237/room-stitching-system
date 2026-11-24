"""Room assembly calculations."""
from __future__ import annotations

import math
from typing import Any
import numpy as np
from shapely.geometry import LineString
from . import candidates as core
from .geometry import Pose, apply_pose, compose, invert
from ..settings import DEFAULTS
LAYOUT_SUGGESTION_CONFIG = DEFAULTS['interactive']['layout_suggestion']
MIN_ROOM_CLEARANCE = DEFAULTS['interactive']['minimum_room_clearance']
MAX_ROOM_OVERLAP_RATIO = DEFAULTS['interactive']['maximum_room_overlap_ratio']
WALL_SNAP_CLEARANCE = DEFAULTS['interactive']['wall_snap_clearance']


def cross_2d(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def polygon_rings(geometry: Any) -> list[list[list[float]]]:
    """Convert polygonal Shapely output into JSON-friendly exterior rings."""
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [[[float(x), float(y)] for x, y in geometry.exterior.coords]]
    if hasattr(geometry, "geoms"):
        rings = []
        for item in geometry.geoms:
            rings.extend(polygon_rings(item))
        return rings
    return []


def collinear_wall_chain(
    points: np.ndarray, wall_index: int, angle_tolerance_deg: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge consecutive polygon segments that represent one straight wall."""
    count = len(points)
    base_a = points[wall_index]
    base_b = points[(wall_index + 1) % count]
    base = base_b - base_a
    base_length = float(np.linalg.norm(base))
    if count < 2 or base_length <= 1e-9:
        return base_a, base_b
    tangent = base / base_length
    sine_limit = math.sin(math.radians(angle_tolerance_deg))
    line_tolerance = max(0.5, 0.02 * base_length)

    def is_collinear(index: int) -> bool:
        start = points[index]
        end = points[(index + 1) % count]
        segment = end - start
        length = float(np.linalg.norm(segment))
        if length <= 1e-9:
            return False
        direction_error = abs(float(np.cross(tangent, segment / length)))
        start_distance = abs(float(np.cross(tangent, start - base_a)))
        end_distance = abs(float(np.cross(tangent, end - base_a)))
        return (
            direction_error <= sine_limit
            and start_distance <= line_tolerance
            and end_distance <= line_tolerance
        )

    first = last = wall_index
    visited = {wall_index}
    while len(visited) < count:
        previous = (first - 1) % count
        if previous in visited or not is_collinear(previous):
            break
        first = previous
        visited.add(previous)
    while len(visited) < count:
        following = (last + 1) % count
        if following in visited or not is_collinear(following):
            break
        last = following
        visited.add(following)
    return points[first], points[(last + 1) % count]


class PlacementEngine:
    """Private calculation context, with no file I/O, actions or session state."""
    def __init__(self, polygons, poses):
        self.polygons = polygons
        self.poses = poses
        self.placed = set(poses)
        self.conflict_overlap_threshold = DEFAULTS['interactive']['conflict_overlap_threshold']
        self.conflict_gap_threshold = DEFAULTS['interactive']['conflict_gap_threshold']
        self.endpoint_alignment_weight = DEFAULTS['interactive']['endpoint_alignment_weight']

    def _layout_options(self, edge: core.Edge, parent: str, child: str) -> list[dict[str, Any]]:
        original = edge.measurement if edge.room_a == parent else invert(edge.measurement)
        variants: list[dict[str, Any]] = []
        for index, candidate in enumerate(edge.candidates):
            measurement = candidate.measurement if edge.room_a == parent else invert(candidate.measurement)
            has_wall_metadata = bool(
                len(edge.candidates) > 1
                or candidate.overlap_ratio > 0.0
                or candidate.polygon_overlap_ratio > 0.0
                or abs(candidate.slide) > 1e-9
            )
            variants.append({
                "candidate_id": f"{candidate.candidate_id}@{index:03d}",
                "measurement": measurement,
                "intrinsic_cost": candidate.intrinsic_cost,
                "slide_cost": candidate.slide_cost,
                "parent_wall_index": (
                    candidate.wall_a_index if edge.room_a == parent else candidate.wall_b_index
                ) if has_wall_metadata else None,
                "child_wall_index": (
                    candidate.wall_b_index if edge.room_a == parent else candidate.wall_a_index
                ) if has_wall_metadata else None,
            })
        variants.append({
            "candidate_id": edge.selected_candidate_id or "current",
            "measurement": original,
            "intrinsic_cost": 0.0,
            "slide_cost": 0.0,
            "parent_wall_index": None,
            "child_wall_index": None,
        })
        parent_points = apply_pose(self.polygons[parent], self.poses[parent])
        parent_shape = core.valid_polygon(parent_points)
        placed_shapes = {
            room: core.valid_polygon(apply_pose(self.polygons[room], self.poses[room]))
            for room in self.placed
        }
        ranked: list[dict[str, Any]] = []
        seen = set()
        for variant in variants:
            candidate_id = variant["candidate_id"]
            measurement = variant["measurement"]
            key = tuple(round(value, 6) for value in measurement)
            if key in seen:
                continue
            seen.add(key)
            child_pose = compose(self.poses[parent], measurement)
            snapped_pose, snap_info = self._snap_world_pose(child, child_pose)
            if snap_info and snap_info.get("target_room") == parent:
                child_pose = snapped_pose
                measurement = compose(invert(self.poses[parent]), child_pose)
            child_points = apply_pose(self.polygons[child], child_pose)
            child_shape = core.valid_polygon(child_points)
            placement_metrics = self._placement_metrics(
                child, child_pose, allowed_contact_room=parent,
            )
            overlaps = []
            conflicting_rooms = []
            for room, shape in placed_shapes.items():
                overlap = child_shape.intersection(shape).area / max(
                    min(child_shape.area, shape.area), 1e-9,
                )
                overlaps.append(float(overlap))
                if overlap >= self.conflict_overlap_threshold:
                    conflicting_rooms.append(room[-6:])
            overlap_sum = sum(overlaps)
            overlap_max = max(overlaps, default=0.0)
            parent_gap = float(child_shape.boundary.distance(parent_shape.boundary))
            wall_gap = wall_overlap = endpoint_gap = None
            parent_wall_index = variant["parent_wall_index"]
            child_wall_index = variant["child_wall_index"]
            if parent_wall_index is not None and child_wall_index is not None:
                parent_a, parent_b = collinear_wall_chain(parent_points, parent_wall_index)
                child_a, child_b = collinear_wall_chain(child_points, child_wall_index)
                parent_length = float(np.linalg.norm(parent_b - parent_a))
                child_length = float(np.linalg.norm(child_b - child_a))
                if parent_length > 1e-9 and child_length > 1e-9:
                    wall_gap = float(LineString([parent_a, parent_b]).distance(
                        LineString([child_a, child_b])
                    ))
                    tangent = (parent_b - parent_a) / parent_length
                    overlap_length = core.interval_overlap(
                        np.dot(parent_a, tangent), np.dot(parent_b, tangent),
                        np.dot(child_a, tangent), np.dot(child_b, tangent),
                    )
                    wall_overlap = float(overlap_length / min(parent_length, child_length))
                    endpoint_gap = float(min(
                        np.linalg.norm(parent_a - child_a),
                        np.linalg.norm(parent_a - child_b),
                        np.linalg.norm(parent_b - child_a),
                        np.linalg.norm(parent_b - child_b),
                    ))
                    wall_scale = max(min(parent_length, child_length), 1.0)
                else:
                    wall_scale = 1.0
            else:
                wall_scale = 1.0
            delta = compose(invert(original), measurement)
            prior_cost = 0.0 if edge.manual_action == "layout_suggestion_pending" else (
                math.hypot(delta[0], delta[1]) / 50.0
                + abs(delta[2]) / math.radians(45.0)
            )
            endpoint_cost = (
                0.0 if endpoint_gap is None
                else self.endpoint_alignment_weight * min(endpoint_gap / wall_scale, 2.0)
            )
            wall_contact_cost = 0.0 if wall_gap is None else 4.0 * min(wall_gap / wall_scale, 2.0)
            wall_overlap_cost = 0.0 if wall_overlap is None else 5.0 * max(0.0, 1.0 - wall_overlap)
            score = (
                20.0 * overlap_sum + 20.0 * overlap_max
                + 2.0 * min(parent_gap / 10.0, 5.0)
                + prior_cost + endpoint_cost + wall_contact_cost + wall_overlap_cost
                + 0.3 * variant["intrinsic_cost"] + 0.2 * variant["slide_cost"]
            )
            record = {
                "measurement": measurement,
                "candidate_id": candidate_id,
                "candidate_count": len(seen),
                "layout_adjusted": any(abs(a - b) > 1e-6 for a, b in zip(measurement, original)),
                "layout_score": float(score),
                "overlap_sum": float(overlap_sum),
                "overlap_max": float(overlap_max),
                "parent_gap": parent_gap,
                "wall_gap": wall_gap,
                "wall_overlap": wall_overlap,
                "endpoint_gap": endpoint_gap,
                "conflicting_rooms": conflicting_rooms,
                "placement_status": placement_metrics["status"],
                "overlaps": placement_metrics["overlaps"],
                "boundary_gap": placement_metrics["boundary_gap"],
                "nearest_room_short": placement_metrics["nearest_room_short"],
                "clearance_required": placement_metrics["clearance_required"],
                "clearance_conflict": placement_metrics["clearance_conflict"],
                "penetration_conflict": placement_metrics["penetration_conflict"],
                "parent_gap_conflict": parent_gap > self.conflict_gap_threshold,
                "hard_conflict": (
                    placement_metrics["hard_conflict"]
                    or parent_gap > self.conflict_gap_threshold
                ),
            }
            ranked.append(record)
        if not ranked:
            raise ValueError(f"no layout option for {parent[-6:]} -> {child[-6:]}")
        ranked.sort(key=lambda item: (item["layout_score"], item["candidate_id"]))
        for record in ranked:
            record["candidate_count"] = len(seen)
        return ranked


    def _placement_metrics(
        self, room: str, pose: Pose, allowed_contact_room: str | None = None,
    ) -> dict[str, Any]:
        points = apply_pose(self.polygons[room], pose)
        shape = core.valid_polygon(points)
        overlaps = []
        intersections = []
        nearest_boundary_gap = math.inf
        nearest_corner_gap = math.inf
        nearest_room = None
        for other in sorted(self.placed):
            if other == room:
                continue
            other_points = apply_pose(self.polygons[other], self.poses[other])
            other_shape = core.valid_polygon(other_points)
            boundary_gap = float(shape.boundary.distance(other_shape.boundary))
            if other != allowed_contact_room and boundary_gap < nearest_boundary_gap:
                nearest_boundary_gap = boundary_gap
                nearest_room = other
            corner_gap = min(
                float(np.linalg.norm(a - b))
                for a in points for b in other_points
            )
            nearest_corner_gap = min(nearest_corner_gap, corner_gap)
            intersection = shape.intersection(other_shape)
            ratio = float(intersection.area / max(min(shape.area, other_shape.area), 1e-9))
            if ratio > 1e-8:
                overlaps.append({
                    "room": other,
                    "room_short": other[-6:],
                    "ratio": ratio,
                })
                intersections.extend(polygon_rings(intersection))
        overlaps.sort(key=lambda item: -item["ratio"])
        overlap_max = max((item["ratio"] for item in overlaps), default=0.0)
        boundary_gap = None if math.isinf(nearest_boundary_gap) else nearest_boundary_gap
        corner_gap = None if math.isinf(nearest_corner_gap) else nearest_corner_gap
        clearance_conflict = (
            boundary_gap is not None
            and boundary_gap < MIN_ROOM_CLEARANCE - 1e-3
        )
        penetration_conflict = overlap_max > MAX_ROOM_OVERLAP_RATIO
        if penetration_conflict:
            status = "conflict"
        elif clearance_conflict:
            status = "clearance_conflict"
        elif boundary_gap is not None and boundary_gap <= self.conflict_gap_threshold:
            status = "near"
        else:
            status = "free"
        return {
            "status": status,
            "overlap_max": overlap_max,
            "overlaps": overlaps,
            "overlap_tolerance": MAX_ROOM_OVERLAP_RATIO,
            "intersections": intersections,
            "boundary_gap": boundary_gap,
            "corner_gap": corner_gap,
            "nearest_room": nearest_room,
            "nearest_room_short": nearest_room[-6:] if nearest_room else None,
            "clearance_required": MIN_ROOM_CLEARANCE,
            "clearance_conflict": clearance_conflict,
            "penetration_conflict": penetration_conflict,
            "hard_conflict": penetration_conflict or clearance_conflict,
        }


    def _snap_world_pose(
        self, room: str, pose: Pose, allowed_contact_room: str | None = None,
    ) -> tuple[Pose, dict[str, Any] | None]:
        """Snap a moving room to nearby placed-room walls or corners."""
        local_points = self.polygons[room]
        base_points = apply_pose(local_points, pose)
        candidates: list[tuple[float, Pose, dict[str, Any]]] = []
        wall_distance_limit = 10.0
        endpoint_distance_limit = 12.0
        angle_limit = math.radians(8.0)

        def consider(candidate_pose: Pose, metadata: dict[str, Any], geometric_cost: float) -> None:
            metrics = self._placement_metrics(
                room, candidate_pose, allowed_contact_room=allowed_contact_room,
            )
            if metrics["hard_conflict"]:
                return
            overlap_penalty = 30.0 * metrics["overlap_max"]
            candidates.append((geometric_cost + overlap_penalty, candidate_pose, metadata))

        for target_room in sorted(self.placed):
            if target_room == room:
                continue
            target_points = apply_pose(self.polygons[target_room], self.poses[target_room])

            # Corner-to-corner snap preserves the current rotation.
            for moving_index, moving_corner in enumerate(base_points):
                for target_index, target_corner in enumerate(target_points):
                    delta = target_corner - moving_corner
                    distance = float(np.linalg.norm(delta))
                    if distance > endpoint_distance_limit:
                        continue
                    snapped = (pose[0] + float(delta[0]), pose[1] + float(delta[1]), pose[2])
                    consider(snapped, {
                        "type": "corner",
                        "target_room": target_room,
                        "target_room_short": target_room[-6:],
                        "moving_corner_index": moving_index,
                        "target_corner_index": target_index,
                        "point": [float(target_corner[0]), float(target_corner[1])],
                    }, distance / endpoint_distance_limit + 0.08)

            # Wall snap may apply a small rotation, then moves only as much as
            # needed to align the supporting lines. A nearby endpoint is also
            # aligned along the wall, avoiding forced center alignment.
            for moving_index in range(len(local_points)):
                moving_a = base_points[moving_index]
                moving_b = base_points[(moving_index + 1) % len(base_points)]
                moving_vector = moving_b - moving_a
                moving_length = float(np.linalg.norm(moving_vector))
                if moving_length <= 1e-9:
                    continue
                moving_tangent = moving_vector / moving_length
                for target_index in range(len(target_points)):
                    target_a = target_points[target_index]
                    target_b = target_points[(target_index + 1) % len(target_points)]
                    target_vector = target_b - target_a
                    target_length = float(np.linalg.norm(target_vector))
                    if target_length <= 1e-9:
                        continue
                    target_tangent = target_vector / target_length
                    rotation_delta = math.atan2(
                        cross_2d(moving_tangent, target_tangent),
                        float(np.dot(moving_tangent, target_tangent)),
                    )
                    if rotation_delta > math.pi / 2:
                        rotation_delta -= math.pi
                    elif rotation_delta < -math.pi / 2:
                        rotation_delta += math.pi
                    if abs(rotation_delta) > angle_limit:
                        continue

                    rotated_pose = (pose[0], pose[1], core.wrap_pi(pose[2] + rotation_delta))
                    rotated_points = apply_pose(local_points, rotated_pose)
                    rotated_a = rotated_points[moving_index]
                    rotated_b = rotated_points[(moving_index + 1) % len(rotated_points)]
                    normal = np.asarray([-target_tangent[1], target_tangent[0]])
                    normal_amount = float(np.dot(target_a - rotated_a, normal))
                    if abs(normal_amount) > wall_distance_limit:
                        continue
                    target_center = target_points.mean(axis=0)
                    target_side = float(np.dot(target_center - target_a, normal))
                    target_side_sign = 1.0 if target_side >= 0.0 else -1.0
                    normal_shift = (
                        normal_amount - target_side_sign * WALL_SNAP_CLEARANCE
                    ) * normal
                    shifted_a = rotated_a + normal_shift
                    shifted_b = rotated_b + normal_shift
                    moving_interval = sorted((
                        float(np.dot(shifted_a, target_tangent)),
                        float(np.dot(shifted_b, target_tangent)),
                    ))
                    target_interval = sorted((
                        float(np.dot(target_a, target_tangent)),
                        float(np.dot(target_b, target_tangent)),
                    ))
                    overlap_length = max(
                        0.0,
                        min(moving_interval[1], target_interval[1])
                        - max(moving_interval[0], target_interval[0]),
                    )
                    endpoint_pairs = [
                        (shifted_a, target_a, moving_index, target_index),
                        (shifted_a, target_b, moving_index, (target_index + 1) % len(target_points)),
                        (shifted_b, target_a, (moving_index + 1) % len(rotated_points), target_index),
                        (shifted_b, target_b, (moving_index + 1) % len(rotated_points), (target_index + 1) % len(target_points)),
                    ]
                    moving_corner, target_corner, moving_corner_index, target_corner_index = min(
                        endpoint_pairs, key=lambda item: float(np.linalg.norm(item[0] - item[1]))
                    )
                    endpoint_distance = float(np.linalg.norm(moving_corner - target_corner))
                    minimum_overlap = min(5.0, 0.15 * min(moving_length, target_length))
                    if overlap_length < minimum_overlap and endpoint_distance > endpoint_distance_limit:
                        continue
                    tangent_amount = 0.0
                    snap_type = "wall"
                    if endpoint_distance <= endpoint_distance_limit:
                        tangent_amount = float(np.dot(target_corner - moving_corner, target_tangent))
                        snap_type = "wall_corner"
                    shift = normal_shift + tangent_amount * target_tangent
                    snapped = (
                        rotated_pose[0] + float(shift[0]),
                        rotated_pose[1] + float(shift[1]),
                        rotated_pose[2],
                    )
                    move_cost = float(np.linalg.norm(shift)) / wall_distance_limit
                    angle_cost = abs(rotation_delta) / angle_limit
                    consider(snapped, {
                        "type": snap_type,
                        "target_room": target_room,
                        "target_room_short": target_room[-6:],
                        "moving_wall_index": moving_index,
                        "target_wall_index": target_index,
                        "moving_corner_index": moving_corner_index if snap_type == "wall_corner" else None,
                        "target_corner_index": target_corner_index if snap_type == "wall_corner" else None,
                        "target_wall": [
                            [float(target_a[0]), float(target_a[1])],
                            [float(target_b[0]), float(target_b[1])],
                        ],
                        "clearance": WALL_SNAP_CLEARANCE,
                    }, move_cost + 0.35 * angle_cost)

        if not candidates:
            return pose, None
        _, snapped_pose, metadata = min(candidates, key=lambda item: item[0])
        return snapped_pose, metadata
