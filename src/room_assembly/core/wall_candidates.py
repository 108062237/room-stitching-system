"""Room assembly calculations."""
from __future__ import annotations

import argparse
import math
from typing import Any
import numpy as np
from shapely.geometry import LineString, Polygon

def normalize_angle(theta: float) -> float:
    return (float(theta) + math.pi) % (2.0 * math.pi) - math.pi


def normalize_deg(theta: float) -> float:
    return (float(theta) + 180.0) % 360.0 - 180.0


def rotation2(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.asarray([[c, -s], [s, c]], dtype=float)


def transform_points(points: np.ndarray, dx: float, dy: float, theta: float) -> np.ndarray:
    return np.asarray(points, dtype=float) @ rotation2(theta).T + np.asarray([dx, dy])


def angular_error_deg(a: float, b: float) -> float:
    return abs(normalize_deg(math.degrees(float(a) - float(b))))


def dominant_manhattan_angle(polygon: np.ndarray) -> float:
    vectors = np.roll(polygon, -1, axis=0) - polygon
    lengths = np.linalg.norm(vectors, axis=1)
    valid = lengths > 1e-9
    if not valid.any():
        raise ValueError("polygon has no nonzero walls")
    angles = np.arctan2(vectors[valid, 1], vectors[valid, 0])
    sx = float(np.sum(lengths[valid] * np.cos(4.0 * angles)))
    sy = float(np.sum(lengths[valid] * np.sin(4.0 * angles)))
    return math.atan2(sy, sx) / 4.0


def rotation_candidates(polygon_a: np.ndarray, polygon_b: np.ndarray) -> list[tuple[int, float]]:
    base = dominant_manhattan_angle(polygon_a) - dominant_manhattan_angle(polygon_b)
    return [(quadrant, normalize_angle(base + math.radians(quadrant))) for quadrant in (0, 90, 180, 270)]


def polygon_walls(polygon: np.ndarray) -> list[dict[str, Any]]:
    walls: list[dict[str, Any]] = []
    for index, p0 in enumerate(polygon):
        p1 = polygon[(index + 1) % len(polygon)]
        vector = np.asarray(p1 - p0, dtype=float)
        length = float(np.linalg.norm(vector))
        if length <= 1e-9:
            continue
        walls.append({
            "id": f"wall_{index}", "index": index,
            "p0": np.asarray(p0, dtype=float), "p1": np.asarray(p1, dtype=float),
            "length": length, "tangent": vector / length,
            "angle": math.atan2(float(vector[1]), float(vector[0])),
        })
    return walls


def axis_class(angle: float, manhattan_angle: float) -> str:
    relative = normalize_angle(angle - manhattan_angle)
    return "horizontal" if abs(math.cos(relative)) >= abs(math.sin(relative)) else "vertical"


def undirected_angle_error_deg(angle_a: float, angle_b: float) -> float:
    delta = abs(normalize_angle(angle_a - angle_b))
    delta = min(delta, abs(math.pi - delta))
    return math.degrees(delta)


def interval_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    alo, ahi = sorted((float(a0), float(a1)))
    blo, bhi = sorted((float(b0), float(b1)))
    return max(0.0, min(ahi, bhi) - max(alo, blo))


def valid_polygon(points: np.ndarray) -> Polygon:
    polygon = Polygon(np.asarray(points, dtype=float)).buffer(0)
    if polygon.is_empty or polygon.area <= 1e-9:
        raise ValueError("invalid or empty room polygon")
    if polygon.geom_type != "Polygon":
        polygon = max(polygon.geoms, key=lambda item: item.area)
    return polygon


def side_sign(value: float, epsilon: float = 1e-7) -> int:
    if value > epsilon:
        return 1
    if value < -epsilon:
        return -1
    return 0


def snap_candidate(
    polygon_a: np.ndarray,
    polygon_b_rotated: np.ndarray,
    polygon_a_shape: Polygon,
    wall_a: dict[str, Any],
    wall_b: dict[str, Any],
    theta: float,
    quadrant: int,
    manhattan_a: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    pa0, pa1 = wall_a["p0"], wall_a["p1"]
    qb0 = polygon_b_rotated[wall_b["index"]]
    qb1 = polygon_b_rotated[(wall_b["index"] + 1) % len(polygon_b_rotated)]
    tangent = wall_a["tangent"]
    normal = np.asarray([-tangent[1], tangent[0]], dtype=float)

    pre_distance = float(LineString([pa0, pa1]).distance(LineString([qb0, qb1])))
    normal_translation = float(np.dot((pa0 + pa1 - qb0 - qb1) * 0.5, normal))
    tangential_translation = float(np.dot((pa0 + pa1 - qb0 - qb1) * 0.5, tangent))
    translation = normal_translation * normal + tangential_translation * tangent
    transformed_b = polygon_b_rotated + translation
    tb0 = transformed_b[wall_b["index"]]
    tb1 = transformed_b[(wall_b["index"] + 1) % len(transformed_b)]

    post_distance = float(LineString([pa0, pa1]).distance(LineString([tb0, tb1])))
    overlap_length = interval_overlap(
        np.dot(pa0, tangent), np.dot(pa1, tangent),
        np.dot(tb0, tangent), np.dot(tb1, tangent),
    )
    overlap_ratio = overlap_length / min(float(wall_a["length"]), float(wall_b["length"]))

    polygon_b_shape = valid_polygon(transformed_b)
    overlap_area = float(polygon_a_shape.intersection(polygon_b_shape).area)
    polygon_overlap_ratio = overlap_area / min(float(polygon_a_shape.area), float(polygon_b_shape.area))

    midpoint = (pa0 + pa1) * 0.5
    rep_a = polygon_a_shape.representative_point()
    rep_b = polygon_b_shape.representative_point()
    signed_a = float(np.dot(np.asarray([rep_a.x, rep_a.y]) - midpoint, normal))
    signed_b = float(np.dot(np.asarray([rep_b.x, rep_b.y]) - midpoint, normal))
    sign_a, sign_b = side_sign(signed_a), side_sign(signed_b)
    opposite_side = sign_a * sign_b < 0

    wall_b_angle = math.atan2(float(qb1[1] - qb0[1]), float(qb1[0] - qb0[0]))
    orientation_error = undirected_angle_error_deg(float(wall_a["angle"]), wall_b_angle)
    orientation_a = axis_class(float(wall_a["angle"]), manhattan_a)
    orientation_b = axis_class(wall_b_angle, manhattan_a)
    pass_orientation = orientation_a == orientation_b and orientation_error < args.wall_angle_threshold
    pass_distance = pre_distance < args.wall_distance_threshold
    pass_overlap = overlap_length > args.wall_min_overlap and overlap_ratio > args.wall_min_overlap_ratio
    pass_polygon = polygon_overlap_ratio < args.polygon_overlap_threshold

    distance_norm = pre_distance / max(args.wall_distance_threshold, 1e-9)
    angle_norm = orientation_error / max(args.wall_angle_threshold, 1e-9)
    score = (
        args.w_distance * distance_norm
        + args.w_angle * angle_norm
        - args.w_overlap * overlap_ratio
        + args.w_polygon_overlap * polygon_overlap_ratio
    )
    return {
        "room_a": args._room_a, "room_b": args._room_b,
        "rotation_quadrant_deg": quadrant,
        "rotation_candidate_deg": normalize_deg(math.degrees(theta)),
        "wall_a_id": wall_a["id"], "wall_b_id": wall_b["id"],
        "wall_a_length": float(wall_a["length"]), "wall_b_length": float(wall_b["length"]),
        "wall_a_orientation": orientation_a, "wall_b_orientation": orientation_b,
        "orientation_error_deg": orientation_error,
        "pre_snap_wall_distance": pre_distance, "post_snap_wall_distance": post_distance,
        "normal_translation": normal_translation,
        "tangential_translation": tangential_translation,
        "shared_wall_overlap_length": overlap_length,
        "shared_wall_overlap_ratio": overlap_ratio,
        "polygon_overlap_area": overlap_area,
        "polygon_overlap_ratio": polygon_overlap_ratio,
        "room_a_side_sign": sign_a, "room_b_side_sign": sign_b,
        "opposite_side_valid": opposite_side,
        "dx": float(translation[0]), "dy": float(translation[1]),
        "dtheta": normalize_angle(theta),
        "translation_magnitude": float(np.linalg.norm(translation)),
        "pass_orientation": pass_orientation, "pass_distance": pass_distance,
        "pass_overlap": pass_overlap, "pass_polygon_overlap": pass_polygon,
        "pass_opposite_side": opposite_side,
        "candidate_valid": bool(pass_orientation and pass_distance and pass_overlap and pass_polygon and opposite_side),
        "candidate_score": float(score),
        "_transformed_b": transformed_b, "_wall_a": np.vstack([pa0, pa1]),
        "_wall_b": np.vstack([tb0, tb1]),
    }


def enumerate_candidates(
    polygon_a: np.ndarray,
    polygon_b: np.ndarray,
    room_a: str,
    room_b: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    polygon_a_shape = valid_polygon(polygon_a)
    walls_a = polygon_walls(polygon_a)
    walls_b = polygon_walls(polygon_b)
    manhattan_a = dominant_manhattan_angle(polygon_a)
    args._room_a, args._room_b = room_a, room_b
    candidates: list[dict[str, Any]] = []
    for quadrant, theta in rotation_candidates(polygon_a, polygon_b):
        rotated_b = polygon_b @ rotation2(theta).T
        for wall_a in walls_a:
            for wall_b in walls_b:
                candidates.append(snap_candidate(
                    polygon_a, rotated_b, polygon_a_shape, wall_a, wall_b,
                    theta, quadrant, manhattan_a, args,
                ))
    return candidates
