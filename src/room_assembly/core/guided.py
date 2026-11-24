"""Room assembly calculations."""
from __future__ import annotations

import argparse
import math
from typing import Any
import numpy as np
from shapely.geometry import LineString
from .wall_candidates import (angular_error_deg, axis_class, dominant_manhattan_angle,
    interval_overlap, normalize_deg, polygon_walls, rotation2, rotation_candidates,
    side_sign, undirected_angle_error_deg, valid_polygon)


def unit2(vector: np.ndarray) -> np.ndarray | None:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    return None if vector.shape != (2,) or norm <= 1e-9 else vector / norm


def vector_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    ua, ub = unit2(a), unit2(b)
    if ua is None or ub is None:
        return float("nan")
    return math.degrees(math.acos(float(np.clip(np.dot(ua, ub), -1.0, 1.0))))


def room_bearing_deg(vector: np.ndarray) -> float:
    """Use the verified Week-1 room-bearing convention: atan2(x, y)."""
    return (math.degrees(math.atan2(float(vector[0]), float(vector[1]))) + 180.0) % 360.0 - 180.0


def cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def outward_wall_normal(wall: dict[str, Any], interior_point: np.ndarray) -> np.ndarray:
    """Return the wall normal pointing away from the polygon interior."""
    tangent = np.asarray(wall["tangent"], dtype=float)
    normal = np.asarray([-tangent[1], tangent[0]], dtype=float)
    midpoint = (np.asarray(wall["p0"]) + np.asarray(wall["p1"])) * 0.5
    if float(np.dot(normal, interior_point - midpoint)) > 0.0:
        normal = -normal
    return normal


def ray_wall_hit_distance(
    wall: dict[str, Any], direction: np.ndarray, origin: np.ndarray | None = None,
) -> float | None:
    """Return forward ray distance to a wall segment, or None when it misses."""
    ray_origin = np.zeros(2, dtype=float) if origin is None else np.asarray(origin, dtype=float)
    ray_direction = unit2(np.asarray(direction, dtype=float))
    if ray_direction is None:
        return None
    p0 = np.asarray(wall["p0"], dtype=float)
    segment = np.asarray(wall["p1"], dtype=float) - p0
    denominator = cross2(ray_direction, segment)
    if abs(denominator) <= 1e-9:
        return None
    offset = p0 - ray_origin
    distance = cross2(offset, segment) / denominator
    segment_fraction = cross2(offset, ray_direction) / denominator
    if distance < -1e-7 or segment_fraction < -1e-7 or segment_fraction > 1.0 + 1e-7:
        return None
    return max(0.0, distance)


def preselect_walls_by_bearing(
    polygon: np.ndarray,
    walls: list[dict[str, Any]],
    target_direction: np.ndarray,
    top_k: int,
    max_angle_deg: float,
) -> list[dict[str, Any]]:
    """Keep walls most plausibly facing the other panorama.

    A direct bearing-ray hit is preferred.  Outward-normal alignment then ranks
    walls that are not directly hit, which preserves alternatives near corners.
    """
    if top_k <= 0:
        return []
    target = unit2(np.asarray(target_direction, dtype=float))
    if target is None:
        return []
    shape = valid_polygon(polygon)
    representative = shape.representative_point()
    interior = np.asarray([representative.x, representative.y], dtype=float)
    ranked: list[dict[str, Any]] = []
    for full_position, source in enumerate(walls):
        wall = dict(source)
        normal = outward_wall_normal(wall, interior)
        alignment = float(np.clip(np.dot(normal, target), -1.0, 1.0))
        angle = math.degrees(math.acos(alignment))
        hit_distance = ray_wall_hit_distance(wall, target)
        wall.update({
            "_full_position": full_position,
            "_bearing_alignment": alignment,
            "_bearing_angle_deg": angle,
            "_ray_hit_distance": hit_distance,
        })
        if angle <= max_angle_deg:
            ranked.append(wall)
    ranked.sort(key=lambda wall: (
        wall["_ray_hit_distance"] is None,
        float("inf") if wall["_ray_hit_distance"] is None else wall["_ray_hit_distance"],
        -wall["_bearing_alignment"],
        -wall["length"],
        wall["index"],
    ))
    limit = min(top_k, len(ranked))
    selected = ranked[:limit]
    if selected and limit < len(ranked):
        boundary = selected[-1]
        for wall in ranked[limit:]:
            same_hit_type = (wall["_ray_hit_distance"] is None) == (
                boundary["_ray_hit_distance"] is None
            )
            same_alignment = abs(
                float(wall["_bearing_alignment"])
                - float(boundary["_bearing_alignment"])
            ) <= 1e-6
            if not (same_hit_type and same_alignment):
                break
            selected.append(wall)
    for rank, wall in enumerate(selected, 1):
        wall["_preselection_rank"] = rank
    return selected


def placement_metrics(
    polygon_a_shape: Any,
    polygon_b_rotated: np.ndarray,
    wall_a: dict[str, Any],
    wall_b: dict[str, Any],
    translation: np.ndarray,
) -> dict[str, Any]:
    transformed_b = polygon_b_rotated + translation
    shape_b = valid_polygon(transformed_b)
    pa0, pa1 = wall_a["p0"], wall_a["p1"]
    tb0 = transformed_b[wall_b["index"]]
    tb1 = transformed_b[(wall_b["index"] + 1) % len(transformed_b)]
    tangent = wall_a["tangent"]
    post_distance = float(LineString([pa0, pa1]).distance(LineString([tb0, tb1])))
    overlap_length = interval_overlap(
        np.dot(pa0, tangent), np.dot(pa1, tangent),
        np.dot(tb0, tangent), np.dot(tb1, tangent),
    )
    overlap_ratio = overlap_length / min(float(wall_a["length"]), float(wall_b["length"]))
    overlap_area = float(polygon_a_shape.intersection(shape_b).area)
    overlap_ratio_polygon = overlap_area / min(float(polygon_a_shape.area), float(shape_b.area))
    normal = np.asarray([-tangent[1], tangent[0]], dtype=float)
    midpoint = (pa0 + pa1) * 0.5
    rep_a, rep_b = polygon_a_shape.representative_point(), shape_b.representative_point()
    sign_a = side_sign(float(np.dot(np.asarray([rep_a.x, rep_a.y]) - midpoint, normal)))
    sign_b = side_sign(float(np.dot(np.asarray([rep_b.x, rep_b.y]) - midpoint, normal)))
    return {
        "transformed_b": transformed_b,
        "wall_a": np.vstack([pa0, pa1]), "wall_b": np.vstack([tb0, tb1]),
        "post_distance": post_distance, "overlap_length": overlap_length,
        "overlap_ratio": overlap_ratio, "polygon_overlap_ratio": overlap_ratio_polygon,
        "opposite_side": sign_a * sign_b < 0,
    }


def closest_manhattan_rotation(
    polygon_a: np.ndarray, polygon_b: np.ndarray, geoverify_yaw: float,
) -> tuple[int, float, float]:
    candidates = rotation_candidates(polygon_a, polygon_b)
    quadrant, theta = min(candidates, key=lambda item: angular_error_deg(item[1], geoverify_yaw))
    return quadrant, theta, angular_error_deg(theta, geoverify_yaw)


def make_candidate(
    polygon_a: np.ndarray,
    polygon_b_rotated: np.ndarray,
    polygon_a_shape: Any,
    wall_a: dict[str, Any],
    wall_b: dict[str, Any],
    theta: float,
    geo: dict[str, Any],
    room_a: str,
    room_b: str,
    index: int,
    yaw_residual: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    pa_mid = (wall_a["p0"] + wall_a["p1"]) * 0.5
    qb0 = polygon_b_rotated[wall_b["index"]]
    qb1 = polygon_b_rotated[(wall_b["index"] + 1) % len(polygon_b_rotated)]
    qb_mid = (qb0 + qb1) * 0.5
    tangent = wall_a["tangent"]
    normal = np.asarray([-tangent[1], tangent[0]], dtype=float)
    normal_displacement = float(np.dot(pa_mid - qb_mid, normal))
    center_tangent = float(np.dot(pa_mid - qb_mid, tangent))
    center_translation = normal_displacement * normal + center_tangent * tangent

    bearing = np.asarray(geo["bearing"], dtype=float)
    incidence = float(np.dot(normal, bearing))
    stable_incidence = abs(incidence) >= args.min_normal_bearing_cos
    lambda_scale = normal_displacement / incidence if abs(incidence) > 1e-12 else float("nan")
    positive_scale = math.isfinite(lambda_scale) and lambda_scale > 0.0
    translation = lambda_scale * bearing if math.isfinite(lambda_scale) else np.asarray([float("nan"), float("nan")])

    wall_b_angle = math.atan2(float(qb1[1] - qb0[1]), float(qb1[0] - qb0[0]))
    manhattan_a = dominant_manhattan_angle(polygon_a)
    orientation_error = undirected_angle_error_deg(float(wall_a["angle"]), wall_b_angle)
    orientation_ok = (
        axis_class(float(wall_a["angle"]), manhattan_a) == axis_class(wall_b_angle, manhattan_a)
        and orientation_error < args.wall_angle_threshold
    )

    guided: dict[str, Any] | None = None
    if np.isfinite(translation).all():
        guided = placement_metrics(polygon_a_shape, polygon_b_rotated, wall_a, wall_b, translation)
    center = placement_metrics(polygon_a_shape, polygon_b_rotated, wall_a, wall_b, center_translation)

    contact_ok = guided is not None and guided["post_distance"] < args.wall_contact_threshold
    overlap_ok = guided is not None and guided["overlap_length"] > args.wall_min_overlap
    polygon_ok = guided is not None and guided["polygon_overlap_ratio"] < args.polygon_overlap_threshold
    side_ok = guided is not None and bool(guided["opposite_side"])
    candidate_valid = bool(
        orientation_ok and stable_incidence and positive_scale and contact_ok
        and overlap_ok and polygon_ok and side_ok
    )
    guided_overlap = 0.0 if guided is None else float(guided["overlap_ratio"])
    guided_polygon_overlap = 1.0 if guided is None else float(guided["polygon_overlap_ratio"])
    score = (
        0.5 * orientation_error / max(args.wall_angle_threshold, 1e-9)
        + args.w_incidence * (1.0 - min(abs(incidence), 1.0))
        + args.w_preselection * (
            2.0
            - float(wall_a.get("_bearing_alignment", 0.0))
            - float(wall_b.get("_bearing_alignment", 0.0))
        ) / 2.0
        - args.w_overlap * guided_overlap
        + args.w_polygon_overlap * guided_polygon_overlap
    )

    center_bearing = unit2(center_translation)
    center_bearing_diff = 180.0 if center_bearing is None else vector_angle_deg(center_bearing, bearing)
    center_valid = bool(
        orientation_ok
        and center["post_distance"] < args.wall_contact_threshold
        and center["overlap_length"] > args.wall_min_overlap
        and center["polygon_overlap_ratio"] < args.polygon_overlap_threshold
        and center["opposite_side"]
    )
    center_score = (
        center_bearing_diff / max(args.bearing_tau, 1e-9)
        + 0.5 * orientation_error / max(args.wall_angle_threshold, 1e-9)
        - args.w_overlap * float(center["overlap_ratio"])
        + args.w_polygon_overlap * float(center["polygon_overlap_ratio"])
    )
    guided_tangent = float("nan") if guided is None else float(np.dot(translation, tangent))
    return {
        "room_a": room_a, "room_b": room_b, "candidate_id": f"candidate_{index:03d}",
        "wall_a_id": wall_a["id"], "wall_b_id": wall_b["id"],
        "wall_a_preselection_rank": wall_a.get("_preselection_rank"),
        "wall_b_preselection_rank": wall_b.get("_preselection_rank"),
        "wall_a_bearing_alignment": wall_a.get("_bearing_alignment"),
        "wall_b_bearing_alignment": wall_b.get("_bearing_alignment"),
        "wall_a_ray_hit_distance": wall_a.get("_ray_hit_distance"),
        "wall_b_ray_hit_distance": wall_b.get("_ray_hit_distance"),
        "geoverify_yaw_deg": float(geo["yaw_deg"]),
        "manhattan_yaw_deg": normalize_deg(math.degrees(theta)),
        "yaw_snap_residual_deg": yaw_residual,
        "yaw_snap_within_threshold": yaw_residual <= args.max_yaw_snap_residual,
        "geoverify_bearing_deg": float(geo["bearing_deg"]),
        "orientation_error_deg": orientation_error, "normal_bearing_cos": incidence,
        "normal_displacement": normal_displacement, "lambda_scale": lambda_scale,
        "dx": None if guided is None else float(translation[0]),
        "dy": None if guided is None else float(translation[1]), "dtheta": theta,
        "translation_magnitude": None if guided is None else float(np.linalg.norm(translation)),
        "center_dx": float(center_translation[0]), "center_dy": float(center_translation[1]),
        "center_tangential_translation": center_tangent,
        "guided_tangential_translation": guided_tangent,
        "along_wall_slide_from_center": None if guided is None else guided_tangent - center_tangent,
        "post_snap_wall_distance": None if guided is None else guided["post_distance"],
        "shared_wall_overlap_length": None if guided is None else guided["overlap_length"],
        "shared_wall_overlap_ratio": None if guided is None else guided["overlap_ratio"],
        "polygon_overlap_ratio": None if guided is None else guided["polygon_overlap_ratio"],
        "opposite_side_valid": False if guided is None else guided["opposite_side"],
        "bearing_diff_deg": None if guided is None else vector_angle_deg(translation, bearing),
        "pass_orientation": orientation_ok, "pass_bearing_incidence": stable_incidence,
        "pass_positive_scale": positive_scale, "pass_wall_contact": contact_ok,
        "pass_overlap": overlap_ok, "pass_polygon_overlap": polygon_ok,
        "pass_opposite_side": side_ok, "candidate_valid": candidate_valid,
        "candidate_score": score, "center_bearing_diff_deg": center_bearing_diff,
        "center_overlap_length": center["overlap_length"],
        "center_overlap_ratio": center["overlap_ratio"],
        "center_polygon_overlap_ratio": center["polygon_overlap_ratio"],
        "center_opposite_side_valid": center["opposite_side"],
        "center_candidate_valid": center_valid, "center_candidate_score": center_score,
        "_guided": guided, "_center": center,
    }


def enumerate_pair(
    polygon_a: np.ndarray,
    polygon_b: np.ndarray,
    geo: dict[str, Any],
    room_a: str,
    room_b: str,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], float, float, dict[str, Any]]:
    shape_a = valid_polygon(polygon_a)
    valid_polygon(polygon_b)
    _, theta, residual = closest_manhattan_rotation(polygon_a, polygon_b, float(geo["yaw_rad"]))
    rotated_b = polygon_b @ rotation2(theta).T
    all_walls_a = polygon_walls(polygon_a)
    all_walls_b = polygon_walls(rotated_b)
    bearing = np.asarray(geo["bearing"], dtype=float)
    if args.disable_wall_preselection:
        walls_a = [dict(wall, _full_position=index) for index, wall in enumerate(all_walls_a)]
        walls_b = [dict(wall, _full_position=index) for index, wall in enumerate(all_walls_b)]
    else:
        walls_a = preselect_walls_by_bearing(
            polygon_a, all_walls_a, bearing,
            args.wall_preselection_top_k, args.wall_preselection_max_angle,
        )
        walls_b = preselect_walls_by_bearing(
            rotated_b, all_walls_b, -bearing,
            args.wall_preselection_top_k, args.wall_preselection_max_angle,
        )
    rows: list[dict[str, Any]] = []
    full_b_count = len(all_walls_b)
    for wall_a in walls_a:
        for wall_b in walls_b:
            stable_index = int(wall_a["_full_position"]) * full_b_count + int(wall_b["_full_position"]) + 1
            rows.append(make_candidate(
                polygon_a, rotated_b, shape_a, wall_a, wall_b, theta, geo,
                room_a, room_b, stable_index, residual, args,
            ))
    full_pairs = len(all_walls_a) * len(all_walls_b)
    selected_pairs = len(walls_a) * len(walls_b)
    preselection = {
        "num_wall_a_total": len(all_walls_a),
        "num_wall_b_total": len(all_walls_b),
        "num_wall_a_preselected": len(walls_a),
        "num_wall_b_preselected": len(walls_b),
        "num_full_wall_pairs": full_pairs,
        "num_preselected_wall_pairs": selected_pairs,
        "wall_pair_reduction_ratio": (
            1.0 - selected_pairs / full_pairs if full_pairs else 0.0
        ),
    }
    return rows, theta, residual, preselection


def rank_valid(rows: list[dict[str, Any]], valid_field: str, score_field: str) -> list[dict[str, Any]]:
    return sorted(
        (row for row in rows if row[valid_field]),
        key=lambda row: (float(row[score_field]), row["candidate_id"]),
    )
