#!/usr/bin/env python3
"""
Floor-plan-level layout objective analyzer.

This module is intentionally independent from the GTSAM optimizer. It evaluates
layout terms that depend on optimized room SE(2) poses:
  - door/opening consistency
  - wall alignment / collinearity
  - room overlap
  - room adjacency distance
  - approximate narrow gaps
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union


Point = Tuple[float, float]
Pose = Tuple[float, float, float]


DEFAULT_WEIGHTS = {
    "door": 3.0,
    "wall_align": 2.0,
    "overlap": 5.0,
    "adjacency": 2.0,
    "gap": 1.0,
}


DEFAULT_PARAMS = {
    "wall_axis_angle_tol_deg": 10.0,
    "wall_near_distance": 2.0,
    "wall_align_tolerance": 0.05,
    "wall_min_overlap": 0.25,
    "adjacency_distance_threshold": 0.1,
    "gap_width_threshold": 0.25,
    "global_gap_closing_radii": [0.2, 0.5, 1.0],
    "matched_gap_tolerance": 0.5,
    "candidate_wall_gap_tolerance": 0.5,
    "candidate_wall_max_distance": 2.0,
    "candidate_wall_min_overlap": 0.25,
    "candidate_wall_penalty_power": 2.0,
    "candidate_wall_exclude_matched_pairs": True,
}


@dataclass
class Room:
    room_id: str
    polygon: np.ndarray


def wrap_pi(theta: float) -> float:
    return (theta + math.pi) % (2.0 * math.pi) - math.pi


def undirected_angle_diff(a: float, b: float) -> float:
    diff = abs(wrap_pi(a - b))
    return min(diff, abs(math.pi - diff))


def as_point(value: Sequence[float], field_name: str) -> Point:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{field_name} must be a 2-number list")
    return float(value[0]), float(value[1])


def as_polygon(value: Any, field_name: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) < 3:
        raise ValueError(f"{field_name} must be a list with at least 3 points")
    pts = np.array([as_point(p, f"{field_name}[]") for p in value], dtype=np.float64)
    if pts.shape[1] != 2:
        raise ValueError(f"{field_name} must be Nx2")
    return pts


def normalize_room_id(room_id: str) -> str:
    return str(room_id).replace(".txt", "")


def load_rooms(data: Dict[str, Any]) -> Dict[str, Room]:
    raw = data.get("rooms")
    if raw is None:
        raise ValueError("input JSON missing 'rooms'")

    rooms: Dict[str, Room] = {}
    if isinstance(raw, dict):
        iterator = raw.items()
        for room_id, item in iterator:
            if isinstance(item, dict):
                polygon = item.get("polygon", item.get("local_polygon"))
            else:
                polygon = item
            rid = normalize_room_id(room_id)
            rooms[rid] = Room(rid, as_polygon(polygon, f"rooms.{room_id}.polygon"))
    elif isinstance(raw, list):
        for idx, item in enumerate(raw):
            if not isinstance(item, dict) or "id" not in item:
                raise ValueError(f"rooms[{idx}] must contain id")
            rid = normalize_room_id(item["id"])
            polygon = item.get("polygon", item.get("local_polygon"))
            rooms[rid] = Room(rid, as_polygon(polygon, f"rooms[{idx}].polygon"))
    else:
        raise ValueError("'rooms' must be a dict or list")

    return rooms


def load_poses(data: Dict[str, Any]) -> Dict[str, Pose]:
    raw = data.get("poses")
    if raw is None:
        raise ValueError("input JSON missing 'poses'")
    out: Dict[str, Pose] = {}
    for room_id, pose in raw.items():
        rid = normalize_room_id(room_id)
        try:
            out[rid] = (float(pose["x"]), float(pose["y"]), float(pose["theta"]))
        except Exception as exc:
            raise ValueError(f"invalid pose for room {room_id}") from exc
    return out


def apply_pose_to_points(pose: Pose, points: np.ndarray) -> np.ndarray:
    x, y, th = pose
    c = math.cos(th)
    s = math.sin(th)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    return points @ rot.T + np.array([x, y], dtype=np.float64)


def apply_pose_to_point(pose: Pose, point: Point) -> Point:
    arr = apply_pose_to_points(pose, np.array([point], dtype=np.float64))[0]
    return float(arr[0]), float(arr[1])


def polygon_to_shapely(points: np.ndarray) -> Polygon:
    poly = Polygon([(float(x), float(y)) for x, y in points])
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def transformed_polygons(rooms: Dict[str, Room], poses: Dict[str, Pose]) -> Dict[str, Polygon]:
    out: Dict[str, Polygon] = {}
    for room_id, room in rooms.items():
        if room_id not in poses:
            raise KeyError(f"missing pose for room {room_id}")
        world = apply_pose_to_points(poses[room_id], room.polygon)
        out[room_id] = polygon_to_shapely(world)
    return out


def safe_unary_union(polygons: Iterable[Polygon]):
    geoms = [poly for poly in polygons if not poly.is_empty and poly.area > 1e-12]
    if not geoms:
        return Polygon()
    union = unary_union(geoms)
    if not union.is_valid:
        union = union.buffer(0)
    return union


def normalize_adjacency_edges(raw: Any) -> List[Dict[str, str]]:
    if raw is None:
        return []
    edges = raw.get("edges", raw) if isinstance(raw, dict) else raw
    if not isinstance(edges, list):
        raise ValueError("adjacency_edges must be a list")
    out = []
    for idx, edge in enumerate(edges):
        if isinstance(edge, dict):
            src = edge.get("src", edge.get("i"))
            dst = edge.get("dst", edge.get("j"))
            out_edge = {"src": normalize_room_id(src), "dst": normalize_room_id(dst)}
            if "weight" in edge:
                out_edge["weight"] = edge["weight"]
        elif isinstance(edge, (list, tuple)) and len(edge) == 2:
            src, dst = edge
            out_edge = {"src": normalize_room_id(src), "dst": normalize_room_id(dst)}
        else:
            raise ValueError(f"invalid adjacency edge at index {idx}")
        out.append(out_edge)
    return out


def segment_from_entry(entry: Dict[str, Any], prefix: str) -> Tuple[Point, Point]:
    segment = entry.get(f"{prefix}_segment")
    if segment is None:
        segment = entry.get(prefix, {}).get("segment") if isinstance(entry.get(prefix), dict) else None
    if not isinstance(segment, list) or len(segment) != 2:
        raise ValueError(f"door/opening entry missing {prefix}_segment")
    return as_point(segment[0], f"{prefix}_segment[0]"), as_point(segment[1], f"{prefix}_segment[1]")


def segment_center(seg: Tuple[Point, Point]) -> Point:
    return (0.5 * (seg[0][0] + seg[1][0]), 0.5 * (seg[0][1] + seg[1][1]))


def segment_length(seg: Tuple[Point, Point]) -> float:
    return math.hypot(seg[1][0] - seg[0][0], seg[1][1] - seg[0][1])


def segment_angle(seg: Tuple[Point, Point]) -> float:
    return math.atan2(seg[1][1] - seg[0][1], seg[1][0] - seg[0][0])


def transform_segment(pose: Pose, seg: Tuple[Point, Point]) -> Tuple[Point, Point]:
    return apply_pose_to_point(pose, seg[0]), apply_pose_to_point(pose, seg[1])


def evaluate_door_consistency(raw_doors: Any, poses: Dict[str, Pose]) -> Dict[str, Any]:
    doors = raw_doors or []
    score = 0.0
    details = []
    for idx, entry in enumerate(doors):
        src = normalize_room_id(entry.get("src", entry.get("i")))
        dst = normalize_room_id(entry.get("dst", entry.get("j")))
        if src not in poses or dst not in poses:
            raise KeyError(f"door correspondence {idx} references missing pose")
        src_world = transform_segment(poses[src], segment_from_entry(entry, "src"))
        dst_world = transform_segment(poses[dst], segment_from_entry(entry, "dst"))
        center_dist = math.hypot(
            segment_center(src_world)[0] - segment_center(dst_world)[0],
            segment_center(src_world)[1] - segment_center(dst_world)[1],
        )
        angle_diff = undirected_angle_diff(segment_angle(src_world), segment_angle(dst_world))
        len_src = segment_length(src_world)
        len_dst = segment_length(dst_world)
        length_diff = abs(len_src - len_dst)
        item_score = center_dist**2 + angle_diff**2 + length_diff**2
        score += item_score
        details.append(
            {
                "idx": idx,
                "id": entry.get("id", f"door_{idx}"),
                "src": src,
                "dst": dst,
                "center_distance": center_dist,
                "angle_diff": angle_diff,
                "length_src": len_src,
                "length_dst": len_dst,
                "length_diff": length_diff,
                "score": item_score,
                "src_world_segment": src_world,
                "dst_world_segment": dst_world,
            }
        )
    return {"score": score, "num_doors": len(details), "details": details}


def extract_wall_segments(poly: Polygon, room_id: str) -> List[Dict[str, Any]]:
    coords = list(poly.exterior.coords)
    out = []
    for idx in range(len(coords) - 1):
        p0 = (float(coords[idx][0]), float(coords[idx][1]))
        p1 = (float(coords[idx + 1][0]), float(coords[idx + 1][1]))
        length = segment_length((p0, p1))
        if length <= 1e-9:
            continue
        angle = segment_angle((p0, p1))
        out.append({"room_id": room_id, "idx": idx, "p0": p0, "p1": p1, "length": length, "angle": angle})
    return out


def wall_axis(seg: Dict[str, Any], angle_tol: float) -> Optional[str]:
    angle = abs(wrap_pi(seg["angle"]))
    horizontal = min(angle, abs(math.pi - angle))
    vertical = abs(angle - math.pi / 2.0)
    if horizontal <= angle_tol:
        return "horizontal"
    if vertical <= angle_tol:
        return "vertical"
    return None


def interval_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    lo = max(min(a0, a1), min(b0, b1))
    hi = min(max(a0, a1), max(b0, b1))
    return max(0.0, hi - lo)


def evaluate_wall_alignment(polygons: Dict[str, Polygon], params: Dict[str, float]) -> Dict[str, Any]:
    angle_tol = math.radians(float(params["wall_axis_angle_tol_deg"]))
    near_distance = float(params["wall_near_distance"])
    align_tol = float(params["wall_align_tolerance"])
    min_overlap = float(params["wall_min_overlap"])

    walls = []
    for room_id, poly in polygons.items():
        for seg in extract_wall_segments(poly, room_id):
            axis = wall_axis(seg, angle_tol)
            if axis is not None:
                seg["axis"] = axis
                walls.append(seg)

    score = 0.0
    details = []
    for a, b in itertools.combinations(walls, 2):
        if a["room_id"] == b["room_id"] or a["axis"] != b["axis"]:
            continue
        if a["axis"] == "horizontal":
            overlap = interval_overlap(a["p0"][0], a["p1"][0], b["p0"][0], b["p1"][0])
            offset = abs(0.5 * (a["p0"][1] + a["p1"][1]) - 0.5 * (b["p0"][1] + b["p1"][1]))
        else:
            overlap = interval_overlap(a["p0"][1], a["p1"][1], b["p0"][1], b["p1"][1])
            offset = abs(0.5 * (a["p0"][0] + a["p1"][0]) - 0.5 * (b["p0"][0] + b["p1"][0]))
        if overlap < min_overlap or offset >= near_distance:
            continue
        if offset <= align_tol:
            item_score = 0.0
        else:
            item_score = (offset - align_tol) ** 2 * overlap
        score += item_score
        if item_score > 0.0:
            details.append(
                {
                    "room_a": a["room_id"],
                    "room_b": b["room_id"],
                    "wall_a_idx": a["idx"],
                    "wall_b_idx": b["idx"],
                    "axis": a["axis"],
                    "offset": offset,
                    "overlap_length": overlap,
                    "score": item_score,
                    "wall_a": [a["p0"], a["p1"]],
                    "wall_b": [b["p0"], b["p1"]],
                }
            )
    return {"score": score, "num_misaligned_pairs": len(details), "details": details}


def evaluate_overlap(polygons: Dict[str, Polygon]) -> Dict[str, Any]:
    score = 0.0
    details = []
    for a, b in itertools.combinations(sorted(polygons), 2):
        inter = polygons[a].intersection(polygons[b])
        area = float(inter.area)
        if area <= 1e-9:
            continue
        score += area
        details.append({"room_a": a, "room_b": b, "overlap_area": area, "score": area})
    return {"score": score, "num_overlaps": len(details), "details": details}


def evaluate_room_overlap_ratio(polygons: Dict[str, Polygon]) -> Dict[str, Any]:
    room_areas = {room_id: float(poly.area) for room_id, poly in polygons.items()}
    total_area = float(sum(room_areas.values()))
    union = safe_unary_union(polygons.values())
    union_area = float(union.area)
    overlap_area = max(0.0, total_area - union_area)
    ratio = overlap_area / total_area if total_area > 0.0 else 0.0

    pairwise_details = []
    pairwise_overlap_sum = 0.0
    for a, b in itertools.combinations(sorted(polygons), 2):
        area = float(polygons[a].intersection(polygons[b]).area)
        if area <= 1e-9:
            continue
        pairwise_overlap_sum += area
        pairwise_details.append({"room_a": a, "room_b": b, "overlap_area": area})

    return {
        "ratio": ratio,
        "overlap_area": overlap_area,
        "total_room_area": total_area,
        "union_area": union_area,
        "room_areas": room_areas,
        "pairwise_overlap_sum": pairwise_overlap_sum,
        "pairwise_details": pairwise_details,
        "definition": "(sum(room areas) - area(union rooms)) / sum(room areas)",
    }


def closing_envelope(union_geom, radius: float):
    if radius <= 0.0 or union_geom.is_empty:
        return union_geom
    envelope = union_geom.buffer(radius).buffer(-radius)
    if not envelope.is_valid:
        envelope = envelope.buffer(0)
    return envelope


def evaluate_global_gap_ratio(
    polygons: Dict[str, Polygon], closing_radii: Iterable[float]
) -> Dict[str, Any]:
    union = safe_unary_union(polygons.values())
    union_area = float(union.area)
    details = []

    for radius_raw in closing_radii:
        radius = float(radius_raw)
        envelope = closing_envelope(union, radius)
        envelope_area = float(envelope.area)
        gap_geom = envelope.difference(union) if not envelope.is_empty else Polygon()
        gap_area = max(0.0, float(gap_geom.area))
        ratio = gap_area / envelope_area if envelope_area > 0.0 else 0.0
        details.append(
            {
                "closing_radius": radius,
                "gap_area": gap_area,
                "envelope_area": envelope_area,
                "union_area": union_area,
                "ratio": ratio,
            }
        )

    return {
        "union_area": union_area,
        "details": details,
        "definition": "area(closing(union, r) - union) / area(closing(union, r))",
    }


def edge_weight(edge: Dict[str, Any]) -> float:
    try:
        return float(edge.get("weight", 1.0))
    except (TypeError, ValueError):
        return 1.0


def evaluate_matched_gap_ratio(
    polygons: Dict[str, Polygon],
    pose_edges: List[Dict[str, str]],
    tolerance: float,
) -> Dict[str, Any]:
    tolerance = float(tolerance)
    if tolerance <= 0.0:
        raise ValueError("matched_gap_tolerance must be positive")

    weighted_sum = 0.0
    total_weight = 0.0
    details = []

    for idx, edge in enumerate(pose_edges):
        src, dst = edge["src"], edge["dst"]
        if src not in polygons or dst not in polygons:
            raise KeyError(f"pose edge {idx} references missing room")

        distance = float(polygons[src].distance(polygons[dst]))
        penalty = min(distance / tolerance, 1.0)
        weight = edge_weight(edge)
        weighted_sum += weight * penalty
        total_weight += weight
        details.append(
            {
                "idx": idx,
                "src": src,
                "dst": dst,
                "distance": distance,
                "tolerance": tolerance,
                "penalty": penalty,
                "weight": weight,
                "weighted_penalty": weight * penalty,
            }
        )

    ratio = weighted_sum / total_weight if total_weight > 0.0 else 0.0
    return {
        "ratio": ratio,
        "weighted_penalty_sum": weighted_sum,
        "total_weight": total_weight,
        "num_edges": len(details),
        "details": details,
        "definition": "weighted average over pose edges of min(distance(P_i, P_j) / tau_gap, 1)",
    }


def bool_param(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def wall_candidate_gap_strip(a: Dict[str, Any], b: Dict[str, Any], axis: str) -> Polygon:
    if axis == "horizontal":
        lo = max(min(a["p0"][0], a["p1"][0]), min(b["p0"][0], b["p1"][0]))
        hi = min(max(a["p0"][0], a["p1"][0]), max(b["p0"][0], b["p1"][0]))
        y0 = 0.5 * (a["p0"][1] + a["p1"][1])
        y1 = 0.5 * (b["p0"][1] + b["p1"][1])
        if hi <= lo or abs(y1 - y0) <= 1e-12:
            return Polygon()
        return Polygon([(lo, y0), (hi, y0), (hi, y1), (lo, y1)]).buffer(0)

    lo = max(min(a["p0"][1], a["p1"][1]), min(b["p0"][1], b["p1"][1]))
    hi = min(max(a["p0"][1], a["p1"][1]), max(b["p0"][1], b["p1"][1]))
    x0 = 0.5 * (a["p0"][0] + a["p1"][0])
    x1 = 0.5 * (b["p0"][0] + b["p1"][0])
    if hi <= lo or abs(x1 - x0) <= 1e-12:
        return Polygon()
    return Polygon([(x0, lo), (x1, lo), (x1, hi), (x0, hi)]).buffer(0)


def detect_candidate_wall_pairs(
    polygons: Dict[str, Polygon],
    pose_edges: List[Dict[str, str]],
    params: Dict[str, Any],
) -> List[Dict[str, Any]]:
    angle_tol = math.radians(float(params["wall_axis_angle_tol_deg"]))
    max_distance = float(params["candidate_wall_max_distance"])
    min_overlap = float(params["candidate_wall_min_overlap"])
    exclude_matched_pairs = bool_param(params["candidate_wall_exclude_matched_pairs"])
    matched_pairs = {adjacency_key(edge["src"], edge["dst"]) for edge in pose_edges}

    walls = []
    for room_id, poly in polygons.items():
        for seg in extract_wall_segments(poly, room_id):
            axis = wall_axis(seg, angle_tol)
            if axis is not None:
                seg["axis"] = axis
                seg["line"] = LineString([seg["p0"], seg["p1"]])
                walls.append(seg)

    candidates = []
    for a, b in itertools.combinations(walls, 2):
        if a["room_id"] == b["room_id"] or a["axis"] != b["axis"]:
            continue
        if exclude_matched_pairs and adjacency_key(a["room_id"], b["room_id"]) in matched_pairs:
            continue

        if a["axis"] == "horizontal":
            overlap = interval_overlap(a["p0"][0], a["p1"][0], b["p0"][0], b["p1"][0])
            distance = abs(
                0.5 * (a["p0"][1] + a["p1"][1])
                - 0.5 * (b["p0"][1] + b["p1"][1])
            )
        else:
            overlap = interval_overlap(a["p0"][1], a["p1"][1], b["p0"][1], b["p1"][1])
            distance = abs(
                0.5 * (a["p0"][0] + a["p1"][0])
                - 0.5 * (b["p0"][0] + b["p1"][0])
            )

        if overlap < min_overlap or distance > max_distance:
            continue

        strip = wall_candidate_gap_strip(a, b, a["axis"])
        candidates.append(
            {
                "room_a": a["room_id"],
                "room_b": b["room_id"],
                "wall_a_idx": a["idx"],
                "wall_b_idx": b["idx"],
                "axis": a["axis"],
                "distance": distance,
                "overlap_length": overlap,
                "weight": 1.0,
                "wall_a": [a["p0"], a["p1"]],
                "wall_b": [b["p0"], b["p1"]],
                "gap_strip": strip,
            }
        )

    return candidates


def evaluate_candidate_wall_gap_ratio(
    polygons: Dict[str, Polygon],
    pose_edges: List[Dict[str, str]],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    tolerance = float(params["candidate_wall_gap_tolerance"])
    if tolerance <= 0.0:
        raise ValueError("candidate_wall_gap_tolerance must be positive")
    power = float(params["candidate_wall_penalty_power"])
    candidates = detect_candidate_wall_pairs(polygons, pose_edges, params)

    numerator = 0.0
    denominator = 0.0
    details = []
    strips = []
    for idx, item in enumerate(candidates):
        distance = float(item["distance"])
        overlap_length = float(item["overlap_length"])
        weight = float(item["weight"])
        penalty = min(distance / tolerance, 1.0) ** power
        support = weight * overlap_length
        numerator += support * penalty
        denominator += support
        if not item["gap_strip"].is_empty:
            strips.append(item["gap_strip"])
        details.append(
            {
                "idx": idx,
                "room_a": item["room_a"],
                "room_b": item["room_b"],
                "wall_a_idx": item["wall_a_idx"],
                "wall_b_idx": item["wall_b_idx"],
                "axis": item["axis"],
                "distance": distance,
                "overlap_length": overlap_length,
                "weight": weight,
                "support": support,
                "tolerance": tolerance,
                "penalty": penalty,
                "weighted_penalty": support * penalty,
                "wall_a": item["wall_a"],
                "wall_b": item["wall_b"],
            }
        )

    ratio = numerator / denominator if denominator > 0.0 else 0.0
    explained_geom = safe_unary_union(strips)
    return {
        "ratio": ratio,
        "weighted_penalty_sum": numerator,
        "total_support": denominator,
        "num_candidates": len(details),
        "details": details,
        "definition": "sum(w*l*min(distance/tau,1)^power) / sum(w*l)",
        "_explained_gap_geometry": explained_geom,
    }


def evaluate_unexplained_gap_ratio(
    polygons: Dict[str, Polygon],
    global_gap: Dict[str, Any],
    explained_gap_geometry,
) -> Dict[str, Any]:
    union = safe_unary_union(polygons.values())
    details = []
    for item in global_gap["details"]:
        radius = float(item["closing_radius"])
        envelope = closing_envelope(union, radius)
        gap_geom = envelope.difference(union) if not envelope.is_empty else Polygon()
        gap_area = max(0.0, float(gap_geom.area))
        explained_area = 0.0
        if gap_area > 0.0 and not explained_gap_geometry.is_empty:
            explained_area = max(0.0, float(gap_geom.intersection(explained_gap_geometry).area))
        unexplained_area = max(0.0, gap_area - explained_area)
        ratio = unexplained_area / gap_area if gap_area > 0.0 else 0.0
        details.append(
            {
                "closing_radius": radius,
                "gap_area": gap_area,
                "explained_gap_area": explained_area,
                "unexplained_gap_area": unexplained_area,
                "ratio": ratio,
            }
        )
    return {
        "details": details,
        "definition": "area(global_gap - candidate_wall_explained_gap) / area(global_gap)",
    }


def evaluate_gap_metrics(
    polygons: Dict[str, Polygon],
    pose_edges: List[Dict[str, str]],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    global_gap = evaluate_global_gap_ratio(
        polygons,
        params.get("global_gap_closing_radii", DEFAULT_PARAMS["global_gap_closing_radii"]),
    )
    candidate_wall_gap = evaluate_candidate_wall_gap_ratio(polygons, pose_edges, params)
    explained_gap_geometry = candidate_wall_gap.pop("_explained_gap_geometry")
    return {
        "room_overlap_ratio": evaluate_room_overlap_ratio(polygons),
        "global_gap_ratio": global_gap,
        "matched_gap_ratio": evaluate_matched_gap_ratio(
            polygons,
            pose_edges,
            float(params.get("matched_gap_tolerance", DEFAULT_PARAMS["matched_gap_tolerance"])),
        ),
        "candidate_wall_gap_ratio": candidate_wall_gap,
        "unexplained_gap_ratio": evaluate_unexplained_gap_ratio(
            polygons, global_gap, explained_gap_geometry
        ),
    }


def adjacency_key(a: str, b: str) -> Tuple[str, str]:
    return tuple(sorted((a, b)))


def evaluate_adjacency_distance(
    polygons: Dict[str, Polygon], adjacency_edges: List[Dict[str, str]], params: Dict[str, float]
) -> Dict[str, Any]:
    threshold = float(params["adjacency_distance_threshold"])
    score = 0.0
    details = []
    for idx, edge in enumerate(adjacency_edges):
        src, dst = edge["src"], edge["dst"]
        if src not in polygons or dst not in polygons:
            raise KeyError(f"adjacency edge {idx} references missing room")
        dist = float(polygons[src].distance(polygons[dst]))
        excess = max(0.0, dist - threshold)
        item_score = excess**2
        score += item_score
        details.append({"idx": idx, "src": src, "dst": dst, "distance": dist, "threshold": threshold, "score": item_score})
    return {"score": score, "num_edges": len(details), "details": details}


def evaluate_narrow_gaps(
    polygons: Dict[str, Polygon], adjacency_edges: List[Dict[str, str]], params: Dict[str, float]
) -> Dict[str, Any]:
    threshold = float(params["gap_width_threshold"])
    adjacent = {adjacency_key(e["src"], e["dst"]) for e in adjacency_edges}
    score = 0.0
    details = []
    for a, b in itertools.combinations(sorted(polygons), 2):
        if adjacency_key(a, b) in adjacent:
            continue
        if polygons[a].intersects(polygons[b]):
            continue
        dist = float(polygons[a].distance(polygons[b]))
        if 0.0 < dist < threshold:
            item_score = (threshold - dist) ** 2
            score += item_score
            details.append({"room_a": a, "room_b": b, "gap_distance": dist, "threshold": threshold, "score": item_score})
    return {"score": score, "num_gaps": len(details), "details": details}


def evaluate_layout_objective(data: Dict[str, Any]) -> Dict[str, Any]:
    weights = {**DEFAULT_WEIGHTS, **data.get("weights", {})}
    params = {**DEFAULT_PARAMS, **data.get("params", {})}
    rooms = load_rooms(data)
    poses = load_poses(data)
    adjacency_edges = normalize_adjacency_edges(data.get("adjacency_edges", []))
    pose_edges = normalize_adjacency_edges(data.get("pose_edges", adjacency_edges))
    polygons = transformed_polygons(rooms, poses)

    door = evaluate_door_consistency(data.get("door_correspondences", []), poses)
    wall_align = evaluate_wall_alignment(polygons, params)
    overlap = evaluate_overlap(polygons)
    adjacency = evaluate_adjacency_distance(polygons, adjacency_edges, params)
    gap = evaluate_narrow_gaps(polygons, adjacency_edges, params)
    gap_metrics = evaluate_gap_metrics(polygons, pose_edges, params)

    weighted_terms = {
        "door": weights["door"] * door["score"],
        "wall_align": weights["wall_align"] * wall_align["score"],
        "overlap": weights["overlap"] * overlap["score"],
        "adjacency": weights["adjacency"] * adjacency["score"],
        "gap": weights["gap"] * gap["score"],
    }
    total = float(sum(weighted_terms.values()))
    return {
        "total_score": total,
        "weights": weights,
        "params": params,
        "weighted_terms": weighted_terms,
        "terms": {
            "door": door,
            "wall_align": wall_align,
            "overlap": overlap,
            "adjacency": adjacency,
            "gap": gap,
        },
        "gap_metrics": gap_metrics,
    }
