"""Room assembly calculations."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from shapely.geometry import LineString, MultiPoint, Polygon
from .geometry import Pose, apply_pose, compose, invert, wrap_pi
from .risk import RISK_SIGNALS, extract_metrics

@dataclass
class Candidate:
    candidate_id: str
    measurement: Pose
    wall_a_index: int
    wall_b_index: int
    candidate_score: float
    intrinsic_cost: float = 0.0
    slide: float = 0.0
    slide_cost: float = 0.0
    overlap_ratio: float = 0.0
    polygon_overlap_ratio: float = 0.0


@dataclass
class Edge:
    index: int
    room_a: str
    room_b: str
    measurement: Pose
    selected_candidate_id: str
    yaw_residual_deg: float
    valid_candidate_count: int
    confidence: float = 0.05
    automatic_confidence: float = 0.05
    candidates: list[Candidate] = field(default_factory=list)
    risk_signal_count: int = 0
    risk_tier: str = "low"
    risk_reasons: list[str] = field(default_factory=list)
    estimated_wrong_rate: float = 0.182
    geoverify_metrics: dict[str, Any] = field(default_factory=dict)
    geoverify_json: str = ""
    manual_action: str = ""
    manual_note: str = ""


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def valid_polygon(points: np.ndarray) -> Polygon:
    polygon = Polygon(np.asarray(points, dtype=float)).buffer(0)
    if polygon.is_empty or polygon.area <= 1e-9:
        polygon = MultiPoint(np.asarray(points, dtype=float)).convex_hull
    if (polygon.geom_type != "Polygon" or polygon.area <= 1e-9) and not polygon.is_empty:
        polygon = polygon.buffer(1e-3)
    if polygon.is_empty or polygon.area <= 1e-9 or polygon.geom_type != "Polygon":
        raise ValueError("invalid room polygon")
    if polygon.geom_type != "Polygon":
        polygon = max(polygon.geoms, key=lambda item: item.area)
    return polygon


def interval_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    a_low, a_high = sorted((float(a0), float(a1)))
    b_low, b_high = sorted((float(b0), float(b1)))
    return max(0.0, min(a_high, b_high) - max(a_low, b_low))


def wall_index(value: str) -> int:
    return int(value.rsplit("_", 1)[1])


def expand_slides(
    candidate: Candidate,
    polygon_a: np.ndarray,
    polygon_b: np.ndarray,
    slide_samples: int,
    max_slide_ratio: float,
    min_wall_overlap: float,
    polygon_overlap_threshold: float,
) -> list[Candidate]:
    wall_a0 = polygon_a[candidate.wall_a_index]
    wall_a1 = polygon_a[(candidate.wall_a_index + 1) % len(polygon_a)]
    tangent = wall_a1 - wall_a0
    wall_a_length = float(np.linalg.norm(tangent))
    if wall_a_length <= 1e-9:
        return [candidate]
    tangent /= wall_a_length
    rotated_b = apply_pose(polygon_b, (0.0, 0.0, candidate.measurement[2]))
    wall_b0 = rotated_b[candidate.wall_b_index]
    wall_b1 = rotated_b[(candidate.wall_b_index + 1) % len(rotated_b)]
    wall_b_length = float(np.linalg.norm(wall_b1 - wall_b0))
    slide_limit = max_slide_ratio * min(wall_a_length, wall_b_length)
    sample_count = max(1, slide_samples)
    shifts = np.linspace(-slide_limit, slide_limit, sample_count).tolist()
    shifts.append(0.0)
    midpoint_shift = float(np.dot((wall_a0 + wall_a1 - wall_b0 - wall_b1) * 0.5, tangent))
    midpoint_shift -= float(np.dot(np.asarray(candidate.measurement[:2]), tangent))
    shifts.append(float(np.clip(midpoint_shift, -slide_limit, slide_limit)))

    polygon_a_shape = valid_polygon(polygon_a)
    output: list[Candidate] = []
    seen: set[int] = set()
    for shift in sorted(shifts):
        key = round(shift * 1000)
        if key in seen:
            continue
        seen.add(key)
        translation = np.asarray(candidate.measurement[:2]) + shift * tangent
        transformed_b = rotated_b + translation
        transformed_shape = valid_polygon(transformed_b)
        transformed_b0 = transformed_b[candidate.wall_b_index]
        transformed_b1 = transformed_b[(candidate.wall_b_index + 1) % len(transformed_b)]
        wall_distance = LineString([wall_a0, wall_a1]).distance(LineString([transformed_b0, transformed_b1]))
        overlap = interval_overlap(
            np.dot(wall_a0, tangent), np.dot(wall_a1, tangent),
            np.dot(transformed_b0, tangent), np.dot(transformed_b1, tangent),
        )
        overlap_ratio = overlap / max(min(wall_a_length, wall_b_length), 1e-9)
        polygon_overlap = polygon_a_shape.intersection(transformed_shape).area
        polygon_overlap_ratio = polygon_overlap / min(polygon_a_shape.area, transformed_shape.area)
        wall_midpoint = (wall_a0 + wall_a1) * 0.5
        normal = np.asarray([-tangent[1], tangent[0]])
        center_a = np.asarray(polygon_a_shape.representative_point().coords[0])
        center_b = np.asarray(transformed_shape.representative_point().coords[0])
        opposite_side = np.dot(center_a - wall_midpoint, normal) * np.dot(center_b - wall_midpoint, normal) < 0.0
        if wall_distance > 1e-4 or overlap < min_wall_overlap:
            continue
        if polygon_overlap_ratio >= polygon_overlap_threshold or not opposite_side:
            continue
        output.append(Candidate(
            candidate_id=candidate.candidate_id,
            measurement=(float(translation[0]), float(translation[1]), candidate.measurement[2]),
            wall_a_index=candidate.wall_a_index,
            wall_b_index=candidate.wall_b_index,
            candidate_score=candidate.candidate_score,
            intrinsic_cost=candidate.intrinsic_cost,
            slide=float(shift),
            slide_cost=abs(float(shift)) / max(min(wall_a_length, wall_b_length), 1.0),
            overlap_ratio=float(overlap_ratio),
            polygon_overlap_ratio=float(polygon_overlap_ratio),
        ))
    return output or [candidate]


def apply_geoverify_risk(
    house: str,
    edge: Edge,
    geoverify_index: dict[tuple[str, tuple[str, str]], tuple[Path, dict[str, Any]]],
    args: argparse.Namespace,
) -> None:
    key = (house, tuple(sorted((edge.room_a, edge.room_b))))
    located = geoverify_index.get(key)
    if located is None:
        edge.risk_signal_count = 1
        edge.risk_tier = "high"
        edge.risk_reasons = ["missing_geoverify_json"]
        edge.estimated_wrong_rate = 0.892
        edge.confidence *= args.high_risk_multiplier
        return
    path, data = located
    metrics = extract_metrics(data)
    reasons = []
    for name, predicate in RISK_SIGNALS.items():
        try:
            if predicate(metrics):
                reasons.append(name)
        except (KeyError, TypeError, ValueError):
            continue
    if metrics.get("flag_low_matches"):
        reasons.append("geoverify_flag_low_matches")
    signal_count = sum(reason in RISK_SIGNALS for reason in reasons)
    if "geoverify_flag_low_matches" in reasons or signal_count >= 4:
        tier, estimated_rate, multiplier = "high", 0.892, args.high_risk_multiplier
    elif signal_count >= 2:
        tier, estimated_rate, multiplier = "medium", 0.381, args.medium_risk_multiplier
    else:
        tier, estimated_rate, multiplier = "low", 0.182, 1.0
    edge.risk_signal_count = signal_count
    edge.risk_tier = tier
    edge.risk_reasons = reasons
    edge.estimated_wrong_rate = estimated_rate
    edge.geoverify_metrics = metrics
    edge.geoverify_json = str(path)
    edge.confidence *= multiplier


def enrich_edges_from_data(
    house: str,
    edges: list[Edge],
    polygons: dict[str, np.ndarray],
    grouped: dict[tuple[str, str], list[Candidate]],
    geoverify_index: dict[tuple[str, tuple[str, str]], tuple[Path, dict[str, Any]]],
    args: argparse.Namespace,
) -> None:
    for edge in edges:
        bases = grouped.get((edge.room_a, edge.room_b), [])
        variants: list[Candidate] = []
        for base in bases:
            variants.extend(expand_slides(
                base, polygons[edge.room_a], polygons[edge.room_b], args.slide_samples,
                args.max_slide_ratio, args.wall_min_overlap, args.polygon_overlap_threshold,
            ))
        if not variants:
            variants = [Candidate(
                candidate_id=edge.selected_candidate_id or "original",
                measurement=edge.measurement,
                wall_a_index=0,
                wall_b_index=0,
                candidate_score=0.0,
            )]
        edge.candidates = variants
        selected = min(
            variants,
            key=lambda candidate: (
                candidate.candidate_id != edge.selected_candidate_id,
                abs(candidate.slide),
                candidate.intrinsic_cost,
            ),
        )
        edge.measurement = selected.measurement
        yaw_quality = math.exp(-0.5 * (edge.yaw_residual_deg / 10.0) ** 2)
        incidence = 0.2
        overlap_quality = max(0.0, min(1.0, selected.overlap_ratio))
        if bases:
            source = min(bases, key=lambda candidate: (
                candidate.candidate_id != edge.selected_candidate_id,
                candidate.candidate_score,
            ))
            overlap_quality = max(0.0, min(1.0, source.overlap_ratio))
            incidence = 0.5
        ambiguity = 1.0 / math.sqrt(max(edge.valid_candidate_count, 1))
        edge.confidence = max(0.03, min(1.0,
            yaw_quality * (0.55 + 0.45 * overlap_quality) * (0.65 + 0.35 * ambiguity) * (0.8 + 0.2 * incidence)
        ))
        apply_geoverify_risk(house, edge, geoverify_index, args)
        edge.automatic_confidence = edge.confidence
        if edge.confidence >= args.candidate_switch_confidence and edge.selected_candidate_id:
            selected_variants = [
                candidate for candidate in edge.candidates
                if candidate.candidate_id == edge.selected_candidate_id
            ]
            if selected_variants:
                edge.candidates = selected_variants
