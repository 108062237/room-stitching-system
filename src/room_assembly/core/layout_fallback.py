"""Room assembly calculations."""
from __future__ import annotations

import argparse
from typing import Any
import numpy as np
from . import candidates as core
from . import wall_candidates as layout_snap
from ..settings import DEFAULTS
LAYOUT_SUGGESTION_CONFIG = DEFAULTS['interactive']['layout_suggestion']
MIN_ROOM_CLEARANCE = DEFAULTS['interactive']['minimum_room_clearance']
MAX_ROOM_OVERLAP_RATIO = DEFAULTS['interactive']['maximum_room_overlap_ratio']
WALL_SNAP_CLEARANCE = DEFAULTS['interactive']['wall_snap_clearance']


def generate_layout_suggestion_candidates(
    room_a: str,
    room_b: str,
    polygons: dict[str, np.ndarray],
) -> list[core.Candidate]:
    """Generate layout-only wall contacts directly from two room polygons."""
    args = argparse.Namespace(
        wall_angle_threshold=LAYOUT_SUGGESTION_CONFIG["wall_angle_threshold"],
        wall_distance_threshold=LAYOUT_SUGGESTION_CONFIG["wall_distance_threshold"],
        wall_min_overlap=LAYOUT_SUGGESTION_CONFIG["wall_min_overlap"],
        wall_min_overlap_ratio=LAYOUT_SUGGESTION_CONFIG["wall_min_overlap_ratio"],
        polygon_overlap_threshold=LAYOUT_SUGGESTION_CONFIG["polygon_overlap_threshold"],
        w_distance=LAYOUT_SUGGESTION_CONFIG["distance_weight"],
        w_angle=LAYOUT_SUGGESTION_CONFIG["angle_weight"],
        w_overlap=LAYOUT_SUGGESTION_CONFIG["overlap_weight"],
        w_polygon_overlap=LAYOUT_SUGGESTION_CONFIG["polygon_overlap_weight"],
    )
    raw_candidates = layout_snap.enumerate_candidates(
        polygons[room_a], polygons[room_b], room_a, room_b, args,
    )
    candidates = []
    for index, row in enumerate(raw_candidates):
        if not all(bool(row[name]) for name in (
            "pass_orientation", "pass_overlap", "pass_polygon_overlap",
            "pass_opposite_side",
        )):
            continue
        score = (
            float(row["orientation_error_deg"]) / LAYOUT_SUGGESTION_CONFIG["score_orientation_scale"]
            - LAYOUT_SUGGESTION_CONFIG["score_overlap_weight"] * float(row["shared_wall_overlap_ratio"])
            + LAYOUT_SUGGESTION_CONFIG["score_polygon_overlap_weight"] * float(row["polygon_overlap_ratio"])
        )
        candidates.append(core.Candidate(
            candidate_id=f"layout_{index:04d}",
            measurement=(float(row["dx"]), float(row["dy"]), float(row["dtheta"])),
            wall_a_index=core.wall_index(row["wall_a_id"]),
            wall_b_index=core.wall_index(row["wall_b_id"]),
            candidate_score=score,
            overlap_ratio=float(row["shared_wall_overlap_ratio"]),
            polygon_overlap_ratio=float(row["polygon_overlap_ratio"]),
        ))
    candidates.sort(key=lambda item: (item.candidate_score, item.candidate_id))
    if candidates:
        values = [item.candidate_score for item in candidates]
        low, span = min(values), max(max(values) - min(values), 1e-9)
        for candidate in candidates:
            candidate.intrinsic_cost = (candidate.candidate_score - low) / span
    return candidates
