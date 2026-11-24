"""Room assembly calculations."""
from __future__ import annotations

import csv
from pathlib import Path
from collections import defaultdict
from ..core.candidates import Candidate, Edge, parse_bool, parse_float, wall_index

def load_edges(path: Path) -> dict[str, list[Edge]]:
    grouped: dict[str, list[Edge]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "success":
                continue
            values = [row.get(name, "") for name in ("guided_dx", "guided_dy", "guided_dtheta")]
            if any(value == "" for value in values):
                continue
            house = row["house"]
            grouped[house].append(Edge(
                index=len(grouped[house]),
                room_a=row["room_a"],
                room_b=row["room_b"],
                measurement=(float(values[0]), float(values[1]), float(values[2])),
                selected_candidate_id=row.get("best_guided_candidate_id", ""),
                yaw_residual_deg=parse_float(row.get("yaw_snap_residual_deg"), 180.0),
                valid_candidate_count=max(1, int(parse_float(row.get("num_valid_guided_candidates"), 1))),
            ))
    return grouped


def load_base_candidates(path: Path) -> dict[tuple[str, str], list[Candidate]]:
    grouped: dict[tuple[str, str], list[Candidate]] = defaultdict(list)
    if not path.is_file():
        return grouped
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not parse_bool(row.get("candidate_valid")):
                continue
            candidate = Candidate(
                candidate_id=row["candidate_id"],
                measurement=(float(row["dx"]), float(row["dy"]), float(row["dtheta"])),
                wall_a_index=wall_index(row["wall_a_id"]),
                wall_b_index=wall_index(row["wall_b_id"]),
                candidate_score=parse_float(row.get("candidate_score")),
                overlap_ratio=parse_float(row.get("shared_wall_overlap_ratio")),
                polygon_overlap_ratio=parse_float(row.get("polygon_overlap_ratio")),
            )
            grouped[(row["room_a"], row["room_b"])].append(candidate)
    for candidates in grouped.values():
        candidates.sort(key=lambda candidate: (candidate.candidate_score, candidate.candidate_id))
        low = candidates[0].candidate_score
        high = candidates[-1].candidate_score
        span = max(high - low, 1e-9)
        for candidate in candidates:
            candidate.intrinsic_cost = (candidate.candidate_score - low) / span if len(candidates) > 1 else 0.0
    return grouped
