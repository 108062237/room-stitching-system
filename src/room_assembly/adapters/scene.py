"""Normalize external data or a versioned JSON scene. Explicit I/O only here."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
from collections.abc import Mapping

import numpy as np

from ..core import guided
from ..core.candidates import Candidate, Edge, enrich_edges_from_data
from ..core.layout_fallback import generate_layout_suggestion_candidates
from ..core.measurement import build_room_measurement
from ..errors import AssemblyError
from ..models import Scene, finite_number
from ..settings import DEFAULTS

SCHEMA_VERSION = "1.0"


def _json_read(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (OSError, ValueError, UnicodeError) as exc:
        raise AssemblyError("INPUT_READ_ERROR", "cannot read valid JSON input", path=str(path)) from exc


def _write_json(path, data):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _args():
    values = DEFAULTS["layout_refinement"].copy()
    values["candidate_switch_confidence"] = DEFAULTS["interactive"]["candidate_switch_confidence"]
    return argparse.Namespace(**values)


def build_scene(scene_id, room_polygons, *, geoverify_results=(), da3_pairs=()):
    """Generate candidates from layouts + precomputed model results, without files.

    Does not run DA3/GeoVerify models, enumerate unknown connectivity, read a
    manifest or consume GT poses. Polygon ordering defines stable wall indices.
    """
    scene = Scene(scene_id, room_polygons)
    edges, grouped, index = [], {}, {}
    seen, unavailable = set(), []
    args = argparse.Namespace(**DEFAULTS["guided"])
    for raw in geoverify_results:
        if not isinstance(raw, Mapping):
            raise AssemblyError("INVALID_INPUT", "GeoVerify result must be an object")
        a, b = raw.get("img_a"), raw.get("img_b")
        if not isinstance(a, str) or not isinstance(b, str) or a not in scene.polygons or b not in scene.polygons or a == b:
            raise AssemblyError("INVALID_INPUT", "GeoVerify img_a/img_b must identify two scene rooms")
        key = tuple(sorted((a, b)))
        if key in seen:
            raise AssemblyError("INVALID_INPUT", "duplicate GeoVerify pair, including reverse duplicates", rooms=list(key))
        seen.add(key)
        index[(scene_id, key)] = (Path(f"pair_{len(index):04d}.json"), dict(raw))
        try:
            measurement = build_room_measurement(dict(raw), a, b)
            if not all(measurement.get(k) for k in ("measurement_valid", "rotation_valid", "bearing_valid")):
                unavailable.append({"room_a": a, "room_b": b, "reason": "invalid_geoverify_measurement"})
                continue
            yaw_deg = finite_number(measurement["relative_yaw_deg"], "GeoVerify yaw")
            bearing = np.array(measurement["room_bearing_ab"], dtype=float)
            if bearing.shape != (2,) or not np.isfinite(bearing).all() or np.linalg.norm(bearing) <= 1e-9:
                raise ValueError("invalid bearing")
            bearing /= np.linalg.norm(bearing)
            geo = {"yaw_rad": math.radians(yaw_deg), "yaw_deg": yaw_deg,
                   "bearing": bearing, "bearing_deg": guided.room_bearing_deg(bearing)}
            rows, _, residual, _ = guided.enumerate_pair(scene.polygons[a], scene.polygons[b], geo, a, b, args)
            valid = guided.rank_valid(rows, "candidate_valid", "candidate_score")
        except (KeyError, TypeError, ValueError, IndexError, OverflowError) as exc:
            raise AssemblyError("INVALID_INPUT", "malformed GeoVerify measurement", room_a=a, room_b=b) from exc
        if not valid:
            unavailable.append({"room_a": a, "room_b": b, "reason": "no_valid_guided_candidate"})
            continue
        best = valid[0]
        candidates = [Candidate(candidate_id=row["candidate_id"],
                                measurement=(row["dx"], row["dy"], row["dtheta"]),
                                wall_a_index=int(row["wall_a_id"].rsplit("_", 1)[1]),
                                wall_b_index=int(row["wall_b_id"].rsplit("_", 1)[1]),
                                candidate_score=row["candidate_score"],
                                overlap_ratio=row["shared_wall_overlap_ratio"],
                                polygon_overlap_ratio=row["polygon_overlap_ratio"]) for row in valid]
        low = candidates[0].candidate_score
        span = max(candidates[-1].candidate_score - low, 1e-9)
        for candidate in candidates:
            candidate.intrinsic_cost = (candidate.candidate_score - low) / span
        grouped[(a, b)] = candidates
        edges.append(Edge(index=len(edges), room_a=a, room_b=b,
                          measurement=(best["dx"], best["dy"], best["dtheta"]),
                          selected_candidate_id=best["candidate_id"], yaw_residual_deg=residual,
                          valid_candidate_count=len(valid)))
    try:
        enrich_edges_from_data(scene_id, edges, scene.polygons, grouped, index, _args())
    except (KeyError, TypeError, ValueError, IndexError, OverflowError) as exc:
        raise AssemblyError("INVALID_INPUT", "GeoVerify quality fields are malformed") from exc
    # Choose the starting room before adding layout-only fallback connections.
    root = choose_root(scene.polygons, edges)
    append_da3_edges(edges, scene.polygons, da3_pairs)
    return Scene(scene_id, scene.polygons, edges, root,
                 {"candidate_origin": "generated_from_model_results", "unavailable_pairs": unavailable,
                  "models_executed": False, "layout_provenance": "caller_supplied"})


def choose_root(polygons, edges):
    scores = {room: 0.0 for room in polygons}
    for edge in edges:
        scores[edge.room_a] += edge.confidence
        scores[edge.room_b] += edge.confidence
    return min(scores, key=lambda room: (-scores[room], room))


def append_da3_edges(edges, polygons, pairs):
    existing = {tuple(sorted((edge.room_a, edge.room_b))) for edge in edges}
    next_index = max((edge.index for edge in edges), default=-1) + 1
    for pair in pairs:
        if not isinstance(pair, Mapping) or set(pair) != {"room_a", "room_b", "probability"}:
            raise AssemblyError("INVALID_INPUT", "DA3 pair needs room_a, room_b, probability")
        a, b = pair["room_a"], pair["room_b"]
        if not isinstance(a, str) or not isinstance(b, str) or a not in polygons or b not in polygons or a == b:
            raise AssemblyError("INVALID_INPUT", "DA3 pair references invalid rooms")
        probability = finite_number(pair["probability"], "DA3 probability")
        if not 0 <= probability <= 1:
            raise AssemblyError("INVALID_INPUT", "DA3 probability must be between 0 and 1")
        key = tuple(sorted((a, b)))
        if key in existing or probability < DEFAULTS["interactive"]["da3_suggestion_threshold"]:
            continue
        candidates = generate_layout_suggestion_candidates(a, b, polygons)
        if not candidates:
            continue
        reasons = ["geoverify_pose_missing_or_rejected", "layout_only_pose_ambiguous"]
        if probability < 0.5:
            reasons.append("da3_below_connectivity_threshold")
        confidence = max(0.001, probability * 0.25)
        edges.append(Edge(index=next_index, room_a=a, room_b=b,
                          measurement=candidates[0].measurement, selected_candidate_id=candidates[0].candidate_id,
                          yaw_residual_deg=180.0, valid_candidate_count=len(candidates),
                          confidence=confidence, automatic_confidence=confidence, candidates=candidates,
                          risk_signal_count=len(reasons), risk_tier="high", risk_reasons=reasons,
                          estimated_wrong_rate=1-probability,
                          geoverify_metrics={"da3_probability": probability, "layout_suggestion_count": len(candidates)},
                          manual_action="layout_suggestion_pending",
                          manual_note="DA3 candidate with layout-only pose suggestions; user review required"))
        existing.add(key)
        next_index += 1


def scene_to_dict(scene):
    """Normalized interchange schema. Internal measurements are radians."""
    return {"schema_version": SCHEMA_VERSION, "scene_id": scene.scene_id,
            "units": "layout_units", "angle_unit": "radians", "root_id": scene.root_id,
            "room_polygons": {room: points.tolist() for room, points in scene.polygons.items()},
            "pairs": [asdict(edge) for edge in scene.edges], "metadata": scene.metadata}


def scene_from_dict(data):
    if not isinstance(data, Mapping) or data.get("schema_version") != SCHEMA_VERSION:
        raise AssemblyError("UNSUPPORTED_SCHEMA", "expected scene schema_version 1.0")
    if data.get("units") != "layout_units" or data.get("angle_unit") != "radians":
        raise AssemblyError("INVALID_INPUT", "normalized scene uses layout_units and radians")
    try:
        edges = []
        for item in data["pairs"]:
            values = dict(item)
            values["measurement"] = tuple(values["measurement"])
            values["candidates"] = [Candidate(**{**candidate, "measurement": tuple(candidate["measurement"])})
                                    for candidate in values.get("candidates", [])]
            edges.append(Edge(**values))
        return Scene(data["scene_id"], data["room_polygons"], edges, data.get("root_id"), dict(data.get("metadata", {})))
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, AssemblyError):
            raise
        raise AssemblyError("INVALID_INPUT", "invalid normalized scene") from exc


def load_scene(path):
    return scene_from_dict(_json_read(path))


def save_scene(scene, path):
    _write_json(path, scene_to_dict(scene))
