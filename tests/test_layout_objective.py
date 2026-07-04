import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.layout_objective import evaluate_layout_objective


def square(size=2.0):
    return [[0.0, 0.0], [size, 0.0], [size, size], [0.0, size]]


def base_input():
    return {
        "rooms": {
            "A": {"polygon": square()},
            "B": {"polygon": square()},
        },
        "poses": {
            "A": {"x": 0.0, "y": 0.0, "theta": 0.0},
            "B": {"x": 2.0, "y": 0.0, "theta": 0.0},
        },
        "adjacency_edges": [{"src": "A", "dst": "B"}],
        "door_correspondences": [],
    }


def test_aligned_adjacent_rooms_have_zero_adjacency_overlap_and_wall_scores():
    report = evaluate_layout_objective(base_input())

    assert report["terms"]["overlap"]["score"] == pytest.approx(0.0)
    assert report["terms"]["adjacency"]["score"] == pytest.approx(0.0)
    assert report["terms"]["wall_align"]["score"] == pytest.approx(0.0)


def test_misaligned_parallel_walls_create_wall_alignment_penalty():
    data = base_input()
    data["poses"]["B"] = {"x": 2.2, "y": 0.2, "theta": 0.0}
    data["params"] = {"wall_near_distance": 0.5, "wall_align_tolerance": 0.05}

    report = evaluate_layout_objective(data)

    assert report["terms"]["wall_align"]["score"] > 0.0
    assert report["terms"]["wall_align"]["num_misaligned_pairs"] >= 1


def test_overlapping_rooms_create_overlap_penalty():
    data = base_input()
    data["poses"]["B"] = {"x": 1.0, "y": 0.0, "theta": 0.0}

    report = evaluate_layout_objective(data)

    assert report["terms"]["overlap"]["score"] > 0.0
    assert report["terms"]["overlap"]["details"][0]["overlap_area"] == pytest.approx(2.0)
    assert report["gap_metrics"]["room_overlap_ratio"]["overlap_area"] == pytest.approx(2.0)
    assert report["gap_metrics"]["room_overlap_ratio"]["ratio"] == pytest.approx(0.25)


def test_separated_adjacent_rooms_create_adjacency_distance_penalty():
    data = base_input()
    data["poses"]["B"] = {"x": 3.0, "y": 0.0, "theta": 0.0}
    data["params"] = {"adjacency_distance_threshold": 0.1}

    report = evaluate_layout_objective(data)

    assert report["terms"]["adjacency"]["score"] == pytest.approx(0.81)
    assert report["terms"]["adjacency"]["details"][0]["distance"] == pytest.approx(1.0)


def test_matched_gap_ratio_normalizes_pose_edge_distance_by_tolerance():
    data = base_input()
    data["poses"]["B"] = {"x": 3.0, "y": 0.0, "theta": 0.0}
    data["params"] = {"matched_gap_tolerance": 2.0}

    report = evaluate_layout_objective(data)
    metric = report["gap_metrics"]["matched_gap_ratio"]

    assert metric["ratio"] == pytest.approx(0.5)
    assert metric["details"][0]["distance"] == pytest.approx(1.0)
    assert metric["details"][0]["penalty"] == pytest.approx(0.5)


def test_global_gap_ratio_closing_detects_gap_between_rooms():
    data = base_input()
    data["poses"]["B"] = {"x": 3.0, "y": 0.0, "theta": 0.0}
    data["params"] = {"global_gap_closing_radii": [0.1, 1.0]}

    report = evaluate_layout_objective(data)
    details = report["gap_metrics"]["global_gap_ratio"]["details"]

    assert details[0]["ratio"] == pytest.approx(0.0)
    assert details[1]["ratio"] > 0.0


def test_candidate_wall_gap_ratio_detects_unmatched_parallel_wall_gap():
    data = {
        "rooms": {
            "A": {"polygon": square()},
            "B": {"polygon": square()},
            "C": {"polygon": square()},
        },
        "poses": {
            "A": {"x": 0.0, "y": 0.0, "theta": 0.0},
            "B": {"x": 2.0, "y": 0.0, "theta": 0.0},
            "C": {"x": 0.0, "y": 3.0, "theta": 0.0},
        },
        "pose_edges": [{"src": "A", "dst": "B"}],
        "adjacency_edges": [{"src": "A", "dst": "B"}],
        "door_correspondences": [],
        "params": {
            "candidate_wall_gap_tolerance": 2.0,
            "candidate_wall_max_distance": 2.0,
            "candidate_wall_min_overlap": 0.5,
            "candidate_wall_exclude_matched_pairs": True,
        },
    }

    report = evaluate_layout_objective(data)
    metric = report["gap_metrics"]["candidate_wall_gap_ratio"]

    assert metric["num_candidates"] >= 1
    assert metric["ratio"] > 0.0
    assert all(
        {detail["room_a"], detail["room_b"]} != {"A", "B"}
        for detail in metric["details"]
    )
    assert any(
        {detail["room_a"], detail["room_b"]} == {"A", "C"}
        and detail["distance"] == pytest.approx(1.0)
        for detail in metric["details"]
    )


def test_unexplained_gap_ratio_is_reduced_when_candidate_wall_explains_gap():
    data = {
        "rooms": {
            "A": {"polygon": square()},
            "B": {"polygon": square()},
        },
        "poses": {
            "A": {"x": 0.0, "y": 0.0, "theta": 0.0},
            "B": {"x": 0.0, "y": 3.0, "theta": 0.0},
        },
        "pose_edges": [],
        "adjacency_edges": [],
        "door_correspondences": [],
        "params": {
            "global_gap_closing_radii": [1.0],
            "candidate_wall_gap_tolerance": 2.0,
            "candidate_wall_max_distance": 2.0,
            "candidate_wall_min_overlap": 0.5,
        },
    }

    report = evaluate_layout_objective(data)
    unexplained = report["gap_metrics"]["unexplained_gap_ratio"]["details"][0]

    assert unexplained["gap_area"] > 0.0
    assert unexplained["explained_gap_area"] > 0.0
    assert unexplained["ratio"] < 1.0


def test_mismatched_door_segments_create_door_penalty():
    data = base_input()
    data["door_correspondences"] = [
        {
            "src": "A",
            "dst": "B",
            "src_segment": [[2.0, 0.75], [2.0, 1.25]],
            "dst_segment": [[0.0, 1.25], [0.0, 1.75]],
        }
    ]

    report = evaluate_layout_objective(data)

    assert report["terms"]["door"]["score"] > 0.0
    detail = report["terms"]["door"]["details"][0]
    assert detail["center_distance"] == pytest.approx(0.5)
    assert detail["length_diff"] == pytest.approx(0.0)


def test_cli_writes_report_and_visualization(tmp_path):
    input_path = tmp_path / "layout_input.json"
    out_path = tmp_path / "report.json"
    viz_path = tmp_path / "viz.png"
    input_path.write_text(json.dumps(base_input()), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "src/evaluate_layout_objective.py",
            "--input",
            str(input_path),
            "--out",
            str(out_path),
            "--viz",
            str(viz_path),
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["total_score"] == pytest.approx(0.0)
    assert viz_path.exists()
