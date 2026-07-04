import json
import math
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def run_analyzer(tmp_path, constraints, before_poses, after_poses):
    scene_dir = tmp_path / "scene"
    constraints_path = tmp_path / "constraints.json"
    before_path = tmp_path / "before.json"
    after_path = tmp_path / "after.json"
    out_json = tmp_path / "candidate_residuals.json"
    out_csv = tmp_path / "candidate_residuals.csv"
    summary = tmp_path / "summary.json"

    write_json(constraints_path, constraints)
    write_json(before_path, before_poses)
    write_json(after_path, after_poses)

    subprocess.run(
        [
            sys.executable,
            "src/analyze_candidate_residuals.py",
            "--scene_dir",
            str(scene_dir),
            "--constraints",
            str(constraints_path),
            "--before",
            str(before_path),
            "--after",
            str(after_path),
            "--out_json",
            str(out_json),
            "--out_csv",
            str(out_csv),
            "--summary",
            str(summary),
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return read_json(out_json), read_json(summary), out_csv


def test_clean_free_point_candidate_residual_is_zero(tmp_path):
    rows, summary, out_csv = run_analyzer(
        tmp_path,
        constraints=[
            {
                "src": "A.txt",
                "dst": "B.txt",
                "constraint_type": "single_point_adjacency",
                "point_mode": "free",
                "point_pairs": [{"src_xy": [1.0, 0.0], "dst_xy": [0.0, 0.0]}],
            }
        ],
        before_poses={
            "A": {"x": 0.0, "y": 0.0, "theta": 0.0},
            "B": {"x": 1.0, "y": 0.0, "theta": 0.0},
        },
        after_poses={
            "A": {"x": 0.0, "y": 0.0, "theta": 0.0},
            "B": {"x": 1.0, "y": 0.0, "theta": 0.0},
        },
    )

    row = rows[0]
    assert row["mean_point_residual_before"] == pytest.approx(0.0, abs=1e-9)
    assert row["rmse_point_residual_after"] == pytest.approx(0.0, abs=1e-9)
    assert row["segment_len_diff_before"] is None
    assert summary["single_point_adjacency"]["num_constraints"] == 1
    assert out_csv.exists()


def test_noisy_before_clean_after_is_improved(tmp_path):
    rows, summary, _ = run_analyzer(
        tmp_path,
        constraints=[
            {
                "src": "A",
                "dst": "B",
                "constraint_type": "wall_alignment",
                "point_pairs": [
                    {"src_xy": [0.0, 0.0], "dst_xy": [0.0, 0.0]},
                    {"src_xy": [1.0, 0.0], "dst_xy": [1.0, 0.0]},
                ],
            }
        ],
        before_poses={
            "A": {"x": 0.0, "y": 0.0, "theta": 0.0},
            "B": {"x": 2.0, "y": 0.0, "theta": 0.0},
        },
        after_poses={
            "A": {"x": 0.0, "y": 0.0, "theta": 0.0},
            "B": {"x": 0.0, "y": 0.0, "theta": 0.0},
        },
    )

    row = rows[0]
    assert row["rmse_point_residual_before"] == pytest.approx(2.0, abs=1e-9)
    assert row["rmse_point_residual_after"] == pytest.approx(0.0, abs=1e-9)
    assert row["segment_len_diff_after"] == pytest.approx(0.0, abs=1e-9)
    assert row["segment_angle_diff_after"] == pytest.approx(0.0, abs=1e-9)
    assert row["improved"] is True
    assert summary["wall_alignment"]["num_improved"] == 1


def test_wall_segment_angle_uses_undirected_difference(tmp_path):
    rows, _, _ = run_analyzer(
        tmp_path,
        constraints=[
            {
                "src": "A",
                "dst": "B",
                "constraint_type": "wall_alignment",
                "point_pairs": [
                    {"src_xy": [0.0, 0.0], "dst_xy": [1.0, 0.0]},
                    {"src_xy": [2.0, 0.0], "dst_xy": [-1.0, 0.0]},
                ],
            }
        ],
        before_poses={
            "A": {"x": 0.0, "y": 0.0, "theta": 0.0},
            "B": {"x": 0.0, "y": 0.0, "theta": 0.0},
        },
        after_poses={
            "A": {"x": 0.0, "y": 0.0, "theta": 0.0},
            "B": {"x": 0.0, "y": 0.0, "theta": 0.0},
        },
    )

    row = rows[0]
    assert row["segment_len_src_before"] == pytest.approx(2.0, abs=1e-9)
    assert row["segment_len_dst_before"] == pytest.approx(2.0, abs=1e-9)
    assert row["segment_angle_diff_before"] == pytest.approx(0.0, abs=1e-9)


def test_summary_contains_default_groups(tmp_path):
    _, summary, _ = run_analyzer(
        tmp_path,
        constraints=[
            {
                "src": "A",
                "dst": "B",
                "constraint_type": "candidate",
                "point_pairs": [{"src_xy": [0.0, 0.0], "dst_xy": [0.0, 0.0]}],
            }
        ],
        before_poses={
            "A": {"x": 0.0, "y": 0.0, "theta": 0.0},
            "B": {"x": 0.0, "y": 0.0, "theta": 0.0},
        },
        after_poses={
            "A": {"x": 0.0, "y": 0.0, "theta": 0.0},
            "B": {"x": 0.0, "y": 0.0, "theta": 0.0},
        },
    )

    assert set(summary) == {
        "all",
        "wall_alignment",
        "structural_adjacency",
        "single_point_adjacency",
        "connectivity",
        "candidate",
        "unknown",
    }
    assert summary["candidate"]["num_constraints"] == 1
    assert summary["all"]["num_constraints"] == 1
