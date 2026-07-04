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


def run_analyzer(tmp_path, edges, before_poses, after_poses):
    edges_path = tmp_path / "edges.json"
    before_path = tmp_path / "before.json"
    after_path = tmp_path / "after.json"
    out_json = tmp_path / "edge_residuals.json"
    out_csv = tmp_path / "edge_residuals.csv"
    summary = tmp_path / "summary.json"

    write_json(edges_path, edges)
    write_json(before_path, before_poses)
    write_json(after_path, after_poses)

    subprocess.run(
        [
            sys.executable,
            "src/analyze_edge_residuals.py",
            "--edges",
            str(edges_path),
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


def test_clean_edge_residual_is_zero(tmp_path):
    rows, summary, out_csv = run_analyzer(
        tmp_path,
        edges=[
            {
                "src": "A",
                "dst": "B",
                "dx": 1.0,
                "dy": 0.0,
                "dtheta": 0.0,
                "edge_type": "tree",
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
    assert row["edge_type"] == "tree"
    assert row["trans_residual_before"] == pytest.approx(0.0, abs=1e-9)
    assert row["rot_residual_before"] == pytest.approx(0.0, abs=1e-9)
    assert row["trans_residual_after"] == pytest.approx(0.0, abs=1e-9)
    assert row["rot_residual_after"] == pytest.approx(0.0, abs=1e-9)
    assert summary["tree"]["num_edges"] == 1
    assert out_csv.exists()


def test_noisy_before_and_clean_after_is_improved(tmp_path):
    rows, summary, _ = run_analyzer(
        tmp_path,
        edges={"edges": [{"src": "A", "dst": "B", "dx": 1.0, "dy": 0.0, "dtheta": 0.0}]},
        before_poses={
            "A": {"x": 0.0, "y": 0.0, "theta": 0.0},
            "B": {"x": 2.0, "y": 0.5, "theta": 0.3},
        },
        after_poses={
            "A": {"x": 0.0, "y": 0.0, "theta": 0.0},
            "B": {"x": 1.0, "y": 0.0, "theta": 0.0},
        },
    )

    row = rows[0]
    assert row["trans_residual_before"] > 1.0
    assert row["rot_residual_before"] == pytest.approx(0.3, abs=1e-9)
    assert row["trans_residual_after"] == pytest.approx(0.0, abs=1e-9)
    assert row["rot_residual_after"] == pytest.approx(0.0, abs=1e-9)
    assert row["improved"] is True
    assert row["worsened"] is False
    assert summary["unknown"]["num_edges_improved"] == 1


def test_rotation_edge_uses_local_frame_relative_pose(tmp_path):
    rows, _, _ = run_analyzer(
        tmp_path,
        edges={
            "edges": [
                {"src": "A", "dst": "B", "dx": 0.0, "dy": 0.0, "dtheta": math.pi / 2},
                {"src": "B", "dst": "C", "dx": 1.0, "dy": 0.0, "dtheta": 0.0},
            ]
        },
        before_poses={
            "A": {"x": 0.0, "y": 0.0, "theta": 0.0},
            "B": {"x": 0.0, "y": 0.0, "theta": math.pi / 2},
            "C": {"x": 0.0, "y": 1.0, "theta": math.pi / 2},
        },
        after_poses={
            "A": {"x": 0.0, "y": 0.0, "theta": 0.0},
            "B": {"x": 0.0, "y": 0.0, "theta": math.pi / 2},
            "C": {"x": 0.0, "y": 1.0, "theta": math.pi / 2},
        },
    )

    bc = rows[1]
    assert bc["dx_pred_before"] == pytest.approx(1.0, abs=1e-9)
    assert bc["dy_pred_before"] == pytest.approx(0.0, abs=1e-9)
    assert bc["dtheta_pred_before"] == pytest.approx(0.0, abs=1e-9)
    assert bc["trans_residual_before"] == pytest.approx(0.0, abs=1e-9)


def test_angle_wrapping_near_pi_boundary(tmp_path):
    eps = 1e-6
    rows, _, _ = run_analyzer(
        tmp_path,
        edges=[{"src": "A", "dst": "B", "dx": 0.0, "dy": 0.0, "dtheta": -math.pi + eps}],
        before_poses={
            "A": {"x": 0.0, "y": 0.0, "theta": 0.0},
            "B": {"x": 0.0, "y": 0.0, "theta": math.pi - eps},
        },
        after_poses={
            "A": {"x": 0.0, "y": 0.0, "theta": 0.0},
            "B": {"x": 0.0, "y": 0.0, "theta": math.pi - eps},
        },
    )

    assert rows[0]["rot_residual_before"] == pytest.approx(2e-6, abs=1e-9)
    assert abs(rows[0]["dtheta_error_before"]) < 1e-5


def test_group_summary_contains_tree_loop_and_unknown(tmp_path):
    rows, summary, _ = run_analyzer(
        tmp_path,
        edges=[
            {"src": "A", "dst": "B", "dx": 1.0, "dy": 0.0, "dtheta": 0.0, "edge_type": "tree"},
            {"src": "B", "dst": "C", "dx": 1.0, "dy": 0.0, "dtheta": 0.0, "type": "loop"},
            {"src": "A", "dst": "C", "dx": 2.0, "dy": 0.0, "dtheta": 0.0},
        ],
        before_poses={
            "A": {"x": 0.0, "y": 0.0, "theta": 0.0},
            "B": {"x": 1.0, "y": 0.0, "theta": 0.0},
            "C": {"x": 2.0, "y": 0.0, "theta": 0.0},
        },
        after_poses={
            "A": {"x": 0.0, "y": 0.0, "theta": 0.0},
            "B": {"x": 1.0, "y": 0.0, "theta": 0.0},
            "C": {"x": 2.0, "y": 0.0, "theta": 0.0},
        },
    )

    assert set(summary.keys()) == {"all", "tree", "loop", "unknown"}
    assert summary["all"]["num_edges"] == 3
    assert summary["tree"]["num_edges"] == 1
    assert summary["loop"]["num_edges"] == 1
    assert summary["unknown"]["num_edges"] == 1
    assert [row["edge_type"] for row in rows] == ["tree", "loop", "unknown"]
