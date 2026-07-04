import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from src.system.preview_pipeline import transform_from_pairs
from src.system.room_loader import load_layout_txt_as_local_xy
from src.utils.geom import invert_measurement, rectify_polygon, se2_compose, wrap_pi
from src.utils.geom import load_layout_gt_txt_as_local_xy
from src.utils.post_proc import np_coor2xy


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def assert_pose_close(actual, expected, tol=1e-6):
    assert actual["x"] == pytest.approx(expected["x"], abs=tol)
    assert actual["y"] == pytest.approx(expected["y"], abs=tol)
    assert wrap_pi(actual["theta"] - expected["theta"]) == pytest.approx(0.0, abs=tol)


def format_pose(pose):
    return "x={:.6f}, y={:.6f}, theta={:.6f}".format(
        pose["x"], pose["y"], pose["theta"]
    )


def make_edges(edges):
    return {
        "scene_id": "synthetic",
        "edges": [
            {
                "i": i,
                "j": j,
                "measurement": {"dx": dx, "dy": dy, "dtheta": dtheta},
                "noise_sigma": {"sigma_xy": sigma_xy, "sigma_theta": sigma_theta},
                "meta": {"source": "synthetic"},
            }
            for i, j, dx, dy, dtheta, sigma_xy, sigma_theta in edges
        ],
    }


def run_script(*args):
    return subprocess.run(
        [sys.executable, *args],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )


def run_gtsam_case(tmp_path, edges, poses, max_iters=50, extra_args=None):
    edges_path = tmp_path / "edges.json"
    init_path = tmp_path / "initial_poses.json"
    out_path = tmp_path / "optimized_poses.json"
    report_path = tmp_path / "report.json"

    write_json(edges_path, make_edges(edges))
    write_json(
        init_path,
        {
            "scene_id": "synthetic",
            "root": "A",
            "poses": poses,
        },
    )

    args = [
        "src/03_optimize_pose_graph_gtsam.py",
        "--edges",
        str(edges_path),
        "--init",
        str(init_path),
        "--out",
        str(out_path),
        "--report",
        str(report_path),
        "--lm_max_iters",
        str(max_iters),
    ]
    if extra_args:
        args.extend(extra_args)
    run_script(*args)

    return read_json(out_path), read_json(report_path)


def run_residual_analyzer(tmp_path, edges, before_poses, after_poses):
    edges_path = tmp_path / "edges.json"
    before_path = tmp_path / "before.json"
    after_path = tmp_path / "after.json"
    out_json = tmp_path / "residuals.json"
    out_csv = tmp_path / "residuals.csv"
    summary = tmp_path / "summary.json"

    write_json(edges_path, make_edges(edges))
    write_json(before_path, {"scene_id": "synthetic", "root": "A", "poses": before_poses})
    write_json(after_path, {"scene_id": "synthetic", "root": "A", "poses": after_poses})

    run_script(
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
    )

    return read_json(out_json), read_json(summary), out_csv


def test_layout_loader_matches_canonical_geom_loader(tmp_path):
    layout_txt = tmp_path / "room.txt"
    layout_txt.write_text(
        "\n".join(
            [
                "100 150",
                "100 350",
                "300 150",
                "300 350",
                "500 150",
                "500 350",
                "700 150",
                "700 350",
            ]
        ),
        encoding="utf-8",
    )

    room_loader_poly = load_layout_txt_as_local_xy(
        layout_txt,
        np_coor2xy_func=np_coor2xy,
        rectify_polygon_func=rectify_polygon,
        pano_w=1024,
        pano_h=512,
        layout_z=50.0,
    )
    canonical_poly = load_layout_gt_txt_as_local_xy(
        layout_txt, pano_w=1024, pano_h=512, z=50.0
    )

    assert canonical_poly is not None
    assert np.max(np.abs(room_loader_poly - canonical_poly)) == pytest.approx(
        0.0, abs=1e-9
    )


def test_pair_transform_matches_gtsam_between_convention():
    pose_i = (2.0, -1.0, math.radians(30.0))
    z_ij = (1.25, -0.5, math.radians(40.0))
    pose_j = se2_compose(pose_i, z_ij)

    room_j_local = np.array(
        [[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]], dtype=np.float64
    )

    # z_ij = x_i^-1 * x_j maps j-local coordinates into i-local coordinates.
    room_i_local = np.array([se2_compose(z_ij, (x, y, 0.0))[:2] for x, y in room_j_local])
    pairs = [(1, 1), (2, 2), (3, 3), (4, 4)]

    h_j_to_i = transform_from_pairs(room_i_local, room_j_local, pairs)
    measured = (
        float(h_j_to_i[0, 2]),
        float(h_j_to_i[1, 2]),
        math.atan2(float(h_j_to_i[1, 0]), float(h_j_to_i[0, 0])),
    )
    predicted_pose_j = se2_compose(pose_i, measured)

    assert measured[0] == pytest.approx(z_ij[0], abs=1e-9)
    assert measured[1] == pytest.approx(z_ij[1], abs=1e-9)
    assert wrap_pi(measured[2] - z_ij[2]) == pytest.approx(0.0, abs=1e-9)
    assert predicted_pose_j[0] == pytest.approx(pose_j[0], abs=1e-9)
    assert predicted_pose_j[1] == pytest.approx(pose_j[1], abs=1e-9)
    assert wrap_pi(predicted_pose_j[2] - pose_j[2]) == pytest.approx(0.0, abs=1e-9)


def test_se2_compose_rotates_local_translation_into_world_frame():
    pose_a = (10.0, 20.0, math.pi / 2.0)
    local_measurement_ab = (2.0, 0.0, math.pi / 2.0)

    x, y, theta = se2_compose(pose_a, local_measurement_ab)

    assert x == pytest.approx(10.0, abs=1e-9)
    assert y == pytest.approx(22.0, abs=1e-9)
    assert wrap_pi(theta - math.pi) == pytest.approx(0.0, abs=1e-9)


def test_invert_measurement_round_trips_with_compose():
    measurement_ab = (1.25, -0.5, math.radians(35.0))
    measurement_ba = invert_measurement(measurement_ab)

    identity = se2_compose(measurement_ab, measurement_ba)

    assert identity[0] == pytest.approx(0.0, abs=1e-9)
    assert identity[1] == pytest.approx(0.0, abs=1e-9)
    assert identity[2] == pytest.approx(0.0, abs=1e-9)


def test_bfs_three_node_chain_composes_edges(tmp_path):
    edges_path = tmp_path / "edges.json"
    out_path = tmp_path / "initial_poses.json"
    write_json(
        edges_path,
        make_edges(
            [
                ("A", "B", 1.0, 0.0, math.pi / 2.0, 1.0, 0.1),
                ("B", "C", 2.0, 0.0, 0.0, 1.0, 0.1),
            ]
        ),
    )

    run_script(
        "src/02_init_poses_bfs.py",
        "--edges",
        str(edges_path),
        "--out",
        str(out_path),
        "--root",
        "A",
    )

    poses = read_json(out_path)["poses"]
    assert_pose_close(poses["A"], {"x": 0.0, "y": 0.0, "theta": 0.0})
    assert_pose_close(poses["B"], {"x": 1.0, "y": 0.0, "theta": math.pi / 2.0})
    assert_pose_close(poses["C"], {"x": 1.0, "y": 2.0, "theta": math.pi / 2.0})


def test_bfs_backward_traversal_uses_inverse_transform(tmp_path):
    edges_path = tmp_path / "edges.json"
    out_path = tmp_path / "initial_poses.json"
    write_json(
        edges_path,
        make_edges([("A", "B", 1.0, 0.0, math.pi / 2.0, 1.0, 0.1)]),
    )

    run_script(
        "src/02_init_poses_bfs.py",
        "--edges",
        str(edges_path),
        "--out",
        str(out_path),
        "--root",
        "B",
    )

    poses = read_json(out_path)["poses"]
    assert_pose_close(poses["B"], {"x": 0.0, "y": 0.0, "theta": 0.0})
    assert_pose_close(poses["A"], {"x": 0.0, "y": 1.0, "theta": -math.pi / 2.0})


def test_bfs_loop_edge_does_not_overwrite_existing_tree_pose(tmp_path):
    edges_path = tmp_path / "edges.json"
    out_path = tmp_path / "initial_poses.json"
    write_json(
        edges_path,
        make_edges(
            [
                ("A", "B", 1.0, 0.0, 0.0, 1.0, 0.1),
                ("B", "C", 1.0, 0.0, 0.0, 1.0, 0.1),
                ("A", "C", 10.0, 0.0, 0.0, 10.0, 0.1),
            ]
        ),
    )

    run_script(
        "src/02_init_poses_bfs.py",
        "--edges",
        str(edges_path),
        "--out",
        str(out_path),
        "--root",
        "A",
    )

    data = read_json(out_path)
    assert_pose_close(data["poses"]["C"], {"x": 2.0, "y": 0.0, "theta": 0.0})
    assert data["loop_closures_found"] >= 1


def test_residual_analyzer_reports_zero_for_clean_edge(tmp_path):
    residuals, summary, out_csv = run_residual_analyzer(
        tmp_path,
        edges=[("A", "B", 1.0, 0.0, 0.0, 0.05, 0.02)],
        before_poses={
            "A": {"x": 0.0, "y": 0.0, "theta": 0.0},
            "B": {"x": 1.0, "y": 0.0, "theta": 0.0},
        },
        after_poses={
            "A": {"x": 0.0, "y": 0.0, "theta": 0.0},
            "B": {"x": 1.0, "y": 0.0, "theta": 0.0},
        },
    )

    row = residuals[0]
    assert row["edge_type"] == "unknown"
    assert row["trans_residual_before"] == pytest.approx(0.0, abs=1e-9)
    assert row["trans_residual_after"] == pytest.approx(0.0, abs=1e-9)
    assert summary["all"]["mean_trans_residual_after"] == pytest.approx(0.0, abs=1e-9)
    assert out_csv.exists()


def test_residual_analyzer_identifies_conflicting_loop_residuals(tmp_path):
    residuals, summary, _ = run_residual_analyzer(
        tmp_path,
        edges=[
            ("A", "B", 1.0, 0.0, 0.0, 0.2, 0.2),
            ("B", "C", 1.0, 0.0, 0.0, 0.2, 0.2),
            ("A", "C", 2.8, 0.0, 0.0, 0.2, 0.2),
        ],
        before_poses={
            "A": {"x": 0.0, "y": 0.0, "theta": 0.0},
            "B": {"x": 1.0, "y": 0.0, "theta": 0.0},
            "C": {"x": 2.0, "y": 0.0, "theta": 0.0},
        },
        after_poses={
            "A": {"x": 0.0, "y": 0.0, "theta": 0.0},
            "B": {"x": 1.2666666667, "y": 0.0, "theta": 0.0},
            "C": {"x": 2.5333333333, "y": 0.0, "theta": 0.0},
        },
    )

    by_pair = {(row["src"], row["dst"]): row for row in residuals}
    assert by_pair[("A", "B")]["edge_type"] == "unknown"
    assert by_pair[("B", "C")]["edge_type"] == "unknown"
    assert by_pair[("A", "C")]["edge_type"] == "unknown"
    assert by_pair[("A", "C")]["trans_residual_before"] == pytest.approx(
        0.8, abs=1e-9
    )
    assert by_pair[("A", "C")]["trans_residual_after"] == pytest.approx(
        0.2666666667, abs=1e-6
    )
    assert by_pair[("A", "B")]["trans_residual_after"] == pytest.approx(
        0.2666666667, abs=1e-6
    )
    assert summary["unknown"]["num_edges_worsened"] == 2
    assert summary["unknown"]["num_edges_improved"] == 1


@pytest.mark.skipif(
    pytest.importorskip("importlib.util").find_spec("gtsam") is None,
    reason="gtsam is not installed",
)
def test_gtsam_two_node_clean_edge_recovers_noisy_initial_pose(tmp_path):
    edges_path = tmp_path / "edges.json"
    init_path = tmp_path / "initial_poses.json"
    out_path = tmp_path / "optimized_poses.json"
    report_path = tmp_path / "report.json"

    write_json(
        edges_path,
        make_edges([("A", "B", 1.0, 0.0, 0.0, 0.05, 0.02)]),
    )
    write_json(
        init_path,
        {
            "scene_id": "synthetic",
            "root": "A",
            "poses": {
                "A": {"x": 0.0, "y": 0.0, "theta": 0.0},
                "B": {"x": 2.0, "y": 0.5, "theta": 0.3},
            },
        },
    )

    run_script(
        "src/03_optimize_pose_graph_gtsam.py",
        "--edges",
        str(edges_path),
        "--init",
        str(init_path),
        "--out",
        str(out_path),
        "--report",
        str(report_path),
        "--lm_max_iters",
        "50",
    )

    poses = read_json(out_path)["poses"]
    report = read_json(report_path)
    before_error = report["graph_diagnostics"]["raw_graph_error_before_stage_a"]
    after_error = report["graph_diagnostics"]["raw_graph_error_after_stage_a"]

    print("\n[synthetic gtsam two-node clean edge]")
    print("graph error before: {:.12f}".format(before_error))
    print("graph error after : {:.12f}".format(after_error))
    print("optimized A:", format_pose(poses["A"]))
    print("optimized B:", format_pose(poses["B"]))

    assert_pose_close(poses["A"], {"x": 0.0, "y": 0.0, "theta": 0.0}, tol=1e-5)
    assert_pose_close(poses["B"], {"x": 1.0, "y": 0.0, "theta": 0.0}, tol=1e-4)
    assert after_error < before_error


@pytest.mark.skipif(
    pytest.importorskip("importlib.util").find_spec("gtsam") is None,
    reason="gtsam is not installed",
)
def test_gtsam_conflicting_loop_reduces_error_but_keeps_nonzero_residual(tmp_path):
    edges_path = tmp_path / "edges.json"
    init_path = tmp_path / "initial_poses.json"
    out_path = tmp_path / "optimized_poses.json"
    report_path = tmp_path / "report.json"

    write_json(
        edges_path,
        make_edges(
            [
                ("A", "B", 1.0, 0.0, 0.0, 0.2, 0.2),
                ("B", "C", 1.0, 0.0, 0.0, 0.2, 0.2),
                ("A", "C", 2.8, 0.0, 0.0, 0.2, 0.2),
            ]
        ),
    )
    write_json(
        init_path,
        {
            "scene_id": "synthetic",
            "root": "A",
            "poses": {
                "A": {"x": 0.0, "y": 0.0, "theta": 0.0},
                "B": {"x": 1.0, "y": 0.0, "theta": 0.0},
                "C": {"x": 2.0, "y": 0.0, "theta": 0.0},
            },
        },
    )

    run_script(
        "src/03_optimize_pose_graph_gtsam.py",
        "--edges",
        str(edges_path),
        "--init",
        str(init_path),
        "--out",
        str(out_path),
        "--report",
        str(report_path),
        "--lm_max_iters",
        "50",
    )

    poses = read_json(out_path)["poses"]
    report = read_json(report_path)
    before = report["graph_diagnostics"]["raw_graph_error_before_stage_a"]
    after = report["graph_diagnostics"]["raw_graph_error_after_stage_a"]
    after_rmse = report["standard_edges_after_umeyama_aligned"]["translation"]["rmse"]
    top_edges = report["top_standard_edges_after_by_translation"]

    print("\n[synthetic gtsam conflicting loop]")
    print("graph error before: {:.12f}".format(before))
    print("graph error after : {:.12f}".format(after))
    print("translation residual rmse after: {:.12f}".format(after_rmse))
    print("optimized A:", format_pose(poses["A"]))
    print("optimized B:", format_pose(poses["B"]))
    print("optimized C:", format_pose(poses["C"]))
    print("top residual edges after:")
    for edge in top_edges:
        print(
            "  {} -> {}: trans_l2={:.6f}, rot_abs={:.6f}, residual=({:.6f}, {:.6f}, {:.6f})".format(
                edge["i"],
                edge["j"],
                edge["trans_l2"],
                edge["rot_abs"],
                edge["residual"]["dx"],
                edge["residual"]["dy"],
                edge["residual"]["dtheta"],
            )
        )

    assert after < before
    assert after_rmse > 0.01


@pytest.mark.skipif(
    pytest.importorskip("importlib.util").find_spec("gtsam") is None,
    reason="gtsam is not installed",
)
def test_gtsam_three_node_chain_recovers_global_poses_from_noisy_initial(tmp_path):
    out, report = run_gtsam_case(
        tmp_path,
        edges=[
            ("A", "B", 1.0, 0.0, 0.0, 0.05, 0.02),
            ("B", "C", 1.0, 0.0, 0.0, 0.05, 0.02),
        ],
        poses={
            "A": {"x": 0.0, "y": 0.0, "theta": 0.0},
            "B": {"x": 2.0, "y": 0.5, "theta": 0.2},
            "C": {"x": 4.0, "y": -0.5, "theta": -0.3},
        },
    )

    poses = out["poses"]
    before = report["graph_diagnostics"]["raw_graph_error_before_stage_a"]
    after = report["graph_diagnostics"]["raw_graph_error_after_stage_a"]

    print("\n[synthetic gtsam three-node chain]")
    print("graph error before: {:.12f}".format(before))
    print("graph error after : {:.12f}".format(after))
    for node_id in ["A", "B", "C"]:
        print("optimized {}: {}".format(node_id, format_pose(poses[node_id])))

    assert_pose_close(poses["A"], {"x": 0.0, "y": 0.0, "theta": 0.0}, tol=1e-5)
    assert_pose_close(poses["B"], {"x": 1.0, "y": 0.0, "theta": 0.0}, tol=1e-4)
    assert_pose_close(poses["C"], {"x": 2.0, "y": 0.0, "theta": 0.0}, tol=1e-4)
    assert after < before


@pytest.mark.skipif(
    pytest.importorskip("importlib.util").find_spec("gtsam") is None,
    reason="gtsam is not installed",
)
def test_gtsam_rotation_edge_uses_local_frame_translation_convention(tmp_path):
    out, report = run_gtsam_case(
        tmp_path,
        edges=[
            ("A", "B", 0.0, 0.0, math.pi / 2.0, 0.05, 0.02),
            ("B", "C", 1.0, 0.0, 0.0, 0.05, 0.02),
        ],
        poses={
            "A": {"x": 0.0, "y": 0.0, "theta": 0.0},
            "B": {"x": 0.5, "y": 0.5, "theta": 0.3},
            "C": {"x": 2.0, "y": 0.5, "theta": -0.2},
        },
    )

    poses = out["poses"]
    before = report["graph_diagnostics"]["raw_graph_error_before_stage_a"]
    after = report["graph_diagnostics"]["raw_graph_error_after_stage_a"]

    print("\n[synthetic gtsam rotation edge]")
    print("graph error before: {:.12f}".format(before))
    print("graph error after : {:.12f}".format(after))
    for node_id in ["A", "B", "C"]:
        print("optimized {}: {}".format(node_id, format_pose(poses[node_id])))

    assert_pose_close(poses["A"], {"x": 0.0, "y": 0.0, "theta": 0.0}, tol=1e-5)
    assert_pose_close(poses["B"], {"x": 0.0, "y": 0.0, "theta": math.pi / 2.0}, tol=1e-4)
    assert_pose_close(poses["C"], {"x": 0.0, "y": 1.0, "theta": math.pi / 2.0}, tol=1e-4)
    assert after < before


@pytest.mark.skipif(
    pytest.importorskip("importlib.util").find_spec("gtsam") is None,
    reason="gtsam is not installed",
)
def test_gtsam_square_loop_clean_case_has_near_zero_residual(tmp_path):
    out, report = run_gtsam_case(
        tmp_path,
        edges=[
            ("A", "B", 1.0, 0.0, 0.0, 0.05, 0.02),
            ("B", "C", 0.0, 1.0, 0.0, 0.05, 0.02),
            ("C", "D", -1.0, 0.0, 0.0, 0.05, 0.02),
            ("D", "A", 0.0, -1.0, 0.0, 0.05, 0.02),
        ],
        poses={
            "A": {"x": 0.0, "y": 0.0, "theta": 0.0},
            "B": {"x": 1.2, "y": -0.2, "theta": 0.1},
            "C": {"x": 0.8, "y": 1.3, "theta": -0.1},
            "D": {"x": -0.3, "y": 0.8, "theta": 0.2},
        },
    )

    poses = out["poses"]
    after = report["graph_diagnostics"]["raw_graph_error_after_stage_a"]
    after_rmse = report["standard_edges_after_umeyama_aligned"]["translation"]["rmse"]

    print("\n[synthetic gtsam square loop clean]")
    print("graph error after: {:.12f}".format(after))
    print("translation residual rmse after: {:.12f}".format(after_rmse))
    for node_id in ["A", "B", "C", "D"]:
        print("optimized {}: {}".format(node_id, format_pose(poses[node_id])))

    assert_pose_close(poses["A"], {"x": 0.0, "y": 0.0, "theta": 0.0}, tol=1e-5)
    assert_pose_close(poses["B"], {"x": 1.0, "y": 0.0, "theta": 0.0}, tol=1e-4)
    assert_pose_close(poses["C"], {"x": 1.0, "y": 1.0, "theta": 0.0}, tol=1e-4)
    assert_pose_close(poses["D"], {"x": 0.0, "y": 1.0, "theta": 0.0}, tol=1e-4)
    assert after < 1e-8
    assert after_rmse < 1e-6


def run_conflicting_loop_with_loop_sigma(tmp_path, loop_sigma):
    out, report = run_gtsam_case(
        tmp_path,
        edges=[
            ("A", "B", 1.0, 0.0, 0.0, 0.05, 0.02),
            ("B", "C", 1.0, 0.0, 0.0, 0.05, 0.02),
            ("A", "C", 2.8, 0.0, 0.0, loop_sigma, 0.02),
        ],
        poses={
            "A": {"x": 0.0, "y": 0.0, "theta": 0.0},
            "B": {"x": 1.0, "y": 0.0, "theta": 0.0},
            "C": {"x": 2.0, "y": 0.0, "theta": 0.0},
        },
    )

    per_edge = report["top_standard_edges_after_by_translation"]
    edge_residuals = {
        (edge["i"], edge["j"]): edge["trans_l2"]
        for edge in per_edge
    }
    tree_residual = max(edge_residuals[("A", "B")], edge_residuals[("B", "C")])
    loop_residual = edge_residuals[("A", "C")]
    return out, report, tree_residual, loop_residual


@pytest.mark.skipif(
    pytest.importorskip("importlib.util").find_spec("gtsam") is None,
    reason="gtsam is not installed",
)
def test_gtsam_different_sigma_weighting_changes_tree_and_loop_residuals(tmp_path):
    _, strong_report, strong_tree, strong_loop = run_conflicting_loop_with_loop_sigma(
        tmp_path / "strong", loop_sigma=0.05
    )
    _, medium_report, medium_tree, medium_loop = run_conflicting_loop_with_loop_sigma(
        tmp_path / "medium", loop_sigma=0.2
    )
    _, weak_report, weak_tree, weak_loop = run_conflicting_loop_with_loop_sigma(
        tmp_path / "weak", loop_sigma=1.0
    )

    print("\n[synthetic gtsam sigma weighting]")
    for label, report, tree_residual, loop_residual in [
        ("loop sigma 0.05", strong_report, strong_tree, strong_loop),
        ("loop sigma 0.20", medium_report, medium_tree, medium_loop),
        ("loop sigma 1.00", weak_report, weak_tree, weak_loop),
    ]:
        after = report["graph_diagnostics"]["raw_graph_error_after_stage_a"]
        print(
            "{}: error_after={:.12f}, tree_residual={:.6f}, loop_residual={:.6f}".format(
                label, after, tree_residual, loop_residual
            )
        )

    assert strong_tree > medium_tree > weak_tree
    assert strong_loop < medium_loop < weak_loop


@pytest.mark.skipif(
    pytest.importorskip("importlib.util").find_spec("gtsam") is None,
    reason="gtsam is not installed",
)
def test_gtsam_root_prior_stability_changes_with_prior_strength(tmp_path):
    common_edges = [
        ("A", "B", 1.0, 0.0, 0.0, 0.2, 0.2),
        ("B", "C", 1.0, 0.0, 0.0, 0.2, 0.2),
        ("A", "C", 2.8, 0.0, 0.0, 0.2, 0.2),
    ]
    initial_poses = {
        "A": {"x": 0.0, "y": 0.0, "theta": 0.0},
        "B": {"x": 1.0, "y": 0.0, "theta": 0.0},
        "C": {"x": 2.0, "y": 0.0, "theta": 0.0},
    }

    strong_out, _ = run_gtsam_case(
        tmp_path / "strong_prior",
        common_edges,
        initial_poses,
        extra_args=["--prior_sigma_xy", "0.001", "--prior_sigma_theta_deg", "0.001"],
    )
    weak_out, _ = run_gtsam_case(
        tmp_path / "weak_prior",
        common_edges,
        initial_poses,
        extra_args=["--prior_sigma_xy", "10.0", "--prior_sigma_theta_deg", "90.0"],
    )

    strong_root = strong_out["poses"]["A"]
    weak_root = weak_out["poses"]["A"]
    strong_root_move = math.hypot(strong_root["x"], strong_root["y"])
    weak_root_move = math.hypot(weak_root["x"], weak_root["y"])

    print("\n[synthetic gtsam root prior stability]")
    print("strong prior root:", format_pose(strong_root))
    print("weak prior root  :", format_pose(weak_root))
    print("strong root translation drift: {:.12f}".format(strong_root_move))
    print("weak root translation drift  : {:.12f}".format(weak_root_move))

    assert strong_root_move < 1e-4
    assert abs(wrap_pi(strong_root["theta"])) < 1e-4
    assert weak_root_move > strong_root_move
