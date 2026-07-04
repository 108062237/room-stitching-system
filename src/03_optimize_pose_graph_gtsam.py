#!/usr/bin/env python3
"""
Step 3: Pose Graph Optimization (PGO) using GTSAM Pose2.

Extended version:
1. Stage A: optimize using normal pose edges (e.g. perfect_edges.json)
2. Stage B: optionally add extra point coincidence constraints as SOFT pseudo-edges
   derived from the Stage A baseline relative angle.

This is a pragmatic MVP:
- does NOT require writing a custom GTSAM factor yet
- lets you test whether extra cross-room point constraints help

Usage:
  python src/03_optimize_pose_graph_gtsam.py \
    --edges  data/group/58472_Floor1/perfect_edges.json \
    --init   data/group/58472_Floor1/initial_poses.json \
    --out    data/group/58472_Floor1/perfect_plus_points_poses.json \
    --report data/group/58472_Floor1/perfect_plus_points_report.json \
    --extra_points data/group/58472_Floor1/extra_point_constraints.json \
    --layout_dir data/group/58472_Floor1/layout_gt \
    --point_sigma_xy 0.5 \
    --point_sigma_theta 0.5
"""

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import gtsam

sys.path.append(str(Path(__file__).parent.parent))
from src.utils.geom import wrap_pi, rectify_polygon


def load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text())


def save_json(p: Path, data: Dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def pose_dict_to_pose2(d: Dict[str, float]) -> gtsam.Pose2:
    return gtsam.Pose2(float(d["x"]), float(d["y"]), float(d["theta"]))


def pose2_to_dict(p: gtsam.Pose2) -> Dict[str, float]:
    return {"x": float(p.x()), "y": float(p.y()), "theta": float(p.theta())}


def make_symbol_map(node_ids: List[str]) -> Dict[str, int]:
    node_ids_sorted = sorted(node_ids)
    return {nid: k for k, nid in enumerate(node_ids_sorted)}


def build_noise_model(
    sigma_xy: float, sigma_theta: float, use_robust: bool, huber_k: float
):
    base = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([sigma_xy, sigma_xy, sigma_theta], dtype=np.float64)
    )
    if not use_robust:
        return base
    huber = gtsam.noiseModel.mEstimator.Huber.Create(huber_k)
    return gtsam.noiseModel.Robust.Create(huber, base)


def relative_error_pose2(
    pi: gtsam.Pose2, pj: gtsam.Pose2, measured_relative_pose: gtsam.Pose2
) -> Tuple[float, float, float]:
    pred = pi.between(pj)
    err = measured_relative_pose.between(pred)
    return float(err.x()), float(err.y()), float(wrap_pi(err.theta()))


def align_umeyama(
    moving: Dict[str, gtsam.Pose2], fixed: Dict[str, gtsam.Pose2]
) -> Dict[str, gtsam.Pose2]:
    """
    Align `moving` to `fixed` in SE(2) using Umeyama-like rigid alignment on XY only.
    This helps keep evaluation focused on structural deformation instead of global drift.
    """
    common_keys = list(set(moving.keys()) & set(fixed.keys()))
    if len(common_keys) < 2:
        if len(common_keys) == 1:
            k = common_keys[0]
            T_align = fixed[k].between(moving[k]).inverse()
            return {nid: T_align.compose(p) for nid, p in moving.items()}
        return moving

    P = np.array([[fixed[k].x(), fixed[k].y()] for k in common_keys]).T
    Q = np.array([[moving[k].x(), moving[k].y()] for k in common_keys]).T

    mu_P = np.mean(P, axis=1, keepdims=True)
    mu_Q = np.mean(Q, axis=1, keepdims=True)

    P_centered = P - mu_P
    Q_centered = Q - mu_Q

    H = Q_centered @ P_centered.T
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = mu_P - R @ mu_Q
    theta = math.atan2(R[1, 0], R[0, 0])

    T_align = gtsam.Pose2(float(t[0, 0]), float(t[1, 0]), float(theta))
    return {nid: T_align.compose(p) for nid, p in moving.items()}


def compute_residual_stats(
    poses: Dict[str, gtsam.Pose2],
    edges: List[Dict[str, Any]],
    initial_poses: Optional[Dict[str, gtsam.Pose2]] = None,
) -> Dict[str, Any]:
    eval_poses = poses
    if initial_poses is not None:
        eval_poses = align_umeyama(poses, initial_poses)

    per_edge = []
    trans_errs = []
    rot_errs = []

    for e in edges:
        i, j = e["i"], e["j"]
        z = e["measurement"]
        measured_relative_pose = gtsam.Pose2(
            float(z["dx"]), float(z["dy"]), float(z["dtheta"])
        )

        pi = eval_poses[i]
        pj = eval_poses[j]
        ex, ey, eth = relative_error_pose2(pi, pj, measured_relative_pose)

        trans = math.sqrt(ex * ex + ey * ey)
        rot = abs(eth)

        per_edge.append(
            {
                "i": i,
                "j": j,
                "residual": {"dx": ex, "dy": ey, "dtheta": eth},
                "trans_l2": trans,
                "rot_abs": rot,
                "meta": e.get("meta", {}),
            }
        )
        trans_errs.append(trans)
        rot_errs.append(rot)

    trans_arr = np.array(trans_errs, dtype=np.float64) if trans_errs else np.zeros((0,))
    rot_arr = np.array(rot_errs, dtype=np.float64) if rot_errs else np.zeros((0,))

    def safe_stats(arr: np.ndarray):
        if arr.size == 0:
            return {"rmse": 0.0, "mean": 0.0, "median": 0.0, "max": 0.0}
        return {
            "rmse": float(np.sqrt(np.mean(arr**2))),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "max": float(np.max(arr)),
        }

    return {
        "translation": safe_stats(trans_arr),
        "rotation_rad": safe_stats(rot_arr),
        "per_edge": per_edge,
    }


def compute_pose_deltas(
    poses_before: Dict[str, gtsam.Pose2], poses_after: Dict[str, gtsam.Pose2]
) -> Dict[str, Any]:
    per_node = []
    trans_norms = []
    rot_abs_vals = []

    common_ids = sorted(set(poses_before.keys()) & set(poses_after.keys()))
    for nid in common_ids:
        p0 = poses_before[nid]
        p1 = poses_after[nid]

        delta = p0.between(p1)
        dx = float(delta.x())
        dy = float(delta.y())
        dtheta = float(wrap_pi(delta.theta()))
        trans = math.sqrt(dx * dx + dy * dy)

        per_node.append(
            {
                "node_id": nid,
                "delta": {"dx": dx, "dy": dy, "dtheta": dtheta},
                "trans_l2": trans,
                "rot_abs": abs(dtheta),
            }
        )

        trans_norms.append(trans)
        rot_abs_vals.append(abs(dtheta))

    trans_arr = (
        np.array(trans_norms, dtype=np.float64) if trans_norms else np.zeros((0,))
    )
    rot_arr = (
        np.array(rot_abs_vals, dtype=np.float64) if rot_abs_vals else np.zeros((0,))
    )

    def safe_stats(arr: np.ndarray):
        if arr.size == 0:
            return {"rmse": 0.0, "mean": 0.0, "median": 0.0, "max": 0.0}
        return {
            "rmse": float(np.sqrt(np.mean(arr**2))),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "max": float(np.max(arr)),
        }

    return {
        "translation": safe_stats(trans_arr),
        "rotation_rad": safe_stats(rot_arr),
        "per_node": per_node,
    }


# ---------------------------------------------------------------------
# Layout loader for extra point constraints
# Must match the same projection convention as edge generation / drawing.
# ---------------------------------------------------------------------
def load_layout_txt(
    txt_path: Path, pano_w: int = 1024, pano_h: int = 512, z: float = 50.0
) -> np.ndarray:
    """
    Same convention as tool_generate_gtsam_edges.py / draw_floorplan_overlay.py:
    np_coor2xy + center shift + Y flip + rectify_polygon
    """
    _LAYOUTHUB = Path(__file__).resolve().parent.parent.parent / "LayoutHub"
    _POST_PROC_PY = _LAYOUTHUB / "utils" / "post_proc.py"

    if not _POST_PROC_PY.exists():
        raise RuntimeError("Cannot find LayoutHub/utils/post_proc.py for np_coor2xy")

    spec = importlib.util.spec_from_file_location("post_proc", str(_POST_PROC_PY))
    post_proc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(post_proc)
    np_coor2xy = post_proc.np_coor2xy

    pts = []
    for line in txt_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) >= 2:
            pts.append([float(parts[0]), float(parts[1])])

    if len(pts) < 2:
        raise ValueError("TXT point count too small: {}".format(txt_path))

    if len(pts) % 2 == 0:
        floor_pixel = [
            pts[i] if pts[i][1] > pts[i + 1][1] else pts[i + 1]
            for i in range(0, len(pts), 2)
        ]
    else:
        floor_pixel = pts

    floor_pixel = np.array(floor_pixel, dtype=np.float64)
    floor_xy = np_coor2xy(
        floor_pixel, z=z, coorW=pano_w, coorH=pano_h, floorW=pano_w, floorH=pano_w
    )

    center = pano_w / 2 - 0.5
    floor_xy[:, 0] -= center
    floor_xy[:, 1] -= center
    floor_xy[:, 1] = -floor_xy[:, 1]

    try:
        floor_xy = rectify_polygon(floor_xy)
    except Exception:
        pass

    return floor_xy.astype(np.float64)


def normalize_extra_point_constraints(raw_constraints: Any) -> List[Dict[str, Any]]:
    """
    Supports either:
    [
      {"src": "...txt", "dst": "...txt", "pairs": [[5,5],[4,3]]}
    ]
    or free local-coordinate pairs from floorplan_constraint_tool.py:
    [
      {
        "src": "...txt",
        "dst": "...txt",
        "point_pairs": [
          {"src_xy": [1.0, 2.0], "dst_xy": [3.0, 4.0]}
        ]
      }
    ]
    or:
    [
      {"src": "...txt", "dst": "...txt", "idx_src": [5], "idx_dst": [5]}
    ]
    """
    if isinstance(raw_constraints, dict):
        if "constraints" in raw_constraints:
            raw_constraints = raw_constraints["constraints"]
        else:
            raise RuntimeError(
                "extra_points json must be a list or a dict with key 'constraints'"
            )

    out = []
    for item in raw_constraints:
        src = item["src"].replace(".txt", "")
        dst = item["dst"].replace(".txt", "")

        if "point_pairs" in item:
            point_pairs = []
            for pair in item["point_pairs"]:
                point_pairs.append(
                    {
                        "src_xy": [float(pair["src_xy"][0]), float(pair["src_xy"][1])],
                        "dst_xy": [float(pair["dst_xy"][0]), float(pair["dst_xy"][1])],
                    }
                )
            out.append({"src": src, "dst": dst, "point_pairs": point_pairs})
            continue

        if "pairs" in item:
            pairs = item["pairs"]
        else:
            idx_src = item.get("idx_src", [])
            idx_dst = item.get("idx_dst", [])
            if len(idx_src) != len(idx_dst):
                raise RuntimeError(
                    "extra point constraint idx_src / idx_dst length mismatch"
                )
            pairs = [[s, d] for s, d in zip(idx_src, idx_dst)]

        out.append({"src": src, "dst": dst, "pairs": pairs})
    return out


def point_pair_to_soft_pose_measurement(
    p_i: np.ndarray, p_j: np.ndarray, dtheta: float
) -> gtsam.Pose2:
    """
    Given a single point correspondence and a fixed relative angle dtheta,
    construct the translation (dx,dy) that would align point j onto point i.
    """
    c, s = math.cos(dtheta), math.sin(dtheta)
    rx = c * p_j[0] - s * p_j[1]
    ry = s * p_j[0] + c * p_j[1]
    dx = p_i[0] - rx
    dy = p_i[1] - ry
    return gtsam.Pose2(float(dx), float(dy), float(dtheta))


def add_standard_pose_edges(
    graph: gtsam.NonlinearFactorGraph,
    initial: gtsam.Values,
    edges: List[Dict[str, Any]],
    id_to_idx: Dict[str, int],
    th_map: Dict[str, float],
    use_robust: bool,
    huber_k_arg: Optional[float],
) -> Tuple[int, List[Dict[str, Any]]]:
    """
    Add the normal BetweenFactorPose2 edges.
    Returns:
      missing_nodes_count, actual_edges_used
    """
    missing_nodes = 0
    actual_edges_used = []

    for e in edges:
        i, j = e["i"], e["j"]

        if i not in id_to_idx or j not in id_to_idx:
            missing_nodes += 1
            continue

        key_i = gtsam.symbol("x", id_to_idx[i])
        key_j = gtsam.symbol("x", id_to_idx[j])

        if not initial.exists(key_i) or not initial.exists(key_j):
            missing_nodes += 1
            print(
                "[WARNING] Edge {}->{} omitted: target missing from initialization scope.".format(
                    i, j
                )
            )
            continue

        meas = e["measurement"]
        measured_relative_pose = gtsam.Pose2(
            float(meas["dx"]), float(meas["dy"]), float(meas["dtheta"])
        )

        # Optional Manhattan angle snapping
        if th_map and i in th_map and j in th_map:
            th_i = float(th_map[i])
            th_j = float(th_map[j])
            original_dtheta = float(meas["dtheta"])
            half_pi = math.pi / 2.0
            k = round((original_dtheta - th_i + th_j) / half_pi)
            snapped_dtheta = wrap_pi(th_i - th_j + k * half_pi)
            measured_relative_pose = gtsam.Pose2(
                float(meas["dx"]), float(meas["dy"]), snapped_dtheta
            )

        sig = e.get("noise_sigma", {})
        sigma_xy = float(sig.get("sigma_xy", 1.0))
        sigma_theta = float(sig.get("sigma_theta", math.radians(30.0)))

        huber_k = huber_k_arg
        if use_robust and huber_k is None:
            err_x, err_y, err_th = relative_error_pose2(
                initial.atPose2(key_i), initial.atPose2(key_j), measured_relative_pose
            )
            baseline_rmse = math.sqrt((err_x**2 + err_y**2) / 2.0)
            huber_k = max(1.0, baseline_rmse * 1.5)

        if huber_k is None:
            huber_k = 1.345

        model = build_noise_model(sigma_xy, sigma_theta, use_robust, huber_k)
        graph.add(gtsam.BetweenFactorPose2(key_i, key_j, measured_relative_pose, model))
        actual_edges_used.append(
            {
                "i": i,
                "j": j,
                "measurement": {
                    "dx": float(measured_relative_pose.x()),
                    "dy": float(measured_relative_pose.y()),
                    "dtheta": float(measured_relative_pose.theta()),
                },
                "noise_sigma": {"sigma_xy": sigma_xy, "sigma_theta": sigma_theta},
                "meta": e.get("meta", {}),
            }
        )

    return missing_nodes, actual_edges_used


def add_extra_point_constraints_as_soft_edges(
    graph: gtsam.NonlinearFactorGraph,
    baseline_result: gtsam.Values,
    node_ids: List[str],
    id_to_idx: Dict[str, int],
    layout_dir: Path,
    extra_constraints: List[Dict[str, Any]],
    point_sigma_xy: float,
    point_sigma_theta: float,
    pano_w: int = 1024,
    pano_h: int = 512,
    layout_z: float = 50.0,
) -> Tuple[int, List[Dict[str, Any]]]:
    """
    Stage B:
    Use Stage A result to freeze a baseline relative angle dtheta_0 for each (i,j),
    then convert each extra point pair into a soft BetweenFactorPose2 pseudo-edge.

    Returns:
      num_pairs_added, pseudo_edges_used_for_reporting
    """
    local_polys = {}
    for nid in node_ids:
        txt_path = layout_dir / "{}.txt".format(nid)
        if txt_path.exists():
            local_polys[nid] = load_layout_txt(
                txt_path, pano_w=pano_w, pano_h=pano_h, z=layout_z
            )

    num_added = 0
    pseudo_edges = []

    for item in extra_constraints:
        i = item["src"]
        j = item["dst"]

        if i not in id_to_idx or j not in id_to_idx:
            continue
        uses_free_points = "point_pairs" in item
        if not uses_free_points and (i not in local_polys or j not in local_polys):
            continue

        key_i = gtsam.symbol("x", id_to_idx[i])
        key_j = gtsam.symbol("x", id_to_idx[j])

        pose_i = baseline_result.atPose2(key_i)
        pose_j = baseline_result.atPose2(key_j)
        rel_ij = pose_i.between(pose_j)
        dtheta0 = float(rel_ij.theta())

        if uses_free_points:
            point_pairs_iter = [
                (
                    np.array(pair["src_xy"], dtype=np.float64),
                    np.array(pair["dst_xy"], dtype=np.float64),
                    {"point_mode": "free", "point_pair": pair},
                )
                for pair in item["point_pairs"]
            ]
        else:
            point_pairs_iter = []
            for src_idx, dst_idx in item["pairs"]:
                if src_idx < 1 or dst_idx < 1:
                    continue
                if src_idx > len(local_polys[i]) or dst_idx > len(local_polys[j]):
                    continue
                point_pairs_iter.append(
                    (
                        local_polys[i][src_idx - 1],
                        local_polys[j][dst_idx - 1],
                        {"point_mode": "vertex", "pair": [int(src_idx), int(dst_idx)]},
                    )
                )

        for p_i, p_j, pair_meta in point_pairs_iter:

            meas = point_pair_to_soft_pose_measurement(p_i, p_j, dtheta0)
            model = gtsam.noiseModel.Diagonal.Sigmas(
                np.array(
                    [point_sigma_xy, point_sigma_xy, point_sigma_theta],
                    dtype=np.float64,
                )
            )

            graph.add(gtsam.BetweenFactorPose2(key_i, key_j, meas, model))
            num_added += 1

            pseudo_edges.append(
                {
                    "i": i,
                    "j": j,
                    "measurement": {
                        "dx": float(meas.x()),
                        "dy": float(meas.y()),
                        "dtheta": float(meas.theta()),
                    },
                    "noise_sigma": {
                        "sigma_xy": float(point_sigma_xy),
                        "sigma_theta": float(point_sigma_theta),
                    },
                    "meta": {
                        "source": "extra_point_constraint_soft_edge",
                        **pair_meta,
                    },
                }
            )

    return num_added, pseudo_edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--edges",
        type=str,
        required=True,
        help="edges_measurements.json / perfect_edges.json",
    )
    ap.add_argument("--init", type=str, required=True, help="initial_poses.json")
    ap.add_argument("--out", type=str, required=True, help="optimized_poses.json")
    ap.add_argument("--report", type=str, required=True, help="residual_report.json")

    ap.add_argument(
        "--use_robust", action="store_true", help="Enable robust Huber kernel"
    )
    ap.add_argument(
        "--huber_k",
        type=float,
        default=None,
        help="Huber kernel parameter (Auto if None)",
    )
    ap.add_argument(
        "--prior_sigma_xy",
        type=float,
        default=1e-3,
        help="Prior sigma for root translation",
    )
    ap.add_argument(
        "--prior_sigma_theta_deg",
        type=float,
        default=1e-3,
        help="Prior sigma for root rotation",
    )
    ap.add_argument(
        "--lm_max_iters",
        type=int,
        default=100,
        help="Levenberg-Marquardt max iterations",
    )
    ap.add_argument("--lm_lambda", type=float, default=1e-3, help="Initial LM lambda")
    ap.add_argument(
        "--lm_verbosity",
        type=str,
        default="SUMMARY",
        choices=[
            "SILENT",
            "SUMMARY",
            "TERMINATION",
            "LAMBDA",
            "TRYLAMBDA",
            "TRYCONFIG",
            "DAMPED",
        ],
        help="GTSAM Levenberg-Marquardt verbosity level",
    )
    ap.add_argument(
        "--optimizer_verbosity",
        type=str,
        default="SILENT",
        choices=["SILENT", "TERMINATION", "ERROR", "VALUES", "DELTA", "LINEAR"],
        help="GTSAM nonlinear optimizer verbosity level",
    )
    ap.add_argument(
        "--optimizer_log",
        type=str,
        default="",
        help="Optional GTSAM optimizer log file path",
    )
    ap.add_argument(
        "--theta_priors", default=None, help="Optional json containing theta_priors"
    )

    # New: extra point constraints
    ap.add_argument(
        "--extra_points",
        type=str,
        default=None,
        help="extra point coincidence constraints json",
    )
    ap.add_argument(
        "--layout_dir",
        type=str,
        default=None,
        help="layout_gt directory for loading local polygons",
    )
    ap.add_argument(
        "--point_sigma_xy",
        type=float,
        default=0.5,
        help="soft XY sigma for extra point constraints",
    )
    ap.add_argument(
        "--point_sigma_theta",
        type=float,
        default=0.5,
        help="soft theta sigma (rad) for extra point constraints",
    )
    ap.add_argument(
        "--layout_z",
        type=float,
        default=50.0,
        help="projection z used for layout txt -> XY",
    )
    ap.add_argument("--pano_w", type=int, default=1024)
    ap.add_argument("--pano_h", type=int, default=512)

    args = ap.parse_args()

    edges_path = Path(args.edges)
    init_path = Path(args.init)
    out_path = Path(args.out)
    report_path = Path(args.report)

    edges_data = load_json(edges_path)
    init_data = load_json(init_path)

    edges = edges_data.get("edges", [])
    if not edges:
        raise RuntimeError("edges json contains no edges")

    root_id = init_data.get("root", "")
    poses_init_dict = init_data.get("poses", {})
    if not poses_init_dict:
        raise RuntimeError("initial_poses.json contains no poses")

    node_ids = sorted(list(poses_init_dict.keys()))
    if root_id == "" or root_id not in poses_init_dict:
        root_id = node_ids[0]

    id_to_idx = make_symbol_map(node_ids)

    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    for nid in node_ids:
        key = gtsam.symbol("x", id_to_idx[nid])
        p = pose_dict_to_pose2(poses_init_dict[nid])
        initial.insert(key, p)

    # Root prior to fix gauge
    prior_sigma_theta = math.radians(args.prior_sigma_theta_deg)
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array(
            [args.prior_sigma_xy, args.prior_sigma_xy, prior_sigma_theta],
            dtype=np.float64,
        )
    )
    root_key = gtsam.symbol("x", id_to_idx[root_id])
    root_pose = initial.atPose2(root_key)
    graph.add(gtsam.PriorFactorPose2(root_key, root_pose, prior_noise))

    # Optional theta priors
    th_map = {}
    if args.theta_priors:
        priors_data = load_json(Path(args.theta_priors))
        th_map = priors_data.get("theta_priors", {})
        print("[INFO] Loaded {} layout theta priors.".format(len(th_map)))

    # Add standard pose edges
    missing_nodes, standard_edges_used = add_standard_pose_edges(
        graph=graph,
        initial=initial,
        edges=edges,
        id_to_idx=id_to_idx,
        th_map=th_map,
        use_robust=bool(args.use_robust),
        huber_k_arg=args.huber_k,
    )

    if missing_nodes > 0:
        print(
            "\n[CRITICAL WARNING] {} edges dropped! Graph topology may be fractured.\n".format(
                missing_nodes
            )
        )

    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(args.lm_max_iters)
    params.setlambdaInitial(args.lm_lambda)
    params.setVerbosityLM(args.lm_verbosity)
    params.setVerbosity(args.optimizer_verbosity)
    if args.optimizer_log:
        Path(args.optimizer_log).parent.mkdir(parents=True, exist_ok=True)
        params.setLogFile(args.optimizer_log)

    # Debug: graph error before Stage A
    graph_error_before_stage_a = float(graph.error(initial))
    print(
        "[DEBUG] Raw graph error BEFORE Stage A optimization: {:.12f}".format(
            graph_error_before_stage_a
        )
    )

    # Stage A optimize
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
    result_stage_a = optimizer.optimize()

    graph_error_after_stage_a = float(graph.error(result_stage_a))
    print(
        "[DEBUG] Raw graph error AFTER Stage A optimization:  {:.12f}".format(
            graph_error_after_stage_a
        )
    )

    # Stage B: add extra point constraints as soft pseudo-edges
    extra_pseudo_edges_used = []
    num_extra_pairs_added = 0
    result_final = result_stage_a

    if args.extra_points:
        if not args.layout_dir:
            raise RuntimeError("--layout_dir is required when --extra_points is used")

        raw_extra = load_json(Path(args.extra_points))
        extra_constraints = normalize_extra_point_constraints(raw_extra)

        num_extra_pairs_added, extra_pseudo_edges_used = (
            add_extra_point_constraints_as_soft_edges(
                graph=graph,
                baseline_result=result_stage_a,
                node_ids=node_ids,
                id_to_idx=id_to_idx,
                layout_dir=Path(args.layout_dir),
                extra_constraints=extra_constraints,
                point_sigma_xy=args.point_sigma_xy,
                point_sigma_theta=args.point_sigma_theta,
                pano_w=args.pano_w,
                pano_h=args.pano_h,
                layout_z=args.layout_z,
            )
        )

        print(
            "[INFO] Added {} extra point-pair soft constraints.".format(
                num_extra_pairs_added
            )
        )

        graph_error_before_stage_b = float(graph.error(result_stage_a))
        print(
            "[DEBUG] Raw graph error BEFORE Stage B optimization: {:.12f}".format(
                graph_error_before_stage_b
            )
        )

        optimizer2 = gtsam.LevenbergMarquardtOptimizer(graph, result_stage_a, params)
        result_final = optimizer2.optimize()

        graph_error_after_stage_b = float(graph.error(result_final))
        print(
            "[DEBUG] Raw graph error AFTER Stage B optimization:  {:.12f}".format(
                graph_error_after_stage_b
            )
        )
    else:
        graph_error_before_stage_b = graph_error_after_stage_a
        graph_error_after_stage_b = graph_error_after_stage_a

    # Collect poses
    poses_opt = {}
    poses_init = {}
    poses_stage_a = {}

    for nid in node_ids:
        key = gtsam.symbol("x", id_to_idx[nid])
        poses_opt[nid] = result_final.atPose2(key)
        poses_init[nid] = initial.atPose2(key)
        poses_stage_a[nid] = result_stage_a.atPose2(key)

    # Debug: pose delta
    pose_deltas_init_to_final = compute_pose_deltas(poses_init, poses_opt)
    top_moved_by_translation = sorted(
        pose_deltas_init_to_final["per_node"], key=lambda d: -d["trans_l2"]
    )[: min(10, len(pose_deltas_init_to_final["per_node"]))]

    top_moved_by_rotation = sorted(
        pose_deltas_init_to_final["per_node"], key=lambda d: -d["rot_abs"]
    )[: min(10, len(pose_deltas_init_to_final["per_node"]))]

    print("[DEBUG] Top moved nodes by translation:")
    for item in top_moved_by_translation:
        print(
            "  {}: dx={:.6f}, dy={:.6f}, dtheta={:.6f}, trans_l2={:.6f}".format(
                item["node_id"],
                item["delta"]["dx"],
                item["delta"]["dy"],
                item["delta"]["dtheta"],
                item["trans_l2"],
            )
        )

    # Evaluate standard edges only
    before_standard = compute_residual_stats(poses_init, standard_edges_used)
    after_standard = compute_residual_stats(
        poses_opt, standard_edges_used, initial_poses=poses_init
    )

    # Evaluate extra pseudo-edges if any
    if extra_pseudo_edges_used:
        extra_after = compute_residual_stats(
            poses_opt, extra_pseudo_edges_used, initial_poses=poses_stage_a
        )
    else:
        extra_after = {
            "translation": {"rmse": 0.0, "mean": 0.0, "median": 0.0, "max": 0.0},
            "rotation_rad": {"rmse": 0.0, "mean": 0.0, "median": 0.0, "max": 0.0},
            "per_edge": [],
        }

    is_tree_like = len(standard_edges_used) == len(node_ids) - 1

    report = {
        "scene_id": edges_data.get("scene_id", ""),
        "root": root_id,
        "num_nodes": len(node_ids),
        "num_standard_edges": len(standard_edges_used),
        "num_input_edges": len(edges),
        "missing_nodes_in_edges": missing_nodes,
        "graph_diagnostics": {
            "raw_graph_error_before_stage_a": graph_error_before_stage_a,
            "raw_graph_error_after_stage_a": graph_error_after_stage_a,
            "raw_graph_error_before_stage_b": graph_error_before_stage_b,
            "raw_graph_error_after_stage_b": graph_error_after_stage_b,
            "is_tree_like_by_count": is_tree_like,
        },
        "optimizer": {
            "type": "LevenbergMarquardt",
            "max_iters": args.lm_max_iters,
            "lambda_initial": args.lm_lambda,
            "use_robust": bool(args.use_robust),
            "huber_k_auto": args.huber_k is None,
        },
        "extra_point_constraints": {
            "enabled": bool(args.extra_points),
            "num_pairs_added": int(num_extra_pairs_added),
            "point_sigma_xy": float(args.point_sigma_xy),
            "point_sigma_theta": float(args.point_sigma_theta),
            "layout_dir": args.layout_dir,
            "layout_z": float(args.layout_z),
        },
        "standard_edges_before": {
            "translation": before_standard["translation"],
            "rotation_rad": before_standard["rotation_rad"],
        },
        "standard_edges_after_umeyama_aligned": {
            "translation": after_standard["translation"],
            "rotation_rad": after_standard["rotation_rad"],
        },
        "extra_point_constraints_after": {
            "translation": extra_after["translation"],
            "rotation_rad": extra_after["rotation_rad"],
        },
        "pose_delta_initial_to_final": {
            "translation": pose_deltas_init_to_final["translation"],
            "rotation_rad": pose_deltas_init_to_final["rotation_rad"],
        },
        "top_moved_nodes_by_translation": top_moved_by_translation,
        "top_moved_nodes_by_rotation": top_moved_by_rotation,
        "top_standard_edges_after_by_translation": sorted(
            after_standard["per_edge"], key=lambda d: -d["trans_l2"]
        )[: min(5, len(after_standard["per_edge"]))],
        "top_standard_edges_after_by_rotation": sorted(
            after_standard["per_edge"], key=lambda d: -d["rot_abs"]
        )[: min(5, len(after_standard["per_edge"]))],
        "top_extra_constraints_after_by_translation": sorted(
            extra_after["per_edge"], key=lambda d: -d["trans_l2"]
        )[: min(5, len(extra_after["per_edge"]))],
    }

    out = {
        "scene_id": edges_data.get("scene_id", ""),
        "root": root_id,
        "num_nodes": len(node_ids),
        "poses": {
            nid: pose2_to_dict(poses_opt[nid]) for nid in sorted(poses_opt.keys())
        },
        "note": "Optimized poses via GTSAM Pose2 PGO. Includes optional extra point soft constraints.",
    }

    save_json(out_path, out)
    save_json(report_path, report)

    print("[OK] wrote optimized poses -> {}".format(out_path))
    print("[OK] wrote residual report -> {}".format(report_path))
    print(
        "Standard edges BEFORE  (trans rmse, rot rmse):",
        report["standard_edges_before"]["translation"]["rmse"],
        report["standard_edges_before"]["rotation_rad"]["rmse"],
    )
    print(
        "Standard edges AFTER   (trans rmse, rot rmse):",
        report["standard_edges_after_umeyama_aligned"]["translation"]["rmse"],
        report["standard_edges_after_umeyama_aligned"]["rotation_rad"]["rmse"],
    )

    if args.extra_points:
        print(
            "Extra-point constraints AFTER (trans rmse, rot rmse):",
            report["extra_point_constraints_after"]["translation"]["rmse"],
            report["extra_point_constraints_after"]["rotation_rad"]["rmse"],
        )


if __name__ == "__main__":
    main()
