#!/usr/bin/env python3
"""
Standalone floorplan-level pose optimizer using scipy.optimize.least_squares.

This prototype keeps each room layout fixed and optimizes only one SE(2) pose
per room:
  x_i = (tx_i, ty_i, theta_i)

Objective residual vector:
  r_total = [r_pose, r_wall_alignment, r_regularization]

The wall-alignment candidates are intentionally modeled as weak line constraints
instead of rigid room-to-room pose edges.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.utils.geom import load_layout_gt_txt_as_local_xy


Pose = Tuple[float, float, float]
Point = Tuple[float, float]

DEFAULT_CANDIDATE_IDS = {"C1", "C4"}


def wrap_pi(theta: float) -> float:
    return (theta + math.pi) % (2.0 * math.pi) - math.pi


def normalize_room_id(room_id: Any) -> str:
    return str(room_id).replace(".txt", "")


def read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def require_number(mapping: Dict[str, Any], key: str, context: str) -> float:
    if key not in mapping:
        raise ValueError(f"Missing {key} in {context}")
    try:
        return float(mapping[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric {key} in {context}: {mapping[key]!r}") from exc


def load_pose_map(path: Path) -> Dict[str, Pose]:
    raw = read_json(path)
    poses_raw = raw.get("poses", raw) if isinstance(raw, dict) else None
    if not isinstance(poses_raw, dict) or not poses_raw:
        raise ValueError(f"Pose file contains no poses: {path}")

    poses: Dict[str, Pose] = {}
    for room_id, pose in poses_raw.items():
        if not isinstance(pose, dict):
            raise ValueError(f"Invalid pose object for {room_id!r} in {path}")
        context = f"pose {room_id!r} in {path}"
        poses[normalize_room_id(room_id)] = (
            require_number(pose, "x", context),
            require_number(pose, "y", context),
            wrap_pi(require_number(pose, "theta", context)),
        )
    return poses


def se2_compose(a: Pose, b: Pose) -> Pose:
    ax, ay, ath = a
    bx, by, bth = b
    c = math.cos(ath)
    s = math.sin(ath)
    return (
        ax + c * bx - s * by,
        ay + s * bx + c * by,
        wrap_pi(ath + bth),
    )


def se2_inverse(pose: Pose) -> Pose:
    x, y, theta = pose
    c = math.cos(theta)
    s = math.sin(theta)
    return (
        -(c * x + s * y),
        -(-s * x + c * y),
        wrap_pi(-theta),
    )


def se2_between(src_pose: Pose, dst_pose: Pose) -> Pose:
    return se2_compose(se2_inverse(src_pose), dst_pose)


def apply_pose(pose: Pose, xy: Point) -> np.ndarray:
    x, y, theta = pose
    px, py = xy
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([x + c * px - s * py, y + s * px + c * py], dtype=np.float64)


def load_edges(path: Path) -> List[Dict[str, Any]]:
    raw = read_json(path)
    if isinstance(raw, list):
        edges_raw = raw
    elif isinstance(raw, dict) and isinstance(raw.get("edges"), list):
        edges_raw = raw["edges"]
    else:
        raise ValueError(f"Edges file must be a list or contain an 'edges' list: {path}")

    edges: List[Dict[str, Any]] = []
    for idx, edge in enumerate(edges_raw):
        if not isinstance(edge, dict):
            raise ValueError(f"edge {idx} in {path} must be an object")
        src = edge.get("src", edge.get("i"))
        dst = edge.get("dst", edge.get("j"))
        if src is None or dst is None:
            raise ValueError(f"edge {idx} in {path} missing src/i or dst/j")
        measurement = edge.get("measurement", edge)
        if not isinstance(measurement, dict):
            raise ValueError(f"edge {idx} in {path} has invalid measurement")
        edges.append(
            {
                "edge_idx": idx,
                "edge_id": str(edge.get("edge_id", edge.get("id", f"C{idx}"))),
                "src": normalize_room_id(src),
                "dst": normalize_room_id(dst),
                "measurement": (
                    require_number(measurement, "dx", f"edge {idx}"),
                    require_number(measurement, "dy", f"edge {idx}"),
                    wrap_pi(require_number(measurement, "dtheta", f"edge {idx}")),
                ),
            }
        )
    return edges


def normalize_candidate_entries(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, dict):
        raw = raw.get("constraints", raw.get("candidates", [raw]))
    if not isinstance(raw, list):
        raise ValueError("candidates must be a list or an object with constraints/candidates")
    return [entry for entry in raw if isinstance(entry, dict)]


def parse_candidate_ids(raw: Optional[str]) -> Optional[set[str]]:
    if raw is None or not raw.strip():
        return None
    out = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if token.upper().startswith("C"):
            out.add(token.upper())
            continue
        try:
            out.add(f"C{int(token)}")
        except ValueError:
            out.add(token)
    return out


def load_local_polygon(layout_dir: Path, room_id: str) -> List[Point]:
    txt_path = layout_dir / f"{room_id}.txt"
    if not txt_path.exists():
        raise FileNotFoundError(f"Missing layout file for {room_id}: {txt_path}")
    poly = load_layout_gt_txt_as_local_xy(txt_path)
    if poly is None:
        raise ValueError(f"Could not parse layout file: {txt_path}")
    return [(float(x), float(y)) for x, y in poly.tolist()]


def as_xy(value: Any, field_name: str) -> Point:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{field_name} must be a 2-number list")
    return float(value[0]), float(value[1])


def extract_candidate_points(
    entry: Dict[str, Any],
    src: str,
    dst: str,
    layout_dir: Path,
    polygon_cache: Dict[str, List[Point]],
) -> Tuple[str, List[Point], List[Point]]:
    if "point_pairs" in entry:
        src_points: List[Point] = []
        dst_points: List[Point] = []
        for idx, pair in enumerate(entry.get("point_pairs", [])):
            if not isinstance(pair, dict):
                raise ValueError(f"Invalid point_pairs[{idx}] for {src}->{dst}")
            src_points.append(as_xy(pair.get("src_xy"), f"point_pairs[{idx}].src_xy"))
            dst_points.append(as_xy(pair.get("dst_xy"), f"point_pairs[{idx}].dst_xy"))
        return "free", src_points, dst_points

    if src not in polygon_cache:
        polygon_cache[src] = load_local_polygon(layout_dir, src)
    if dst not in polygon_cache:
        polygon_cache[dst] = load_local_polygon(layout_dir, dst)

    src_points = []
    dst_points = []
    pairs = entry.get("pairs", [])
    if not isinstance(pairs, list):
        raise ValueError(f"pairs must be a list for {src}->{dst}")
    for idx, pair in enumerate(pairs):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"Invalid pairs[{idx}] for {src}->{dst}")
        src_idx = int(pair[0])
        dst_idx = int(pair[1])
        if src_idx < 1 or src_idx > len(polygon_cache[src]):
            raise IndexError(f"src vertex index {src_idx} out of range for {src}")
        if dst_idx < 1 or dst_idx > len(polygon_cache[dst]):
            raise IndexError(f"dst vertex index {dst_idx} out of range for {dst}")
        src_points.append(polygon_cache[src][src_idx - 1])
        dst_points.append(polygon_cache[dst][dst_idx - 1])
    return "vertex", src_points, dst_points


def load_wall_candidates(
    path: Path,
    layout_dir: Path,
    selected_ids: Optional[set[str]],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    entries = normalize_candidate_entries(read_json(path))
    polygon_cache: Dict[str, List[Point]] = {}
    candidates: List[Dict[str, Any]] = []
    skipped: List[str] = []
    wanted_ids = selected_ids if selected_ids is not None else DEFAULT_CANDIDATE_IDS

    for idx, entry in enumerate(entries):
        candidate_id = f"C{idx}"
        source_candidate_id = entry.get("id")
        match_ids = {candidate_id}
        if source_candidate_id is not None:
            match_ids.add(str(source_candidate_id))
        if not (match_ids & wanted_ids):
            continue
        constraint_type = str(entry.get("constraint_type", "unknown"))
        src = normalize_room_id(entry.get("src"))
        dst = normalize_room_id(entry.get("dst"))
        if constraint_type != "wall_alignment":
            skipped.append(f"{candidate_id}: skipped non-wall candidate ({constraint_type})")
            continue
        point_mode, src_points, dst_points = extract_candidate_points(
            entry, src, dst, layout_dir, polygon_cache
        )
        if len(src_points) < 2 or len(dst_points) < 2:
            skipped.append(f"{candidate_id}: skipped because it has fewer than two points")
            continue
        candidates.append(
            {
                "candidate_idx": idx,
                "candidate_id": candidate_id,
                "source_candidate_id": source_candidate_id,
                "src": src,
                "dst": dst,
                "constraint_type": constraint_type,
                "point_mode": entry.get("point_mode", point_mode),
                "src_points": src_points[:2],
                "dst_points": dst_points[:2],
            }
        )
    return candidates, skipped


def pack_poses(room_ids: Sequence[str], poses: Dict[str, Pose]) -> np.ndarray:
    values = []
    for room_id in room_ids:
        x, y, theta = poses[room_id]
        values.extend([x, y, theta])
    return np.array(values, dtype=np.float64)


def unpack_poses(room_ids: Sequence[str], values: np.ndarray) -> Dict[str, Pose]:
    poses: Dict[str, Pose] = {}
    for idx, room_id in enumerate(room_ids):
        base = 3 * idx
        poses[room_id] = (
            float(values[base]),
            float(values[base + 1]),
            wrap_pi(float(values[base + 2])),
        )
    return poses


def residuals_pose(
    poses: Dict[str, Pose],
    edges: Iterable[Dict[str, Any]],
    sigma_t: float,
    sigma_theta: float,
) -> List[float]:
    out: List[float] = []
    for edge in edges:
        src = edge["src"]
        dst = edge["dst"]
        if src not in poses or dst not in poses:
            raise KeyError(f"Missing pose for reliable edge {src}->{dst}")
        predicted = se2_between(poses[src], poses[dst])
        error = se2_compose(se2_inverse(edge["measurement"]), predicted)
        out.extend([error[0] / sigma_t, error[1] / sigma_t, wrap_pi(error[2]) / sigma_theta])
    return out


def residuals_wall(
    poses: Dict[str, Pose],
    candidates: Iterable[Dict[str, Any]],
    sigma_dist: float,
    sigma_angle: float,
) -> List[float]:
    out: List[float] = []
    for candidate in candidates:
        src = candidate["src"]
        dst = candidate["dst"]
        if src not in poses or dst not in poses:
            raise KeyError(f"Missing pose for candidate {candidate['candidate_id']}: {src}->{dst}")
        a0 = apply_pose(poses[src], candidate["src_points"][0])
        a1 = apply_pose(poses[src], candidate["src_points"][1])
        b0 = apply_pose(poses[dst], candidate["dst_points"][0])
        b1 = apply_pose(poses[dst], candidate["dst_points"][1])

        u_a = a1 - a0
        u_b = b1 - b0
        len_a = float(np.linalg.norm(u_a))
        len_b = float(np.linalg.norm(u_b))
        if len_a <= 1e-12 or len_b <= 1e-12:
            raise ValueError(f"Degenerate wall segment in {candidate['candidate_id']}")
        u_a /= len_a
        u_b /= len_b
        n_a = np.array([-u_a[1], u_a[0]], dtype=np.float64)

        cross = float(u_a[0] * u_b[1] - u_a[1] * u_b[0])
        line_0 = float(np.dot(n_a, b0 - a0))
        line_1 = float(np.dot(n_a, b1 - a0))
        out.extend([cross / sigma_angle, line_0 / sigma_dist, line_1 / sigma_dist])
    return out


def residuals_regularization(
    poses: Dict[str, Pose],
    initial_poses: Dict[str, Pose],
    room_ids: Iterable[str],
    sigma_t: float,
    sigma_theta: float,
) -> List[float]:
    out: List[float] = []
    for room_id in room_ids:
        x, y, theta = poses[room_id]
        x0, y0, theta0 = initial_poses[room_id]
        out.extend([(x - x0) / sigma_t, (y - y0) / sigma_t, wrap_pi(theta - theta0) / sigma_theta])
    return out


def make_residual_function(
    room_ids: Sequence[str],
    initial_poses: Dict[str, Pose],
    edges: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    args: argparse.Namespace,
):
    def residual(values: np.ndarray) -> np.ndarray:
        poses = unpack_poses(room_ids, values)
        parts = []
        parts.extend(residuals_pose(poses, edges, args.sigma_pose_t, args.sigma_pose_theta))
        parts.extend(residuals_wall(poses, candidates, args.sigma_wall_dist, args.sigma_wall_angle))
        parts.extend(
            residuals_regularization(
                poses, initial_poses, room_ids, args.sigma_reg_t, args.sigma_reg_theta
            )
        )
        return np.asarray(parts, dtype=np.float64)

    return residual


def squared_norm(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.dot(arr, arr))


def summarize_residuals(
    poses: Dict[str, Pose],
    initial_poses: Dict[str, Pose],
    room_ids: Sequence[str],
    edges: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    pose_r = residuals_pose(poses, edges, args.sigma_pose_t, args.sigma_pose_theta)
    wall_r = residuals_wall(poses, candidates, args.sigma_wall_dist, args.sigma_wall_angle)
    reg_r = residuals_regularization(
        poses, initial_poses, room_ids, args.sigma_reg_t, args.sigma_reg_theta
    )

    return {
        "num_pose_residuals": len(pose_r),
        "num_wall_residuals": len(wall_r),
        "num_regularization_residuals": len(reg_r),
        "pose_loss": 0.5 * squared_norm(pose_r),
        "wall_loss": 0.5 * squared_norm(wall_r),
        "regularization_loss": 0.5 * squared_norm(reg_r),
        "total_loss": 0.5 * squared_norm([*pose_r, *wall_r, *reg_r]),
        "max_abs_pose_residual": max((abs(v) for v in pose_r), default=0.0),
        "max_abs_wall_residual": max((abs(v) for v in wall_r), default=0.0),
        "max_abs_regularization_residual": max((abs(v) for v in reg_r), default=0.0),
    }


def build_pose_output(
    source: Any,
    room_ids: Sequence[str],
    optimized_poses: Dict[str, Pose],
    args: argparse.Namespace,
    candidate_ids: List[str],
) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(source) if isinstance(source, dict) else {}
    out["poses"] = {
        room_id: {"x": pose[0], "y": pose[1], "theta": pose[2]}
        for room_id, pose in ((room_id, optimized_poses[room_id]) for room_id in room_ids)
    }
    out["num_nodes"] = len(room_ids)
    out["optimizer"] = "scipy.optimize.least_squares"
    out["objective"] = "r_total = [r_pose, r_wall_alignment, r_regularization]"
    out["used_candidate_ids"] = candidate_ids
    out["sigmas"] = {
        "sigma_pose_t": args.sigma_pose_t,
        "sigma_pose_theta": args.sigma_pose_theta,
        "sigma_wall_dist": args.sigma_wall_dist,
        "sigma_wall_angle": args.sigma_wall_angle,
        "sigma_reg_t": args.sigma_reg_t,
        "sigma_reg_theta": args.sigma_reg_theta,
    }
    out["note"] = "Optimized poses via standalone floorplan-level objective. Room polygons are fixed."
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", required=True)
    parser.add_argument("--layout_dir", default=None)
    parser.add_argument("--edges", required=True)
    parser.add_argument("--init", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--candidate_ids", default=None)
    parser.add_argument("--sigma_pose_t", type=float, default=0.005)
    parser.add_argument("--sigma_pose_theta", type=float, default=0.001)
    parser.add_argument("--sigma_wall_dist", type=float, default=5.0)
    parser.add_argument("--sigma_wall_angle", type=float, default=0.05)
    parser.add_argument("--sigma_reg_t", type=float, default=10.0)
    parser.add_argument("--sigma_reg_theta", type=float, default=0.25)
    parser.add_argument(
        "--loss",
        default="linear",
        choices=["linear", "soft_l1", "huber", "cauchy", "arctan"],
    )
    parser.add_argument("--f_scale", type=float, default=1.0)
    return parser.parse_args()


def validate_sigmas(args: argparse.Namespace) -> None:
    for name in [
        "sigma_pose_t",
        "sigma_pose_theta",
        "sigma_wall_dist",
        "sigma_wall_angle",
        "sigma_reg_t",
        "sigma_reg_theta",
        "f_scale",
    ]:
        value = float(getattr(args, name))
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")


def main() -> None:
    args = parse_args()
    validate_sigmas(args)

    scene_dir = Path(args.scene_dir)
    layout_dir = Path(args.layout_dir) if args.layout_dir else scene_dir / "layout_gt"
    init_path = Path(args.init)

    init_raw = read_json(init_path)
    initial_poses = load_pose_map(init_path)
    room_ids = sorted(initial_poses)
    x0 = pack_poses(room_ids, initial_poses)

    edges = load_edges(Path(args.edges))
    selected_ids = parse_candidate_ids(args.candidate_ids)
    candidates, skipped = load_wall_candidates(Path(args.candidates), layout_dir, selected_ids)

    missing_edge_rooms = sorted(
        {room for edge in edges for room in [edge["src"], edge["dst"]] if room not in initial_poses}
    )
    if missing_edge_rooms:
        raise KeyError(f"Reliable edges reference rooms missing from initial poses: {missing_edge_rooms}")

    missing_candidate_rooms = sorted(
        {
            room
            for candidate in candidates
            for room in [candidate["src"], candidate["dst"]]
            if room not in initial_poses
        }
    )
    if missing_candidate_rooms:
        raise KeyError(f"Candidates reference rooms missing from initial poses: {missing_candidate_rooms}")

    residual_fn = make_residual_function(room_ids, initial_poses, edges, candidates, args)
    before_summary = summarize_residuals(
        initial_poses, initial_poses, room_ids, edges, candidates, args
    )

    result = least_squares(
        residual_fn,
        x0,
        loss=args.loss,
        f_scale=args.f_scale,
        max_nfev=2000,
    )
    optimized_poses = unpack_poses(room_ids, result.x)
    after_summary = summarize_residuals(
        optimized_poses, initial_poses, room_ids, edges, candidates, args
    )

    candidate_ids = [candidate["candidate_id"] for candidate in candidates]
    write_json(
        Path(args.out),
        build_pose_output(init_raw, room_ids, optimized_poses, args, candidate_ids),
    )

    report = {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "cost": float(result.cost),
        "optimality": float(result.optimality),
        "loss": args.loss,
        "f_scale": args.f_scale,
        "num_rooms": len(room_ids),
        "num_reliable_edges": len(edges),
        "num_wall_candidates": len(candidates),
        "used_candidate_ids": candidate_ids,
        "skipped_candidates": skipped,
        "default_candidate_ids": sorted(DEFAULT_CANDIDATE_IDS),
        "before": before_summary,
        "after": after_summary,
        "candidate_details": [
            {
                "candidate_idx": candidate["candidate_idx"],
                "candidate_id": candidate["candidate_id"],
                "source_candidate_id": candidate["source_candidate_id"],
                "src": candidate["src"],
                "dst": candidate["dst"],
                "constraint_type": candidate["constraint_type"],
                "point_mode": candidate["point_mode"],
            }
            for candidate in candidates
        ],
    }
    write_json(Path(args.report), report)

    print(f"[OK] wrote optimized poses -> {args.out}")
    print(f"[OK] wrote report -> {args.report}")
    print(f"[OK] used wall candidates: {', '.join(candidate_ids) if candidate_ids else '(none)'}")


if __name__ == "__main__":
    main()
