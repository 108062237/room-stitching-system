#!/usr/bin/env python3
"""
Analyze candidate structural constraints before and after optimization.

This script reads candidate_structural_matches.json entries produced by
floorplan_constraint_tool.py and measures how well the annotated point pairs
agree under two pose files.

It supports:
  - vertex pairs: {"pairs": [[src_idx, dst_idx], ...]}
  - free point pairs: {"point_pairs": [{"src_xy": [...], "dst_xy": [...]}, ...]}

The residuals are measured in the current root-prior coordinate frame. No global
alignment, GTSAM, or optimization is performed here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.utils.geom import load_layout_gt_txt_as_local_xy


DEFAULT_GROUPS = [
    "all",
    "wall_alignment",
    "structural_adjacency",
    "single_point_adjacency",
    "connectivity",
    "candidate",
    "unknown",
]

CSV_COLUMNS = [
    "candidate_idx",
    "candidate_id",
    "src",
    "dst",
    "constraint_type",
    "point_mode",
    "num_pairs",
    "mean_point_residual_before",
    "rmse_point_residual_before",
    "max_point_residual_before",
    "segment_len_src_before",
    "segment_len_dst_before",
    "segment_len_diff_before",
    "segment_angle_diff_before",
    "combined_residual_before",
    "mean_point_residual_after",
    "rmse_point_residual_after",
    "max_point_residual_after",
    "segment_len_src_after",
    "segment_len_dst_after",
    "segment_len_diff_after",
    "segment_angle_diff_after",
    "combined_residual_after",
    "improved",
    "worsened",
]


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def normalize_room_id(room_id: str) -> str:
    return room_id.replace(".txt", "")


def normalize_entries(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, dict):
        if "constraints" in raw:
            raw = raw["constraints"]
        else:
            raw = [raw]
    if not isinstance(raw, list):
        raise ValueError("constraints must be a list or a dict with key 'constraints'")
    return [item for item in raw if isinstance(item, dict)]


def load_pose_map(path: Path) -> Dict[str, Tuple[float, float, float]]:
    raw = load_json(path)
    poses_raw = raw.get("poses", raw)
    if not isinstance(poses_raw, dict):
        raise ValueError(f"Invalid pose file: {path}")
    out: Dict[str, Tuple[float, float, float]] = {}
    for room_id, pose in poses_raw.items():
        if not isinstance(pose, dict):
            raise ValueError(f"Invalid pose entry for {room_id}: expected object")
        try:
            out[normalize_room_id(room_id)] = (
                float(pose["x"]),
                float(pose["y"]),
                float(pose["theta"]),
            )
        except KeyError as exc:
            raise ValueError(f"Invalid pose entry for {room_id}: missing {exc}") from exc
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid numeric pose fields for {room_id}") from exc
    return out


def as_xy(value: Any, field_name: str) -> Tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{field_name} must be a 2-number list")
    return float(value[0]), float(value[1])


def apply_pose(pose: Tuple[float, float, float], xy: Tuple[float, float]) -> Tuple[float, float]:
    x, y, theta = pose
    px, py = xy
    c = math.cos(theta)
    s = math.sin(theta)
    return x + c * px - s * py, y + s * px + c * py


def distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def wrap_pi(theta: float) -> float:
    return (theta + math.pi) % (2 * math.pi) - math.pi


def undirected_angle_diff(a: float, b: float) -> float:
    diff = abs(wrap_pi(a - b))
    return min(diff, abs(math.pi - diff))


def segment_metrics(points_src: List[Tuple[float, float]], points_dst: List[Tuple[float, float]]) -> Dict[str, Optional[float]]:
    if len(points_src) < 2 or len(points_dst) < 2:
        return {
            "segment_len_src": None,
            "segment_len_dst": None,
            "segment_len_diff": None,
            "segment_angle_diff": None,
        }

    s0, s1 = points_src[0], points_src[1]
    d0, d1 = points_dst[0], points_dst[1]
    sv = (s1[0] - s0[0], s1[1] - s0[1])
    dv = (d1[0] - d0[0], d1[1] - d0[1])
    len_s = math.hypot(sv[0], sv[1])
    len_d = math.hypot(dv[0], dv[1])
    if len_s <= 1e-12 or len_d <= 1e-12:
        angle_diff = None
    else:
        angle_diff = undirected_angle_diff(math.atan2(sv[1], sv[0]), math.atan2(dv[1], dv[0]))
    return {
        "segment_len_src": len_s,
        "segment_len_dst": len_d,
        "segment_len_diff": abs(len_s - len_d),
        "segment_angle_diff": angle_diff,
    }


def point_stats(points_src: List[Tuple[float, float]], points_dst: List[Tuple[float, float]]) -> Dict[str, float]:
    if len(points_src) != len(points_dst):
        raise ValueError("src and dst point count mismatch")
    if not points_src:
        return {
            "mean_point_residual": 0.0,
            "rmse_point_residual": 0.0,
            "max_point_residual": 0.0,
        }
    dists = [distance(a, b) for a, b in zip(points_src, points_dst)]
    return {
        "mean_point_residual": sum(dists) / len(dists),
        "rmse_point_residual": math.sqrt(sum(d * d for d in dists) / len(dists)),
        "max_point_residual": max(dists),
    }


def combined_residual(stats: Dict[str, Optional[float]]) -> float:
    # A simple diagnostic scalar. Point RMSE is the primary term; segment terms
    # only participate when the candidate has at least two pairs.
    value = float(stats["rmse_point_residual"])
    if stats.get("segment_len_diff") is not None:
        value += float(stats["segment_len_diff"])
    if stats.get("segment_angle_diff") is not None:
        value += float(stats["segment_angle_diff"])
    return value


def load_local_polys(scene_dir: Path, room_ids: Iterable[str]) -> Dict[str, List[Tuple[float, float]]]:
    out: Dict[str, List[Tuple[float, float]]] = {}
    layout_dir = scene_dir / "layout_gt"
    for room_id in sorted(set(room_ids)):
        txt_path = layout_dir / f"{room_id}.txt"
        if not txt_path.exists():
            raise FileNotFoundError(f"Missing layout file for {room_id}: {txt_path}")
        poly = load_layout_gt_txt_as_local_xy(txt_path)
        if poly is None:
            raise ValueError(f"Could not parse layout file: {txt_path}")
        out[room_id] = [(float(x), float(y)) for x, y in poly.tolist()]
    return out


def extract_local_pairs(
    entry: Dict[str, Any],
    src: str,
    dst: str,
    local_polys: Dict[str, List[Tuple[float, float]]],
) -> Tuple[str, List[Tuple[float, float]], List[Tuple[float, float]]]:
    if "point_pairs" in entry:
        src_points = []
        dst_points = []
        for idx, pair in enumerate(entry.get("point_pairs", [])):
            if not isinstance(pair, dict):
                raise ValueError(f"Invalid point_pairs[{idx}] for {src}->{dst}")
            src_points.append(as_xy(pair.get("src_xy"), f"point_pairs[{idx}].src_xy"))
            dst_points.append(as_xy(pair.get("dst_xy"), f"point_pairs[{idx}].dst_xy"))
        return "free", src_points, dst_points

    pairs = entry.get("pairs", [])
    if not isinstance(pairs, list):
        raise ValueError(f"pairs must be a list for {src}->{dst}")
    src_points = []
    dst_points = []
    for idx, pair in enumerate(pairs):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"Invalid pairs[{idx}] for {src}->{dst}")
        src_idx = int(pair[0])
        dst_idx = int(pair[1])
        if src_idx < 1 or src_idx > len(local_polys[src]):
            raise IndexError(f"src vertex index {src_idx} out of range for {src} 1..{len(local_polys[src])}")
        if dst_idx < 1 or dst_idx > len(local_polys[dst]):
            raise IndexError(f"dst vertex index {dst_idx} out of range for {dst} 1..{len(local_polys[dst])}")
        src_points.append(local_polys[src][src_idx - 1])
        dst_points.append(local_polys[dst][dst_idx - 1])
    return "vertex", src_points, dst_points


def evaluate_points(
    src_points_local: List[Tuple[float, float]],
    dst_points_local: List[Tuple[float, float]],
    pose_src: Tuple[float, float, float],
    pose_dst: Tuple[float, float, float],
) -> Dict[str, Optional[float]]:
    src_world = [apply_pose(pose_src, p) for p in src_points_local]
    dst_world = [apply_pose(pose_dst, p) for p in dst_points_local]
    out: Dict[str, Optional[float]] = {}
    out.update(point_stats(src_world, dst_world))
    out.update(segment_metrics(src_world, dst_world))
    out["combined_residual"] = combined_residual(out)
    return out


def analyze(
    scene_dir: Path,
    constraints_path: Path,
    before_path: Path,
    after_path: Path,
) -> List[Dict[str, Any]]:
    entries = normalize_entries(load_json(constraints_path))
    before_poses = load_pose_map(before_path)
    after_poses = load_pose_map(after_path)

    vertex_room_ids = []
    for idx, entry in enumerate(entries):
        if "src" not in entry or "dst" not in entry:
            raise ValueError(f"constraint {idx} missing src or dst")
        if "point_pairs" not in entry:
            vertex_room_ids.extend([normalize_room_id(str(entry["src"])), normalize_room_id(str(entry["dst"]))])

    local_polys = load_local_polys(scene_dir, vertex_room_ids) if vertex_room_ids else {}
    rows: List[Dict[str, Any]] = []

    for idx, entry in enumerate(entries):
        src = normalize_room_id(str(entry["src"]))
        dst = normalize_room_id(str(entry["dst"]))
        if src not in before_poses:
            raise KeyError(f"Missing before pose for src: {src}")
        if dst not in before_poses:
            raise KeyError(f"Missing before pose for dst: {dst}")
        if src not in after_poses:
            raise KeyError(f"Missing after pose for src: {src}")
        if dst not in after_poses:
            raise KeyError(f"Missing after pose for dst: {dst}")

        point_mode, src_points, dst_points = extract_local_pairs(entry, src, dst, local_polys)
        if len(src_points) != len(dst_points):
            raise ValueError(f"point count mismatch for {src}->{dst}")
        if not src_points:
            raise ValueError(f"constraint {idx} has no point pairs: {src}->{dst}")

        before_stats = evaluate_points(src_points, dst_points, before_poses[src], before_poses[dst])
        after_stats = evaluate_points(src_points, dst_points, after_poses[src], after_poses[dst])
        before_combined = float(before_stats["combined_residual"])
        after_combined = float(after_stats["combined_residual"])

        row: Dict[str, Any] = {
            "candidate_idx": idx,
            "candidate_id": entry.get("id", f"C{idx}"),
            "src": src,
            "dst": dst,
            "constraint_type": entry.get("constraint_type", "unknown"),
            "point_mode": entry.get("point_mode", point_mode),
            "num_pairs": len(src_points),
            "improved": after_combined < before_combined,
            "worsened": after_combined > before_combined,
        }
        for key, value in before_stats.items():
            row[f"{key}_before"] = value
        for key, value in after_stats.items():
            row[f"{key}_after"] = value
        rows.append(row)

    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col) for col in CSV_COLUMNS})


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize_group(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def vals(name: str) -> List[float]:
        out = []
        for row in rows:
            value = row.get(name)
            if value is not None:
                out.append(float(value))
        return out

    return {
        "num_constraints": len(rows),
        "num_pairs": sum(int(row["num_pairs"]) for row in rows),
        "mean_point_residual_before": mean(vals("mean_point_residual_before")),
        "mean_point_residual_after": mean(vals("mean_point_residual_after")),
        "mean_rmse_point_residual_before": mean(vals("rmse_point_residual_before")),
        "mean_rmse_point_residual_after": mean(vals("rmse_point_residual_after")),
        "max_point_residual_before": max(vals("max_point_residual_before"), default=0.0),
        "max_point_residual_after": max(vals("max_point_residual_after"), default=0.0),
        "mean_segment_len_diff_before": mean(vals("segment_len_diff_before")),
        "mean_segment_len_diff_after": mean(vals("segment_len_diff_after")),
        "mean_segment_angle_diff_before": mean(vals("segment_angle_diff_before")),
        "mean_segment_angle_diff_after": mean(vals("segment_angle_diff_after")),
        "num_improved": sum(1 for row in rows if row["improved"]),
        "num_worsened": sum(1 for row in rows if row["worsened"]),
        "top_by_point_residual_after": sorted(
            rows,
            key=lambda row: float(row["rmse_point_residual_after"]),
            reverse=True,
        )[:10],
        "top_by_worsening": sorted(
            rows,
            key=lambda row: float(row["combined_residual_after"]) - float(row["combined_residual_before"]),
            reverse=True,
        )[:10],
    }


def write_summary(path: Path, rows: List[Dict[str, Any]]) -> None:
    groups = {name: [] for name in DEFAULT_GROUPS}
    for row in rows:
        groups["all"].append(row)
        ctype = str(row.get("constraint_type", "unknown"))
        groups.setdefault(ctype, []).append(row)
    summary = {name: summarize_group(group_rows) for name, group_rows in groups.items()}
    save_json(path, summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", required=True)
    parser.add_argument("--constraints", required=True)
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--summary", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = analyze(
        scene_dir=Path(args.scene_dir),
        constraints_path=Path(args.constraints),
        before_path=Path(args.before),
        after_path=Path(args.after),
    )
    save_json(Path(args.out_json), rows)
    write_csv(Path(args.out_csv), rows)
    write_summary(Path(args.summary), rows)
    print(f"[OK] wrote {args.out_json}")
    print(f"[OK] wrote {args.out_csv}")
    print(f"[OK] wrote {args.summary}")


if __name__ == "__main__":
    main()
