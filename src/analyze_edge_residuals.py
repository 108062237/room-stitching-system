#!/usr/bin/env python3
"""
Analyze per-edge SE(2) residuals before and after optimization.

Convention:
  z_ij = x_i^-1 * x_j

For each edge, predicted_ij is computed from the provided poses as:
  inverse(pose_src) compose pose_dst

Residual is:
  inverse(measured_edge_pose) compose predicted_ij

This script intentionally does not import GTSAM and does not perform any global
alignment. Residuals are reported in the actual root-prior coordinate frame.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


Pose = Tuple[float, float, float]

CSV_COLUMNS = [
    "edge_idx",
    "edge_id",
    "src",
    "dst",
    "edge_type",
    "dx_measured",
    "dy_measured",
    "dtheta_measured",
    "dx_pred_before",
    "dy_pred_before",
    "dtheta_pred_before",
    "dx_error_before",
    "dy_error_before",
    "dtheta_error_before",
    "trans_residual_before",
    "rot_residual_before",
    "dx_pred_after",
    "dy_pred_after",
    "dtheta_pred_after",
    "dx_error_after",
    "dy_error_after",
    "dtheta_error_after",
    "trans_residual_after",
    "rot_residual_after",
    "improved",
    "worsened",
]


def wrap_pi(theta: float) -> float:
    return (theta + math.pi) % (2.0 * math.pi) - math.pi


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


def read_json_file(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file: {path}: {exc}") from exc


def write_json_file(path: Path, data: Any) -> None:
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
    raw = read_json_file(path)
    if not isinstance(raw, dict):
        raise ValueError(f"Pose file must be a JSON object: {path}")

    poses_raw = raw.get("poses", raw)
    if not isinstance(poses_raw, dict) or not poses_raw:
        raise ValueError(f"Pose file contains no poses: {path}")

    poses: Dict[str, Pose] = {}
    for node_id, pose in poses_raw.items():
        context = f"pose {node_id!r} in {path}"
        if not isinstance(pose, dict):
            raise ValueError(f"Invalid pose object for {node_id!r} in {path}")
        poses[str(node_id)] = (
            require_number(pose, "x", context),
            require_number(pose, "y", context),
            require_number(pose, "theta", context),
        )
    return poses


def get_edge_type(edge: Dict[str, Any]) -> str:
    meta = edge.get("meta", {})
    candidates = [
        edge.get("edge_type"),
        edge.get("type"),
        meta.get("edge_type") if isinstance(meta, dict) else None,
    ]
    for candidate in candidates:
        if candidate is not None:
            return str(candidate)
    return "unknown"


def load_edges(path: Path) -> List[Dict[str, Any]]:
    raw = read_json_file(path)
    if isinstance(raw, list):
        edges_raw = raw
    elif isinstance(raw, dict) and isinstance(raw.get("edges"), list):
        edges_raw = raw["edges"]
    else:
        raise ValueError(
            f"Edges file must be a list or an object with an 'edges' list: {path}"
        )

    edges = []
    for idx, edge in enumerate(edges_raw):
        context = f"edge {idx} in {path}"
        if not isinstance(edge, dict):
            raise ValueError(f"{context} must be a JSON object")

        src = edge.get("src", edge.get("i"))
        dst = edge.get("dst", edge.get("j"))
        if src is None:
            raise ValueError(f"Missing src in {context}")
        if dst is None:
            raise ValueError(f"Missing dst in {context}")

        measurement = edge.get("measurement", edge)
        if not isinstance(measurement, dict):
            raise ValueError(f"Invalid measurement object in {context}")

        dx = require_number(measurement, "dx", context)
        dy = require_number(measurement, "dy", context)
        dtheta = require_number(measurement, "dtheta", context)
        edge_id = edge.get("edge_id", edge.get("id", f"{src}->{dst}#{idx}"))

        edges.append(
            {
                "edge_idx": idx,
                "edge_id": str(edge_id),
                "src": str(src),
                "dst": str(dst),
                "edge_type": get_edge_type(edge),
                "measurement": (dx, dy, wrap_pi(dtheta)),
            }
        )
    if not edges:
        raise ValueError(f"Edges file contains no edges: {path}")
    return edges


def analyze_state(
    poses: Dict[str, Pose], edge: Dict[str, Any], state_name: str
) -> Tuple[Pose, Pose, float, float]:
    src = edge["src"]
    dst = edge["dst"]
    if src not in poses:
        raise ValueError(f"Missing pose for src {src!r} while analyzing {state_name}")
    if dst not in poses:
        raise ValueError(f"Missing pose for dst {dst!r} while analyzing {state_name}")

    predicted = se2_between(poses[src], poses[dst])
    residual = se2_compose(se2_inverse(edge["measurement"]), predicted)
    residual = (residual[0], residual[1], wrap_pi(residual[2]))
    trans_residual = math.hypot(residual[0], residual[1])
    rot_residual = abs(wrap_pi(residual[2]))
    return predicted, residual, trans_residual, rot_residual


def analyze_edges(
    edges: Iterable[Dict[str, Any]],
    poses_before: Dict[str, Pose],
    poses_after: Dict[str, Pose],
) -> List[Dict[str, Any]]:
    rows = []
    for edge in edges:
        pred_before, err_before, trans_before, rot_before = analyze_state(
            poses_before, edge, "before"
        )
        pred_after, err_after, trans_after, rot_after = analyze_state(
            poses_after, edge, "after"
        )

        combined_before = trans_before + rot_before
        combined_after = trans_after + rot_after

        rows.append(
            {
                "edge_idx": edge["edge_idx"],
                "edge_id": edge["edge_id"],
                "src": edge["src"],
                "dst": edge["dst"],
                "edge_type": edge["edge_type"],
                "dx_measured": edge["measurement"][0],
                "dy_measured": edge["measurement"][1],
                "dtheta_measured": edge["measurement"][2],
                "dx_pred_before": pred_before[0],
                "dy_pred_before": pred_before[1],
                "dtheta_pred_before": pred_before[2],
                "dx_error_before": err_before[0],
                "dy_error_before": err_before[1],
                "dtheta_error_before": err_before[2],
                "trans_residual_before": trans_before,
                "rot_residual_before": rot_before,
                "dx_pred_after": pred_after[0],
                "dy_pred_after": pred_after[1],
                "dtheta_pred_after": pred_after[2],
                "dx_error_after": err_after[0],
                "dy_error_after": err_after[1],
                "dtheta_error_after": err_after[2],
                "trans_residual_after": trans_after,
                "rot_residual_after": rot_after,
                "improved": combined_after < combined_before,
                "worsened": combined_after > combined_before,
            }
        )
    return rows


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize_group(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    top_after = sorted(rows, key=lambda row: row["trans_residual_after"], reverse=True)
    top_worsening = sorted(
        rows,
        key=lambda row: (
            row["trans_residual_after"]
            + row["rot_residual_after"]
            - row["trans_residual_before"]
            - row["rot_residual_before"]
        ),
        reverse=True,
    )
    return {
        "num_edges": len(rows),
        "mean_trans_residual_before": mean(
            [row["trans_residual_before"] for row in rows]
        ),
        "mean_trans_residual_after": mean(
            [row["trans_residual_after"] for row in rows]
        ),
        "mean_rot_residual_before": mean([row["rot_residual_before"] for row in rows]),
        "mean_rot_residual_after": mean([row["rot_residual_after"] for row in rows]),
        "max_trans_residual_before": max(
            [row["trans_residual_before"] for row in rows], default=0.0
        ),
        "max_trans_residual_after": max(
            [row["trans_residual_after"] for row in rows], default=0.0
        ),
        "max_rot_residual_before": max(
            [row["rot_residual_before"] for row in rows], default=0.0
        ),
        "max_rot_residual_after": max(
            [row["rot_residual_after"] for row in rows], default=0.0
        ),
        "num_edges_improved": sum(1 for row in rows if row["improved"]),
        "num_edges_worsened": sum(1 for row in rows if row["worsened"]),
        "top_edges_by_trans_residual_after": top_after[:10],
        "top_edges_by_worsening": top_worsening[:10],
    }


def build_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        group: summarize_group(
            rows
            if group == "all"
            else [row for row in rows if row["edge_type"] == group]
        )
        for group in ["all", "tree", "loop", "unknown"]
    }


def write_csv_file(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in CSV_COLUMNS})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edges", required=True, help="Input pose graph edges JSON")
    parser.add_argument("--before", required=True, help="Poses before optimization")
    parser.add_argument("--after", required=True, help="Poses after optimization")
    parser.add_argument("--out_json", required=True, help="Output per-edge JSON")
    parser.add_argument("--out_csv", required=True, help="Output per-edge CSV")
    parser.add_argument("--summary", required=True, help="Output summary JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    edges = load_edges(Path(args.edges))
    poses_before = load_pose_map(Path(args.before))
    poses_after = load_pose_map(Path(args.after))

    rows = analyze_edges(edges, poses_before, poses_after)
    summary = build_summary(rows)

    write_json_file(Path(args.out_json), rows)
    write_csv_file(Path(args.out_csv), rows)
    write_json_file(Path(args.summary), summary)

    print(f"[OK] wrote edge residuals JSON -> {args.out_json}")
    print(f"[OK] wrote edge residuals CSV  -> {args.out_csv}")
    print(f"[OK] wrote summary JSON         -> {args.summary}")
    print(
        "[SUMMARY] edges={} improved={} worsened={} mean_after_trans={:.6g}".format(
            summary["all"]["num_edges"],
            summary["all"]["num_edges_improved"],
            summary["all"]["num_edges_worsened"],
            summary["all"]["mean_trans_residual_after"],
        )
    )


if __name__ == "__main__":
    main()
