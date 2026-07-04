#!/usr/bin/env python3
"""
Step 2: Build initial global poses (x,y,theta) by Maximum Reliability Spanning Tree (Dijkstra)
over edges_measurements.

Input:
  - edges_measurements.json (from Step 1)

Output:
  - initial_poses.json: pose for each pano_id in a common world frame

Usage:
  python src/02_init_poses_bfs.py \
    --edges data/group/58472_Floor1/edges_measurements.json \
    --out   data/group/58472_Floor1/initial_poses.json
"""

import argparse
import json
import heapq
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import defaultdict

# Import geometric functions from shared utils
sys.path.append(str(Path(__file__).parent.parent))
from src.utils.geom import se2_compose, invert_measurement


def load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text())


def save_json(p: Path, data: Dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def pick_root_node(edges: List[Dict[str, Any]]) -> str:
    """Pick a root: node with largest degree (more stable)."""
    deg = defaultdict(int)
    for e in edges:
        deg[e["i"]] += 1
        deg[e["j"]] += 1
    # choose max degree; stable tie-break: lexicographic
    return sorted(deg.keys(), key=lambda k: (-deg[k], k))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", type=str, required=True, help="edges_measurements.json")
    ap.add_argument("--out", type=str, required=True, help="initial_poses.json")
    ap.add_argument(
        "--root",
        type=str,
        default="",
        help="Optional root pano_id (default: auto pick by degree)",
    )
    args = ap.parse_args()

    edges_path = Path(args.edges)
    out_path = Path(args.out)

    data = load_json(edges_path)
    edges = data.get("edges", [])
    if not edges:
        raise RuntimeError("No edges found in edges_measurements.json")

    # Build adjacency (directed edge + inverse for traversal robustness)
    adj = defaultdict(list)
    nodes = set()
    for e in edges:
        i, j = e["i"], e["j"]
        m = e["measurement"]
        meas_ij = (float(m["dx"]), float(m["dy"]), float(m["dtheta"]))

        # Read or infer covariance/confidence for Priority Queue weight
        # Smaller sigma_xy => smaller weight => higher priority
        cost_ij = e.get("noise_sigma", {}).get("sigma_xy", 1.0)

        adj[i].append((j, meas_ij, cost_ij, "forward"))
        adj[j].append((i, invert_measurement(meas_ij), cost_ij, "inverse"))
        nodes.add(i)
        nodes.add(j)

    root = args.root.strip() if args.root.strip() else pick_root_node(edges)
    if root not in nodes:
        raise RuntimeError(f"Root {root} not found in graph nodes.")

    # Modified Dijkstra (Maximum Reliability Spanning Tree)
    poses: Dict[str, Tuple[float, float, float]] = {}
    visited = set()
    loop_closures = []

    component_id = 0
    for start in [root] + sorted([n for n in nodes if n != root]):
        if start in visited:
            continue

        component_id += 1

        # Offset disconnected components to prevent overlayting
        offset_x = (component_id - 1) * 20.0
        poses[start] = (offset_x, 0.0, 0.0)

        # Priority queue stores: (cumulative_cost, current_node, current_pose)
        pq = [(0.0, start, poses[start])]

        while pq:
            cost, u, pose_u = heapq.heappop(pq)

            # If we arrive at a visited node, it's either an old queue item or a loop closure
            if u in visited and u != start:
                continue

            poses[u] = pose_u
            visited.add(u)

            for v, meas_uv, edge_cost, tag in adj[u]:
                if v in visited:
                    # Record loop closure edge
                    if (v, u) not in loop_closures and (u, v) not in loop_closures:
                        loop_closures.append((u, v))
                    continue

                new_pose = se2_compose(pose_u, meas_uv)
                new_cost = cost + edge_cost
                heapq.heappush(pq, (new_cost, v, new_pose))

    # Alert User for disconnected components
    if component_id > 1:
        print(
            f"\\n[WARNING] Graph is disconnected! Found {component_id} separate components."
        )
        print(
            "[WARNING] Disconnected components have been artificially shifted by 20m to prevent visual overlap.\\n"
        )

    out = {
        "scene_id": data.get("scene_id", ""),
        "root": root,
        "num_nodes": len(nodes),
        "num_edges": len(edges),
        "components": component_id,
        "loop_closures_found": len(loop_closures),
        "poses": {
            k: {"x": poses[k][0], "y": poses[k][1], "theta": poses[k][2]}
            for k in sorted(poses.keys())
        },
        "note": "Initial poses via Dijkstra Spanning Tree. Loop closures bypassed.",
    }

    save_json(out_path, out)
    print(f"[OK] Wrote initial poses for {len(out['poses'])} nodes -> {out_path}")
    print(
        f"     root: {root}, nodes: {len(nodes)}, components: {component_id}, loops: {len(loop_closures)}"
    )


if __name__ == "__main__":
    main()
