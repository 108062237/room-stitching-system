# JSON / Text File Formats (v1)

This document defines the canonical input/output formats used in the multi-room floorplan stitching system.

## Global Conventions

### Room ID
- Canonical room identifiers use **bare pano IDs** without file extensions.
- Example:
  - Correct: `08ddb587-468c-45b7-8681-e213539a1710`
  - Legacy accepted: `08ddb587-468c-45b7-8681-e213539a1710.txt`

### Corner Index
- All corner indices are **1-based**.
- Example: the first corner is `1`, not `0`.

### Angles
- All angles are stored in **radians**.
- Applies to:
  - `theta`
  - `dtheta`
  - `sigma_theta`

### Pairwise Direction
- All pairwise transforms are defined as **source -> destination**.
- In edge files:
  - `i` = source room
  - `j` = destination room
  - `measurement = (dx, dy, dtheta)` means the relative pose from `i` to `j`

---

## 1. Scene Manifest

### File
`manifest.json`

### Purpose
Defines the rooms in a scene, file paths, hotspot-based connectivity, and scene-level metadata.

### Canonical Schema
```json
{
  "scene_id": "58472_Floor1",
  "scene_dir": "data/group/58472_Floor1",
  "panos_dir": "data/group/58472_Floor1/panos",
  "layout_gt_dir": "data/group/58472_Floor1/layout_gt",
  "hotspot_json": "data/group/58472_Floor1/Floor1_HOTSPOT.json",
  "nodes": [
    {
      "pano_id": "08ddb587-058e-49f4-8e21-2644539d741d",
      "image_path": "data/group/58472_Floor1/panos/08ddb587-058e-49f4-8e21-2644539d741d.jpg",
      "layout_gt_path": "data/group/58472_Floor1/layout_gt/08ddb587-058e-49f4-8e21-2644539d741d.txt",
      "room_idx": 3,
      "connections": [
        {
          "neighbor": "08ddb587-468c-45b7-8681-e213539a1710",
          "hotspot_xy": [0.4543, -2.2076],
          "index_in_json": 0
        }
      ]
    }
  ],
  "edges_raw": [],
  "stats": {}
}
```

### Required Fields
- `scene_id`
- `nodes`

### Required Node Fields
- `pano_id`
- `layout_gt_path`

### Notes
- `connections` are hotspot-based adjacency hints and are **not equivalent** to manually verified corner correspondences.
- `edges_raw` may store raw connectivity extracted from hotspot metadata.

---

## 2. Layout Ground Truth Text File

### File
`layout_gt/<room_id>.txt`

### Purpose
Stores room layout corner observations in panorama pixel coordinates.

### Current Format
Each line contains two numbers:
```text
x y
```

### Example
```text
107 163
107 353
153 174
153 342
...
```

### Notes
- The current pipeline assumes alternating ceiling/floor points when the number of rows is even.
- The floor points are extracted by taking the point with the larger image y-coordinate in each pair.
- These points are then projected into local floor XY coordinates using:
  - `np_coor2xy(z=50)`
  - center shift
  - Y flip
  - optional `rectify_polygon`

---

## 3. Pose Edge Matches

### File
`pose_edge_matches.json`

### Purpose
Stores manually annotated room-to-room correspondences that contain **at least two point pairs**, so they can be used to generate a rigid SE(2) relative pose edge.

### Canonical Schema
```json
{
  "scene_id": "58472_Floor1",
  "version": "1.0",
  "items": [
    {
      "src": "08ddb586-b06d-4fab-8ee6-90c4fb85c6c5",
      "dst": "08ddb586-be87-4f65-8199-4f9e21ff613b",
      "pairs": [[1, 7], [6, 8]],
      "source": "manual",
      "comment": ""
    }
  ]
}
```

### Required Fields
- `src`
- `dst`
- `pairs`

### Rules
- `pairs.length >= 2`
- each item in `pairs` is `[src_corner_idx, dst_corner_idx]`
- all indices are 1-based

### Legacy Compatibility
The previous format:
```json
{
  "src": "roomA.txt",
  "dst": "roomB.txt",
  "idx_src": [1, 6],
  "idx_dst": [7, 8]
}
```

should be normalized internally into:
```json
{
  "src": "roomA",
  "dst": "roomB",
  "pairs": [[1, 7], [6, 8]]
}
```

---

## 4. Extra Point Constraints

### File
`extra_point_constraints.json`

### Purpose
Stores additional point-level constraints used to supplement the main pose graph.
These constraints may contain only one pair and are **not required** to define a full rigid transform.

### Canonical Schema
```json
{
  "scene_id": "58472_Floor1",
  "version": "1.0",
  "items": [
    {
      "src": "08ddb587-8ae1-436b-8182-eb0738a27e69",
      "dst": "08ddb587-058e-49f4-8e21-2644539d741d",
      "pairs": [[4, 3]],
      "weight": 1.0,
      "source": "manual",
      "comment": ""
    }
  ]
}
```

### Rules
- `pairs.length >= 1`
- can represent single-point or multi-point supplemental constraints
- should not be interpreted as a rigid SE(2) edge unless `pairs.length >= 2` and a fitting step is explicitly performed

---

## 5. Pose Graph Edges

### File
`edges.json` or `perfect_edges.json`

### Purpose
Stores the relative pose constraints used by GTSAM pose graph optimization.

### Canonical Schema
```json
{
  "scene_id": "58472_Floor1",
  "version": "1.0",
  "coordinate_convention": "src_to_dst_pose2",
  "edges": [
    {
      "i": "08ddb586-b06d-4fab-8ee6-90c4fb85c6c5",
      "j": "08ddb586-be87-4f65-8199-4f9e21ff613b",
      "measurement": {
        "dx": -58.3188,
        "dy": 149.3835,
        "dtheta": -2.5655
      },
      "noise_sigma": {
        "sigma_xy": 0.005,
        "sigma_theta": 0.001
      },
      "source_match": {
        "pairs": [[1, 7], [6, 8]]
      },
      "meta": {
        "source": "perfect_manual_match",
        "confidence": 1.0
      }
    }
  ]
}
```

### Required Fields
- `scene_id`
- `edges`

### Required Edge Fields
- `i`
- `j`
- `measurement.dx`
- `measurement.dy`
- `measurement.dtheta`
- `noise_sigma.sigma_xy`
- `noise_sigma.sigma_theta`

### Notes
- `measurement` always means the relative transform from room `i` to room `j`
- `source_match` is recommended for traceability, even if it is optional

---

## 6. Initial Poses

### File
`initial_poses.json`

### Purpose
Stores the initial global absolute poses used to initialize pose graph optimization.

### Canonical Schema
```json
{
  "scene_id": "58472_Floor1",
  "root": "08ddb587-058e-49f4-8e21-2644539d741d",
  "num_nodes": 9,
  "num_edges": 8,
  "components": 1,
  "loop_closures_found": 8,
  "poses": {
    "08ddb587-058e-49f4-8e21-2644539d741d": {
      "x": 0.0,
      "y": 0.0,
      "theta": 0.0,
      "component": "main"
    }
  },
  "note": "Initial poses via spanning tree expansion."
}
```

### Required Fields
- `scene_id`
- `root`
- `poses`

### Required Pose Fields
- `x`
- `y`
- `theta`

### Recommended Fields
- `component`:
  - `"main"` for the root-connected component
  - `"island"` for disconnected components

---

## 7. Optimized Poses

### File
`optimized_poses.json`

### Purpose
Stores the final optimized room poses after pose graph optimization.

### Canonical Schema
```json
{
  "scene_id": "58472_Floor1",
  "root": "08ddb587-058e-49f4-8e21-2644539d741d",
  "num_nodes": 9,
  "poses": {
    "08ddb587-058e-49f4-8e21-2644539d741d": {
      "x": 0.0,
      "y": 0.0,
      "theta": 0.0
    }
  },
  "note": "Optimized poses via GTSAM Pose2."
}
```

---

## 8. Residual Report

### File
`residual_report.json`

### Purpose
Stores optimization diagnostics and summary statistics for experiments.

### Canonical Schema
```json
{
  "scene_id": "58472_Floor1",
  "root": "08ddb587-058e-49f4-8e21-2644539d741d",
  "num_nodes": 9,
  "num_edges": 8,
  "graph_diagnostics": {
    "raw_graph_error_before": 12345.6,
    "raw_graph_error_after": 12.3
  },
  "optimizer": {
    "type": "LevenbergMarquardt",
    "max_iters": 100
  },
  "before": {
    "translation": {},
    "rotation_rad": {}
  },
  "after": {
    "translation": {},
    "rotation_rad": {}
  }
}
```

---

## Summary

### Main Distinction
- `pose_edge_matches.json`:
  manual correspondences used to generate rigid pairwise edges
- `extra_point_constraints.json`:
  supplemental geometric constraints, not necessarily full rigid edges
- `edges.json`:
  optimized graph constraints
- `initial_poses.json`:
  spanning-tree-based initial guess
- `optimized_poses.json`:
  final global room poses

### Recommended Practice
- Use **bare room IDs** internally
- Keep indices **1-based**
- Keep all angles in **radians**
- Keep transform direction consistent as **source -> destination**