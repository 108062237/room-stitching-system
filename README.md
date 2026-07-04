# Room Stitching System

這個專案用來做多房間的 floorplan stitching。輸入是一個場景中多張 panorama 對應的房間輪廓，流程會把每個房間的局部 layout 轉成 2D 平面 polygon，透過人工標註的房間對應點建立 pairwise constraints，再用 pose graph optimization 把所有房間拼到同一個座標系中。


## 專案目標

本系統主要處理以下問題：

1. 從 `layout_gt/*.txt` 讀取單一房間的 layout corner。
2. 將 panorama pixel 座標轉成房間局部 XY 平面座標。
3. 手動標註兩個房間之間的對應 corner 或 point。
4. 把人工標註轉成 SE(2) relative pose edge。
5. 建立所有房間的初始全域 pose。
6. 使用 GTSAM Pose2 pose graph optimization 最佳化房間位置。
7. 輸出 stitched floorplan overlay，檢查拼接結果。

## 環境安裝


```bash
cd room-stitching-system
conda env create -f environment.yml
conda activate room-stitching
```


## 資料結構

一個場景資料夾通常放在 `data/group/<scene_name>/`，例如 `data/group/58472_Floor1/`。

```text
data/group/<scene_name>/
├── manifest.json
├── panos/
│   └── <room_id>.jpg
├── layout/
│   └── <room_id>.json
├── layout_gt/
│   └── <room_id>.txt
├── matches/
│   └── pose_edge_matches.json
├── edges/
│   └── edges_measurements.json
├── poses/
│   ├── initial_poses.json
│   └── optimized_poses.json
└── viz/
    └── floorplan_overlay.png
```

主要流程使用的是 `layout_gt/*.txt`。`panos/` 用在互動式標註與人工檢查；`layout/` 是原始或中間格式，主流程不一定會直接讀。

## 常用檔案格式

詳細格式請看 [docs/json_formats.md](docs/json_formats.md)。這裡只列最常遇到的幾個。

### `layout_gt/<room_id>.txt`

每一行是一個 panorama pixel 座標：

```text
x y
```


### `matches/pose_edge_matches.json`

人工標註的房間對應點。每組 pair 使用 1-based corner index。

```json
{
  "scene_id": "58472_Floor1",
  "items": [
    {
      "src": "room_a_id",
      "dst": "room_b_id",
      "pairs": [[1, 7], [6, 8]]
    }
  ]
}
```

至少要有兩組 point pair，才能穩定估計一條 SE(2) relative pose edge。

### `edges/edges_measurements.json`

給 GTSAM 用的 pairwise pose constraints。

```json
{
  "scene_id": "58472_Floor1",
  "edges": [
    {
      "i": "room_a_id",
      "j": "room_b_id",
      "measurement": {
        "dx": 0.0,
        "dy": 0.0,
        "dtheta": 0.0
      },
      "noise_sigma": {
        "sigma_xy": 0.005,
        "sigma_theta": 0.001
      }
    }
  ]
}
```

方向約定是 `i -> j`，也就是 `z_ij = x_i^-1 * x_j`。

### `poses/*.json`

每個房間在全域座標中的 pose：

```json
{
  "poses": {
    "room_id": {
      "x": 0.0,
      "y": 0.0,
      "theta": 0.0
    }
  }
}
```

角度單位都是 radians。

## 快速開始

以下用 `data/group/58472_Floor1` 當範例。

### 1. 檢查 layout 是否能正確讀取

```bash
python -m src.system.test_room_loader_manual \
  --scene_dir data/group/58472_Floor1
```

這個工具會載入 `layout_gt/` 中的房間 polygon，並顯示每個 corner index。之後標註時會用到這些 index。

### 2. 標註兩個房間的對應點

```bash
python -m src.system.annotation_tool \
  --scene_dir data/group/58472_Floor1 \
  --src_room <src_room_id> \
  --dst_room <dst_room_id> \
  --out data/group/58472_Floor1/matches/pose_edge_matches.json
```

標註時請注意：

- corner index 是 1-based。
- 一條 pose edge 至少需要兩組對應點。
- 對應點品質會直接影響後續拼接結果。
- 如果同一個 scene 需要多組房間配對，可以多次執行並累積到同一份 matches。

### 3. 由 matches 產生 pose graph edges

```bash
python src/tool_generate_gtsam_edges.py \
  --matches data/group/58472_Floor1/matches/pose_edge_matches.json \
  --layout_dir data/group/58472_Floor1/layout_gt \
  --out data/group/58472_Floor1/edges/edges_measurements.json
```

### 4. 建立初始 poses

```bash
python src/02_init_poses_bfs.py \
  --edges data/group/58472_Floor1/edges/edges_measurements.json \
  --out data/group/58472_Floor1/poses/initial_poses.json
```

### 5. 執行 GTSAM 最佳化

```bash
python src/03_optimize_pose_graph_gtsam.py \
  --edges data/group/58472_Floor1/edges/edges_measurements.json \
  --init data/group/58472_Floor1/poses/initial_poses.json \
  --out data/group/58472_Floor1/poses/optimized_poses.json \
  --report data/group/58472_Floor1/poses/optimizer_report.json
```

### 6. 畫出拼接結果

```bash
python src/05_draw_floorplan_overlay.py \
  --scene_dir data/group/58472_Floor1 \
  --poses data/group/58472_Floor1/poses/optimized_poses.json \
  --out data/group/58472_Floor1/viz/floorplan_overlay.png
```

輸出的 overlay 圖是最重要的檢查結果。請確認相鄰房間是否接在合理位置、是否有大面積重疊，以及標註過的 corner 是否真的對齊。


```


## 重要座標約定

- room id 使用不含副檔名的 pano id。
- corner index 使用 1-based。
- 所有角度使用 radians。
- pose edge 方向是 `i -> j`。
- `layout_z`、`pano_w`、`pano_h` 在 edge generation、optimization、overlay visualization 中要保持一致。
- 預設參數是 `layout_z=50.0`、`pano_w=1024`、`pano_h=512`。

## 程式結構

```text
src/
├── system/
│   ├── room_loader.py
│   ├── annotation_tool.py
│   ├── floorplan_constraint_tool.py
│   ├── preview_pipeline.py
│   └── corner_editor.py
├── utils/
│   ├── geom.py
│   ├── post_proc.py
│   ├── axis_align.py
│   ├── labels.py
│   └── panostretch.py
├── tool_generate_gtsam_edges.py
├── 02_init_poses_bfs.py
├── 03_optimize_pose_graph_gtsam.py
├── 04_viz_pose_graph.py
├── 05_draw_floorplan_overlay.py
├── analyze_edge_residuals.py
├── analyze_candidate_residuals.py
├── evaluate_layout_objective.py
└── optimize_floorplan_objective.py
```

## 測試

修改程式後，至少跑一次：

```bash
pytest
```

格式檢查可以使用：

```bash
ruff check src tests
black src tests
```

如果在 server 或 sandbox 中執行 matplotlib 相關程式，可能會看到 font/cache 目錄不可寫的 warning。可以指定可寫的 cache 位置：

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache python src/05_draw_floorplan_overlay.py --help
```

## 建議工作流程

1. 確認 `manifest.json`、`panos/`、`layout_gt/` 都齊全。
2. 用 `test_room_loader_manual.py` 檢查每個房間 polygon 和 corner index。
3. 先標註最有把握的房間 pair。
4. 產生 edges、initial poses、optimized poses。
5. 畫 overlay 檢查結果。
6. 若有明顯錯位，回頭檢查標註或 edge residual。
7. 主流程穩定後，再嘗試加入 extra constraints。

