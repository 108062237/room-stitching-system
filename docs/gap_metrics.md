# Gap Metrics Design

這份文件定義 stitched floorplan 的量化指標，用來評估房間重疊、整體空隙、已 match 房間是否貼合，以及尚未被 pose edge 處理的 candidate wall gap。

指標會輸出在 `evaluate_layout_objective()` report 的 `gap_metrics` 欄位。

## Inputs

每個房間有：

```text
x_i = (t_x, t_y, theta)
P_i_local
P_i = T(x_i) P_i_local
```

所有轉到 global frame 後的房間 polygons 為：

```text
{P_1, P_2, ..., P_N}
```

定義：

```text
A_total = sum_i Area(P_i)
U = Union(P_1, P_2, ..., P_N)
A_union = Area(U)
```

## 1. Room Overlap Ratio

房間重疊比例用來衡量不同房間互相壓到的程度。

```text
A_overlap = A_total - A_union
R_overlap = A_overlap / A_total
```

意義：

```text
R_overlap = 0      沒有重疊
R_overlap 越大     房間重疊越嚴重
```

輸出欄位：

```json
{
  "ratio": 0.0,
  "overlap_area": 0.0,
  "total_room_area": 0.0,
  "union_area": 0.0,
  "room_areas": {},
  "pairwise_overlap_sum": 0.0,
  "pairwise_details": []
}
```

注意：`overlap_area = A_total - A_union` 是 union-based overlap；`pairwise_overlap_sum` 是每兩間房的 pairwise overlap 加總。當三間以上房間在同一區域重疊時，兩者可能不同。

## 2. Global Gap Ratio

全域空隙比例用來估計整體 building envelope 內的空洞或縫隙。

沒有 ground-truth building outline 時，使用 morphological closing 估計 envelope：

```text
E_r = Closing(U, r) = Erode(Dilate(U, r), r)
A_gap_r = Area(E_r - U)
R_global_gap(r) = A_gap_r / Area(E_r)
```

實作使用 Shapely：

```text
Closing(U, r) = U.buffer(r).buffer(-r)
```

`r` 是 closing radius。建議做 sweep：

```text
r = 0.2m, 0.5m, 1.0m
```

意義：

```text
小 r 有 gap      代表非常細的縫也被偵測到
大 r 才有 gap    代表縫比較寬或房間分離較明顯
ratio 越大       envelope 裡空白越嚴重
```

## 3. Matched Gap Ratio

這是原本 `Adjacency Gap Ratio` 的改名版本。它不是主要 gap 指標，而是 sanity check，用來確認已經有 pose edge / manual match 的房間對是否真的貼合。

給定 pose edges：

```text
E_pose = {(i, j)}
```

定義：

```text
R_matched_gap =
sum_(i,j in E_pose) w_ij min(Distance(P_i, P_j) / tau_gap, 1)
/
sum_(i,j in E_pose) w_ij
```

若沒有設定權重，預設：

```text
w_ij = 1
```

意義：

```text
R_matched_gap ≈ 0
=> pose edge 對應的房間確實被拼在一起。
```

但它不能代表整張 floorplan 沒有 gap，因為它只看已經有 match 的房間 pair。

輸出欄位：

```json
{
  "ratio": 0.0,
  "weighted_penalty_sum": 0.0,
  "total_weight": 0.0,
  "num_edges": 0,
  "details": [
    {
      "src": "room_a",
      "dst": "room_b",
      "distance": 0.0,
      "tolerance": 0.5,
      "penalty": 0.0,
      "weight": 1.0
    }
  ]
}
```

## 4. Candidate Wall Gap Ratio

這是主要 gap 指標，用來衡量「幾何上可能應該貼合，但目前沒有被 pose edge 處理」的牆段還有多少縫。

目前實作會自動偵測 candidate wall pairs：

- 不同房間。
- 預設排除已經有 `pose_edges` 的 room pair。
- 牆段方向同軸，使用 Manhattan axis 判斷。
- 投影有足夠重疊長度。
- 兩牆距離小於 `candidate_wall_max_distance`。

對每一組 candidate wall pair：

```text
d_ab = distance between wall segment a and b
l_ab = projected overlap length
w_ab = confidence / weight
```

定義：

```text
R_candidate_wall_gap =
sum_(a,b in C) w_ab l_ab min(d_ab / tau_gap, 1)^p
/
sum_(a,b in C) w_ab l_ab
```

目前預設：

```text
p = 2
w_ab = 1
```

使用平方可以讓大 gap 更明顯。

輸出欄位：

```json
{
  "ratio": 0.0,
  "weighted_penalty_sum": 0.0,
  "total_support": 0.0,
  "num_candidates": 0,
  "details": [
    {
      "room_a": "room_a",
      "room_b": "room_b",
      "wall_a_idx": 0,
      "wall_b_idx": 2,
      "axis": "horizontal",
      "distance": 0.0,
      "overlap_length": 0.0,
      "weight": 1.0,
      "support": 0.0,
      "tolerance": 0.5,
      "penalty": 0.0,
      "wall_a": [[0.0, 0.0], [1.0, 0.0]],
      "wall_b": [[0.0, 1.0], [1.0, 1.0]]
    }
  ]
}
```

## 5. Unexplained Gap Ratio

Global gap 只能說「有 gap」，但不能說 gap 是哪兩面牆造成的。`Unexplained Gap Ratio` 用來檢查 candidate wall detection 是否有抓到 global gap 的原因。

定義 global gap region：

```text
G_r = Closing(U, r) - U
```

由 candidate wall pairs 產生可解釋 gap 區域：

```text
G_explained
```

剩下：

```text
G_unexplained = G_r - G_explained
```

定義：

```text
R_unexplained_gap(r) =
Area(G_unexplained) / Area(G_r)
```

意義：

```text
R_unexplained_gap 高
=> global gap 很多，但 candidate wall detection 沒有抓到原因。
```

輸出欄位：

```json
{
  "details": [
    {
      "closing_radius": 1.0,
      "gap_area": 0.0,
      "explained_gap_area": 0.0,
      "unexplained_gap_area": 0.0,
      "ratio": 0.0
    }
  ]
}
```

## Parameters

可在 layout objective input JSON 的 `params` 覆蓋：

```json
{
  "params": {
    "global_gap_closing_radii": [0.2, 0.5, 1.0],
    "matched_gap_tolerance": 0.5,
    "candidate_wall_gap_tolerance": 0.5,
    "candidate_wall_max_distance": 2.0,
    "candidate_wall_min_overlap": 0.25,
    "candidate_wall_penalty_power": 2.0,
    "candidate_wall_exclude_matched_pairs": true
  }
}
```

## How To Interpret

建議一起看：

```text
High R_overlap + low R_matched_gap
=> 已 match 的房間貼合，但可能貼太近造成重疊。

Low R_overlap + low R_matched_gap + high R_candidate_wall_gap
=> 已有 edge 的地方沒問題，但未 match 的候選牆段有縫。

High R_global_gap + high R_candidate_wall_gap
=> 整體有 gap，且可由 candidate wall pair 解釋，適合拿來做 optimization。

High R_global_gap + high R_unexplained_gap
=> 有 gap，但 candidate wall detection 沒抓到原因，需要改善 candidate detection 或補 adjacency/match。

Low all
=> 從這組幾何指標看，拼接結果較合理。
```

