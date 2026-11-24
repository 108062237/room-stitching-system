# API 規格

適用套件版本：0.1.1。

穩定入口為 `room_assembly.recommend_rooms`、`suggest_placements`、`preview_placement`。資料讀寫在 `adapters`；繪圖在 `rendering`，不是額外的拼接服務。三項計算均不讀寫任務檔案、不發網路請求、不執行模型、不重新最佳化已放置房間。

## 共用資料

`scene: Scene`：房間局部 XY 輪廓、已準備的位置候選與配對評分。

`current_state: AssemblyState | dict`：

```json
{
  "placed": {
    "A": {"x": 0.0, "y": 0.0, "theta_deg": 0.0}
  },
  "excluded_pair_ids": []
}
```

`excluded_pair_ids` 是選用的推薦過濾清單，可由整合端管理略後處理／停用；它不是 toolkit 保存的狀態。未知房間或 pair ID 會報錯。`pair_id` 僅在同一份準備好的 scene 中有效，不保證重新生成或重排輸入後仍相同。

## 1. recommend_rooms(scene, current_state, top_k=5)

| 參數 | 型別 | 規則 |
| --- | --- | --- |
| scene | Scene | 必填 |
| current_state | AssemblyState / dict | 必填；開始時 `placed={}` |
| top_k | int | 選用，正整數，不接受 bool |

有已放置房間時的輸出範例（數值僅為示意）：

```json
{
  "scene_id": "demo",
  "mode": "insertion",
  "status": "available",
  "remaining_rooms": 2,
  "score_direction": "higher_is_better",
  "score_type": "connection_confidence",
  "total": 2,
  "items": [{
    "rank": 1,
    "room_id": "B",
    "connect_to": "A",
    "pair_id": "edge_0000",
    "score": 0.225,
    "risk_tier": "high",
    "risk_reasons": ["layout_only_pose_ambiguous"],
    "source": "da3_layout_suggestion"
  }]
}
```

- `total`：截取 top_k 前的連接建議數，不是剩餘房間數。
- `remaining_rooms`：目前尚未放置房間數；`mode=start` 時不回傳此欄位。
- `status=available`：存在可用建議。
- `status=no_available_connections`：仍有房間未放置，現有資料無可用建議。
- `status=all_rooms_placed`：全部房間有位置；不表示經人工確認正確。
- 尚未放置房間時 `mode=start`，`score_type=weighted_degree`；`connect_to`、`pair_id` 是 null，item 不包含 risk/source 欄位。
- 相同待加入房間可因連接對象不同而重複出現。這是連接方案排名，不是互斥的房間清單。

## 2. suggest_placements(scene, current_state, room_a, room_b, top_k=5, *, pair_id=None)

房間 A 必須已放置，B 必須未放置。`room_a`、`room_b` 為非空字串；`top_k` 為正整數；`pair_id` 為選用的配對識別字串。建議將推薦結果中的 `connect_to`、`room_id`、`pair_id` 原樣傳入。

```json
{
  "scene_id": "demo",
  "pair_id": "edge_0000",
  "room_a": "A",
  "room_b": "B",
  "score_direction": "lower_is_better",
  "score_type": "layout_cost",
  "total": 16,
  "items": [{
    "rank": 1,
    "candidate_id": "layout_0002@000",
    "relative_pose": {"dx": 10.0, "dy": -35.0, "dtheta_deg": 0.0},
    "world_pose": {"x": 10.0, "y": -35.0, "theta_deg": 0.0},
    "score": 0.0
  }]
}
```

上例 A 位於原點且角度為 0，兩種座標數值剛好一致。一般情況下不同。

- 按完整位置代價排序取前 top_k；`total` 為截取前方案數。
- 位置推薦使用幾何評分與內部吸附；回傳資料不包含通過／拒絕決定或衝突判定。
- 不檢查 `excluded_pair_ids` 是否禁止直接查位置：該欄位只過濾推薦。整合端可直接查詢 scene 中 pair 的位置。
- 同一對房間有多筆 edge 時，必須指定 pair_id。

## 3. preview_placement(scene, current_state, room_id, world_pose)

`world_pose` 是 `Pose2D` 或恰含 `x, y, theta_deg` 的 dict。房間可以是新房間或既有房間；後者只替換該房間的預覽位置。

```json
{
  "scene_id": "demo",
  "coordinate_frame": "world_xy",
  "units": "layout_units",
  "preview_room_id": "B",
  "rooms": [{
    "room_id": "B",
    "preview": true,
    "world_pose": {"x": 50.0, "y": 0.0, "theta_deg": 0.0},
    "polygon": [[30.0,-15.0],[70.0,-15.0],[70.0,15.0],[30.0,15.0]]
  }],
  "bounds": {"min_x": -30.0, "min_y": -20.0, "max_x": 70.0, "max_y": 20.0}
}
```

`rooms` 實際包含所有已放置房間及本次預覽房間，上例省略 A。bounds 根據全部輸出輪廓計算，不含顯示留白。

**無 saved、accepted、valid、conflicts 欄位。** 重疊、接觸、遠離都依指定 pose 顯示。輸入型別／有限值／輪廓格式檢查仍存在，這是資料契約驗證，不是位置幾何檢查。

## 錯誤

三個介面的預期輸入錯誤使用 `AssemblyError`，為 ValueError 子類別：

```python
from room_assembly import AssemblyError
try:
    result = suggest_placements(scene, state, "A", "B")
except AssemblyError as error:
    payload = error.to_dict()
    # {"code": "NO_PAIR_DATA", "message": "...", "details": {}}
```

| code | 意義 |
| --- | --- |
| INVALID_INPUT | 型別、欄位、數值、輪廓或索引不符 |
| UNKNOWN_ROOM | 指定或配置中的房間不存在 |
| INVALID_STATE | A 未放置、B 已放置等狀態不符 |
| NO_PAIR_DATA | 找不到該配對的位置資料 |
| AMBIGUOUS_PAIR | 同一配對有多筆 edge，需指定 pair_id |
| INPUT_READ_ERROR | adapter 無法讀取指定檔案 |
| UNSUPPORTED_SCHEMA | scene JSON 版本不支援 |

檔案寫入的權限／磁碟錯誤由 I/O helper 拋出 OSError；未預期的計算錯誤不偽裝成「沒有推薦」。本套件沒有 HTTP status code，整合端自行映射。
