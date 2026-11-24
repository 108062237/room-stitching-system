# 輸入輸出規格

## 1. 資料類型

| 類型 | 內容 | 用途 |
| --- | --- | --- |
| `Scene` | 房間局部輪廓、配對、位置候選與評分資料 | 三個公開介面的共用輸入 |
| `AssemblyState` | 目前已放置房間的整體位置、排除配對清單 | 每次操作的配置輸入 |
| 推薦結果 | 排名、房間、連接對象、分數及狀態 | 下一步選擇 |
| 位置結果 | 相對 pose、world pose、代價及排名 | 擺放方案選擇 |
| 預覽結果 | 整體座標輪廓、預覽房間標記及 bounds | 繪製配置 |

三個公開介面的完整回傳格式見 [API.md](API.md)。程式內使用的輸入與實際示範輸出見 [examples/data](../examples/data/)。

## 2. 局部輪廓與配對資料

可使用以下 JSON 作為應用程式的資料準備輸入，再傳給 `build_scene(**payload)`：

```json
{
  "scene_id": "example_floor",
  "room_polygons": {
    "A": [[-30, -20], [30, -20], [30, 20], [-30, 20]],
    "B": [[-20, -15], [20, -15], [20, 15], [-20, 15]]
  },
  "geoverify_results": [],
  "da3_pairs": [{"room_a": "A", "room_b": "B", "probability": 0.9}]
}
```

| 欄位 | 型別 | 規則 |
| --- | --- | --- |
| scene_id | string | 非空的場景識別 |
| room_polygons | object | room ID → Nx2 局部 XY 輪廓，至少一間房 |
| geoverify_results | array | 選用；完整單對幾何結果物件 |
| da3_pairs | array | 選用；正規化的連通配對物件 |

輪廓必須符合：

- 至少三點，座標為有限數值，面積大於 0。
- 簡單多邊形，無自交及零長邊。
- 末點不重複首點，頂點順序固定。
- 所有房間使用相同尺度及相容的局部座標定義。

XY 輸入不會再自動校正或重新置中。若輸入是全景圖角點 TXT，請使用第 5 節的投影轉接器。

### 2.1 正規化配對

每筆 `da3_pairs` 恰含 `room_a`、`room_b`、`probability`。兩端必須是 scene 中不同的房間；probability 為 0…1 的有限數值。

本欄位不是其他專案整包輸出檔的直接入口。呼叫端需先解析來源檔案、對應房間 ID，並依來源模型的判斷門檻選擇要交給套件的配對。照片觀看方向或 attention 不能填入房間旋轉角度或連通機率。

### 2.2 完整單對幾何結果

| 欄位 | 用途 |
| --- | --- |
| img_a、img_b | 配對方向，對應 room ID |
| pose.R | 3×3 相對旋轉矩陣 |
| pose.cam_b_in_a 或 pose.az_ab_deg | A 看向 B 的方位 |
| pose.az_ba_deg | B 看向 A 的方位 |
| flags.no_covisibility、degenerate_rot_only | 量測有效性 |
| n、inlier_rate、quality、regions | 候選信心與風險特徵 |

本版本 `build_scene()` 使用此完整單對格式。整房 JSON、僅含共視區域的公開格式，以及其他模型輸出的相機 pose，必須由整合端另行轉換，不會自動合併進 scene。

缺少有效旋轉或方位時，原因記錄於 `scene.metadata.unavailable_pairs`；有正規化配對時才可能使用 layout-only 備援。沒有可用位置不等於確定不相通。重複或反向重複的完整單對幾何結果會報錯，避免不明確地選用其中一份。

## 3. 準備後的 Scene JSON

由 `save_scene(scene, path)` 產生，再由 `load_scene(path)` 讀取。其 `schema_version` 是資料格式版本，和套件版本分開管理。

| 欄位 | 型別 | 說明 |
| --- | --- | --- |
| schema_version | string | 固定 `1.0` |
| scene_id | string | 場景識別 |
| units | string | 固定 `layout_units` |
| angle_unit | string | 固定 `radians`，指候選 measurement |
| root_id | string / null | 起始房間識別 |
| room_polygons | object | room ID → 局部 Nx2 輪廓 |
| pairs | array | 配對及位置候選 |
| metadata | object | 資料來源與候選準備紀錄 |

`pairs` 由套件準備，建議透過 `build_scene` 或相容性載入器產生，避免手動修改評分欄位。配對主要欄位如下：

| 欄位 | 說明 |
| --- | --- |
| index | scene 內唯一、非負整數，對外形成 pair_id |
| room_a、room_b | 量測的正向端點 |
| measurement | `[dx, dy, dtheta_rad]`，B 在 A 座標下的位置 |
| selected_candidate_id | 基礎位置方案識別 |
| confidence、automatic_confidence | 0…1 的連接評分 |
| risk_tier、risk_reasons | 風險等級與原因 |
| candidates | 位置候選清單 |

每個 candidate 包含 `candidate_id`、`measurement`、`wall_a_index`、`wall_b_index`、`candidate_score`、`intrinsic_cost`、`slide`、`slide_cost`、`overlap_ratio`、`polygon_overlap_ratio`。牆索引必須和輸入輪廓的頂點順序一致。

資料需要重現所有欄位時，請使用 `scene_to_dict()`，或以示範輸出的 `scene.json` 為準。

## 4. 目前配置

```json
{
  "placed": {
    "A": {"x": 0.0, "y": 0.0, "theta_deg": 0.0}
  },
  "excluded_pair_ids": []
}
```

- `placed` 必填，初次使用可為空物件；每個 room ID 必須存在於 scene。
- 每個 pose 恰含 `x`、`y`、`theta_deg`，值為有限數值。
- `excluded_pair_ids` 選填，僅影響房間推薦，值必須存在於同一 scene。
- 輸入使用 degree；不要將 scene 內的 radian measurement 直接填入 `theta_deg`。

## 5. 檔案轉接器

### Layout TXT

`read_layout_directory()` 讀取目錄下 `*.txt`，以檔名（不含副檔名）作為 room ID。每行為一組全景像素座標 `x y`。天花／地板角點交錯時，取每對 y 較大的地板點。

預設投影圖尺寸為 1024×512、投影高度 `layout_z=50`。檔案尺寸不同時，使用 `pano_width`、`pano_height` 指定。此載入路徑會進行直角校正；直接提供 XY 輪廓時不進行該校正。

### 正規化前的 CSV

相容性工具 `load_legacy_scene()` 支援已有的候選產物，不需產生候選的其他專案原始碼：

| 檔案 | 使用欄位 |
| --- | --- |
| combined_results.csv | house、room_a/b、status、guided_dx/dy/dtheta、best_guided_candidate_id、yaw_snap_residual_deg、num_valid_guided_candidates |
| candidates.csv | room_a/b、candidate_valid、candidate_id、dx/dy/dtheta、wall_a_id/b_id、candidate_score、shared_wall_overlap_ratio、polygon_overlap_ratio |
| DA3 predictions.csv | pano_a_path、pano_b_path、probability |

CSV 的 `dtheta` 是 radian。DA3 CSV 透過 `.../<scene>/panos/<room>.jpg` 路徑解析識別，不讀取影像。GT 誤差與 GT 最佳位置等額外欄位不參與選解。

缺少候選 CSV 時，使用已有相對量測作為基礎候選；缺少幾何品質資料時可能降低配對分數。資料缺席造成的結果差異不應解讀為相同輸入下的結果改變。

## 6. 資料來源與結果解讀

輪廓及配對結果由整合端提供。若輪廓經人工修正、配對清單經人工指定或來自已知連接，結果亦受這些輸入條件影響。應保存模型版本、輪廓版本與配對來源，不以輸出分數取代獨立驗收。

預覽不輸出位置是否合理的判斷。`save_scene`、範例程式及 PNG helper 會在明確呼叫時寫檔；三個公開計算介面只回傳資料。
