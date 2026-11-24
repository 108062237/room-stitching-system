# 整合流程

## 1. 系統責任

| 套件負責 | 整合端負責 |
| --- | --- |
| 房間與連接方案排序 | 準備一致尺度的房間輪廓及配對資料 |
| 產生建議位置與旋轉角度 | 操作介面與方案選擇 |
| 將指定位置轉為預覽輪廓 | 確認位置及保存進度 |
| 回報輸入或狀態錯誤 | 使用者識別、任務識別及存取控制 |

三個計算介面不保存狀態。可由同一個 `Scene` 搭配不同 `AssemblyState` 處理不同任務；呼叫期間請勿由另一執行緒修改輸入物件。

## 2. 建議呼叫順序

```text
準備 Scene
    ↓
recommend_rooms(scene, 空配置) → 起始房間
    ↓
整合端設定起始房間 pose
    ↓
recommend_rooms(scene, 目前配置) → 下一間與連接對象
    ↓
suggest_placements(...) → 位置方案
    ↓
preview_placement(...) → 顯示輪廓
    ↓
整合端確認並保存 world_pose
    ↓
更新目前配置後，再次推薦
```

若使用者改變位置，直接把新的絕對位置傳給 `preview_placement`；預覽不會偷偷吸附或修改其他房間。

## 3. 確認位置後的配置更新

```python
# 僅在整合端已確認該位置後執行。
chosen = placements["items"][0]
state.placed[choice["room_id"]] = Pose2D.parse(chosen["world_pose"])

# 由整合端寫入資料庫或檔案。
payload = state.to_dict()

# 後續推薦使用更新後的配置。
next_choices = recommend_rooms(scene, state)
```

`save_scene()` 保存的是準備好的輪廓與候選輸入，不是這份操作進度。套件不會在加入後執行全局最佳化，其他已放置房間保持呼叫端提供的位置。

## 4. 無建議與錯誤處理

- `status=available`：可顯示推薦清單。
- `status=no_available_connections`：仍有房間未放置，但現有配對資料無可用連接；應提示補資料或由整合端處理。
- `status=all_rooms_placed`：所有房間有位置；是否進一步確認完成由整合端決定。
- `AssemblyError`：輸入或呼叫狀態不符，依 [API](API.md) 的錯誤碼處理，不視為推薦成功。

```python
from room_assembly import AssemblyError

try:
    result = suggest_placements(scene, state, room_a, room_b)
except AssemblyError as error:
    result = {"error": error.to_dict()}
```

檔案寫入的權限或磁碟錯誤可能由 I/O helper 拋出 `OSError`。若自行包裝 HTTP，HTTP 狀態碼、逾時及保存交易由服務端定義；套件本身不執行網路請求。

## 5. 識別與快取

- `scene_id` 是輸入場景識別，不是套件建立的 Session ID。
- 同一場景可用多份獨立配置，不共享保存狀態。
- `pair_id`、`candidate_id` 只在同一份準備好的 scene 中有效。
- 更換輪廓、頂點順序或候選資料後，需更新 scene 版本並清除依賴舊索引的快取。
- `excluded_pair_ids` 只過濾推薦；指定 pair 仍可直接查詢位置。停用權限或操作規則由整合端控制。

## 6. 結果呈現

`preview.json` 可由整合端的 Canvas、SVG 或 3D 引擎繪製；每個 `polygon` 已在整體 XY 座標中，不要再次套用該房間的 `world_pose`。

`render_preview()` 是選用 PNG helper：回傳 PNG bytes，只有提供目的路徑才寫檔。虛線表示本次預覽房間，不表示衝突。房間配色固定於 room ID，與推薦分數無關。
