# Room Assembly Toolkit

房間拼接 Python 套件。輸入各房間輪廓、配對資訊及目前配置，取得下一個房間的推薦順序、建議位置與 2D 拼接預覽。

| 功能 | Python 介面 | 輸出 |
| --- | --- | --- |
| 房間推薦與優先順序 | `recommend_rooms()` | 推薦房間、連接對象、分數與排序 |
| 房間拼接位置建議 | `suggest_placements()` | 相對平移、旋轉角度與整體座標位置 |
| 拼接位置預覽 | `preview_placement()` | 各房間的世界座標輪廓與顯示範圍 |

本套件提供計算介面。整合端負責操作畫面、位置確認及進度保存。位置推薦本身使用幾何評分選擇方案。

## 目錄

- [環境需求與安裝](#環境需求與安裝)
- [快速開始](#快速開始)
- [程式整合](#程式整合)
- [輸入與輸出](#輸入與輸出)
- [資料準備](#資料準備)
- [文件](#文件)
- [測試](#測試)
- [版本與使用範圍](#版本與使用範圍)

## 環境需求與安裝

- Python 3.10 以上；已驗證環境為 Linux / Python 3.12.2。
- 核心依賴：NumPy、Shapely。
- PNG 輸出另需 Matplotlib。
- 核心計算在 CPU 執行，不需要 GPU 或模型權重。

取得原始碼後，於專案根目錄執行：

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install '.[render]'
```

**Windows PowerShell**

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install '.[render]'
```

僅需 JSON 結果時使用 `python -m pip install .`。已驗證依賴版本列於 [constraints-tested.txt](constraints-tested.txt)，安裝時可加 `-c constraints-tested.txt`；其他作業系統尚未實機驗證。

亦可安裝交付的 wheel：

```bash
python -m pip install 'room_assembly_toolkit-0.1.1-py3-none-any.whl[render]'
```

## 快速開始

```bash
python examples/demo.py --output outputs/demo
```

範例使用套件內附的合成資料，不需額外資料集。執行後產生：

```text
outputs/demo/
├── scene.json             # 準備好的房間與候選資料
├── start.json             # 起始房間建議
├── state.json             # 本次呼叫使用的配置
├── recommendations.json   # 下一間房間的推薦清單
├── placements.json        # 拼接位置方案
├── preview.json           # 預覽輪廓及顯示範圍
└── preview.png            # 2D 預覽圖
```

合成資料只用於介面示範，不代表實際場景的拼接準確率。

## 程式整合

```python
from room_assembly import (
    AssemblyState, Pose2D,
    recommend_rooms, suggest_placements, preview_placement,
)
from room_assembly.adapters import load_scene
from room_assembly.rendering import render_preview

scene = load_scene("outputs/demo/scene.json")

# 取得起始房間並設定其整體位置。
start = recommend_rooms(scene, AssemblyState())
root = start["items"][0]["room_id"]
state = AssemblyState(placed={root: Pose2D(0, 0, 0)})

# 1. 房間推薦與排序。
ranking = recommend_rooms(scene, state, top_k=5)
if ranking["items"]:
    choice = ranking["items"][0]

    # 2. 查詢指定配對的位置方案。
    placements = suggest_placements(
        scene, state,
        room_a=choice["connect_to"],
        room_b=choice["room_id"],
        pair_id=choice["pair_id"],
        top_k=5,
    )
    pose = placements["items"][0]["world_pose"]

    # 3. 預覽指定位置。
    preview = preview_placement(scene, state, choice["room_id"], pose)
    render_preview(preview, "outputs/preview.png")
else:
    print(ranking["status"])
```

三個介面均回傳可序列化為 JSON 的 `dict`，不修改傳入的配置。整合端確認位置後，將該房間的 `world_pose` 加入下一次呼叫的 `placed`，並自行保存；套件不執行全局位置最佳化。

## 輸入與輸出

| 資料 | 必要性 | 說明 |
| --- | --- | --- |
| 房間識別與局部 XY 輪廓 | 必要 | 每間房使用唯一 ID 與固定頂點順序 |
| 配對與位置候選資料 | 推薦及位置建議需要 | 可讀取準備好的 scene，或由輪廓與模型結果產生 |
| 目前已放置房間的位置 | 必要 | 初次呼叫可為空；後續由整合端更新 |
| GeoVerify 完整單對結果 | 產生幾何引導候選時使用 | 提供相對方向、旋轉與品質資訊 |
| DA3 正規化配對分數 | 選用 | 提供缺少幾何引導候選時的備援配對 |

核心約定：

- 公開操作介面角度為 **degree**；scene JSON 內部 `measurement` 為 **radian**。
- XY 為 **layout units**，未經尺度校準不得當成公尺。
- `relative_pose` 是 B 在 A 局部座標中的位置；`world_pose` 是整體配置的位置。
- 房間推薦分數越高越優先；位置代價越低越優先。分數不是經校準的正確機率。
- `no_available_connections` 表示目前沒有可用連接，仍可能有未放置房間。
- `all_rooms_placed` 只代表所有房間已有位置，不代表已確認位置正確。

完整格式見 [輸入輸出規格](docs/INPUT_OUTPUT.md) 與 [座標規格](docs/COORDINATES.md)。

## 資料準備

**使用局部輪廓與正規化配對資料：**

```python
from room_assembly.adapters import build_scene, save_scene

scene = build_scene(
    scene_id="example_floor",
    room_polygons={
        "A": [[-30, -20], [30, -20], [30, 20], [-30, 20]],
        "B": [[-20, -15], [20, -15], [20, 15], [-20, 15]],
    },
    da3_pairs=[{"room_a": "A", "room_b": "B", "probability": 0.9}],
)
save_scene(scene, "outputs/example_floor.json")
```

**使用 layout TXT 與完整單對模型輸出：**

```bash
python examples/prepare_from_results.py \
  --scene-id example_floor \
  --layout-dir data/example_floor/layout_gt \
  --geoverify-root data/geoverify \
  --da3-predictions data/da3/predictions.csv \
  --output outputs/example_floor.json
```

`data/geoverify/example_floor/pair_results/` 放置完整單對 JSON。`--geoverify-root` 與 `--da3-predictions` 至少提供一項。模型結果的格式需符合 [輸入規格](docs/INPUT_OUTPUT.md)，本版本不會自動辨識其他專案的整房交付格式或執行上游模型。

## 文件

| 文件 | 內容 |
| --- | --- |
| [API](docs/API.md) | 三個介面的參數、回傳值、狀態及錯誤 |
| [INPUT_OUTPUT](docs/INPUT_OUTPUT.md) | 輪廓、配對、scene 及匯出格式 |
| [COORDINATES](docs/COORDINATES.md) | 座標方向、角度、尺度與索引 |
| [INTEGRATION](docs/INTEGRATION.md) | 呼叫流程、資料責任及異常處理 |
| [FUNCTION_MAP](docs/FUNCTION_MAP.md) | 功能細項與程式模組 |
| [ALGORITHM_OVERVIEW](docs/ALGORITHM_OVERVIEW.md) | 推薦排序及位置候選的計算方式 |
| [VALIDATION](docs/VALIDATION.md) | 驗證環境、測試範圍與結果 |

## 測試

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

測試不需要 GPU、私人資料集或其他專案原始碼。程式結構如下：

```text
src/room_assembly/
├── api.py          # 三個公開入口
├── models.py       # Scene、AssemblyState、Pose2D
├── core/           # 排序、位置候選與座標轉換
├── adapters/       # JSON、CSV、TXT 資料轉接
└── rendering.py    # PNG 輸出
examples/           # 呼叫範例與示範資料
tests/              # 自動測試
docs/               # 整合與格式規格
```

## 版本與使用範圍

套件版本：**0.1.1**；scene 資料格式版本：**1.0**。本版本提供 Python function API，整合端可自行包裝 HTTP 服務。套件不提供登入、任務儲存、手動編輯 UI、衝突檢查服務或後處理。

原碼、第三方依賴及資料的使用範圍見 [NOTICE.md](NOTICE.md)。
