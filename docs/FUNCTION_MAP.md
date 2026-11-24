# 功能與模組

## 對外功能

| 入口 | 功能細項 | 模組 |
| --- | --- | --- |
| `recommend_rooms()` | 起始房間排序、可加入連接篩選、排除指定配對、下一步推薦排序 | api.py、core/recommendation.py |
| `suggest_placements()` | 配對方向轉換、位置候選評分、方案排序、相對與整體座標換算 | api.py、core/placement.py、core/geometry.py |
| `preview_placement()` | 套用指定位置、組合輪廓、計算顯示範圍 | api.py、core/preview.py |

## 資料與顯示輔助

| 功能 | 模組 | I/O 行為 |
| --- | --- | --- |
| 從局部輪廓及模型結果準備 Scene | adapters/scene.py、core/guided.py、core/candidates.py | `build_scene` 不讀寫檔案 |
| 無幾何位置時產生 layout-only 備援 | core/layout_fallback.py | 不讀寫檔案 |
| TXT、CSV、JSON 載入與 scene 匯出 | adapters/ | 由指定路徑讀寫 |
| PNG 輸出 | rendering.py | 回傳 bytes；指定路徑時寫檔 |

## 分層

```text
檔案 → adapters → Scene
                   │
目前配置 ───────────┤
                   ↓
                 api.py
                   ↓
                  core
                   ↓
              JSON 相容資料
                   └→ rendering → PNG
```

整合時使用頂層三個介面；`core` 與底線開頭方法屬內部實作，不作為對外穩定 API。
