# 呼叫範例

先於專案根目錄安裝 `python -m pip install '.[render]'`。所有指令皆在專案根目錄執行。

## 完整示範

```bash
python examples/demo.py --output outputs/demo
```

輸入位於 [data/input.json](data/input.json)。內容為三間合成輪廓與兩組正規化配對。完整示範輸出位於 [data/expected](data/expected/)，可用來確認欄位及顯示方式；它不是實際場景的參考答案。

## 單獨查詢推薦

```bash
python examples/recommend_rooms.py \
  --scene outputs/demo/scene.json \
  --state outputs/demo/state.json \
  --top-k 5
```

## 單獨查詢位置方案

```bash
python examples/suggest_placements.py \
  --scene outputs/demo/scene.json \
  --state outputs/demo/state.json \
  --room-a room_A \
  --room-b room_B \
  --pair-id edge_0000 \
  --top-k 5
```

## 預覽指定位置

```bash
python examples/preview_placement.py \
  --scene outputs/demo/scene.json \
  --state outputs/demo/state.json \
  --room room_B \
  --x 50 --y 0 --theta-deg 0 \
  --output outputs/manual_preview.png
```

回傳指定位置的輪廓；此範例不執行吸附、重疊檢查或保存房間位置。

## 檔案格式轉接

- `prepare_from_results.py`：layout TXT、完整單對 GeoVerify JSON 及正規化前的 DA3 CSV → scene。
- `import_legacy.py`：已有的 combined_results/candidates CSV → scene。

兩者的參數可用 `python examples/<script>.py --help` 查詢。外部模型輸出需符合 [輸入規格](../docs/INPUT_OUTPUT.md)，不能將任意相似欄位直接替換。
