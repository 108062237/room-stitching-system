# 驗證報告

套件版本：0.1.1。驗證日期：2026-11-24。

## 環境

| 項目 | 已驗證版本 |
| --- | --- |
| 作業系統 | Linux |
| Python | 3.12.2 |
| NumPy | 1.26.4 |
| Shapely | 2.1.2 |
| Matplotlib | 3.9.2 |
| pytest | 7.4.4 |

完整依賴快照見 [constraints-tested.txt](../constraints-tested.txt)。

## 驗證項目

| 項目 | 結果 |
| --- | --- |
| 自動測試 | 55 項通過，無跳過 |
| README Python 範例 | 可執行 |
| 推薦、位置與預覽 CLI 範例 | 可執行 |
| 合成輸入至 JSON／PNG 輸出 | 可執行 |
| wheel 建置與安裝 | 通過 |
| 非原始碼目錄執行三項介面及繪圖 | 通過 |
| 文件內本地連結 | 已檢查 |

自動測試涵蓋起始與後續推薦、連接排序、位置方案排序、正反配對、相對與整體座標、輪廓預覽、無建議狀態、輸入驗證、JSON 載入保存及 PNG 輸出。預覽含完整重疊配置的測試，確認不因位置衝突而拒絕顯示。

## 執行方式

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python examples/demo.py --output outputs/demo
```

測試不需其他專案、GPU 或私人資料。示範輸出見 [examples/data/expected](../examples/data/expected/)。

## 範圍與限制

- 安裝驗證使用獨立虛擬環境及既有數值依賴，已確認套件從安裝位置載入，不依賴原工作區。
- 尚未在全新主機重新下載依賴，亦未對 Windows、macOS 或其他 Python 版本進行實機驗證。
- 此報告驗證介面與交付包可執行性，不代表真實場景的拼接準確率。
- 本版本未提供網頁或 HTTP server，因此不包含瀏覽器互動測試。
