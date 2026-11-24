# 座標與單位

## 1. 房間局部與整體座標

每個房間輪廓位於自己的局部 XY 座標。`world_pose` 表示該房間的局部座標如何轉換到整體配置：

```text
p_world = R(theta) × p_local + [x, y]
R(theta) = [[cos(theta), -sin(theta)],
            [sin(theta),  cos(theta)]]
```

正角度為 XY 平面逆時針旋轉。`world_pose` 不是照片觀看方向；輪廓不能先轉換一次後，又套用同一 pose。

## 2. A → B 的相對位置

`relative_pose={dx,dy,dtheta_deg}` 表示 B 的原點與朝向在 A 局部座標下的位置：

```text
T_B_world = T_A_world ⊕ T_B_in_A
```

例如 A 位於 (100,200)、角度 +90°；B 在 A 座標中位移 (10,0)、角度 +20°，則 B 的 world pose 為 (100,210,+110°)。反向配對必須取完整 SE(2) 逆變換，不能只把 dx、dy 加負號。

## 3. 角度

| 位置 | 單位 |
| --- | --- |
| 公開 pose 的 `theta_deg` | degree |
| 公開相對 pose 的 `dtheta_deg` | degree |
| scene JSON 的 `measurement[2]` | radian |
| 內部座標運算 | radian |

合成後的內部角度正規化至 [-π,π)，公開相對／整體結果會轉為 degree。套件版本和 scene 的 `angle_unit` 不可混用。

## 4. 尺度

XY 使用 **layout units**。未提供外部尺度校準時，不表示公尺或公分。

全景 TXT 轉接的預設高度為 50，建立相同投影尺度的局部輪廓；影像相對位移方向沒有可直接使用的距離長度。若使用其他來源的 camera-height 或 metric 輪廓，應先轉成一致尺度，再和既有 scene 合併。

## 5. 全景角點轉接

- TXT 座標尺寸必須符合指定的 pano_width、pano_height。
- 以地板點投影至 XY，移除投影中心並翻轉 Y，接著進行直角校正並轉回原朝向。
- 轉接不重排或刪除頂點；輪廓不包含重複閉合點。
- 牆索引 `i` 對應 `polygon[i] → polygon[(i+1) % N]`。

GeoVerify 的 `x_b = R × x_a + t` 是相機相對變換，與本套件的 B-in-A pose 不同。量測轉接使用其旋轉及 bearing，水平 camera `[x,z]` 對應局部 `[-x,-z]`；不將位移向量長度當成 layout 距離。

## 6. 繪圖

`preview_placement()` 回傳的 `polygon` 已是整體 XY 座標。PNG helper 使用等比例座標；顏色由 room ID 決定，虛線標示本次預覽房間，不表示位置衝突或分數高低。
