# 🏓 TTMATCH Dataset 說明

本 Notebook 針對 **乒乓球比賽動作資料集 (TTMATCH)** 進行前處理與 EDA。  
資料包含比賽、回合、擊球等資訊，用於預測：
- 下一擊動作類型（`action_id`）
- 下一擊落點（`point_id`）
- 該回合最終勝者（`server_won_point`）

---

## 📊 資料欄位簡介（主要 features）

| 欄位名稱 | 中文說明 | Definition (English) |
|-----------|-----------|----------------------|
| **rally_uid** | 小分的唯一識別碼 | Unique ID for each rally |
| **sex** | 比賽性別（男=1 / 女=2） | Gender category of the match |
| **match_id** | 比賽唯一識別碼 | Unique ID of the match |
| **game_number** | 小局數（第幾局） | Game (set) number within the match |
| **rally_id** | 小局內的小分編號 | Rally ID within the game |
| **stroke_number** | 小分內的揮拍次序 | Stroke number within the rally |
| **score_self / score_opponent** | 主視角與對手的得分 | Player and opponent scores |
| **server_won_point** | 發球者是否得分（1=是, 0=否） | Whether the server won the point |
| **player_self_id / player_opponent_id** | 雙方球員代碼 | IDs of the two players |
| **server_id** | 發球者 ID | ID of the serving player |
| **serve_number** | 小局中第幾次發球 | nth serve within the rally |
| **stroke_id** | 揮拍狀態或階段（發球、接發、第三板之後…） | Stroke phase indicator |
| **handId** | 揮拍手別（0=無, 1=正手, 2=反手, -1=結束拍） | Forehand/backhand indicator |
| **strength_id** | 力道（1=強, 2=中, 3=弱） | Stroke strength level |
| **spin_id** | 球的旋轉方式 | Type of spin applied |
| **point_id** | 球落點位置 | Landing position on the table |
| **action_id** | 動作類型（拉球、殺球、擋球等） | Stroke or action type |
| **position_id** | 球員站位（1=左, 2=中, 3=右） | Player’s court position |

---

## 🔢 常見 ID 對照（簡表）

| 類別 | ID | 中文意義 | English |
|------|----|-----------|----------|
| **stroke_id** | 1 | 發球 | Serve |
|  | 2 | 接發球 | Receive |
|  | 4 | 第三板之後 | Rally |
|  | -1 | 結束拍 | End Point |
| **handId** | 1 | 正手 | Forehand |
|  | 2 | 反手 | Backhand |
| **strength_id** | 1 | 強 | Strong |
|  | 2 | 中 | Medium |
|  | 3 | 弱 | Slow |
| **spin_id** | 1 | 上旋 | Top Spin |
|  | 2 | 下旋 | Back Spin |
|  | 3 | 不旋 | No Spin |
|  | 4 | 側上旋 | Side Top Spin |
|  | 5 | 側下旋 | Side Back Spin |
| **point_id** | 1 | 正手短球 | Forehand Short |
|  | 5 | 中路半出台球 | Middle Half-Long |
|  | 9 | 反手長球 | Backhand Long |
| **action_id** | 1 | 拉球 | Drive |
|  | 3 | 殺球 | Smash |
|  | 10 | 搓球 | Push |
|  | 13 | 擋球 | Block |
|  | 15–18 | 發球類動作 | Serve Types |

---

## 📋 Notebook 架構
1. **欄位說明**（本區）  
2. **資料讀取與排序**  
3. **前處理：建立前一擊 / 下一擊 / 特徵工程**  
4. **清理與輸出 clean_train.csv, clean_test.csv**  
5. **EDA：動作分佈、回合長度、階段變化**

> ⚙️ 小提醒：  
> - 所有欄位名稱請以實際 `train.csv` 為準。  
> - 有些欄位在 test 可能不存在（要先用 `if col in df.columns:` 判斷）。  
> - `stroke_number` 與 `rally_id` 共同決定回合內順序。
