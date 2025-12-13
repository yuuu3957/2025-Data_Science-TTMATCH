# Kaggle Competition-TTMATCH

## 競賽概述及目標
[競賽連結](https://www.kaggle.com/competitions/introduction-to-data-secience-ttmatch/overview)

本次競賽需要透過以下feature進行三個方向的預測
### Feature Description

| Feature | 說明（中文） | Definition（English） |
|--------|--------------|------------------------|
| rally_uid | 小分的唯一識別碼 | Unique ID for each rally |
| sex | 比賽性別（男(1) / 女(2)） | Gender category of the match (male=1, female=2) |
| match_id | 比賽的唯一識別碼 | Unique ID of the match |
| game_number | 小局數（第幾局） | Game (set) number within the match |
| rally_id | 小局內的小分編號 | Rally ID within the game |
| stroke_number | 小分內的揮拍次序 | Stroke number within the rally |
| score_self | 主視角選手的得分 | Points won by the main-view player |
| score_opponent | 對側視角選手的得分 | Points won by the opponent player |
| server_won_point | 發球者是否得分（1=是, 0=否） | Whether the server won the point (1=yes, 0=no) |
| player_self_id | 主視角選手的 ID | ID of the main-view player |
| player_opponent_id | 對側視角選手的 ID | ID of the opponent player |
| server_id | 發球者的 ID | ID of the serving player |
| serve_number | 小局中第幾次發球 | The nth serve within the current rally |
| stroke_id | 揮拍狀態或動作類型 | Stroke action type or state identifier |
| handId | 正手或反手揮拍 | Forehand or backhand stroke indicator |
| strength_id | 擊球力道 | Stroke strength level |
| spin_id | 球的旋轉方式 | Type of spin applied to the ball |
| point_id | 球的落點位置 | Landing position of the ball on the table |
| action_id | 擊球方式 | Stroke or action type |
| let | 叫暫停或重發球的選手 | Player who called a let or timeout |
| position_id | 球員站位區域 | Player’s court position |


### Task

#### Task1 : Next stroke (n-th shot) action type prediction(actionId)
- 透過前n-1 shot的資料去預測下個action id為何
- Macro F1-Score

#### Task2 : Next stroke (n-th shot) landing location prediction(pointId) 
- 透過前n-1 shot的資料去預測下個point id為和
- Macro F1-Score

#### Task3 : Current rally outcome prediction(ServerGetPoint)
- 透過現有當局資料預測當局誰會得分
- AUC-ROC (Area Under the ROC Curve)


## 方法與流程

我首先先對於資料進行資料前處理、EDA、特徵工程等行為，並使用LightGBM模型，其中資料相關code在`data-handling.ipynb`中，模型訓練及產出預測於`LightGBM.ipynb`中。

### 1.1 Data Processing

先對下載資料，並對資料做基礎的處理

#### 創建基礎特徵
在進行模型訓練前，需先對原始資料做清理、欄位統一、與特徵整理。本階段步驟包括：

- 建立唯一識別碼（`rally_uid`）
- 統一欄位名稱（例如 strickNumber → stroke_number）
- 確保欄位型態正確
- 清理可能存在的缺失值與異常值

```
for df, name in [(train, 'train'), (test, 'test')]:
    # rally_uid（唯一識別符）
    if 'rally_uid' not in df.columns:
        df['rally_uid'] = df['match'].astype(str) + '_' + df['rally_id'].astype(str)
    
    # match_id
    if 'match_id' not in df.columns:
        df['match_id'] = df['match']
    
    # stroke_number（當前是第幾拍）
    if 'stroke_number' not in df.columns and 'strickNumber' in df.columns:
        df['stroke_number'] = df['strickNumber']
    
    print(f"{name}: {df.shape}")

```

#### 創建tag
因後續訓練模型需要target，在測試集創建`next_actionID`、`next_pointID`欄位
```
# ============ 創建標籤（next_actionId, next_pointId）============
print("\n🎯 創建標籤...")

# 為訓練集創建標籤
train['next_actionId'] = train.groupby('rally_id')['actionId'].shift(-1)
train['next_pointId'] = train.groupby('rally_id')['pointId'].shift(-1)

# 處理最後一拍（下一拍是結束）
train['next_actionId'] = train['next_actionId'].fillna(-1).astype(int)
train['next_pointId'] = train['next_pointId'].fillna(-1).astype(int)

print(f"標籤統計:")
print(f"  next_actionId: {train['next_actionId'].value_counts().sort_index()}")
print(f"  next_pointId: {train['next_pointId'].value_counts().sort_index()}")
```

### 1.2 EDA
進行EDA，分析資料以便後續特徵工程及模型訓練

#### 分析欄位分布
對 next_actionId、next_pointId 進行長條圖統計，了解常見動作類型與落點位置。  
結果顯示少數動作佔比明顯較高，部分類別分布不平衡，因此後續使用 **Stratified K-Fold** 作為驗證方式，可以保證每個fold的類別比例一致。

#### 回合長度分布
回合長度和task3有很大的關聯，故這邊分析了回合的長度，結果顯示多數回合集中在 4～6 拍之間，因此「前 1～3 拍」的歷史資訊即可覆蓋大部分回合(故在處理task1、2時使用prev1~3)。

#### 動作落點組合
以我直覺而言，認為action和point應有高關聯。而將 actionId 與 pointId 組合後計算出現頻率，確實可觀察到明顯的固定搭配。  
因此後續加入 **action × point 的交互特徵**。

#### 動作轉換
計算前一拍 → 當前拍的動作序列（prev_action → actionId）。  
常見的轉換呈現固定模式，說明動作具有序列性，因此加入 **action_transition 特徵**。

#### 比分影響
我覺得桌球比賽中，比分可能和選手壓力有關，此處將比分 ≥10 的分數定義為關鍵分，比較其與一般分的回合長度與動作分布。  
果顯示兩者差異不大，但關鍵分的回合長度略高。基於這些小幅趨勢，我們仍保留比分相關特徵（score_diff / is_game_point / score_phase）。


### 1.3 特徵工程
結合EDA結果和我自己思考後認為可能會影響到預測的重要特徵，我創建了以下特徵。

#### 1.3.1 歷史資訊特徵
擊球的動作、落點應與前幾拍有關聯，又根據前面回合長度分析結果，得知多數回合集中在 4～6 拍之間，因此取前 1～3 拍作為主要的歷史訊息來源，包含：
`prev1_actionId`, `prev2_actionId`, `prev3_actionId`

`prev1_pointId`, `prev2_pointId`, `prev3_pointId`

`prev1_handId`, `prev2_handId`, `prev3_handId`

`prev1_spinId` 等

用於預測Task1、2

#### 1.3.2 比分特徵
因覺得比分可能會影響選手的心理跟策略，故加入。雖然EDA顯示影響可能不大。
- `score_diff`：分數差
- `score_sum`：兩者分數和
- `is_game_point`：是否 ≥10 分（局末）
- `is_key_point`：比分緊張（差 ≤2 且總分 ≥16）

用於預測Task3

#### 1.3.3 回合階段特徵
因覺得每一拍在回合中所處的位置不同，選手可能採用不同策略，因此將 stroke_number 做分段：
```
# 回合進程
df['stroke_phase'] = pd.cut(df['stroke_number'], 
                             bins=[0, 3, 7, 100], 
                             labels=['early', 'mid', 'long'])
```
#### 1.3.4 組合特徵
考慮到handID、strengthID及actionID、pointID應該有關連性，故我將其分別組合成一特徵:
```
if 'handId' in df.columns and 'strengthId' in df.columns:
        df['hand_strength_encoded'] = df['handId'] * 10 + df['strengthId']
    
    if 'actionId' in df.columns and 'pointId' in df.columns:
        df['action_point_encoded'] = df['actionId'] * 10 + df['pointId']
```

#### 1.3.5 一階組合特徵
這部分特徵為我訓練後期為提升LB分數故嘗試加入之特徵，包含:
- `prev1_action_point`:前一拍動作-落點組合
- `curr_action_point`:當前動作-落點組合
- `action_transition`:動作轉換模式
- `point_transition`:落點轉換模式
- `hand_point`:手部-落點組合
- `score_phase`:比分階段組合
加入後LB score有略微提升

#### 1.3.6 白名單
因我初期在訓練模型時常出現CV score很高(0.35up)但LB score很低(0.15左右)之情況，我懷疑是因資料洩漏導致，故強制使用白名單以確保僅有安全之特徵會被使用。
```
safe_features = [
    # ID
    'rally_uid', 'match_id', 'rally_id', 'stroke_number',
    
    # 當前拍
    'actionId', 'pointId', 'handId', 'strengthId', 'spinId',
    
    # 前 1-2 拍
    'prev1_actionId', 'prev1_pointId', 'prev1_handId', 'prev1_strengthId', 'prev1_spinId',
    'prev2_actionId', 'prev2_pointId', 'prev2_handId', 'prev2_strengthId', 'prev2_spinId',
    
    # 比分
    'scoreSelf', 'scoreOther', 'score_diff', 'score_sum', 'is_leading',
    
    # 組合
    'hand_strength_encoded', 'action_point_encoded',
    
    # 階段
    'is_early_rally', 'stroke_phase',
    
    # 標籤
    'next_actionId', 'next_pointId', 'serverGetPoint', 'server_won_point',
]
```



### 2 模型建立與訓練

#### 2.1 模型選擇

##### KNN
我最初先想到的模型是課堂上所教之KNN，但因得到的分數非常低，故上網查詢了其他模型。

##### LighGBM
上網搜尋後發現可以使用LightGBM，選擇理由如下:
1. 適合大量類別特徵與交互特徵
2. 訓練速度快
3. 記憶體效率高
4. 天然支持缺失值與不平衡類別
5. 對 tabular data 顯著優勢


#### 2.2 資料切分策略
為避免資料洩漏及考量到比賽間的各小局應有強烈相關性，我採用 **GroupKFold** 並以 `match_id` 作為 group。  
理由如下：
- 同一場比賽的所有拍數必須保持在同一 fold  
- 避免模型在訓練看到該比賽的部分資料、在驗證時看到同一比賽的另一部分  

#### 2.3 模型參數
對於Task1、2、3分別設定不同參數(此處參數更動多次，附上之為最後public score最高之參數)
- Task1
```
params_action = {
    'n_estimators': 1200,     # 900 → 1200
    'learning_rate': 0.02,    # 0.025 → 0.02
    'num_leaves': 63,         # 110 → 63
    'max_depth': 8,           # 9 → 8
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_samples': 25,    # 增加: 20 → 25
    'reg_alpha': 0.12,          # 稍增: 0.1 → 0.12
    'reg_lambda': 0.12,         # 稍增: 0.1 → 0.12
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}
```
- Task2

```
params_point = {
    'n_estimators': 1000,     # 800 → 1000
    'learning_rate': 0.025,   # 0.03 → 0.025
    'num_leaves': 63,         # 110 → 63
    'max_depth': 8,           # 10 → 8
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_samples': 20,    # Task2 保持較小，允許學習稀有落點
    'reg_alpha': 0.12,
    'reg_lambda': 0.12,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}
```
- Task3
```
    params_rally = {
        'n_estimators': 1000,     # 800 → 1000
        'learning_rate': 0.025,   # 0.03 → 0.025
        'num_leaves': 63,         # ✓ 保持
        'max_depth': 8,           # -1 → 8 ⚠️ 最重要！
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
```

#### 2.4 Task的特徵選擇
- Task1、2移除洩漏特徵及target等
```
exclude_cols = [
    'match', 'match_id', 'rally_id', 'stroke_number', 'strickNumber',
    'next_actionId', 'next_pointId', 'serverGetPoint', 'server_won_point',
    'rally_uid', 'actionId', 'pointId',
    # 🔴 移除洩漏特徵
    'rally_length', 'rally_progress', 
    'is_early_rally', 'is_mid_rally', 'is_late_rally',
    'score_rally_interact',  # 基於 rally_progress
    'action_point_combo'  # 可能包含當前拍信息
]
```
- Task3因考慮到得分會因未來訊息影響，故確保指使用現在的特徵
```
exclude_task3 = [
    # 排除的欄位（標籤、ID等）
    'next_actionId', 'next_pointId', 'serverGetPoint', 'server_won_point',
    'rally_uid', 'match', 'match_id', 'rally_id', 'strickNumber',
    
    # 🔴 只排除這些「未來信息」
    'rally_length',           # 總拍數（未來）
    'rally_progress',         # 基於 rally_length
    'is_early_rally',         # 基於 rally_progress
    'is_mid_rally',           # 基於 rally_progress
    'is_late_rally',          # 基於 rally_progress
    'score_rally_interact',   # 基於 rally_progress
]
```

#### 2.5 拍數權重
因注意到測試及地拍數分布和訓練集不太一致，若直接訓練，模型會偏向長回合的行為（因為訓練資料較多），導致 Task1 / Task2 在測試集表現變差。
因此設計了拍數權重讓模型更重視短回合的資料。
```
# 更接近測試集的權重策略
for i, s in enumerate(stroke_numbers):
    if s == 1:
        stroke_weights[i] = 2.1    # 微調：2.0 → 2.1
    elif s == 2:
        stroke_weights[i] = 2.7    # 微調：2.6 → 2.7
    elif s == 3:
        stroke_weights[i] = 2.8    # 微調：2.9 → 2.8
    elif s == 4:
        stroke_weights[i] = 2.0    # 微調：2.1 → 2.0
    elif s == 5:
        stroke_weights[i] = 1.2    # 微調：1.3 → 1.2
    elif s == 6:
        stroke_weights[i] = 0.95   # 微調：1.0 → 0.95
    else:
        stroke_weights[i] = 0.7    # 微調：0.75 → 0.7

# 如果有壓力特徵，進一步調整權重
if 'score_pressure' in clean_train.columns:
    pressure = clean_train['score_pressure'].values
    for i, p in enumerate(pressure):
        if abs(p) >= 2:  # 壓力大的時刻
            stroke_weights[i] *= 1.15
```

#### 2.6 Task1、Task2、Task3 訓練方式

##### Task1:
- 多類別分類（macro F1-score）
```
f1_action_cv = cv_multiclass_f1(
    X_action, y_action, groups, 
    n_splits=10, 
    class_weight=weight_action,
    params=params_action,
    verbose=True
)
```

##### Task2:
- 多類別分類（macro F1-score）
```
f1_point_cv = cv_multiclass_f1(
    X_point, y_point, groups,
    n_splits=10,
    class_weight=weight_point,
    params=params_point,
    verbose=True
)
```

##### Task3:
- 二元分類（AUC-ROC）
```
auc_rally_cv = cv_binary_auc(
    X_rally, y_rally, groups,
    params=params_rally,
    verbose=True
)
```

##### Ensemble策略
單一 random_seed 在 LightGBM 中波動較大，因此我用不同 seed 訓練 3個模型，
最後取平均，能讓 LB 變更穩定。
```
for seed in [42, 123, 456]:
    print(f"  訓練模型 seed={seed}...")
    params_seed = params_point.copy()
    params_seed['random_state'] = seed
    
    model = LGBMClassifier(**params_seed, class_weight=weight_point)
    model.fit(X_point, y_point, sample_weight=stroke_weights)
    models_point.append(model)
    print(f"    ✓ 完成")
```

##### 預測與輸出 submission.csv
最後透過訓練好之模型產生預測結果(必須要做對齊)

### 3 調整過程
整體而言我將完成這個比賽分為兩大階段:
前期CV和LB分數差距過大，public score一直卡在0.2以下無法進步。
後期達到0.3以上開始嘗試微調參數、加權重、特徵以提高分數

#### 3.1 前期
主要處理CV、LB分數過大，經以下調整後分數顯著上升。

##### 3.1.1 資料洩漏
經過多次嘗試後懷疑為資料洩漏導致，故增加白名單且在訓練時針對不同task進行特徵選擇。

##### 3.1.2 -1之處理
一開始覺得-1應該是無異議之結果，故將其視為異常值處理，但後續比較訓練集和測試集資料分布，發現訓練集-1占比很大，故最後未將其視為異常值。

##### 3.1.3 分組之特徵名打錯
未注意到自己在`data-handling.ipymb`中，比賽的column是以match儲存的，在分組時使用了match_id作為分組，造成未確實按比賽場次分組，導致模型在訓練看到該比賽的部分資料、在驗證時看到同一比賽的另一部分，後續將其更改

#### 3.2 後期
進行多方嘗試以提升分數，分數上升幅度小

##### 3.2.1 參數調整
進行多次參數調整，尤其針對以下參數:
- `n_estimators`:增加樹的數量以使模型學習複雜關係，自小慢慢提高。
- `learning_rate`:讓模型學得更穩定，但會搭配較多樹（提高 n_estimators）。嘗試多種組合後，選擇收斂速度與穩定度都較佳的設定。
- `num_leaves`:控制樹的複雜度。 嘗試從較少葉數到較多葉數，過程中發現數字過大可能造成過擬合，因此選擇中間值使模型具有一定表達能力但不會太複雜。
- `max_depth`:限制樹的最大深度。多次嘗試後設定為較淺的深度，使模型更穩定、較不容易過擬合。
- `min_child_samples`:對於Task2特別設定允許其可學習罕見落點

以上參數皆進行多次調整，以提升穩定性。

##### 3.2.2 拍數權重
因發現訓練集跟測試集的拍數分布不均，故嘗試加上權重使兩者更為一致

##### 3.2.3 Ensemble策略
原始只有一個seed，後來有嘗試3、5個seed，但因5 seed跟3 seed結果相距不大(甚至無更好)，故考量到訓練時間問題，最終選用3個模型。

##### 3.2.4 增加特徵
增加了一階交互特徵：

- `prev1_action_point`
- `curr_action_point`
- `action_transition`
- `point_transition`
- `hand_point`
- `score_phase`

以上特徵描述了:
- 動作與落點的搭配  
- 動作序列  
- 攻擊手法的“轉換模式” 
等較為複雜的部份
增加後成功提升LB約0.01

## 操作說明
本專案包含兩個`.ipynb`:`data_handling.ipynb`、`LightGBM.ipynb`
執行方式如下

1. 至[資料連結](https://www.kaggle.com/competitions/introduction-to-data-secience-ttmatch/data)下載`train.csv`、`test.csv`，並將其放置data資料夾
2. 執行`data_handling.ipynb`以獲得`clean_train_v2.csv`、`clean_test_v2.csv`，此兩檔案亦會放於data資料夾
3. 執行`LightGBM.ipynb`即可看到CV score並且將預測結果存成`submission_improved_{ts}.csv`({ts}為時間戳)，此檔會放於result資料夾中