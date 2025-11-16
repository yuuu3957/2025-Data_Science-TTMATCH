# ⚡ 快速測試版本 - 測試移除 -1 的效果
# 預計時間：8-10 分鐘
# 🎯 核心改動：訓練時移除 -1 樣本

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, roc_auc_score
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("⚡ 快速測試版本 - 測試移除 -1 效果")
print("="*60)
print("🎯 核心改動：訓練時移除 -1 樣本")
print("   理論：-1 是「最後一筆記錄」標記，不是預測目標")
print("="*60)

# ============================================================
# 1. 讀取數據
# ============================================================
print("\n📂 讀取數據...")
DATA_DIR = "data"

clean_train = pd.read_csv(f"{DATA_DIR}/clean_train_v2.csv")
clean_test = pd.read_csv(f"{DATA_DIR}/clean_test_v2.csv")

print(f"訓練集: {clean_train.shape}")
print(f"測試集: {clean_test.shape}")

# ============================================================
# 🆕 2. 分析並移除 -1 樣本
# ============================================================
print("\n" + "="*60)
print("🔍 分析 -1 樣本")
print("="*60)

# 統計 -1
neg1_action_count = (clean_train['next_actionId'] == -1).sum()
neg1_point_count = (clean_train['next_pointId'] == -1).sum()

print(f"\n原始訓練集: {len(clean_train)} 樣本")
print(f"  next_actionId = -1: {neg1_action_count} ({neg1_action_count/len(clean_train):.1%})")
print(f"  next_pointId = -1:  {neg1_point_count} ({neg1_point_count/len(clean_train):.1%})")

# 🎯 移除 -1 樣本（核心改動）
print("\n🔧 移除 -1 樣本...")
clean_train_no_neg1 = clean_train[
    (clean_train['next_actionId'] != -1) & 
    (clean_train['next_pointId'] != -1)
].copy()

print(f"移除後訓練集: {len(clean_train_no_neg1)} 樣本")
print(f"  移除了: {len(clean_train) - len(clean_train_no_neg1)} 樣本")
print(f"  保留率: {len(clean_train_no_neg1)/len(clean_train):.1%}")

# ✅ 使用 30% 的訓練數據來快速測試
sample_size = int(len(clean_train_no_neg1) * 0.3)
clean_train_sample = clean_train_no_neg1.sample(n=sample_size, random_state=42).reset_index(drop=True)
print(f"\n使用 30% 樣本快速測試: {clean_train_sample.shape}")

# ============================================================
# 3. 準備特徵和標籤
# ============================================================
print("\n🔧 準備特徵...")

# 排除的欄位
exclude_cols = [
    'match', 'match_id', 'rally_id', 'stroke_number', 'rally_uid',
    'next_actionId', 'next_pointId', 'serverGetPoint', 'server_won_point',
    'actionId', 'pointId',
]

# 特徵列
feature_cols = [c for c in clean_train_sample.columns if c not in exclude_cols]
print(f"特徵數量: {len(feature_cols)}")

# 準備數據
X = clean_train_sample[feature_cols].copy()
y_action = clean_train_sample['next_actionId'].astype(int)
y_point = clean_train_sample['next_pointId'].astype(int)
groups = clean_train_sample['match_id'].values

# 檢查 -1 是否已經完全移除
print(f"\n✅ 標籤檢查:")
print(f"  y_action 包含 -1: {(-1 in y_action.values)}")
print(f"  y_point 包含 -1:  {(-1 in y_point.values)}")
print(f"  y_action 類別數: {y_action.nunique()}")
print(f"  y_point 類別數:  {y_point.nunique()}")

# stroke 權重
stroke_weights = np.ones(len(clean_train_sample))
stroke_numbers = clean_train_sample['stroke_number'].values

for i, s in enumerate(stroke_numbers):
    if s <= 2:
        stroke_weights[i] = 2.5
    elif s == 3:
        stroke_weights[i] = 2.0
    elif s == 4:
        stroke_weights[i] = 1.5
    else:
        stroke_weights[i] = 0.8

# 簡化的參數
params_fast = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'num_leaves': 31,
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_samples': 50,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

# ============================================================
# 4. Task1: ActionId (2-Fold CV)
# ============================================================
print("\n" + "="*60)
print("🎯 Task 1: ActionId (2-Fold CV)")
print("="*60)

cnt_action = Counter(y_action)
weight_action = {cls: 1.0 / np.sqrt(count) for cls, count in cnt_action.items()}

print("⏱️  開始 2-Fold CV...")
gkf = GroupKFold(n_splits=2)
cv_scores_action = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_action, groups), 1):
    X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_va = y_action.iloc[train_idx], y_action.iloc[val_idx]
    w_tr = stroke_weights[train_idx]
    
    train_classes = set(y_tr.values)
    fold_weight = {k: v for k, v in weight_action.items() if k in train_classes}
    
    model = LGBMClassifier(**params_fast, class_weight=fold_weight)
    model.fit(X_tr, y_tr, sample_weight=w_tr)
    
    y_pred = model.predict(X_va)
    score = f1_score(y_va, y_pred, average='macro')
    cv_scores_action.append(score)
    print(f"  Fold {fold}: {score:.4f}")

cv_action = np.mean(cv_scores_action)
print(f"\n📈 Task1 CV Score: {cv_action:.4f}")

print("⏱️  訓練最終模型...")
model_action = LGBMClassifier(**params_fast, class_weight=weight_action)
model_action.fit(X, y_action, sample_weight=stroke_weights)
print(f"✅ Task1 完成 (模型學到 {len(model_action.classes_)} 個類別)")
print(f"   模型包含 -1: {(-1 in model_action.classes_)}")

# ============================================================
# 5. Task2: PointId (2-Fold CV)
# ============================================================
print("\n" + "="*60)
print("🎯 Task 2: PointId (2-Fold CV)")
print("="*60)

cnt_point = Counter(y_point)
weight_point = {cls: 1.0 / np.sqrt(count) for cls, count in cnt_point.items()}

print("⏱️  開始 2-Fold CV...")
cv_scores_point = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_point, groups), 1):
    X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_va = y_point.iloc[train_idx], y_point.iloc[val_idx]
    w_tr = stroke_weights[train_idx]
    
    train_classes = set(y_tr.values)
    fold_weight = {k: v for k, v in weight_point.items() if k in train_classes}
    
    model = LGBMClassifier(**params_fast, class_weight=fold_weight)
    model.fit(X_tr, y_tr, sample_weight=w_tr)
    
    y_pred = model.predict(X_va)
    score = f1_score(y_va, y_pred, average='macro')
    cv_scores_point.append(score)
    print(f"  Fold {fold}: {score:.4f}")

cv_point = np.mean(cv_scores_point)
print(f"\n📈 Task2 CV Score: {cv_point:.4f}")

print("⏱️  訓練最終模型...")
model_point = LGBMClassifier(**params_fast, class_weight=weight_point)
model_point.fit(X, y_point, sample_weight=stroke_weights)
print(f"✅ Task2 完成 (模型學到 {len(model_point.classes_)} 個類別)")
print(f"   模型包含 -1: {(-1 in model_point.classes_)}")

# ============================================================
# 6. Task3: ServerGetPoint (2-Fold CV)
# ============================================================
print("\n" + "="*60)
print("🎯 Task 3: ServerGetPoint (2-Fold CV)")
print("="*60)

if 'serverGetPoint' in clean_train_sample.columns:
    y_rally = clean_train_sample['serverGetPoint'].astype(int)
    
    print("⏱️  開始 2-Fold CV...")
    cv_scores_rally = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_rally, groups), 1):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y_rally.iloc[train_idx], y_rally.iloc[val_idx]
        
        model = LGBMClassifier(**params_fast)
        model.fit(X_tr, y_tr)
        
        y_pred_prob = model.predict_proba(X_va)[:, 1]
        score = roc_auc_score(y_va, y_pred_prob)
        cv_scores_rally.append(score)
        print(f"  Fold {fold}: {score:.4f}")
    
    cv_rally = np.mean(cv_scores_rally)
    print(f"\n📈 Task3 CV Score: {cv_rally:.4f}")
    
    print("⏱️  訓練最終模型...")
    model_rally = LGBMClassifier(**params_fast)
    model_rally.fit(X, y_rally)
    print("✅ Task3 完成")
else:
    cv_rally = 0.5
    model_rally = None
    print("⚠️  Task3 跳過")

# ============================================================
# 整體 CV 分數
# ============================================================
print("\n" + "="*60)
print("📊 整體 CV 分數")
print("="*60)

blended_cv = (cv_action + cv_point + cv_rally) / 3
print(f"Task1 (ActionId)  : {cv_action:.4f}")
print(f"Task2 (PointId)   : {cv_point:.4f}")
print(f"Task3 (ServerWin) : {cv_rally:.4f}")
print(f"\n🎯 Blended Score: {blended_cv:.4f}")

# ============================================================
# 7. 測試集對齊
# ============================================================
print("\n" + "="*60)
print("🔧 測試集對齊")
print("="*60)

print(f"原始測試集: {clean_test.shape}")
print(f"唯一 rally: {clean_test['rally_uid'].nunique()}")

# ✅ 關鍵修正：只保留每個 rally 的最後一拍
if 'stroke_number' in clean_test.columns:
    test_aligned = clean_test.loc[clean_test.groupby('rally_uid')['stroke_number'].idxmax()].copy()
    print(f"\n✅ 對齊策略：保留每個 rally 的最後一拍")
else:
    test_aligned = clean_test.groupby('rally_uid').last().reset_index()
    print(f"\n✅ 對齊策略：保留每個 rally 的最後一行")

print(f"對齊後測試集: {test_aligned.shape}")
print(f"唯一 rally: {test_aligned['rally_uid'].nunique()}")

# 檢查是否還有重複
if test_aligned['rally_uid'].duplicated().sum() > 0:
    print(f"❌ 警告：還有 {test_aligned['rally_uid'].duplicated().sum()} 個重複")
    test_aligned = test_aligned.drop_duplicates(subset='rally_uid', keep='last')
    print(f"   強制去重後: {test_aligned.shape}")
else:
    print(f"✅ 沒有重複的 rally_uid")

# ============================================================
# 8. 測試集預測
# ============================================================
print("\n" + "="*60)
print("🔮 測試集預測")
print("="*60)

# 準備測試集特徵
X_test = test_aligned[feature_cols].copy()
print(f"測試集特徵: {X_test.shape}")

# 填充缺失值
for col in X_test.columns:
    if X_test[col].isna().sum() > 0:
        fill_val = X_test[col].median() if X_test[col].dtype in ['float64', 'int64'] else 0
        X_test[col].fillna(fill_val, inplace=True)

# 預測
print("⏱️  預測 Task1...")
pred_action = model_action.predict(X_test)
pred_action_probs = model_action.predict_proba(X_test)

print("⏱️  預測 Task2...")
pred_point = model_point.predict(X_test)
pred_point_probs = model_point.predict_proba(X_test)

if model_rally:
    print("⏱️  預測 Task3...")
    pred_rally = model_rally.predict_proba(X_test)[:, 1]
else:
    pred_rally = np.full(len(X_test), 0.5)

print("✅ 預測完成")

# ============================================================
# 🆕 9. 檢查預測中是否有 -1
# ============================================================
print("\n" + "="*60)
print("🔍 預測結果檢查")
print("="*60)

neg1_in_pred_action = (pred_action == -1).sum()
neg1_in_pred_point = (pred_point == -1).sum()

print(f"\n預測統計:")
print(f"  actionId 包含 -1: {neg1_in_pred_action} 個")
print(f"  pointId 包含 -1:  {neg1_in_pred_point} 個")

if neg1_in_pred_action == 0 and neg1_in_pred_point == 0:
    print(f"\n✅ 完美！預測中沒有 -1")
    print(f"   這證明移除 -1 樣本訓練是成功的！")
else:
    print(f"\n❌ 警告：預測中仍有 -1")
    print(f"   這不應該發生，因為模型沒學過 -1")

print(f"\n預測範圍:")
print(f"  actionId: [{pred_action.min()}, {pred_action.max()}]")
print(f"  pointId:  [{pred_point.min()}, {pred_point.max()}]")
print(f"  serverGetPoint: mean={pred_rally.mean():.3f}")

# ============================================================
# 10. 生成 Submission
# ============================================================
print("\n" + "="*60)
print("📝 生成 Submission")
print("="*60)

submission = pd.DataFrame({
    'rally_uid': test_aligned['rally_uid'],
    'actionId': pred_action,
    'pointId': pred_point,
    'serverGetPoint': pred_rally
})

# 最終檢查
print(f"\nSubmission 檢查:")
print(f"  大小: {submission.shape}")
print(f"  唯一 rally_uid: {submission['rally_uid'].nunique()}")
print(f"  重複 rally_uid: {submission['rally_uid'].duplicated().sum()}")

if submission['rally_uid'].duplicated().sum() > 0:
    print(f"\n❌ 錯誤：還有重複！")
    submission = submission.drop_duplicates(subset='rally_uid', keep='first')
    print(f"   去重後: {submission.shape}")

# 保存
submission.to_csv('submission_no_minus1.csv', index=False, float_format="%.10f")
print(f"\n✅ 已保存: submission_no_minus1.csv")

# ============================================================
# 11. 診斷報告
# ============================================================
print("\n" + "="*60)
print("🔍 診斷報告")
print("="*60)

print(f"\n📊 CV 分數總結:")
print(f"  Task1 (ActionId)  : {cv_action:.4f}")
print(f"  Task2 (PointId)   : {cv_point:.4f}")
print(f"  Task3 (ServerWin) : {cv_rally:.4f}")
print(f"  Blended           : {blended_cv:.4f}")

print(f"\n🎯 關鍵改動總結:")
print(f"  ✅ 訓練時移除了 -1 樣本")
print(f"  ✅ 模型不會學到 -1 類別")
print(f"  ✅ 預測中沒有 -1")
print(f"  ✅ 所有預測都是實際動作/落點")

print(f"\n📈 預期效果:")
print(f"  如果理論正確：LB 應該比之前的 0.3241 更高")
print(f"  預期範圍：0.33-0.38")
print(f"  關鍵：提交後對比 LB")

print(f"\n🎯 判斷標準:")
print(f"  如果 LB > 0.3241 → ✅ 理論正確，繼續完整訓練")
print(f"  如果 LB < 0.3241 → ❓ 需要重新思考")

print("\n" + "="*60)
print("✅ 快速測試完成！")
print("="*60)
print("\n下一步：")
print("1. 提交 submission_no_minus1.csv")
print("2. 記錄 LB 分數")
print("3. 對比 0.3241")
print("4. 如果提升，執行完整訓練（100% 數據 + 更多樹）")