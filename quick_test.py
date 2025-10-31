"""
超快速訓練 - 使用現有的 clean_train_v2.csv
預計執行時間: 2-3 分鐘
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from lightgbm import LGBMClassifier
from collections import Counter

print("="*60)
print("⚡ 超快速訓練（使用現有數據）")
print("="*60)

# ============ 讀取數據 ============
DATA_DIR = "data"
print("\n📁 讀取數據...")

try:
    clean_train = pd.read_csv(f"{DATA_DIR}/clean_train_v2.csv", low_memory=False)
    clean_test = pd.read_csv(f"{DATA_DIR}/clean_test_v2.csv", low_memory=False)
    print(f"✅ 使用 v2 版本")
except:
    clean_train = pd.read_csv(f"{DATA_DIR}/clean_train.csv", low_memory=False)
    clean_test = pd.read_csv(f"{DATA_DIR}/clean_test.csv", low_memory=False)
    print(f"✅ 使用 v1 版本")

sample_sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv", low_memory=False)

print(f"clean_train: {clean_train.shape}")
print(f"clean_test: {clean_test.shape}")

# ============ 🔥 關鍵：先移除 -1 標籤 ============
ACTION_COL = 'next_actionId'
POINT_COL = 'next_pointId'

print("\n🔧 檢查並移除 -1 標籤...")
before = len(clean_train)

# 移除 -1 標籤
if ACTION_COL in clean_train.columns and POINT_COL in clean_train.columns:
    clean_train = clean_train[
        (clean_train[ACTION_COL] != -1) & 
        (clean_train[POINT_COL] != -1) &
        clean_train[ACTION_COL].notna() &
        clean_train[POINT_COL].notna()
    ].reset_index(drop=True)
    
    clean_train[ACTION_COL] = clean_train[ACTION_COL].astype(int)
    clean_train[POINT_COL] = clean_train[POINT_COL].astype(int)
    
    after = len(clean_train)
    removed = before - after
    
    if removed > 0:
        print(f"  ⚠️ 移除了 {removed} 行 ({removed/before*100:.1f}%)")
    else:
        print(f"  ✅ 無需移除")

# 驗證
action_dist = Counter(clean_train[ACTION_COL])
print(f"\n✅ 標籤統計:")
print(f"  ActionId: {len(action_dist)} 類，範圍 [{min(action_dist)}, {max(action_dist)}]")
print(f"  有 -1 嗎? {-1 in action_dist}")

if -1 in action_dist:
    print("  ❌ 警告：仍有 -1 標籤！")
    print("  這會導致分數很低")
else:
    print("  ✅ 無 -1 標籤，可以繼續")

# ============ 配置 ============
KEY = 'rally_uid'
RALLY_TARGET = 'serverGetPoint' if 'serverGetPoint' in clean_train.columns else None

# 特徵選擇
exclude_cols = {
    'match_id', 'rally_id', 'stroke_number', 'strickNumber',
    'next_actionId', 'next_pointId', 'serverGetPoint', 'server_won_point',
    'rally_uid', 'actionId', 'pointId', 'Unnamed: 0'
}

all_features = [c for c in clean_train.columns if c not in exclude_cols]
print(f"\n📊 可用特徵數: {len(all_features)}")

# ============ Task 1: ActionId ============
print("\n" + "="*60)
print("🎯 Task 1: ActionId (快速版)")
print("="*60)

X_action = clean_train[all_features].copy().fillna(0)
y_action = clean_train[ACTION_COL].astype(int)

# 單次切分（不用 5-fold，更快）
X_tr, X_va, y_tr, y_va = train_test_split(
    X_action, y_action, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_action
)

# 類別權重
cnt = Counter(y_tr)
weight = {cls: 1.0/np.sqrt(c) for cls, c in cnt.items()}

# 快速模型配置
model_action = LGBMClassifier(
    n_estimators=150,      # 少樹數
    learning_rate=0.1,     # 高學習率
    num_leaves=31,         # 簡單樹
    max_depth=5,           # 淺樹
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
    class_weight=weight
)

print("訓練中 (150 棵樹)...")
model_action.fit(X_tr, y_tr)

# 驗證
y_pred = model_action.predict(X_va)
score_action = f1_score(y_va, y_pred, average='macro')
print(f"✅ Task1 Validation Score: {score_action:.4f}")

# 用全量數據訓練最終模型
print("用全量數據訓練最終模型...")
model_action_full = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.08,
    num_leaves=31,
    max_depth=6,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
    class_weight=weight
)
model_action_full.fit(X_action, y_action)

# Top 特徵
importances = pd.Series(
    model_action_full.feature_importances_,
    index=X_action.columns
).sort_values(ascending=False)
print(f"\n🔝 Top 5 重要特徵:")
for feat, imp in importances.head(5).items():
    print(f"  {feat}: {imp:.1f}")

# ============ Task 2: PointId ============
print("\n" + "="*60)
print("🎯 Task 2: PointId (快速版)")
print("="*60)

X_point = clean_train[all_features].copy().fillna(0)
y_point = clean_train[POINT_COL].astype(int)

X_tr, X_va, y_tr, y_va = train_test_split(
    X_point, y_point,
    test_size=0.2,
    random_state=42,
    stratify=y_point
)

cnt = Counter(y_tr)
weight = {cls: 1.0/np.sqrt(c) for cls, c in cnt.items()}

model_point = LGBMClassifier(
    n_estimators=150,
    learning_rate=0.1,
    num_leaves=31,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
    class_weight=weight
)

print("訓練中 (150 棵樹)...")
model_point.fit(X_tr, y_tr)

y_pred = model_point.predict(X_va)
score_point = f1_score(y_va, y_pred, average='macro')
print(f"✅ Task2 Validation Score: {score_point:.4f}")

# 全量訓練
model_point_full = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.08,
    num_leaves=31,
    max_depth=6,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
    class_weight=weight
)
model_point_full.fit(X_point, y_point)

# ============ Task 3: ServerGetPoint ============
print("\n" + "="*60)
print("🎯 Task 3: ServerGetPoint (快速版)")
print("="*60)

if RALLY_TARGET:
    X_rally = clean_train[all_features].copy().fillna(0)
    y_rally = clean_train[RALLY_TARGET].astype(int)
    
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_rally, y_rally,
        test_size=0.2,
        random_state=42,
        stratify=y_rally
    )
    
    model_rally = LGBMClassifier(
        n_estimators=150,
        learning_rate=0.1,
        num_leaves=31,
        max_depth=5,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    print("訓練中 (150 棵樹)...")
    model_rally.fit(X_tr, y_tr)
    
    y_prob = model_rally.predict_proba(X_va)[:, 1]
    score_rally = roc_auc_score(y_va, y_prob)
    print(f"✅ Task3 Validation Score: {score_rally:.4f}")
    
    # 全量訓練
    model_rally_full = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.08,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    model_rally_full.fit(X_rally, y_rally)
else:
    score_rally = 0.5
    model_rally_full = None
    print("⚠️ 找不到 serverGetPoint")

# ============ 總結 ============
print("\n" + "="*60)
print("📊 驗證分數總結")
print("="*60)
print(f"Task1 (ActionId)  : {score_action:.4f}")
print(f"Task2 (PointId)   : {score_point:.4f}")
print(f"Task3 (ServerWin) : {score_rally:.4f}")

blended = 0.4 * score_action + 0.4 * score_point + 0.2 * score_rally
print(f"\n🎯 Blended Score: {blended:.4f}")

# 評估
print("\n💡 評估:")
if blended < 0.20:
    print("❌ 分數很低！可能：")
    print("   1. 數據中仍有 -1 標籤")
    print("   2. 需要重新處理數據")
elif blended < 0.30:
    print("⚠️ 分數偏低但可接受")
    print("   建議：使用更多特徵或調整參數")
elif blended < 0.40:
    print("✅ 分數正常！")
    print("   快速版已足夠驗證修正效果")
else:
    print("🎉 分數很好！")

# ============ 預測 ============
print("\n" + "="*60)
print("🔮 生成 Submission")
print("="*60)

# 對齊測試集
test_df = clean_test.copy()
test_df[KEY] = test_df[KEY].astype(str)
sample_sub[KEY] = sample_sub[KEY].astype(str)

if 'stroke_number' in test_df.columns:
    dedup = (test_df
             .sort_values([KEY, 'stroke_number'])
             .groupby(KEY, as_index=False)
             .tail(1))
elif 'strickNumber' in test_df.columns:
    dedup = (test_df
             .sort_values([KEY, 'strickNumber'])
             .groupby(KEY, as_index=False)
             .tail(1))
else:
    dedup = test_df.drop_duplicates(subset=[KEY], keep='last')

test_aligned = (dedup.set_index(KEY)
                .reindex(sample_sub[KEY])
                .reset_index())

# 特徵準備
features_in_test = [f for f in all_features if f in test_aligned.columns]
X_test = test_aligned[features_in_test].fillna(0)

print(f"測試集特徵數: {len(features_in_test)}")

# 預測
print("預測中...")
pred_action = model_action_full.predict(X_test)
pred_point = model_point_full.predict(X_test)

if model_rally_full:
    pred_rally_prob = model_rally_full.predict_proba(X_test)[:, 1]
else:
    pred_rally_prob = np.full(len(X_test), 0.5)

# 生成 submission
submission = sample_sub.copy()
submission['actionId'] = pred_action.astype(int)
submission['pointId'] = pred_point.astype(int)
submission['serverGetPoint'] = pred_rally_prob

submission.to_csv('submission_fast.csv', index=False)
print(f"✅ 已生成: submission_fast.csv")

print("\n📊 預測統計:")
print(f"ActionId 類別數: {submission['actionId'].nunique()}")
print(f"PointId 類別數: {submission['pointId'].nunique()}")
print(f"ServerGetPoint 平均: {submission['serverGetPoint'].mean():.3f}")

print("\n" + "="*60)
print("🎉 完成！")
print("="*60)
print(f"\n總執行時間: 約 2-3 分鐘")
print(f"使用數據: {DATA_DIR}/clean_train_v2.csv")
print(f"輸出檔案: submission_fast.csv")

if blended >= 0.30:
    print(f"\n✅ 驗證分數正常 ({blended:.4f})")
    print("   如需更高分，可以：")
    print("   1. 增加樹的數量 (n_estimators)")
    print("   2. 降低學習率 (learning_rate)")
    print("   3. 使用 5-Fold CV")
else:
    print(f"\n⚠️ 驗證分數偏低 ({blended:.4f})")
    print("   建議檢查數據是否正確處理")