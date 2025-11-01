"""
驗證：交叉驗證策略對比
比較 rally_id vs match_id 分組的 CV 分數
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🔬 驗證：交叉驗證策略對比")
print("="*80)

# ============================================================
# 1. 載入數據
# ============================================================
print("\n📁 載入數據...")

try:
    train = pd.read_csv("data/clean_train_v2.csv")
    print(f"✓ 訓練集: {len(train)} 行")
except FileNotFoundError:
    print("❌ 找不到 data/clean_train_v2.csv")
    exit(1)

# ============================================================
# 2. 檢查數據結構
# ============================================================
print("\n" + "="*80)
print("📊 數據結構分析")
print("="*80)

print(f"\n比賽數 (match): {train['match'].nunique()}")
print(f"回合數 (rally_id): {train['rally_id'].nunique()}")
print(f"平均每場比賽的回合數: {train['rally_id'].nunique() / train['match'].nunique():.1f}")

# 每場比賽的回合數
rallies_per_match = train.groupby('match')['rally_id'].nunique()
print(f"\n每場比賽的回合數分佈:")
print(f"  最小: {rallies_per_match.min()}")
print(f"  最大: {rallies_per_match.max()}")
print(f"  平均: {rallies_per_match.mean():.1f}")
print(f"  中位數: {rallies_per_match.median():.1f}")

# ============================================================
# 3. 準備特徵和標籤
# ============================================================
print("\n" + "="*80)
print("🔧 準備數據")
print("="*80)

# 排除欄位
exclude_cols = [
    'match', 'rally_id', 'stroke_number',
    'next_actionId', 'next_pointId', 'server_won_point',
    'rally_uid'
]

# 特徵
features = [c for c in train.columns if c not in exclude_cols]
print(f"\n特徵數: {len(features)}")

# 移除缺失值
if 'next_actionId' not in train.columns:
    print("❌ 找不到 next_actionId 欄位")
    exit(1)

# 準備數據
X = train[features].fillna(0)
y = train['next_actionId']

print(f"樣本數: {len(X)}")
print(f"標籤分佈: {y.value_counts().head()}")

# ============================================================
# 4. 測試方式 1：rally_id 分組（你現在的方式）
# ============================================================
print("\n" + "="*80)
print("🔴 方式 1：按 rally_id 分組（你現在的方式）")
print("="*80)

groups_rally = train['rally_id'].values
gkf1 = GroupKFold(n_splits=5)

print(f"分組數: {len(np.unique(groups_rally))}")
print("開始 5-Fold 交叉驗證...")

scores_rally = []
for fold, (train_idx, val_idx) in enumerate(gkf1.split(X, y, groups_rally), 1):
    X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
    
    # 快速訓練（減少樹的數量以加快速度）
    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )
    model.fit(X_tr, y_tr)
    
    pred = model.predict(X_va)
    score = f1_score(y_va, pred, average='macro')
    scores_rally.append(score)
    
    print(f"  Fold {fold}: {score:.4f}")

mean_rally = np.mean(scores_rally)
std_rally = np.std(scores_rally)
print(f"\n平均 CV: {mean_rally:.4f} ± {std_rally:.4f}")

# ============================================================
# 5. 測試方式 2：match 分組（正確的方式）
# ============================================================
print("\n" + "="*80)
print("🟢 方式 2：按 match 分組（正確的方式）")
print("="*80)

groups_match = train['match'].values
gkf2 = GroupKFold(n_splits=5)

print(f"分組數: {len(np.unique(groups_match))}")
print("開始 5-Fold 交叉驗證...")

scores_match = []
for fold, (train_idx, val_idx) in enumerate(gkf2.split(X, y, groups_match), 1):
    X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
    
    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )
    model.fit(X_tr, y_tr)
    
    pred = model.predict(X_va)
    score = f1_score(y_va, pred, average='macro')
    scores_match.append(score)
    
    print(f"  Fold {fold}: {score:.4f}")

mean_match = np.mean(scores_match)
std_match = np.std(scores_match)
print(f"\n平均 CV: {mean_match:.4f} ± {std_match:.4f}")

# ============================================================
# 6. 對比分析
# ============================================================
print("\n" + "="*80)
print("📊 對比分析")
print("="*80)

print(f"\n{'策略':<20} {'CV 分數':<15} {'標準差':<10}")
print("-"*50)
print(f"{'rally_id 分組':<20} {mean_rally:<15.4f} {std_rally:<10.4f}")
print(f"{'match 分組':<20} {mean_match:<15.4f} {std_match:<10.4f}")
print("-"*50)
print(f"{'差距':<20} {mean_rally - mean_match:<15.4f}")

# ============================================================
# 7. 診斷結論
# ============================================================
print("\n" + "="*80)
print("🎯 診斷結論")
print("="*80)

cv_gap = mean_rally - mean_match
lb_score = 0.15  # 你的 LB 分數

print(f"\nCV 分數差距: {cv_gap:.4f}")
print(f"你的 LB 分數: {lb_score:.4f}")
print(f"rally_id CV 與 LB 差距: {mean_rally - lb_score:.4f}")
print(f"match_id CV 與 LB 差距: {mean_match - lb_score:.4f}")

if cv_gap > 0.05:
    print("\n" + "="*80)
    print("❌ 【問題確認】交叉驗證策略導致 CV 虛高！")
    print("="*80)
    
    print(f"\n證據:")
    print(f"  1. rally_id 分組的 CV ({mean_rally:.4f}) 明顯高於 match_id ({mean_match:.4f})")
    print(f"  2. 差距達 {cv_gap:.4f}，超過 0.05 的閾值")
    print(f"  3. match_id 的 CV ({mean_match:.4f}) 更接近你的 LB ({lb_score:.4f})")
    
    print(f"\n原因:")
    print(f"  - rally_id 分組：同一場比賽的回合被分散在訓練集和驗證集")
    print(f"  - 驗證集可以從訓練集中「學到」該場比賽的特徵")
    print(f"  - 導致 CV 分數虛高")
    
    print(f"\n解決方案:")
    print(f"  ✓ 將 GROUP_COL 從 'rally_id' 改為 'match'")
    print(f"  ✓ 重新訓練模型")
    print(f"  ✓ 預期 LB 會提升到 {mean_match - 0.02:.2f} - {mean_match + 0.02:.2f}")
    
    print("\n" + "="*80)
    print("🔧 修改建議")
    print("="*80)
    
    print("\n在 LightGBM.ipynb 中修改:")
    print("""
# 原來的（錯誤）
GROUP_COL = 'rally_id'

# 改為（正確）
GROUP_COL = 'match'
""")
    
    print("\n然後重新運行整個訓練流程")
    
elif cv_gap > 0.02:
    print("\n⚠️ 【可能問題】CV 策略可能有輕微洩漏")
    print(f"   rally_id CV ({mean_rally:.4f}) 略高於 match CV ({mean_match:.4f})")
    print(f"   建議改用 match 分組以獲得更可靠的 CV")
    
else:
    print("\n✓ 兩種策略差異不大")
    print("   交叉驗證策略不是主要問題")
    print("\n   需要檢查其他可能原因:")
    print("   1. Task3 特徵洩漏")
    print("   2. 測試集分佈不同")
    print("   3. 模型過擬合")

# ============================================================
# 8. 保存結果
# ============================================================
results = pd.DataFrame({
    'Strategy': ['rally_id', 'match'],
    'Mean_CV': [mean_rally, mean_match],
    'Std_CV': [std_rally, std_match],
    'Gap_to_LB': [mean_rally - lb_score, mean_match - lb_score]
})

results.to_csv("cv_strategy_comparison.csv", index=False)
print("\n✓ 結果已保存: cv_strategy_comparison.csv")

print("\n" + "="*80)
print("✓ 驗證完成")
print("="*80)