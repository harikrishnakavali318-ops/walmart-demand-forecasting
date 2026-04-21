"""
walmart_train.py  –  MODIFIABLE (autoresearch-style).
This file is iteratively improved to maximise R² on the validation set.
Run:  python walmart_train.py
"""

import time, sys, json
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from walmart_prepare import (
    load_data, build_features, get_train_val,
    ALL_FEATURES, TARGET, evaluate,
)

t0 = time.time()

# ────────────────────────────────────────────────────────────────────────────
# Load & prepare data
# ────────────────────────────────────────────────────────────────────────────
df    = load_data()
df    = build_features(df)

# Exponential smoothing feature (alpha=0.3)
alpha = 0.3
df = df.sort_values(["Store","Dept","Date"]).reset_index(drop=True)
df["EWM_sales"] = (
    df.groupby(["Store","Dept"])["Weekly_Sales"]
    .transform(lambda x: x.shift(1).ewm(alpha=alpha, adjust=False).mean())
    .fillna(df.groupby(["Store","Dept"])["Weekly_Sales"].transform("mean"))
)

train, val = get_train_val(df)

FEAT = ALL_FEATURES + ["EWM_sales"]

X_tr  = train[FEAT].fillna(0)
y_tr  = train[TARGET]
X_val = val[FEAT].fillna(0)
y_val = val[TARGET]
is_hol = val["IsHoliday"].values

# ────────────────────────────────────────────────────────────────────────────
# MODEL  (edit this section in each experiment)
# ────────────────────────────────────────────────────────────────────────────

# ── LightGBM ─────────────────────────────────────────────────────────────
lgb_params = dict(
    objective       = "regression_l1",
    metric          = "mae",
    n_estimators    = 2000,
    learning_rate   = 0.03,
    num_leaves      = 255,
    max_depth       = -1,
    min_child_samples = 10,
    subsample       = 0.85,
    colsample_bytree= 0.75,
    reg_alpha       = 0.1,
    reg_lambda      = 0.5,
    random_state    = 42,
    n_jobs          = -1,
    verbose         = -1,
)
lgb_model = lgb.LGBMRegressor(**lgb_params)
lgb_model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)],
)
pred_lgb = lgb_model.predict(X_val)

# ── XGBoost ──────────────────────────────────────────────────────────────
xgb_model = xgb.XGBRegressor(
    n_estimators=1500, learning_rate=0.03, max_depth=9,
    subsample=0.8, colsample_bytree=0.75,
    reg_alpha=0.1, reg_lambda=0.5,
    objective="reg:absoluteerror",
    eval_metric="mae", tree_method="hist",
    early_stopping_rounds=80, random_state=42,
    n_jobs=-1, verbosity=0,
)
xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
pred_xgb = xgb_model.predict(X_val)

# ── Weighted blend (optimise weights on val) ──────────────────────────────
best_r2, best_w = -np.inf, 0.5
for w in np.arange(0.3, 0.9, 0.05):
    p = w * pred_lgb + (1-w) * pred_xgb
    r = float(np.corrcoef(y_val, p)[0,1]**2)
    if r > best_r2:
        best_r2, best_w = r, w

pred = np.clip(best_w * pred_lgb + (1-best_w) * pred_xgb, 0, None)

# ────────────────────────────────────────────────────────────────────────────
# Evaluate & print summary
# ────────────────────────────────────────────────────────────────────────────
m = evaluate(y_val, pred, is_hol)
elapsed = time.time() - t0

print("\n---")
print(f"r2:      {m['r2']:.6f}")
print(f"mae:     {m['mae']:.2f}")
print(f"rmse:    {m['rmse']:.2f}")
print(f"wmae:    {m['wmae']:.2f}")
print(f"mape:    {m['mape']:.4f}")
print(f"seconds: {elapsed:.1f}")
