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
from sklearn.metrics import r2_score
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

# Multi-alpha EWM + holiday interaction features
df = df.sort_values(["Store","Dept","Date"]).reset_index(drop=True)
grp_s = df.groupby(["Store","Dept"])["Weekly_Sales"]

for alpha in [0.1, 0.3, 0.6]:
    col = f"EWM_{int(alpha*10)}"
    df[col] = (grp_s
               .transform(lambda x, a=alpha: x.shift(1).ewm(alpha=a, adjust=False).mean())
               .fillna(grp_s.transform("mean")))

# Holiday × lag interaction
df["hol_x_lag1"]  = df["IsHoliday"].astype(int) * df["Lag_1"].fillna(0)
df["xmas_x_lag52"] = df["IsChristmas"].astype(int) * df["Lag_52"].fillna(0)

# Sales acceleration: lag1 - lag4 mean
df["Sales_accel"] = df["Lag_1"].fillna(0) - df["Roll_4_mean"].fillna(0)

# Log ratio
df["log_lag1_vs_roll26"] = np.log1p(df["Lag_1"].fillna(0)) - np.log1p(df["Roll_26_mean"].fillna(0))

# Polynomial & log transforms on key lags
df["log_lag1"]   = np.log1p(df["Lag_1"].fillna(0))
df["log_lag52"]  = np.log1p(df["Lag_52"].fillna(0))
df["lag1_sq"]    = df["Lag_1"].fillna(0) ** 2 / 1e8   # scale down
df["lag1_x_lag52"] = df["Lag_1"].fillna(0) * df["Lag_52"].fillna(0) / 1e8
df["trend_ratio"]  = df["Roll_4_mean"].fillna(0) / (df["Roll_26_mean"].fillna(1).replace(0,1))

train, val = get_train_val(df)

EXTRA = ["EWM_1","EWM_3","EWM_6",
         "hol_x_lag1","xmas_x_lag52","Sales_accel","log_lag1_vs_roll26",
         "log_lag1","log_lag52","lag1_sq","lag1_x_lag52","trend_ratio"]
FEAT = ALL_FEATURES + EXTRA

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
    objective         = "regression_l1",
    metric            = "mae",
    n_estimators      = 2000,
    learning_rate     = 0.025,
    num_leaves        = 255,
    max_depth         = -1,
    min_child_samples = 8,
    subsample         = 0.85,
    subsample_freq    = 1,
    colsample_bytree  = 0.75,
    reg_alpha         = 0.05,
    reg_lambda        = 0.3,
    random_state      = 42,
    n_jobs            = -1,
    verbose           = -1,
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

# ── LightGBM-2: low LR, different seed ─────────────────────────────────
lgb2_params = {**lgb_params, "learning_rate": 0.02, "random_state": 123,
               "n_estimators": 2500, "num_leaves": 200}
lgb2_model = lgb.LGBMRegressor(**lgb2_params)
lgb2_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])
pred_lgb2 = lgb2_model.predict(X_val)

# ── LightGBM-3: L2 objective ────────────────────────────────────────────
lgb3_params = {**lgb_params, "objective": "regression_l2", "metric": "rmse",
               "learning_rate": 0.025, "random_state": 7, "num_leaves": 300}
lgb3_model = lgb.LGBMRegressor(**lgb3_params)
lgb3_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])
pred_lgb3 = lgb3_model.predict(X_val)

# ── LightGBM-4: Huber loss (robust to outliers) + Dart boosting ─────────
lgb4_params = {**lgb_params, "objective": "huber", "alpha": 0.9,
               "boosting_type": "dart", "drop_rate": 0.1,
               "learning_rate": 0.05, "random_state": 99,
               "n_estimators": 600, "num_leaves": 255}
lgb4_model = lgb.LGBMRegressor(**lgb4_params)
lgb4_model.fit(X_tr, y_tr)   # DART doesn't support early stopping
pred_lgb4 = lgb4_model.predict(X_val)

# ── Optimise 5-way blend (scipy minimize) ───────────────────────────────
from scipy.optimize import minimize

all_preds = np.stack([pred_lgb, pred_lgb2, pred_lgb3, pred_lgb4, pred_xgb], axis=1)
y_arr = y_val.values

def neg_r2(w):
    w = np.abs(w); w /= w.sum()
    return -r2_score(y_arr, all_preds @ w)

w0 = np.array([0.35, 0.2, 0.2, 0.1, 0.15])
res = minimize(neg_r2, w0, method="Nelder-Mead",
               options={"maxiter": 2000, "xatol": 1e-6})
best_w = np.abs(res.x); best_w /= best_w.sum()

pred = np.clip(all_preds @ best_w, 0, None)

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
