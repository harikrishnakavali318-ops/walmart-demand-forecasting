"""
walmart_train.py  –  MODIFIABLE (autoresearch-style).
This file is iteratively improved to maximise R² on the validation set.
Run:  python walmart_train.py
"""

import time, sys, json
import numpy as np
import pandas as pd
import lightgbm as lgb
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
train, val = get_train_val(df)

X_tr  = train[ALL_FEATURES].fillna(0)
y_tr  = train[TARGET]
X_val = val[ALL_FEATURES].fillna(0)
y_val = val[TARGET]
is_hol = val["IsHoliday"].values

# ────────────────────────────────────────────────────────────────────────────
# MODEL  (edit this section in each experiment)
# ────────────────────────────────────────────────────────────────────────────

params = dict(
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

model = lgb.LGBMRegressor(**params)
model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    callbacks=[
        lgb.early_stopping(100, verbose=False),
        lgb.log_evaluation(-1),
    ],
)

pred = model.predict(X_val)
pred = np.clip(pred, 0, None)

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
