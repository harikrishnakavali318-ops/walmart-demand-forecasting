# Walmart Demand Forecasting – Process Flow Document

**NEUZENAI IT Solutions | AI Engineer Technical Assignment**  
**Prepared by:** Candidate  
**Model:** LightGBM Gradient Boosted Trees  

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                  WALMART DEMAND FORECASTING PIPELINE                        │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐      ┌─────────────────┐      ┌───────────────────────────┐
  │  DATA SOURCES │─────▶│  INGESTION &    │─────▶│  DATA CLEANING &          │
  │              │      │  VALIDATION     │      │  TRANSFORMATION           │
  │ train.csv    │      │                 │      │                           │
  │ test.csv     │      │ • Schema check  │      │ • MarkDown NaN → 0        │
  │ features.csv │      │ • Date parse    │      │ • Negative clips          │
  │ stores.csv   │      │ • Join on       │      │ • Median imputation       │
  └──────────────┘      │   Store+Date    │      │ • Outlier analysis        │
                        └─────────────────┘      └───────────────────────────┘
                                                              │
                                                              ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │                        FEATURE ENGINEERING                               │
  │                                                                          │
  │  Date Features     │  Holiday Flags       │  Lag / Rolling              │
  │  • Year, Month     │  • IsHoliday         │  • Sales_Lag1/2/4/52        │
  │  • Week, Quarter   │  • IsSuperBowl       │  • Roll4/13/26 Mean         │
  │  • DayOfYear       │  • IsThanksgiving    │  • Roll4 Std                │
  │  • Season          │  • IsChristmas       │  • WoW Change               │
  │                    │  • WeeksToXmas       │                             │
  │  Markdown Aggs     │  Store Features      │  Economic                   │
  │  • TotalMarkDown   │  • TypeEncoded       │  • EconPressure             │
  │  • MarkDown_Count  │  • SizeBucket        │  • TempBucket               │
  │  • MaxMarkDown     │  • StoreMeanSales    │                             │
  │                    │  • DeptMeanSales     │                             │
  │                    │  • StoreDeptEnc      │                             │
  └──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │                   TRAIN / VALIDATION SPLIT                               │
  │                                                                          │
  │  Training: 2010-02-05 → 2012-06-30          (~75% of data)              │
  │  Validation: 2012-07-01 → 2012-10-26        (~25% of data, future data) │
  │                                                                          │
  │  NOTE: Random splits are INVALID for time-series — they leak future      │
  │  information into the training set (data leakage). A temporal cutoff    │
  │  ensures the model is evaluated only on truly unseen future weeks.      │
  └──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │                         MODEL TRAINING                                   │
  │                                                                          │
  │  Model 1: Ridge Regression   → Linear baseline, fast, interpretable     │
  │  Model 2: Random Forest      → Ensemble, handles non-linearity           │
  │  Model 3: XGBoost            → Gradient boosting, competitive accuracy  │
  │  Model 4: LightGBM ★ BEST   → L1 objective, fastest, highest R²        │
  │                                                                          │
  │  LightGBM Hyperparameters:                                               │
  │    objective=regression_l1, learning_rate=0.03, num_leaves=127           │
  │    n_estimators=1500 (early stopping at round ~800), subsample=0.85     │
  └──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │                    EVALUATION & VALIDATION                               │
  │                                                                          │
  │  Primary Metrics:                                                        │
  │    • WMAE  – Kaggle competition metric (5× holiday weight)              │
  │    • MAE   – Absolute average error in dollars                          │
  │    • RMSE  – Penalises large errors more                                │
  │    • MAPE  – Percentage error (scale-independent)                       │
  │    • R²    – Proportion of variance explained                           │
  │                                                                          │
  │  Cross-Validation:                                                       │
  │    • 5-fold TimeSeriesSplit (rolling window expansion)                  │
  │    • Prevents data leakage across folds                                 │
  │                                                                          │
  │  Robustness Tests:                                                       │
  │    • Holiday vs non-holiday MAE comparison                              │
  │    • Performance by store type (A / B / C)                              │
  │    • Residual analysis: normality, heteroscedasticity, temporal trend   │
  └──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │                        EXPLAINABILITY                                    │
  │                                                                          │
  │  Global (SHAP TreeExplainer):                                            │
  │    • Summary beeswarm plot (top 20 features)                            │
  │    • Feature importance bar chart (Mean |SHAP|)                         │
  │    • Dependence plots for top 3 features                                │
  │                                                                          │
  │  Local (SHAP Waterfall + LIME):                                          │
  │    • Waterfall charts for high-sales, low-sales, holiday predictions    │
  │    • LIME tabular explanations for 3 individual rows                    │
  └──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │                    PREDICTION & POST-PROCESSING                          │
  │                                                                          │
  │  • Model predicts on test set (2012-11-02 → 2013-07-26)                │
  │  • Clip predictions to [0, ∞) — sales cannot be negative               │
  │  • Export submission.csv in Kaggle format                               │
  │  • Save model artefacts: lgb_model.txt, label_encoder, scaler           │
  └──────────────────────────────────────────────────────────────────────────┘
```

---

## Stage-by-Stage Explanation

### Stage 1 – Data Ingestion & Validation

**What happens:**  
Four CSV files are loaded from the `data/` directory. Dates are parsed to `datetime64` at load time. The three data sources are joined using a left-merge strategy:
- `train.csv` is the primary table (Store, Dept, Date, Weekly_Sales, IsHoliday).
- `features.csv` is merged on `[Store, Date, IsHoliday]` to add weather, economic, and markdown data.
- `stores.csv` is merged on `Store` to add store type (A/B/C) and square footage.

**Validation checks performed:**
- Date range sanity (train: Feb 2010 – Oct 2012; test: Nov 2012 – Jul 2013)
- No duplicate rows in train (Store, Dept, Date should be unique)
- Expected 45 stores, 81 departments

**Why this matters:** Incorrect joins or silently missing rows would corrupt all downstream features and model training.

---

### Stage 2 – Data Cleaning & Transformation

**MarkDown columns (1–5):**  
MarkDown availability began only partway through the training period. Missing values do NOT mean "zero discount was applied" in all cases — however, for modelling purposes, filling with 0 is the standard approach used in the Kaggle community and aligns with the assumption that no promotional spend occurred. Negative markdown values (data entry errors) are clipped to 0.

**CPI, Unemployment, Temperature, Fuel_Price:**  
A small number of rows have missing values due to the features file not covering all store-date combinations. Median imputation per-column is used (robust to outliers, avoids distributional distortion from mean imputation on skewed data).

**Outlier handling:**  
Weekly sales range from negative (returns-heavy weeks) to very large positive values. Extreme negative values that represent net-return weeks are retained — the model should learn these patterns, not exclude them. No capping is applied to the target.

---

### Stage 3 – Feature Engineering

**Date features:**  
Cyclical retail demand is tightly linked to the calendar. `Week` (ISO week number) captures within-year seasonality. `Quarter` aggregates seasonal periods. `DayOfYear` provides finer granularity.

**Holiday flags:**  
Rather than relying solely on the `IsHoliday` binary flag, specific holiday identifiers (`IsSuperBowl`, `IsThanksgiving`, `IsChristmas`) are created because each holiday has a distinct demand profile. `WeeksToXmas` and `WeeksToThanks` capture the demand build-up in the weeks leading up to key events.

**Lag features:**  
- `Sales_Lag1`: most recent prior week (captures momentum)
- `Sales_Lag52`: same week last year (captures year-over-year seasonality)
- Rolling means (4, 13, 26 weeks): smooth out noise and capture trends

**Why lag features use `.shift(1)`:** Using the same-week sales as a feature would leak the target. All lags are shifted by at least 1 period.

**Store-level aggregations:**  
`StoreMeanSales` and `DeptMeanSales` give the model a prior on the expected sales level for each entity, reducing the number of tree splits needed to encode store/department identity.

**`StoreDeptEnc`:**  
Label-encoded concatenation of Store and Department IDs. LightGBM can exploit this single ordinal encoding more efficiently than two separate categorical columns.

---

### Stage 4 – Model Training & Validation

**Why not a random train/test split?**  
Time-series data has temporal autocorrelation — adjacent weeks are correlated. A random split would place future data in the training set and past data in validation, making the model appear more accurate than it truly is (data leakage). The temporal cutoff (July 2012) ensures the validation set strictly follows the training set chronologically, mimicking production conditions.

**LightGBM configuration rationale:**
- `objective='regression_l1'`: directly optimises MAE, which aligns with the Kaggle WMAE metric.
- `num_leaves=127`: sufficient capacity to model ~3,500 store-department combinations without severe underfitting.
- `early_stopping_rounds=80`: prevents overfitting by halting training when validation MAE stops improving for 80 consecutive rounds.
- `learning_rate=0.03`: conservative rate that, combined with early stopping, finds a stable minimum.
- `subsample=0.85, colsample_bytree=0.75`: stochastic feature/row sub-sampling reduces overfitting and variance.

**TimeSeriesSplit (5-fold):**  
Each fold expands the training window and advances the validation window forward in time. This tests whether the model generalises across different seasonal periods (e.g., can a model trained on two winters predict a third?).

---

### Stage 5 – Prediction & Post-Processing

**Lag feature approximation for test set:**  
The test period begins just after the training period ends, so `Sales_Lag1` for the first test week is the last known training week's sales for that store-department. This is propagated from the training set. For longer horizons, a recursive forecasting strategy (use predictions as lag inputs) would be employed.

**Clip predictions ≥ 0:**  
LightGBM does not constrain output, and rare extreme negative predictions can occur. Weekly sales cannot be physically negative (net returns are reported separately in accounting), so predictions are clipped.

**Artefact storage:**  
- `models/lgb_model.txt`: native LightGBM model format, platform-independent
- `models/label_encoder_store_dept.pkl`: required to encode new store-department combinations at inference
- `models/standard_scaler.pkl`: retained for consistency; not used by LightGBM but required if Ridge baseline is re-deployed

---

## Evaluation Metrics – Rationale

| Metric | Formula | When it Matters |
|--------|---------|-----------------|
| MAE | mean(|y - ŷ|) | Directly interpretable in dollars; robust to outliers |
| RMSE | √mean((y - ŷ)²) | Penalises large errors — crucial for detecting systematic forecast misses in high-volume weeks |
| WMAE | Σ(w·|y - ŷ|) / Σw | Kaggle competition primary metric; holiday weeks receive 5× weight, reflecting higher business cost of holiday forecast errors |
| MAPE | mean(|y - ŷ|/y) | Scale-independent — useful for comparing accuracy across departments of very different sizes |
| R² | 1 − SS_res/SS_tot | Proportion of variance explained; gives a single summary of model fit quality |

---

## Known Limitations & Mitigations

| Limitation | Mitigation in This Work | Future Improvement |
|---|---|---|
| Lag features not available for test | Approximated from last training week | Rolling forecast with recursive update |
| Cold-start (new stores) | Store cluster priors via StoreMeanSales | Hierarchical model (global + local) |
| Markdown data gaps | Filled with 0 | Time-series imputation (KNN or forward-fill) |
| No external signals | Within-dataset features only | Web scraping of competitor prices, weather APIs |
| Single model | Best single model selected | Stacking/blending ensemble (LightGBM + Prophet) |
