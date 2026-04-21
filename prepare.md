# walmart_prepare.py — Fixed Pipeline Documentation

> **Mirrors the autoresearch/Karpathy pattern:**
> `walmart_prepare.py` is the **read-only** data preparation and evaluation module.
> `walmart_train.py` is the **modifiable** training file iterated by the research loop.

---

## Purpose

`walmart_prepare.py` provides four things that never change across experiments:

| Export | Type | Description |
|---|---|---|
| `load_data()` | function | Loads real Kaggle CSVs or generates synthetic Walmart-like data |
| `build_features(df)` | function | Full deterministic feature engineering pipeline |
| `get_train_val(df)` | function | Temporal train/val split with target encoding |
| `evaluate(y_true, y_pred, is_holiday)` | function | Returns all 5 metrics (R², MAE, RMSE, WMAE, MAPE) |
| `ALL_FEATURES` | list[str] | Canonical feature column list for models |
| `TARGET` | str | `"Weekly_Sales"` |
| `VAL_CUTOFF` | str | `"2012-07-01"` — temporal split boundary |

---

## Data Loading

### Real data (Kaggle)
Place these four files in `data/`:
```
data/
├── train.csv       # Store, Dept, Date, Weekly_Sales, IsHoliday
├── test.csv        # Same schema without Weekly_Sales
├── features.csv    # Temperature, Fuel_Price, MarkDown1-5, CPI, Unemployment
└── stores.csv      # Store type (A/B/C) and Size (sq ft)
```

Merge strategy:
```
train ──left join── features  (on Store + Date + IsHoliday)
     ──left join── stores     (on Store)
```

### Synthetic data (fallback)
When CSVs are absent the module auto-generates 521,235 rows of realistic
Walmart-like data:
- 45 stores, 81 departments, 143 weekly dates (Feb 2010 – Oct 2012)
- Sales follow: `base × seasonal_multiplier × holiday_spike × noise`
- Seasonal: `sin/cos` of ISO week number
- Holiday spikes: ×1.3–1.8 in weeks 6, 7, 36, 47, 52

---

## Feature Engineering Pipeline

All features are computed inside `build_features(df)`. They are grouped
into 8 families:

### 1. Calendar Features
| Feature | Formula |
|---|---|
| `Year`, `Month`, `Week` | `dt.year`, `dt.month`, `dt.isocalendar().week` |
| `Quarter`, `DayOfYear` | `dt.quarter`, `dt.dayofyear` |
| `Season` | {Winter=3, Spring=0, Summer=1, Fall=2} mapped from Month |

### 2. Fourier Seasonality
Six features: `sin_week_k` and `cos_week_k` for k ∈ {1, 2, 3}
```
sin_week_k = sin(2π·k·Week / 52)
cos_week_k = cos(2π·k·Week / 52)
```
These give the model a smooth representation of within-year cyclicality.

### 3. Holiday Flags
| Feature | Condition |
|---|---|
| `IsHoliday` | Cast to int |
| `IsSuperBowl` | Week ∈ {6, 7} |
| `IsLaborDay` | Week == 36 |
| `IsThanksgiving` | Week == 47 |
| `IsChristmas` | Week == 52 |
| `WeeksToXmas` | `(52 − Week) % 52`, clipped to 12 |
| `WeeksToThanks` | `(47 − Week) % 52`, clipped to 8 |

### 4. Markdown Aggregates
| Feature | Formula |
|---|---|
| `TotalMarkDown` | Sum of MarkDown1–5 (NaN → 0, negative → 0) |
| `MarkDown_Count` | Count of non-zero markdowns |
| `MaxMarkDown` | Max of MarkDown1–5 |

### 5. Store / Economic Features
| Feature | Description |
|---|---|
| `TypeEncoded` | A→2, B→1, C→0 |
| `SizeBucket` | Quartile bin of store size [0–3] |
| `EconPressure` | CPI × Unemployment |
| `TempBucket` | Cold/Cool/Warm/Hot bins |

### 6. Lag Features  ← **Most important family**
All lags use `.shift(1)` within `(Store, Dept)` group to avoid leakage.

| Feature | Lag |
|---|---|
| `Lag_1` | 1 week (prior week — strongest signal) |
| `Lag_2`, `Lag_3`, `Lag_4` | 2–4 weeks |
| `Lag_8`, `Lag_13`, `Lag_26` | ~2, 3, 6 months |
| `Lag_52` | Same week last year |

### 7. Rolling Statistics
Computed on `.shift(1)` to prevent leakage.

| Feature | Window |
|---|---|
| `Roll_4_mean`, `Roll_4_std` | 4 weeks |
| `Roll_8_mean`, `Roll_8_std` | 8 weeks |
| `Roll_13_mean`, `Roll_13_std` | 13 weeks (~quarter) |
| `Roll_26_mean`, `Roll_26_std` | 26 weeks (~half-year) |
| `WoW_Change` | Pct change, clipped to [-2, 2] |
| `YoY_ratio` | Lag_52 / Roll_26_mean |

### 8. Target Encoding (applied in `get_train_val`)
Computed from **training data only** to prevent leakage.

| Feature | Encoding |
|---|---|
| `StoreDept_TE` | Mean Weekly_Sales per (Store, Dept) pair |
| `Store_TE` | Mean Weekly_Sales per Store |
| `Dept_TE` | Mean Weekly_Sales per Dept |

---

## Evaluation Metric: WMAE

The Kaggle competition primary metric is **Weighted Mean Absolute Error**:

```
WMAE = Σ(wᵢ · |yᵢ − ŷᵢ|) / Σ(wᵢ)

where wᵢ = 5  if IsHoliday == True
           1  otherwise
```

Holiday weeks receive 5× weight because forecast errors during peak retail
periods (Thanksgiving, Christmas, Super Bowl, Labor Day) carry 5× the business
cost of a regular-week error — missing a holiday restock can mean lost sales
of hundreds of thousands of dollars.

---

## Train / Validation Split

```
Training  : 2010-02-05 → 2012-06-29   (≈ 75% of rows)
Validation: 2012-07-06 → 2012-10-26   (≈ 25% of rows)
```

**Why not random split?**
Time series data has autocorrelation — adjacent weeks are correlated.
A random 75/25 split would place future weeks in the training set and past
weeks in validation, leaking future information into the model
(data leakage). The temporal cutoff ensures the model only ever trains on
data it would have seen in production before the validation period starts.

---

## Adding Real Kaggle Data

```bash
# Download from Kaggle (requires account + accepted competition rules)
# https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/data
mkdir -p data
# Copy train.csv, test.csv, features.csv, stores.csv into data/

# Verify
python -c "from walmart_prepare import load_data; df = load_data(); print(df.shape)"
# Should print: (421570, 16) for the real dataset
```
