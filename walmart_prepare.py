"""
walmart_prepare.py  –  FIXED. Do not modify.
Mirrors autoresearch/prepare.py: data loading, feature pipeline, evaluation.
The training script (walmart_train.py) imports from here.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

DATA_DIR  = Path("data")
SEED      = 42
VAL_CUTOFF = "2012-07-01"   # temporal split – no random shuffle

# ── Evaluation metric ──────────────────────────────────────────────────────
def wmae(y_true, y_pred, is_holiday):
    """Kaggle Weighted Mean Absolute Error (5× weight on holiday weeks)."""
    w = np.where(is_holiday, 5.0, 1.0)
    return float(np.sum(w * np.abs(y_true - y_pred)) / np.sum(w))

def evaluate(y_true, y_pred, is_holiday):
    """Return dict of all evaluation metrics."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask   = y_true != 0
    mape   = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    return {
        "r2"   : float(r2_score(y_true, y_pred)),
        "mae"  : float(mean_absolute_error(y_true, y_pred)),
        "rmse" : float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "wmae" : wmae(y_true, y_pred, is_holiday),
        "mape" : mape,
    }

# ── Synthetic data (used when real CSVs are absent) ───────────────────────
def _make_synthetic():
    """Generate realistic Walmart-like synthetic data for dev/CI."""
    rng = np.random.default_rng(SEED)
    stores = list(range(1, 46))
    depts  = list(range(1, 82))
    dates  = pd.date_range("2010-02-05", "2012-10-26", freq="W-FRI")
    rows   = []

    store_base = {s: rng.uniform(8000, 80000) for s in stores}
    dept_base  = {d: rng.uniform(500,  12000) for d in depts}
    store_type = {s: rng.choice(["A","B","C"], p=[0.5,0.3,0.2]) for s in stores}
    store_size = {s: int(rng.uniform(40000, 220000)) for s in stores}

    for s in stores:
        for d in depts:
            base = store_base[s] * dept_base[d] / 5000
            for dt in dates:
                wk  = dt.isocalendar()[1]
                mon = dt.month
                # Seasonal
                season = 1 + 0.4 * np.sin(2*np.pi*(wk-1)/52) + 0.15 * np.sin(4*np.pi*(wk-1)/52)
                # Holiday spike
                hol_flag = wk in (6,7,36,47,52)
                hol_mult = rng.uniform(1.3, 1.8) if hol_flag else 1.0
                # Noise
                noise = rng.normal(1.0, 0.06)
                val   = max(0, base * season * hol_mult * noise)
                rows.append({
                    "Store": s, "Dept": d, "Date": dt,
                    "Weekly_Sales": round(val, 2),
                    "IsHoliday": hol_flag,
                    "Temperature": rng.uniform(20, 100),
                    "Fuel_Price" : rng.uniform(2.5, 4.5),
                    "MarkDown1"  : rng.uniform(0, 8000) if rng.random() > 0.6 else 0,
                    "MarkDown2"  : rng.uniform(0, 3000) if rng.random() > 0.7 else 0,
                    "MarkDown3"  : rng.uniform(0, 2000) if rng.random() > 0.7 else 0,
                    "MarkDown4"  : rng.uniform(0, 5000) if rng.random() > 0.65 else 0,
                    "MarkDown5"  : rng.uniform(0, 4000) if rng.random() > 0.65 else 0,
                    "CPI"        : rng.uniform(126, 230),
                    "Unemployment": rng.uniform(3, 14),
                    "Type"       : store_type[s],
                    "Size"       : store_size[s],
                })
    return pd.DataFrame(rows)

# ── Load data ─────────────────────────────────────────────────────────────
def load_data():
    """Load real CSVs if present, else generate synthetic data."""
    if (DATA_DIR / "train.csv").exists():
        train_df    = pd.read_csv(DATA_DIR/"train.csv",    parse_dates=["Date"])
        features_df = pd.read_csv(DATA_DIR/"features.csv", parse_dates=["Date"])
        stores_df   = pd.read_csv(DATA_DIR/"stores.csv")
        df = (train_df
              .merge(features_df, on=["Store","Date","IsHoliday"], how="left")
              .merge(stores_df,   on="Store",                        how="left"))
        print(f"[prepare] Loaded real Kaggle data: {len(df):,} rows")
    else:
        print("[prepare] Real data not found – generating synthetic Walmart-like data …")
        df = _make_synthetic()
        print(f"[prepare] Synthetic data: {len(df):,} rows")
    return df

# ── Feature engineering (fixed pipeline) ──────────────────────────────────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["Store","Dept","Date"]).reset_index(drop=True)

    # MarkDown cleaning
    for md in ["MarkDown1","MarkDown2","MarkDown3","MarkDown4","MarkDown5"]:
        if md in df.columns:
            df[md] = df[md].fillna(0).clip(lower=0)

    for col in ["CPI","Unemployment","Temperature","Fuel_Price"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Calendar
    df["Year"]      = df["Date"].dt.year
    df["Month"]     = df["Date"].dt.month
    df["Week"]      = df["Date"].dt.isocalendar().week.astype(int)
    df["Quarter"]   = df["Date"].dt.quarter
    df["DayOfYear"] = df["Date"].dt.dayofyear

    # Fourier seasonality features
    for k in [1, 2, 3]:
        df[f"sin_week_{k}"] = np.sin(2 * np.pi * k * df["Week"] / 52)
        df[f"cos_week_{k}"] = np.cos(2 * np.pi * k * df["Week"] / 52)

    # Holiday flags
    df["IsHoliday"]      = df["IsHoliday"].astype(int)
    df["IsSuperBowl"]    = df["Week"].isin([6,7]).astype(int)
    df["IsLaborDay"]     = (df["Week"] == 36).astype(int)
    df["IsThanksgiving"] = (df["Week"] == 47).astype(int)
    df["IsChristmas"]    = (df["Week"] == 52).astype(int)
    df["WeeksToXmas"]    = ((52 - df["Week"]) % 52).clip(upper=12)
    df["WeeksToThanks"]  = ((47 - df["Week"]) % 52).clip(upper=8)

    # Markdown aggregates
    md_cols = [c for c in ["MarkDown1","MarkDown2","MarkDown3","MarkDown4","MarkDown5"] if c in df.columns]
    if md_cols:
        df["TotalMarkDown"]  = df[md_cols].sum(axis=1)
        df["MarkDown_Count"] = (df[md_cols] > 0).sum(axis=1)
        df["MaxMarkDown"]    = df[md_cols].max(axis=1)
    else:
        df["TotalMarkDown"] = df["MarkDown_Count"] = df["MaxMarkDown"] = 0

    # Store encoding
    type_map = {"A":2,"B":1,"C":0}
    df["TypeEncoded"] = df["Type"].map(type_map).fillna(1).astype(int)
    df["SizeBucket"]  = pd.qcut(df["Size"], q=4, labels=[0,1,2,3]).astype(int)
    df["EconPressure"] = df.get("CPI", 170) * df.get("Unemployment", 8)
    df["Season"] = df["Month"].map({12:3,1:3,2:3,3:0,4:0,5:0,6:1,7:1,8:1,9:2,10:2,11:2})

    # Lag features (shift by Store, Dept group)
    grp = df.groupby(["Store","Dept"])["Weekly_Sales"]
    for lag in [1, 2, 3, 4, 8, 13, 26, 52]:
        df[f"Lag_{lag}"] = grp.shift(lag)
    for win in [4, 8, 13, 26]:
        df[f"Roll_{win}_mean"] = grp.transform(lambda x: x.shift(1).rolling(win, min_periods=1).mean())
        df[f"Roll_{win}_std"]  = grp.transform(lambda x: x.shift(1).rolling(win, min_periods=1).std().fillna(0))
    df["WoW_Change"] = grp.transform(lambda x: x.pct_change().clip(-2, 2)).fillna(0)
    df["YoY_ratio"]  = (df["Lag_52"] / df["Roll_26_mean"].replace(0, np.nan)).fillna(1).clip(0, 5)

    # Store/Dept means (computed within training data – passed as argument)
    df["StoreDept"] = df["Store"].astype(str) + "_" + df["Dept"].astype(str)

    # Fill remaining NaN (from lags at series start)
    lag_cols = [c for c in df.columns if c.startswith(("Lag_","Roll_","WoW","YoY"))]
    for c in lag_cols:
        df[c] = df[c].fillna(df[c].median())

    return df

FEATURE_COLS = [
    "Store","Dept",
    "Year","Month","Week","Quarter","DayOfYear",
    "sin_week_1","cos_week_1","sin_week_2","cos_week_2","sin_week_3","cos_week_3",
    "Temperature","Fuel_Price","CPI","Unemployment",
    "MarkDown1","MarkDown2","MarkDown3","MarkDown4","MarkDown5",
    "TotalMarkDown","MarkDown_Count","MaxMarkDown",
    "TypeEncoded","Size","SizeBucket","EconPressure","Season",
    "IsHoliday","IsSuperBowl","IsLaborDay","IsThanksgiving","IsChristmas",
    "WeeksToXmas","WeeksToThanks",
    "Lag_1","Lag_2","Lag_3","Lag_4","Lag_8","Lag_13","Lag_26","Lag_52",
    "Roll_4_mean","Roll_4_std","Roll_8_mean","Roll_8_std",
    "Roll_13_mean","Roll_13_std","Roll_26_mean","Roll_26_std",
    "WoW_Change","YoY_ratio",
]
TARGET = "Weekly_Sales"

def get_train_val(df):
    """Temporal split – no data leakage."""
    train = df[df["Date"] <  VAL_CUTOFF].copy()
    val   = df[df["Date"] >= VAL_CUTOFF].copy()

    # Target-encode StoreDept using ONLY training data means
    te = train.groupby("StoreDept")[TARGET].mean().rename("StoreDept_TE")
    train = train.join(te, on="StoreDept")
    val   = val.join(te,   on="StoreDept")
    val["StoreDept_TE"] = val["StoreDept_TE"].fillna(train["StoreDept_TE"].mean())

    # Also encode Store/Dept means
    store_te = train.groupby("Store")[TARGET].mean().rename("Store_TE")
    dept_te  = train.groupby("Dept")[TARGET].mean().rename("Dept_TE")
    train = train.join(store_te, on="Store").join(dept_te, on="Dept")
    val   = val.join(store_te,   on="Store").join(dept_te, on="Dept")
    val["Store_TE"] = val["Store_TE"].fillna(train["Store_TE"].mean())
    val["Dept_TE"]  = val["Dept_TE"].fillna(train["Dept_TE"].mean())

    return train, val

TE_COLS = ["StoreDept_TE", "Store_TE", "Dept_TE"]
ALL_FEATURES = FEATURE_COLS + TE_COLS
