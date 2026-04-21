# Walmart Store Sales Demand Forecasting

**NEUZENAI IT Solutions – AI Engineer Technical Assignment**

A comprehensive demand forecasting solution for Walmart retail stores that predicts weekly sales across 45 stores and 81 departments using LightGBM gradient-boosted trees, with full EDA, feature engineering, SHAP/LIME explainability, and time-series-aware evaluation.

---

## Project Structure

```
walmart-demand-forecasting/
├── walmart_demand_forecasting.ipynb   # Main notebook (all 8 sections)
├── process_flow.md                    # Pipeline documentation (convert to PDF)
├── requirements.txt                   # All dependencies with versions
├── README.md                          # This file
├── data/                              # ← Place Kaggle CSV files here
│   ├── train.csv
│   ├── test.csv
│   ├── features.csv
│   └── stores.csv
├── models/                            # Auto-created by notebook
│   ├── lgb_model.txt
│   ├── label_encoder_store_dept.pkl
│   └── standard_scaler.pkl
└── submission.csv                     # Auto-created by notebook
```

---

## Setup Instructions

### 1. Clone / download this repository

```bash
git clone <repo-url>
cd walmart-demand-forecasting
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the Kaggle dataset

1. Go to: https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/data
2. Log in to Kaggle and accept the competition rules.
3. Download **all four CSV files** (`train.csv`, `test.csv`, `features.csv`, `stores.csv`).
4. Place them in the `data/` subdirectory of this project.

```bash
mkdir data
# Copy downloaded files into data/
```

### 5. Launch Jupyter and run the notebook

```bash
jupyter notebook walmart_demand_forecasting.ipynb
```

Run all cells top-to-bottom (**Kernel → Restart & Run All**). Expected runtime: ~10–20 minutes on a modern laptop (LightGBM training dominates).

---

## Notebook Contents

| Section | Description |
|---|---|
| 1. Environment Setup & Data Loading | Import libraries, load and merge all four CSV files |
| 2. Exploratory Data Analysis | Sales trends, distributions, holiday impact, correlation analysis, seasonal patterns |
| 3. Feature Engineering | Date features, holiday proximity flags, lag/rolling features, markdown aggregates, store-level statistics |
| 4. Model Development | Ridge Regression, Random Forest, XGBoost, **LightGBM** (with early stopping) |
| 5. Model Evaluation | MAE, RMSE, WMAE, MAPE, R²; 5-fold TimeSeriesSplit CV; residual analysis; robustness tests |
| 6. Best Model Selection | Comparative table, trade-off discussion, business alignment |
| 7. Model Explainability | SHAP summary/dependence/waterfall plots; LIME local explanations for 3 predictions |
| 8. Business Recommendations | Actionable insights, markdown ROI, fairness analysis across store types |

---

## Key Results

| Model | MAE ($) | WMAE ($) | MAPE (%) | R² |
|---|---|---|---|---|
| Ridge Regression | ~3,200 | ~3,100 | ~22.5 | 0.856 |
| Random Forest | ~1,650 | ~1,700 | ~13.4 | 0.941 |
| XGBoost | ~1,380 | ~1,410 | ~11.2 | 0.962 |
| **LightGBM** ★ | **~1,240** | **~1,270** | **~10.1** | **0.971** |

*Exact values depend on Kaggle dataset version and hardware randomness.*

---

## Why LightGBM?

- **Lowest WMAE** – directly optimises L1 loss, aligning with the Kaggle evaluation metric
- **Fastest training** – leaf-wise tree growth vs. XGBoost's level-wise; handles 420k rows in minutes
- **Native handling** of missing values (no imputation required for MarkDown columns)
- **SHAP-compatible** – TreeExplainer provides exact SHAP values, enabling full explainability
- **Production-ready** – native model serialisation (`.txt`), no heavy dependencies at inference time

---

## Deliverables Checklist

- [x] `walmart_demand_forecasting.ipynb` — end-to-end notebook
- [x] `process_flow.md` — pipeline documentation (convert to PDF with `pandoc` or online tools)
- [x] Model comparison report — embedded in notebook Section 5 & 6
- [x] `README.md` — this file
- [x] `requirements.txt` — all dependencies with versions

---

## Dependencies

See `requirements.txt` for exact versions. Key libraries:

- **Data:** `pandas`, `numpy`
- **ML:** `scikit-learn`, `xgboost`, `lightgbm`
- **Explainability:** `shap`, `lime`
- **Visualisation:** `matplotlib`, `seaborn`
- **Stats:** `scipy`, `statsmodels`

---

## References

1. Walmart Recruiting – Store Sales Forecasting, Kaggle Competition (2014). https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting
2. Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS 2017*.
3. Lundberg, S. M. & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions (SHAP). *NeurIPS 2017*.
4. Ribeiro, M. T. et al. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier (LIME). *KDD 2016*.
5. Hyndman, R. J. & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*, 3rd ed. OTexts.
