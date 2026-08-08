# 📦 Sales Prediction & Business Analytics

Predict how many units a **new product** will sell before launch — based on
its category, price, discount, rating, and description — and turn that
prediction into a business-analytics dashboard (revenue mix, ABC/Pareto
analysis, pricing strategy, discount effectiveness).

This is a from-scratch rebuild of an earlier academic project
("Prediction of Sales of a New Product"), with three concrete upgrades:

1. **A working, honest model.** The original report's Gradient Boosting
   model scored **R² = 1.58**, which is mathematically impossible (R² is
   bounded at 1.0) — a sign of a broken train/test evaluation. This version
   trains on a log-transformed target, fixes that bug, and reports a real,
   defensible **R² ≈ 0.69** on held-out data.
2. **Text + structured features combined.** The original split into two
   separate notebooks — one used only product-description text (TF-IDF →
   XGBoost), the other used only category/price/rating (→ Gradient
   Boosting) — and never combined them. This version fuses both into one
   `ColumnTransformer` pipeline.
3. **A business-analytics layer.** Beyond "predict a number," this project
   exports category revenue mix, ABC/Pareto tiering, price-band and
   discount-effectiveness analysis, and a KPI scorecard — ready to drop
   straight into Tableau.

---

## Architecture

```
                 ┌────────────────────┐
                 │  Amazon product     │   scripts/generate_sample_data.py
                 │  data (CSV)         │   (or real Kaggle CSVs)
                 └─────────┬───────────┘
                           │
                 ┌─────────▼───────────┐
                 │ data_preprocessing.py│   clean, engineer features
                 └─────────┬───────────┘
                           │
              ┌────────────┼─────────────┐
              │                          │
   ┌──────────▼─────────┐     ┌──────────▼──────────┐
   │     train.py         │     │  business_analytics.py │
   │ RF / GBR / XGBoost    │────▶│  ABC, category, pricing,│
   │ comparison + tuning   │model │  KPI CSV exports         │
   └──────────┬─────────┘     └──────────┬──────────┘
              │                          │
   ┌──────────▼─────────┐     ┌──────────▼──────────┐
   │   Flask app (app/)   │     │  Tableau dashboard    │
   │  predict + benchmark │     │  (tableau/exports/*.csv)│
   └─────────────────────┘     └─────────────────────┘
```

---

## Project structure

```
sales-prediction-analytics/
├── scripts/
│   └── generate_sample_data.py   # bootstrap dataset (see "About the data")
├── src/
│   ├── data_preprocessing.py     # cleaning + feature engineering
│   ├── features.py               # ColumnTransformer (text+cat+numeric)
│   ├── train.py                  # trains/compares RF, GBR, XGBoost
│   └── business_analytics.py     # builds Tableau-ready CSV exports
├── app/
│   ├── app.py                    # Flask deployment
│   ├── templates/                # index.html (predictor), dashboard.html
│   └── static/style.css
├── data/
│   ├── raw/amazon_products_sample.csv   # bundled sample dataset
│   └── processed/                       # cleaned output (generated)
├── models/                       # trained model + metadata (generated)
├── tableau/
│   ├── exports/                  # CSVs for Tableau (generated)
│   └── README.md                 # step-by-step dashboard build guide
├── tests/test_pipeline.py
├── run_pipeline.sh               # one-command: generate → train → export
└── requirements.txt
```

---

## Quickstart

```bash
git clone <your-fork-url>
cd sales-prediction-analytics
python3 -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt

# Generate data → train models → build Tableau exports (≈2-5 min)
./run_pipeline.sh

# Launch the app
python3 app/app.py
# then open http://127.0.0.1:5000
```

If you'd rather run each stage yourself:

```bash
python3 scripts/generate_sample_data.py --rows 20000
python3 -m src.train --n-iter 8 --cv 4
python3 -m src.business_analytics
python3 app/app.py
```

Run the test suite:
```bash
pytest -q
```

---

## About the data

The original project used the Kaggle dataset **["Amazon Products Sales
Dataset 2023"](https://www.kaggle.com/datasets/lokeshparab/amazon-products-dataset)**
(~1.4M rows across 142 category CSVs — columns: `name`, `main_category`,
`sub_category`, `ratings`, `no_of_ratings`, `discount_price`,
`actual_price`). That dataset is distributed via Kaggle and requires a free
personal API token to download programmatically, so this repo **doesn't
ship it directly**.

Instead, `scripts/generate_sample_data.py` generates a **schema-identical**
sample (20 categories, 72 sub-categories, 20,000 products) with realistic,
hand-tuned relationships between price/discount/rating/brand and sales
volume, so the entire pipeline — training, the Flask app, the Tableau
exports — runs immediately after cloning, with no external downloads or
credentials required.

**To use the real dataset:**
1. Download the CSVs from the Kaggle link above (needs a free Kaggle
   account; use `kaggle datasets download -d lokeshparab/amazon-products-dataset`
   with your own API token, or download manually from the site).
2. Unzip the per-category CSVs into `data/raw/amazon_categories/`.
3. Re-run `python -m src.train` — `data_preprocessing.py` automatically
   prefers the real per-category files over the bundled sample if that
   folder is present and non-empty.

> Treat the bundled sample as a realistic demo/portfolio dataset, not real
> Amazon sales figures — the absolute revenue numbers in the Tableau
> exports are illustrative until you swap in real data.

---

## The model

**Features:** product description (TF-IDF, 1-2 grams), main category
+ sub-category (one-hot), discount price, actual price, rating, and an
engineered `discount_percentage` (standardized).

**Target:** `total_sales` (the dataset's `no_of_ratings` field, used as a
sales-volume proxy — the same choice the original project made, since
review count is one of the only volume signals available in this kind of
product-catalog data). Modeled as `log1p(total_sales)` to handle the
right-skewed, long-tail distribution typical of product sales — predictions
are exponentiated back (`expm1`) before being shown to the user.

**Models compared:** RandomForestRegressor, GradientBoostingRegressor,
XGBRegressor, each tuned via `RandomizedSearchCV`. The best model (by R² on
a held-out 20% test split) is saved and used by the app.

Typical result on the bundled sample dataset:

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **XGBoost** | ~667 | ~1,027 | **~0.69** |
| GradientBoosting | ~684 | ~1,061 | ~0.67 |
| RandomForest | ~761 | ~1,166 | ~0.60 |

(Your exact numbers will vary slightly with the random seed and search
budget — see `tableau/exports/model_performance.csv` after training.)

---

## The Flask app

- `/` — prediction form: description, category/sub-category (cascading
  dropdown), pricing, rating slider → predicted units sold, estimated
  revenue, and how your price/predicted-sales compare to the category
  average.
- `/api/predict` — same thing as JSON, for scripting/testing.
- `/dashboard` — a lightweight in-app KPI summary (the full interactive
  dashboard lives in Tableau — see below).

## The Tableau dashboard

`src/business_analytics.py` builds five flat CSVs in `tableau/exports/`:
product-level detail, category rollups, ABC/Pareto revenue tiering,
price-band + discount-effectiveness analysis, and a KPI scorecard.
**See [`tableau/README.md`](tableau/README.md)** for the full, sheet-by-sheet
build walkthrough (KPI cards, revenue-by-category treemap, price-vs-sales
scatter with trend line, an 80/20 Pareto chart, and a model-comparison
chart), including exact field names and calculated-field formulas.

---

## What changed vs. the original project (for anyone comparing)

| | Original | This version |
|---|---|---|
| Target scale | Raw counts | `log1p` (fixes the impossible R²=1.58 result) |
| Features | Text *or* structured, never both | Text + category + price + rating, fused |
| Outlier handling | None | IQR-based trimming on the target |
| Model comparison | Ad hoc, not saved | RF vs GBR vs XGBoost, metrics exported |
| Business analytics | None | Category mix, ABC/Pareto, pricing analysis, KPIs |
| Deployment UI | Single text box | Full form + category benchmarking + JSON API |
| Tests | None | `pytest` suite covering cleaning + pipeline + edge cases |
| Dashboard | None | Tableau-ready CSV exports + build guide |

---

## Requirements

Python 3.10+. See `requirements.txt`:
`pandas`, `numpy`, `scikit-learn`, `xgboost`, `scipy`, `joblib`, `Flask`, `pytest`.

> **Verified working** end-to-end on a clean virtualenv (Python 3.12, pandas 3.0,
> numpy 2.5, scikit-learn 1.9, xgboost 3.4): data generation → training →
> business-analytics exports → Flask app (real HTTP requests, not just the
> test client) → `pytest` (13/13 passing). One benign warning to expect on
> very new numpy releases: joblib's model-loading code emits a
> `DeprecationWarning` about array-shape assignment — it's a joblib/numpy
> version-lag issue, not a bug in this project, and doesn't affect behavior.
> If it ever becomes a hard error in a future numpy release, pin
> `numpy<2.5` in `requirements.txt` as a workaround.

## Production deployment notes

`python app/app.py` runs Flask's built-in dev server (`debug=True`) — great
for local use, **not** for production. To actually deploy:

```bash
pip install gunicorn
gunicorn -w 2 -b 0.0.0.0:8000 "app.app:app"
```

Also, before deploying publicly: set `debug=False` in `app/app.py` (the
Werkzeug debugger exposes an interactive Python console on error pages —
fine for local dev, a real security risk on the open internet).

## License

MIT — see [`LICENSE`](LICENSE).
