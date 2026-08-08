"""
business_analytics.py
----------------------
Turns the cleaned product dataset + trained model into a set of tidy,
denormalised CSV extracts designed to be dropped straight into Tableau
(or Power BI / Looker Studio) with zero further transformation.

Run after src/train.py (needs the saved model for predicted_sales /
model-vs-actual comparisons). Produces, in tableau/exports/:

  1. product_master.csv        - one row per product, every engineered
                                  field + model prediction. Use this as
                                  your primary Tableau data source.
  2. category_summary.csv      - revenue / rating / discount rollups by
                                  main_category x sub_category.
  3. abc_analysis.csv          - Pareto (ABC) classification of products
                                  by revenue contribution.
  4. pricing_analysis.csv      - price-band performance + discount
                                  effectiveness.
  5. kpi_summary.csv           - single-row overall KPI scorecard.
  6. model_performance.csv     - written by train.py (model comparison).

See tableau/README.md for the exact dashboard-build walkthrough.
"""
from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd

from src.data_preprocessing import get_clean_dataset
from src.features import ALL_FEATURE_COLS

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
EXPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "tableau", "exports")


def _load_model():
    model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "No trained model found. Run `python -m src.train` first."
        )
    bundle = joblib.load(model_path)
    return bundle["pipeline"], bundle["model_name"]


def build_product_master(df: pd.DataFrame, pipeline, model_name: str) -> pd.DataFrame:
    X = df[ALL_FEATURE_COLS]
    pred_log = pipeline.predict(X)
    df = df.copy()
    df["predicted_sales"] = np.clip(np.expm1(pred_log), 0, None).round(0)
    df["prediction_error"] = df["predicted_sales"] - df["total_sales"]
    df["prediction_error_pct"] = np.where(
        df["total_sales"] > 0,
        (df["prediction_error"] / df["total_sales"] * 100).round(1),
        np.nan,
    )
    df["model_used"] = model_name
    df["product_id"] = "P" + df.index.astype(str).str.zfill(6)

    cols = ["product_id", "product_description", "main_category", "sub_category",
            "price_band", "ratings", "is_high_rated", "discount_price",
            "actual_price", "discount_percentage", "total_sales",
            "predicted_sales", "prediction_error", "prediction_error_pct",
            "estimated_revenue", "description_word_count", "model_used"]
    return df[cols]


def build_category_summary(product_master: pd.DataFrame) -> pd.DataFrame:
    grp = product_master.groupby(["main_category", "sub_category"], as_index=False).agg(
        product_count=("product_id", "count"),
        avg_rating=("ratings", "mean"),
        avg_discount_pct=("discount_percentage", "mean"),
        avg_actual_price=("actual_price", "mean"),
        avg_discount_price=("discount_price", "mean"),
        total_sales=("total_sales", "sum"),
        avg_predicted_sales=("predicted_sales", "mean"),
        total_estimated_revenue=("estimated_revenue", "sum"),
    )
    total_rev = grp["total_estimated_revenue"].sum()
    grp["revenue_share_pct"] = (grp["total_estimated_revenue"] / total_rev * 100).round(2)
    grp["avg_rating"] = grp["avg_rating"].round(2)
    grp["avg_discount_pct"] = grp["avg_discount_pct"].round(1)
    grp["avg_actual_price"] = grp["avg_actual_price"].round(2)
    grp["avg_discount_price"] = grp["avg_discount_price"].round(2)
    grp["avg_predicted_sales"] = grp["avg_predicted_sales"].round(1)
    grp = grp.sort_values("total_estimated_revenue", ascending=False).reset_index(drop=True)
    grp["category_rank"] = grp.index + 1
    return grp


def build_abc_analysis(product_master: pd.DataFrame) -> pd.DataFrame:
    """Classic Pareto / ABC inventory analysis: rank products by revenue
    contribution and bucket into A (top 80% of revenue), B (next 15%),
    C (final 5%) tiers -- a standard retail business-analytics technique
    the original project never included."""
    d = product_master[["product_id", "product_description", "main_category",
                         "sub_category", "estimated_revenue", "total_sales"]].copy()
    d = d.sort_values("estimated_revenue", ascending=False).reset_index(drop=True)
    total_rev = d["estimated_revenue"].sum()
    d["cumulative_revenue"] = d["estimated_revenue"].cumsum()
    d["cumulative_revenue_pct"] = (d["cumulative_revenue"] / total_rev * 100).round(2)
    d["revenue_rank"] = d.index + 1

    def tier(pct):
        if pct <= 80:
            return "A - Top Revenue Drivers"
        elif pct <= 95:
            return "B - Steady Contributors"
        return "C - Long Tail"

    d["abc_tier"] = d["cumulative_revenue_pct"].apply(tier)
    return d


def build_pricing_analysis(product_master: pd.DataFrame) -> pd.DataFrame:
    bands = product_master.groupby("price_band", observed=True).agg(
        product_count=("product_id", "count"),
        avg_sales=("total_sales", "mean"),
        median_sales=("total_sales", "median"),
        avg_discount_pct=("discount_percentage", "mean"),
        avg_rating=("ratings", "mean"),
        total_estimated_revenue=("estimated_revenue", "sum"),
    ).reset_index()
    bands["avg_sales"] = bands["avg_sales"].round(1)
    bands["avg_discount_pct"] = bands["avg_discount_pct"].round(1)
    bands["avg_rating"] = bands["avg_rating"].round(2)

    # discount-band effectiveness: does deeper discounting actually move units?
    disc_bins = [-0.1, 0, 10, 20, 30, 50, 100]
    disc_labels = ["No discount", "1-10%", "11-20%", "21-30%", "31-50%", "50%+"]
    pm = product_master.copy()
    pm["discount_band"] = pd.cut(pm["discount_percentage"], bins=disc_bins, labels=disc_labels)
    disc_effectiveness = pm.groupby("discount_band", observed=True).agg(
        product_count=("product_id", "count"),
        avg_sales=("total_sales", "mean"),
        avg_rating=("ratings", "mean"),
    ).reset_index()
    disc_effectiveness["avg_sales"] = disc_effectiveness["avg_sales"].round(1)
    disc_effectiveness.insert(0, "analysis_type", "discount_effectiveness")
    bands.insert(0, "analysis_type", "price_band")

    bands = bands.rename(columns={"price_band": "segment"})
    disc_effectiveness = disc_effectiveness.rename(columns={"discount_band": "segment"})
    combined = pd.concat([bands, disc_effectiveness], ignore_index=True, sort=False)
    return combined


def build_kpi_summary(product_master: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    corr_price_sales = product_master[["actual_price", "total_sales"]].corr().iloc[0, 1]
    corr_discount_sales = product_master[["discount_percentage", "total_sales"]].corr().iloc[0, 1]
    corr_rating_sales = product_master[["ratings", "total_sales"]].corr().iloc[0, 1]

    top_category = (
        product_master.groupby("main_category")["estimated_revenue"].sum().idxmax()
    )
    top_model = metadata["metrics"][0]

    kpis = {
        "total_products": int(len(product_master)),
        "total_estimated_revenue": float(product_master["estimated_revenue"].sum()),
        "total_units_sold_est": int(product_master["total_sales"].sum()),
        "avg_rating": float(product_master["ratings"].mean()),
        "avg_discount_pct": float(product_master["discount_percentage"].mean()),
        "avg_selling_price": float(product_master["discount_price"].mean()),
        "top_revenue_category": top_category,
        "num_main_categories": int(product_master["main_category"].nunique()),
        "num_sub_categories": int(product_master["sub_category"].nunique()),
        "corr_price_vs_sales": round(float(corr_price_sales), 3),
        "corr_discount_vs_sales": round(float(corr_discount_sales), 3),
        "corr_rating_vs_sales": round(float(corr_rating_sales), 3),
        "best_model": top_model["model"],
        "best_model_r2": round(float(top_model["R2"]), 3),
        "best_model_mae": round(float(top_model["MAE"]), 1),
    }
    return pd.DataFrame([kpis])


def run_all_exports():
    os.makedirs(EXPORTS_DIR, exist_ok=True)
    df = get_clean_dataset(save=False)
    pipeline, model_name = _load_model()

    with open(os.path.join(MODELS_DIR, "model_metadata.json")) as f:
        metadata = json.load(f)

    product_master = build_product_master(df, pipeline, model_name)
    category_summary = build_category_summary(product_master)
    abc = build_abc_analysis(product_master)
    pricing = build_pricing_analysis(product_master)
    kpis = build_kpi_summary(product_master, metadata)

    exports = {
        "product_master.csv": product_master,
        "category_summary.csv": category_summary,
        "abc_analysis.csv": abc,
        "pricing_analysis.csv": pricing,
        "kpi_summary.csv": kpis,
    }
    for filename, frame in exports.items():
        path = os.path.join(EXPORTS_DIR, filename)
        frame.to_csv(path, index=False)
        print(f"Wrote {len(frame):,} rows -> {path}")

    return exports


if __name__ == "__main__":
    run_all_exports()
