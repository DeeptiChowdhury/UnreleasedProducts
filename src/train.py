"""
train.py
--------
Trains and compares three regressors (Random Forest, Gradient Boosting,
XGBoost) on the log-transformed sales target, tunes the best candidate
with RandomizedSearchCV, and persists:
  - models/best_model.joblib           (full sklearn Pipeline)
  - models/model_metadata.json         (feature lists, categories for the UI)
  - tableau/exports/model_performance.csv  (metrics for every model, for Tableau)

Why log1p(total_sales)?
The target is right-skewed count data (a handful of viral products sell
10-100x the median). Training directly on raw counts is almost certainly
why the original project reported a nonsensical R2 of 1.58 (Chapter 9,
Figure 9.1) -- squared errors on a long-tailed target blow up and can
produce a broken train/test evaluation if any preprocessing leaks or
mismatches between fit/predict. Modeling log1p(sales) and exponentiating
predictions back is standard practice for this kind of target and gives
honest, bounded R2 in [0, 1] on held-out data.
"""
from __future__ import annotations

import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from xgboost import XGBRegressor

from src.data_preprocessing import get_clean_dataset
from src.features import ALL_FEATURE_COLS, TARGET_COL, build_pipeline

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
TABLEAU_EXPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "tableau", "exports")


def _metrics(y_true_log, y_pred_log):
    """Metrics computed back on the ORIGINAL sales scale (not log scale) so
    MAE/RMSE are directly interpretable as "units of sales"."""
    y_true = np.expm1(y_true_log)
    y_pred = np.clip(np.expm1(y_pred_log), 0, None)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


def train_and_compare(sample_frac: float = 1.0, random_state: int = 42,
                       n_iter_search: int = 8, quick: bool = False,
                       cv_folds_override: int | None = None):
    df = get_clean_dataset()
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=random_state)

    X = df[ALL_FEATURE_COLS]
    y_log = np.log1p(df[TARGET_COL])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.2, random_state=random_state
    )

    candidates = {
        "RandomForest": (
            RandomForestRegressor(random_state=random_state, n_jobs=1),
            {
                "model__n_estimators": randint(100, 250),
                "model__max_depth": randint(6, 16),
                "model__min_samples_leaf": randint(1, 6),
            },
        ),
        "GradientBoosting": (
            GradientBoostingRegressor(random_state=random_state),
            {
                "model__n_estimators": randint(100, 350),
                "model__learning_rate": uniform(0.02, 0.2),
                "model__max_depth": randint(2, 5),
            },
        ),
        "XGBoost": (
            XGBRegressor(random_state=random_state, n_jobs=1,
                         objective="reg:squarederror", tree_method="hist"),
            {
                "model__n_estimators": randint(100, 300),
                "model__learning_rate": uniform(0.02, 0.2),
                "model__max_depth": randint(3, 8),
                "model__subsample": uniform(0.7, 0.3),
            },
        ),
    }

    n_iter = 3 if quick else n_iter_search
    cv_folds = cv_folds_override if cv_folds_override else (3 if quick else 4)

    results = []
    fitted_pipelines = {}

    for name, (estimator, param_dist) in candidates.items():
        print(f"\n=== Tuning {name} ===")
        t0 = time.time()
        pipe = build_pipeline(estimator)
        search = RandomizedSearchCV(
            pipe, param_distributions=param_dist, n_iter=n_iter, cv=cv_folds,
            scoring="neg_mean_squared_error", random_state=random_state,
            n_jobs=-1, verbose=0,
        )
        search.fit(X_train, y_train)
        best = search.best_estimator_
        train_time = time.time() - t0

        y_pred = best.predict(X_test)
        m = _metrics(y_test, y_pred)
        m.update({"model": name, "best_params": search.best_params_,
                   "train_time_sec": round(train_time, 2)})
        results.append(m)
        fitted_pipelines[name] = best
        print(f"{name}: MAE={m['MAE']:.1f}  RMSE={m['RMSE']:.1f}  R2={m['R2']:.3f}  "
              f"({train_time:.1f}s, best params={search.best_params_})")

    results_df = pd.DataFrame(results)[
        ["model", "MAE", "MSE", "RMSE", "R2", "train_time_sec", "best_params"]
    ].sort_values("R2", ascending=False).reset_index(drop=True)

    best_model_name = results_df.iloc[0]["model"]
    best_pipeline = fitted_pipelines[best_model_name]
    print(f"\n>>> Best model: {best_model_name} (R2={results_df.iloc[0]['R2']:.3f})")

    # ---- Persist artifacts --------------------------------------------
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(TABLEAU_EXPORTS_DIR, exist_ok=True)

    model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    joblib.dump({"pipeline": best_pipeline, "model_name": best_model_name}, model_path)
    print(f"Saved best model -> {model_path}")

    metadata = {
        "model_name": best_model_name,
        "feature_columns": ALL_FEATURE_COLS,
        "target_col": TARGET_COL,
        "target_transform": "log1p",
        "main_categories": sorted(df["main_category"].unique().tolist()),
        "sub_categories_by_main": {
            mc: sorted(df.loc[df["main_category"] == mc, "sub_category"].unique().tolist())
            for mc in df["main_category"].unique()
        },
        "price_stats": {
            "actual_price": {"min": float(df["actual_price"].min()),
                              "max": float(df["actual_price"].max()),
                              "median": float(df["actual_price"].median())},
        },
        "training_rows": int(len(df)),
        "metrics": results_df.drop(columns="best_params").to_dict(orient="records"),
    }
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved model metadata -> {meta_path}")

    perf_path = os.path.join(TABLEAU_EXPORTS_DIR, "model_performance.csv")
    results_df.drop(columns="best_params").to_csv(perf_path, index=False)
    print(f"Saved model performance (Tableau export) -> {perf_path}")

    return best_pipeline, best_model_name, results_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                         help="fast smoke-test mode: fewer search iterations/folds")
    parser.add_argument("--sample-frac", type=float, default=1.0)
    parser.add_argument("--n-iter", type=int, default=8, help="RandomizedSearchCV iterations per model")
    parser.add_argument("--cv", type=int, default=None, help="cross-validation folds")
    args = parser.parse_args()
    train_and_compare(sample_frac=args.sample_frac, quick=args.quick,
                       n_iter_search=args.n_iter, cv_folds_override=args.cv)
