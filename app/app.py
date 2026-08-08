"""
app.py
------
Flask deployment for the sales-prediction model. Replaces the bare-bones
single-textbox UI from the original project (Chapter 10 of the report)
with a full form (category, sub-category, pricing, rating, description)
plus business-context on the result: estimated revenue, where the price
sits versus the category, and a plain-English confidence note.

Run:
    python app/app.py
Then open http://127.0.0.1:5000
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

# allow `import src...` when running this file directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.features import ALL_FEATURE_COLS  # noqa: E402

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR = os.path.join(BASE_DIR, "models")
EXPORTS_DIR = os.path.join(BASE_DIR, "tableau", "exports")

app = Flask(__name__)

_model_bundle = None
_metadata = None
_category_benchmarks = None


def load_artifacts():
    """Lazy-load the trained model + metadata once, on first request."""
    global _model_bundle, _metadata, _category_benchmarks
    if _model_bundle is None:
        model_path = os.path.join(MODELS_DIR, "best_model.joblib")
        meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
        if not os.path.exists(model_path):
            raise RuntimeError(
                "No trained model found. Run `python -m src.train` "
                "from the project root before starting the app."
            )
        _model_bundle = joblib.load(model_path)
        with open(meta_path) as f:
            _metadata = json.load(f)

        cat_summary_path = os.path.join(EXPORTS_DIR, "category_summary.csv")
        if os.path.exists(cat_summary_path):
            _category_benchmarks = pd.read_csv(cat_summary_path)
        else:
            _category_benchmarks = None
    return _model_bundle, _metadata, _category_benchmarks


@app.route("/")
def index():
    _, metadata, _ = load_artifacts()
    return render_template(
        "index.html",
        main_categories=metadata["main_categories"],
        sub_categories_by_main=metadata["sub_categories_by_main"],
        model_name=metadata["model_name"],
        best_metrics=metadata["metrics"][0],
    )


@app.route("/api/subcategories/<main_category>")
def api_subcategories(main_category):
    _, metadata, _ = load_artifacts()
    subs = metadata["sub_categories_by_main"].get(main_category, [])
    return jsonify(subs)


@app.route("/predict", methods=["POST"])
def predict():
    bundle, metadata, benchmarks = load_artifacts()
    pipeline = bundle["pipeline"]

    form = request.form
    try:
        actual_price = float(form.get("actual_price", 0) or 0)
        discount_price = float(form.get("discount_price", 0) or actual_price)
        rating = float(form.get("rating", 4.0) or 4.0)
        main_category = form.get("main_category", "").strip().lower()
        sub_category = form.get("sub_category", "").strip().lower()
        description = form.get("description", "").strip() or f"{main_category} {sub_category} product"
    except (TypeError, ValueError):
        safe_form_values = {
            "description": form.get("description", ""),
            "main_category": form.get("main_category", ""),
            "sub_category": form.get("sub_category", ""),
            "actual_price": form.get("actual_price", ""),
            "discount_price": form.get("discount_price", ""),
            "rating": form.get("rating", "4.0"),
        }
        return render_template("index.html", error="Please enter valid numeric values.",
                                main_categories=metadata["main_categories"],
                                sub_categories_by_main=metadata["sub_categories_by_main"],
                                model_name=metadata["model_name"],
                                best_metrics=metadata["metrics"][0],
                                form_values=safe_form_values)

    # Explicit validation instead of silently predicting for a nonsensical
    # $0 / no-category "phantom" product.
    validation_errors = []
    if not main_category:
        validation_errors.append("Please select a main category.")
    if not sub_category:
        validation_errors.append("Please select a sub-category.")
    if actual_price <= 0:
        validation_errors.append("Actual price must be greater than 0.")

    safe_form_values = {
        "description": description,
        "main_category": main_category,
        "sub_category": sub_category,
        "actual_price": form.get("actual_price", ""),
        "discount_price": form.get("discount_price", ""),
        "rating": rating,
    }

    if validation_errors:
        return render_template("index.html", error=" ".join(validation_errors),
                                main_categories=metadata["main_categories"],
                                sub_categories_by_main=metadata["sub_categories_by_main"],
                                model_name=metadata["model_name"],
                                best_metrics=metadata["metrics"][0],
                                form_values=safe_form_values)

    if discount_price > actual_price:
        discount_price = actual_price
    discount_percentage = round((actual_price - discount_price) / actual_price * 100, 2) if actual_price else 0.0

    row = pd.DataFrame([{
        "product_description": description,
        "main_category": main_category,
        "sub_category": sub_category,
        "discount_price": discount_price,
        "actual_price": actual_price,
        "ratings": rating,
        "discount_percentage": discount_percentage,
    }])[ALL_FEATURE_COLS]

    pred_log = pipeline.predict(row)[0]
    predicted_sales = max(float(np.expm1(pred_log)), 0)
    estimated_revenue = predicted_sales * discount_price

    # Business context vs. category benchmark
    benchmark_note = None
    if benchmarks is not None:
        cat_rows = benchmarks[
            (benchmarks["main_category"] == main_category) &
            (benchmarks["sub_category"] == sub_category)
        ]
        if not cat_rows.empty:
            b = cat_rows.iloc[0]
            price_delta_pct = ((discount_price - b["avg_discount_price"]) / b["avg_discount_price"] * 100
                                if b["avg_discount_price"] else 0)
            sales_delta_pct = ((predicted_sales - b["avg_predicted_sales"]) / b["avg_predicted_sales"] * 100
                                if b["avg_predicted_sales"] else 0)
            benchmark_note = {
                "category_avg_price": round(float(b["avg_discount_price"]), 2),
                "category_avg_sales": round(float(b["avg_predicted_sales"]), 1),
                "price_delta_pct": round(float(price_delta_pct), 1),
                "sales_delta_pct": round(float(sales_delta_pct), 1),
            }

    result = {
        "predicted_sales": round(predicted_sales),
        "estimated_revenue": round(estimated_revenue, 2),
        "discount_percentage": discount_percentage,
        "model_name": bundle["model_name"],
        "benchmark": benchmark_note,
    }

    return render_template(
        "index.html",
        main_categories=metadata["main_categories"],
        sub_categories_by_main=metadata["sub_categories_by_main"],
        model_name=metadata["model_name"],
        best_metrics=metadata["metrics"][0],
        result=result,
        form_values=safe_form_values,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API version of /predict, for programmatic use / testing."""
    bundle, metadata, benchmarks = load_artifacts()
    pipeline = bundle["pipeline"]
    data = request.get_json(silent=True) or {}

    try:
        actual_price = float(data.get("actual_price", 0) or 0)
        discount_price = float(data.get("discount_price", actual_price) or actual_price)
        rating = float(data.get("rating", 4.0) or 4.0)
    except (TypeError, ValueError):
        return jsonify({"error": "actual_price, discount_price, and rating must be numeric."}), 400

    main_category = str(data.get("main_category", "")).strip().lower()
    sub_category = str(data.get("sub_category", "")).strip().lower()
    description = str(data.get("description", "")).strip() or f"{main_category} {sub_category} product"

    errors = []
    if not main_category:
        errors.append("main_category is required.")
    if not sub_category:
        errors.append("sub_category is required.")
    if actual_price <= 0:
        errors.append("actual_price must be greater than 0.")
    if errors:
        return jsonify({"error": " ".join(errors)}), 400

    if discount_price > actual_price:
        discount_price = actual_price
    discount_percentage = round((actual_price - discount_price) / actual_price * 100, 2) if actual_price else 0.0

    row = pd.DataFrame([{
        "product_description": description,
        "main_category": main_category,
        "sub_category": sub_category,
        "discount_price": discount_price,
        "actual_price": actual_price,
        "ratings": rating,
        "discount_percentage": discount_percentage,
    }])[ALL_FEATURE_COLS]

    pred_log = pipeline.predict(row)[0]
    predicted_sales = max(float(np.expm1(pred_log)), 0)

    return jsonify({
        "predicted_sales": round(predicted_sales),
        "estimated_revenue": round(predicted_sales * discount_price, 2),
        "model_name": bundle["model_name"],
    })


@app.route("/dashboard")
def dashboard():
    """Small in-app analytics summary (the full interactive dashboard lives
    in Tableau -- see tableau/README.md -- this route is a quick sanity
    check / demo view)."""
    kpi_path = os.path.join(EXPORTS_DIR, "kpi_summary.csv")
    cat_path = os.path.join(EXPORTS_DIR, "category_summary.csv")
    if not (os.path.exists(kpi_path) and os.path.exists(cat_path)):
        return render_template("dashboard.html", ready=False)

    kpis = pd.read_csv(kpi_path).iloc[0].to_dict()
    cat_summary = pd.read_csv(cat_path)
    top_categories = (
        cat_summary.groupby("main_category")["total_estimated_revenue"]
        .sum().sort_values(ascending=False).head(10)
    )
    return render_template(
        "dashboard.html", ready=True, kpis=kpis,
        chart_labels=list(top_categories.index),
        chart_values=[round(v, 2) for v in top_categories.values],
    )


if __name__ == "__main__":
    # DEBUG defaults on for local development convenience (auto-reload,
    # interactive tracebacks). Set FLASK_DEBUG=0 before deploying anywhere
    # reachable from the internet -- the Werkzeug debugger's interactive
    # console on error pages is a real security risk in production. Better
    # yet, don't use this dev server in production at all; see README.md
    # > "Production deployment notes" for running under gunicorn instead.
    debug_mode = os.environ.get("FLASK_DEBUG", "1") != "0"
    app.run(debug=debug_mode, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
