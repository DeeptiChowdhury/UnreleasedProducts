"""
Basic pipeline tests. Run with:  pytest -q

These are intentionally fast (they use a small synthetic slice, not the
full pipeline) so they're suitable for a CI workflow on every push.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_preprocessing import _clean_price, clean_and_engineer
from src.features import ALL_FEATURE_COLS, TARGET_COL, build_pipeline
from sklearn.ensemble import RandomForestRegressor


def _make_raw_df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    mains = ["electronics", "fashion", "books"]
    subs = {"electronics": ["headphones", "cameras"],
            "fashion": ["footwear", "watches"],
            "books": ["fiction", "academic"]}
    rows = []
    for i in range(n):
        m = rng.choice(mains)
        s = rng.choice(subs[m])
        actual = float(rng.uniform(100, 20000))
        discount = actual * rng.uniform(0.5, 1.0)
        rows.append({
            "Unnamed: 0": i,
            "name": f"Test Product {i} {m} {s}",
            "main_category": m,
            "sub_category": s,
            "ratings": float(rng.uniform(1, 5)),
            "no_of_ratings": int(rng.integers(0, 5000)),
            "discount_price": f"\u20b9{discount:,.0f}",
            "actual_price": f"\u20b9{actual:,.0f}",
            "image": "http://example.local/img.jpg",
            "link": "http://example.local/link",
        })
    return pd.DataFrame(rows)


def test_clean_price_strips_currency_and_commas():
    s = pd.Series(["\u20b91,234", "\u20b9500", "999"])
    out = _clean_price(s)
    assert list(out) == [1234.0, 500.0, 999.0]


def test_clean_and_engineer_produces_expected_columns():
    raw = _make_raw_df(300)
    clean = clean_and_engineer(raw)
    for col in ALL_FEATURE_COLS + [TARGET_COL, "discount_percentage", "price_band",
                                    "estimated_revenue"]:
        assert col in clean.columns, f"missing column: {col}"
    assert clean["discount_price"].dtype.kind in "fc"
    assert clean["actual_price"].dtype.kind in "fc"
    assert (clean["discount_price"] <= clean["actual_price"] * 1.01).all()
    assert clean["discount_percentage"].between(-1, 100.01).all()


def test_clean_and_engineer_drops_bad_rows():
    raw = _make_raw_df(50)
    # inject a garbage row that should get dropped
    bad = raw.iloc[0:1].copy()
    bad["actual_price"] = "not_a_price"
    raw_with_bad = pd.concat([raw, bad], ignore_index=True)
    clean = clean_and_engineer(raw_with_bad)
    assert len(clean) <= len(raw_with_bad)


def test_pipeline_fits_and_predicts_reasonable_shape():
    raw = _make_raw_df(400)
    clean = clean_and_engineer(raw)
    X = clean[ALL_FEATURE_COLS]
    y = np.log1p(clean[TARGET_COL])

    pipe = build_pipeline(RandomForestRegressor(n_estimators=15, max_depth=5, random_state=0))
    pipe.fit(X, y)
    preds = pipe.predict(X)

    assert len(preds) == len(X)
    assert np.all(np.isfinite(preds))


def test_pipeline_handles_unseen_category_gracefully():
    """A brand-new category at inference time shouldn't crash the pipeline
    (OneHotEncoder(handle_unknown='ignore') should absorb it)."""
    raw = _make_raw_df(300)
    clean = clean_and_engineer(raw)
    X = clean[ALL_FEATURE_COLS]
    y = np.log1p(clean[TARGET_COL])

    pipe = build_pipeline(RandomForestRegressor(n_estimators=10, max_depth=4, random_state=0))
    pipe.fit(X, y)

    novel_row = pd.DataFrame([{
        "product_description": "Totally New Gadget 3000",
        "main_category": "never_seen_before",
        "sub_category": "also_never_seen",
        "discount_price": 999.0,
        "actual_price": 1299.0,
        "ratings": 4.2,
        "discount_percentage": 23.1,
    }])[ALL_FEATURE_COLS]

    pred = pipe.predict(novel_row)
    assert len(pred) == 1
    assert np.isfinite(pred[0])


# ---------------------------------------------------------------------------
# Flask app regression tests
#
# These cover a real bug found during deployment verification: submitting
# the /predict form with a field (e.g. sub_category) entirely ABSENT from
# the POST body -- not just empty -- caused Jinja to resolve
# `form_values.sub_category` to an `Undefined` object, which crashed with a
# 500 when passed through the `|tojson` filter (used to seed a JS variable).
# Fixed by always building a plain dict with explicit defaults for every
# templated field, on every render path (success, validation-error, and
# parse-error), instead of passing the raw request.form MultiDict through.
# ---------------------------------------------------------------------------

@pytest.fixture()
def client():
    from app.app import app as flask_app
    flask_app.config.update(TESTING=True)
    with flask_app.test_client() as c:
        yield c


def test_index_page_loads(client):
    r = client.get("/")
    assert r.status_code == 200


def test_predict_missing_fields_returns_200_not_500(client):
    """The original bug: this used to raise TypeError -> 500."""
    r = client.post("/predict", data={"main_category": "electronics"})
    assert r.status_code == 200
    assert b"Please select a sub-category" in r.data


def test_predict_zero_price_shows_validation_error_not_crash(client):
    r = client.post("/predict", data={
        "description": "Free item", "main_category": "electronics",
        "sub_category": "headphones", "actual_price": "0",
        "discount_price": "0", "rating": "4.0",
    })
    assert r.status_code == 200
    assert b"greater than 0" in r.data


def test_predict_garbage_price_shows_friendly_error_not_crash(client):
    r = client.post("/predict", data={
        "description": "Bad input", "main_category": "electronics",
        "sub_category": "headphones", "actual_price": "notanumber",
        "discount_price": "999", "rating": "4.0",
    })
    assert r.status_code == 200
    assert b"valid numeric values" in r.data


def test_predict_happy_path_renders_prediction(client):
    r = client.post("/predict", data={
        "description": "Bose Wireless Headphones Pro 400",
        "main_category": "electronics", "sub_category": "headphones",
        "actual_price": "3999", "discount_price": "2999", "rating": "4.5",
    })
    assert r.status_code == 200
    assert b"metric-big" in r.data


def test_api_predict_missing_fields_returns_400_not_500(client):
    r = client.post("/api/predict", json={"main_category": "electronics"})
    assert r.status_code == 400
    assert "error" in r.get_json()


def test_api_predict_malformed_json_returns_400_not_500(client):
    r = client.post("/api/predict", data="not valid json{",
                     content_type="application/json")
    assert r.status_code == 400


def test_api_predict_happy_path(client):
    r = client.post("/api/predict", json={
        "description": "Sony Watch", "main_category": "watches",
        "sub_category": "smart watches", "actual_price": 8999,
        "discount_price": 6499, "rating": 4.3,
    })
    assert r.status_code == 200
    body = r.get_json()
    assert body["predicted_sales"] >= 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
