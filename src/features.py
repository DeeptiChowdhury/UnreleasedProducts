"""
features.py
------------
Builds the scikit-learn preprocessing pipeline used by every model.

Improvement over the original project: the original notebook trained on
EITHER pure text (TF-IDF of description -> XGBoost, in CUPID.ipynb) OR
pure structured features (category/price/rating -> GBR, in
Majorproject2.ipynb), never both. Here we combine them with a
ColumnTransformer so the model can use brand/product-name signal
*and* pricing/category/rating signal at the same time -- which is closer
to how you'd actually price and position a brand-new product before
launch.
"""
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TEXT_COL = "product_description"
CATEGORICAL_COLS = ["main_category", "sub_category"]
NUMERIC_COLS = ["discount_price", "actual_price", "ratings", "discount_percentage"]

ALL_FEATURE_COLS = [TEXT_COL] + CATEGORICAL_COLS + NUMERIC_COLS
TARGET_COL = "total_sales"


def build_preprocessor(max_text_features: int = 300) -> ColumnTransformer:
    """ColumnTransformer that fuses text (TF-IDF), categorical (one-hot)
    and numeric (scaled) features into a single sparse matrix."""
    return ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(max_features=max_text_features,
                                      stop_words="english",
                                      ngram_range=(1, 2)), TEXT_COL),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
            ("num", StandardScaler(), NUMERIC_COLS),
        ],
        remainder="drop",
    )


def build_pipeline(estimator, max_text_features: int = 300) -> Pipeline:
    return Pipeline(steps=[
        ("preprocessor", build_preprocessor(max_text_features)),
        ("model", estimator),
    ])
