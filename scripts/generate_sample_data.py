"""
generate_sample_data.py
------------------------
Generates a realistic, schema-accurate stand-in for the Kaggle
"Amazon Products Sales Dataset 2023" (lokeshparab/amazon-products-dataset),
which is the dataset the original project (Majorproject2.ipynb) used
(columns: name, main_category, sub_category, ratings, no_of_ratings,
discount_price, actual_price, image, link).

Why this exists
----------------
The real dataset is ~1.4M rows split across 142 per-category CSVs and is
distributed via Kaggle, which requires a personal Kaggle API token to
download programmatically. Rather than ask you to hand over credentials,
this script creates a smaller, structurally-identical dataset (same
columns, same relationships between price/discount/rating and sales) so
the ENTIRE pipeline runs end-to-end out of the box.

For the real thing: download the dataset yourself from
https://www.kaggle.com/datasets/lokeshparab/amazon-products-dataset
(needs a free Kaggle account + API token in ~/.kaggle/kaggle.json) and
drop the per-category CSVs into data/raw/amazon_categories/ -- the
loader in src/data_preprocessing.py will pick them up automatically
instead of the sample file. See README.md > "Using the real dataset".

Usage:
    python scripts/generate_sample_data.py --rows 20000 --seed 42
"""
import argparse
import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Category taxonomy: main_category -> {sub_category: (price_min, price_max,
# popularity_weight, [brand_tiers])}
# Popularity weight loosely models how many units a "typical" product in
# that category sells -- mirrors real Amazon India category skew (mobiles /
# electronics / fashion dominate volume; industrial/scientific is a long tail).
# ---------------------------------------------------------------------------
CATEGORY_TREE = {
    "electronics": {
        "subs": ["headphones", "smart watches", "cameras", "televisions", "speakers"],
        "price": (499, 45000), "popularity": 1.6,
        "premium_brands": ["Sony", "Bose", "JBL", "Samsung"],
        "budget_brands": ["boAt", "Noise", "Zebronics", "Ptron"],
    },
    "mobiles": {
        "subs": ["smartphones", "mobile accessories", "power banks", "cases & covers"],
        "price": (299, 120000), "popularity": 2.0,
        "premium_brands": ["Apple", "Samsung", "OnePlus"],
        "budget_brands": ["Redmi", "Realme", "Ambrane", "Portronics"],
    },
    "computers": {
        "subs": ["laptops", "keyboards & mice", "monitors", "storage devices"],
        "price": (399, 150000), "popularity": 1.1,
        "premium_brands": ["Apple", "Dell", "HP"],
        "budget_brands": ["Lenovo", "Zebronics", "Logitech", "HP"],
    },
    "appliances": {
        "subs": ["air conditioners", "refrigerators", "washing machines", "microwaves"],
        "price": (2999, 85000), "popularity": 0.9,
        "premium_brands": ["LG", "Samsung", "Bosch"],
        "budget_brands": ["Voltas", "Whirlpool", "Carrier", "Lloyd"],
    },
    "fashion": {
        "subs": ["men's clothing", "women's clothing", "footwear", "watches"],
        "price": (199, 8999), "popularity": 1.8,
        "premium_brands": ["Tommy Hilfiger", "Levis", "Fossil"],
        "budget_brands": ["Redwolf", "HRX", "Roadster", "Fabindia"],
    },
    "home & kitchen": {
        "subs": ["cookware", "home decor", "furniture", "kitchen storage"],
        "price": (149, 25000), "popularity": 1.3,
        "premium_brands": ["Prestige", "Pigeon", "Milton"],
        "budget_brands": ["Cello", "Amazon Basics", "Nirlon", "Wonderchef"],
    },
    "beauty & health": {
        "subs": ["skincare", "haircare", "personal care appliances", "makeup"],
        "price": (99, 4999), "popularity": 1.5,
        "premium_brands": ["Lakme", "Philips", "Mamaearth"],
        "budget_brands": ["WOW", "Nivea", "Dove", "Garnier"],
    },
    "sports & fitness": {
        "subs": ["yoga", "gym equipment", "cycling", "sportswear"],
        "price": (199, 15999), "popularity": 1.0,
        "premium_brands": ["Nike", "Adidas", "Decathlon"],
        "budget_brands": ["BE SAVAGE", "Cockatoo", "Strauss", "Kore"],
    },
    "toys & baby products": {
        "subs": ["toys", "baby care", "infant wear", "educational games"],
        "price": (99, 5999), "popularity": 1.2,
        "premium_brands": ["Fisher-Price", "LEGO", "Chicco"],
        "budget_brands": ["Mothercare", "Luvlap", "Baybee", "Funskool"],
    },
    "books": {
        "subs": ["fiction", "non-fiction", "children's books", "academic"],
        "price": (99, 1999), "popularity": 0.8,
        "premium_brands": ["Penguin", "HarperCollins"],
        "budget_brands": ["Local Press", "Rupa Publications"],
    },
    "grocery & gourmet foods": {
        "subs": ["snacks", "beverages", "staples", "organic foods"],
        "price": (49, 2499), "popularity": 1.4,
        "premium_brands": ["Nestle", "Tata"],
        "budget_brands": ["Local Brand", "24 Mantra"],
    },
    "home improvement": {
        "subs": ["power tools", "hand tools", "electricals", "paint supplies"],
        "price": (199, 22000), "popularity": 0.6,
        "premium_brands": ["Bosch", "Stanley"],
        "budget_brands": ["Cheston", "iBell"],
    },
    "pet supplies": {
        "subs": ["pet food", "pet grooming", "pet accessories"],
        "price": (99, 4999), "popularity": 0.5,
        "premium_brands": ["Pedigree", "Royal Canin"],
        "budget_brands": ["Drools", "Basil"],
    },
    "car & motorbike": {
        "subs": ["car accessories", "bike accessories", "car care"],
        "price": (99, 12999), "popularity": 0.7,
        "premium_brands": ["3M", "Michelin"],
        "budget_brands": ["AutoFurnish", "Spidy Moto"],
    },
    "office products": {
        "subs": ["stationery", "office electronics", "storage & organization"],
        "price": (49, 9999), "popularity": 0.6,
        "premium_brands": ["Parker", "Casio"],
        "budget_brands": ["Classmate", "Cello"],
    },
    "music & instruments": {
        "subs": ["guitars", "keyboards", "accessories"],
        "price": (299, 35000), "popularity": 0.3,
        "premium_brands": ["Yamaha", "Fender"],
        "budget_brands": ["Kadence", "Juarez"],
    },
    "industrial & scientific": {
        "subs": ["lab equipment", "safety gear", "measuring tools"],
        "price": (199, 40000), "popularity": 0.2,
        "premium_brands": ["3M", "Bosch"],
        "budget_brands": ["Generic", "Local"],
    },
    "garden & outdoors": {
        "subs": ["gardening tools", "outdoor furniture", "grills"],
        "price": (149, 15000), "popularity": 0.4,
        "premium_brands": ["Weber", "Fiskars"],
        "budget_brands": ["Truphe", "GARDENIA"],
    },
    "luggage & bags": {
        "subs": ["backpacks", "suitcases", "wallets"],
        "price": (299, 9999), "popularity": 0.9,
        "premium_brands": ["American Tourister", "Samsonite"],
        "budget_brands": ["Skybags", "Wildcraft"],
    },
    "watches": {
        "subs": ["analog watches", "smart watches", "watch accessories"],
        "price": (299, 45000), "popularity": 0.8,
        "premium_brands": ["Fossil", "Titan", "Casio"],
        "budget_brands": ["Fastrack", "Sonata"],
    },
}

ADJECTIVES = ["Premium", "Classic", "Pro", "Advanced", "Compact", "Portable",
              "Ultra", "Everyday", "Deluxe", "Essential", "Smart", "Wireless",
              "Lightweight", "Heavy-Duty", "Eco-Friendly"]


def make_product_name(rng, main_cat, sub_cat, brand):
    adj = rng.choice(ADJECTIVES)
    model = f"{rng.integers(100, 999)}{rng.choice(['X', 'Pro', 'S', 'Max', ''])}"
    return f"{brand} {adj} {sub_cat.title()} {model}".strip()


def generate(n_rows: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    mains = list(CATEGORY_TREE.keys())
    # weight main-category sampling by popularity so volume categories get more SKUs
    weights = np.array([CATEGORY_TREE[m]["popularity"] for m in mains], dtype=float)
    weights = weights / weights.sum()

    rows = []
    for _ in range(n_rows):
        main = rng.choice(mains, p=weights)
        cat = CATEGORY_TREE[main]
        sub = rng.choice(cat["subs"])
        pmin, pmax = cat["price"]

        # log-uniform price sampling (Amazon catalogs are heavily right-skewed)
        actual_price = float(np.exp(rng.uniform(np.log(pmin), np.log(pmax))))
        discount_pct = float(np.clip(rng.beta(2, 5) * 80, 0, 75))  # most discounts 0-40%, some deep
        discount_price = round(actual_price * (1 - discount_pct / 100), 2)
        actual_price = round(actual_price, 2)

        rating = float(np.clip(rng.normal(4.0, 0.55), 1.0, 5.0))
        rating = round(rating * 10) / 10  # one decimal, like real Amazon ratings

        is_premium = rng.random() < 0.35
        brand = rng.choice(cat["premium_brands"] if is_premium else cat["budget_brands"])
        name = make_product_name(rng, main, sub, brand)

        # ---- Latent "true sales" generative model -------------------------
        # base popularity of the category/sub-category
        base = cat["popularity"] * 450
        # rating effect: sales rise sharply once rating clears ~3.5
        rating_effect = np.exp((rating - 3.0) * 0.9)
        # discount effect: diminishing returns (sqrt), deep discounts help but not linearly
        discount_effect = 1 + np.sqrt(discount_pct) * 0.28
        # price effect: cheaper-within-category items sell more (log scale, normalized)
        rel_price = (np.log(actual_price) - np.log(pmin)) / (np.log(pmax) - np.log(pmin) + 1e-9)
        price_effect = np.exp(-1.1 * rel_price)
        # premium brand recognition bonus
        brand_effect = 1.35 if is_premium else 1.0

        mean_sales = base * rating_effect * discount_effect * price_effect * brand_effect
        mean_sales = max(mean_sales, 1.0)

        # Realistic-but-learnable noise: modest multiplicative (log-normal)
        # jitter representing unmodeled real-world factors (seasonality,
        # merchandising placement, etc.), then Poisson sampling for the
        # discreteness of an actual unit count. Keeping the jitter small
        # (sigma=0.35) means the engineered features (price/rating/discount/
        # brand/category) remain genuinely predictive -- as they would be
        # in a real, well-specified sales dataset -- while still leaving
        # plenty of irreducible noise so metrics stay honest (R2 well
        # short of 1.0, the way a real-world model would land).
        noise_multiplier = rng.lognormal(mean=0.0, sigma=0.35)
        lam = mean_sales * noise_multiplier
        total_sales = int(rng.poisson(lam))

        rows.append({
            "name": name,
            "main_category": main,
            "sub_category": sub,
            "ratings": rating,
            "no_of_ratings": total_sales,
            "discount_price": f"\u20b9{discount_price:,.0f}",
            "actual_price": f"\u20b9{actual_price:,.0f}",
            "image": f"https://example-cdn.local/img/{rng.integers(100000, 999999)}.jpg",
            "link": f"https://example.local/product/{rng.integers(100000, 999999)}",
        })

    df = pd.DataFrame(rows)
    df.insert(0, "Unnamed: 0", range(len(df)))
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=20000, help="number of product rows to generate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    out_path = args.out or os.path.join(
        os.path.dirname(__file__), "..", "data", "raw", "amazon_products_sample.csv"
    )
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    df = generate(args.rows, args.seed)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df):,} rows -> {out_path}")
    print(df.head(3).to_string())


if __name__ == "__main__":
    main()
