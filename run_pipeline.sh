#!/usr/bin/env bash
# run_pipeline.sh — one-command setup: generate data -> train -> business
# analytics exports. Run the Flask app separately with `python app/app.py`
# (kept separate so you can iterate on the app without retraining).
#
# Usage:
#   ./run_pipeline.sh                # sample data, default training budget
#   ./run_pipeline.sh --full         # more thorough hyperparameter search
set -euo pipefail
cd "$(dirname "$0")"

ROWS=20000
N_ITER=8
CV=4
if [[ "${1:-}" == "--full" ]]; then
  N_ITER=20
  CV=5
  ROWS=40000
fi

echo "== 1/3  Generating sample data (${ROWS} rows) =="
python3 scripts/generate_sample_data.py --rows "$ROWS"

echo "== 2/3  Training + comparing models (n_iter=${N_ITER}, cv=${CV}) =="
python3 -m src.train --n-iter "$N_ITER" --cv "$CV"

echo "== 3/3  Building business-analytics / Tableau exports =="
python3 -m src.business_analytics

echo
echo "Done. Start the app with:  python3 app/app.py"
echo "Then open:                 http://127.0.0.1:5000"
