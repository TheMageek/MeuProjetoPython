# LGBM Stocks (B3) - Sector Models

## Quick start

```bash
python -m venv .venv
# windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements-ml.txt

# Train (example)
python -m ml.train --tickers PETR4 VALE3 ITUB4 BBDC4 ABEV3 WEGE3 --horizon 10 --range 5y

# Predict (example)
python -m ml.predict --ticker PETR4
```

## Brapi auth (optional)

- `--brapi_token <TOKEN>` (query param token)
- `--brapi_bearer <TOKEN>` (Authorization: Bearer)

## Outputs

- SQLite DB: `data/market.sqlite3`
- Models: `models/lgbm_<SECTOR>.joblib` and `models/lgbm_<SECTOR>.metrics.json`
- Fallback: `models/lgbm_GLOBAL.joblib`
