from ml.modeling import load_bundle

bundle = load_bundle("models/lgbm_ENERGY.joblib")
print("TOTAL FEATURES:", len(bundle.feature_cols))
print("\n".join(bundle.feature_cols))

news_cols = [c for c in bundle.feature_cols if c.startswith("news_")]
print("\nNEWS COLS:")
print(news_cols)