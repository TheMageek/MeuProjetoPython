from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ml.db import DBConfig, connect
from ml.features import build_feature_frame
from ml.modeling import load_bundle
from ml.sources import (
    BrapiAuth,
    fetch_fundamentals_brapi,
    fetch_news_daily,
    fetch_ohlcv_brapi,
    fetch_ohlcv_yfinance,
    fetch_sector_yfinance,
)
from ml.train import _choose_best_source, _fundamentals_to_daily


def _load_macro(con) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM macro ORDER BY date", con).drop(columns=["source"], errors="ignore")


def _show_feature_contributions(row: pd.DataFrame, bundle, top_n: int = 12) -> None:
    X = row[bundle.feature_cols]

    try:
        contrib = bundle.clf.predict(X, pred_contrib=True)
    except TypeError:
        print("\n[Explicação da previsão]")
        print("O modelo atual não suportou pred_contrib nesta configuração.")
        return

    contrib = np.asarray(contrib)

    if contrib.ndim != 2 or contrib.shape[0] != 1:
        print("\n[Explicação da previsão]")
        print("Não foi possível extrair contribuições individuais das features.")
        return

    # LightGBM normalmente devolve última coluna como bias/base value
    raw_contrib = contrib[0]
    if len(raw_contrib) == len(bundle.feature_cols) + 1:
        feature_contrib = raw_contrib[:-1]
        bias = raw_contrib[-1]
    else:
        feature_contrib = raw_contrib[: len(bundle.feature_cols)]
        bias = None

    values = X.iloc[0].to_dict()

    df_exp = pd.DataFrame(
        {
            "feature": bundle.feature_cols,
            "value": [values.get(f) for f in bundle.feature_cols],
            "contribution": feature_contrib,
        }
    )

    df_exp["abs_contribution"] = df_exp["contribution"].abs()
    df_exp = df_exp.sort_values("abs_contribution", ascending=False).reset_index(drop=True)

    top = df_exp.head(top_n).copy()
    pos = top[top["contribution"] > 0].copy()
    neg = top[top["contribution"] < 0].copy()

    print("\n================ EXPLICAÇÃO DA PREVISÃO ================")
    if bias is not None:
        print(f"Bias/base value do classificador: {bias:.6f}")

    print("\nFeatures que mais EMPURRARAM para ALTA:")
    if pos.empty:
        print("  Nenhuma entre as top features.")
    else:
        for _, r in pos.iterrows():
            print(
                f"  {r['feature']}: valor={r['value']:.6f} | contribuição=+{r['contribution']:.6f}"
            )

    print("\nFeatures que mais EMPURRARAM para BAIXA:")
    if neg.empty:
        print("  Nenhuma entre as top features.")
    else:
        for _, r in neg.iterrows():
            print(
                f"  {r['feature']}: valor={r['value']:.6f} | contribuição={r['contribution']:.6f}"
            )

    news_top = df_exp[df_exp["feature"].str.startswith("news_")].head(8)
    print("\nImpacto das FEATURES DE NOTÍCIA:")
    if news_top.empty:
        print("  Nenhuma feature de notícia apareceu entre as contribuições disponíveis.")
    else:
        for _, r in news_top.iterrows():
            sign = "+" if r["contribution"] >= 0 else ""
            print(
                f"  {r['feature']}: valor={r['value']:.6f} | contribuição={sign}{r['contribution']:.6f}"
            )

    print("========================================================\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--db", default="data/market.sqlite3")
    ap.add_argument("--models_dir", default="models")
    ap.add_argument("--range", dest="range_", default="1y")
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--brapi_token", default=None)
    ap.add_argument("--brapi_bearer", default=None)
    ap.add_argument("--asof", default=None, help="YYYY-MM-DD (optional)")
    ap.add_argument("--top_features", type=int, default=12, help="Quantidade de features a mostrar na explicação")
    args = ap.parse_args()

    ticker = args.ticker.strip().upper()
    auth = BrapiAuth(token=args.brapi_token, bearer=args.brapi_bearer)

    sector, _industry = fetch_sector_yfinance(ticker)
    sector_key = (sector or "UNKNOWN").replace(" ", "_").upper()

    models_dir = Path(args.models_dir)
    model_path = models_dir / f"lgbm_{sector_key}.joblib"
    if not model_path.exists():
        model_path = models_dir / "lgbm_GLOBAL.joblib"

    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    bundle = load_bundle(str(model_path))

    df_yf = fetch_ohlcv_yfinance(ticker, range_=args.range_, interval=args.interval)
    df_br = fetch_ohlcv_brapi(ticker, auth=auth, range_=args.range_, interval=args.interval)
    best_df, src = _choose_best_source(df_yf, df_br)

    if best_df.empty:
        raise SystemExit("No price data available.")

    fpay = fetch_fundamentals_brapi(ticker, auth=auth)
    fundamentals_daily = _fundamentals_to_daily(fpay, best_df["date"])

    news_daily = fetch_news_daily(
        ticker=ticker,
        company_name=ticker,
        sector=sector,
    )

    with connect(DBConfig(path=Path(args.db))) as con:
        macro = _load_macro(con)

    feat = build_feature_frame(best_df, fundamentals_daily, macro, news_daily)

    if feat.empty:
        raise SystemExit("Feature frame is empty.")

    if args.asof:
        feat = feat[feat["date"] <= args.asof].copy()

    if feat.empty:
        raise SystemExit("No feature rows available for the requested --asof date.")

    row = feat.iloc[-1:].copy()

    missing_cols = [c for c in bundle.feature_cols if c not in row.columns]
    if missing_cols:
        raise SystemExit(f"Missing feature columns in prediction frame: {missing_cols}")

    X = row[bundle.feature_cols]

    prob_up = float(bundle.clf.predict_proba(X)[:, 1][0])
    sl_hat = float(bundle.reg_sl.predict(X)[0])
    sg_hat = float(bundle.reg_sg.predict(X)[0])
    vol_hat = float(bundle.reg_vol.predict(X)[0])

    entry = float(row["close"].iloc[0])
    stop_loss = entry * (1.0 + sl_hat)
    stop_gain = entry * (1.0 + sg_hat)

    print(
        {
            "ticker": ticker,
            "model": model_path.name,
            "source_used": src,
            "date": row["date"].iloc[0],
            "entry": round(entry, 4),
            "prob_up": round(prob_up, 4),
            "stop_loss_pct": round(sl_hat, 4),
            "stop_gain_pct": round(sg_hat, 4),
            "stop_loss": round(stop_loss, 4),
            "stop_gain": round(stop_gain, 4),
            "future_vol_logstd": round(vol_hat, 6),
        }
    )

    _show_feature_contributions(row=row, bundle=bundle, top_n=args.top_features)


if __name__ == "__main__":
    main()