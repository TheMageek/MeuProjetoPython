from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

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


def _feature_contrib_frame(row: pd.DataFrame, bundle) -> pd.DataFrame:
    X = row[bundle.feature_cols]

    try:
        contrib = bundle.clf.predict(X, pred_contrib=True)
    except TypeError:
        return pd.DataFrame(columns=["feature", "value", "contribution", "abs_contribution"])

    contrib = np.asarray(contrib)
    if contrib.ndim != 2 or contrib.shape[0] != 1:
        return pd.DataFrame(columns=["feature", "value", "contribution", "abs_contribution"])

    raw_contrib = contrib[0]
    if len(raw_contrib) == len(bundle.feature_cols) + 1:
        feature_contrib = raw_contrib[:-1]
    else:
        feature_contrib = raw_contrib[: len(bundle.feature_cols)]

    values = X.iloc[0].to_dict()

    df_exp = pd.DataFrame(
        {
            "feature": bundle.feature_cols,
            "value": [values.get(f) for f in bundle.feature_cols],
            "contribution": feature_contrib,
        }
    )
    df_exp["abs_contribution"] = df_exp["contribution"].abs()
    return df_exp.sort_values("abs_contribution", ascending=False).reset_index(drop=True)


def _show_feature_contributions(row: pd.DataFrame, bundle, top_n: int = 12) -> None:
    df_exp = _feature_contrib_frame(row, bundle)
    if df_exp.empty:
        print("\n[Explicação da decisão]")
        print("Não foi possível extrair contribuições individuais das features.")
        return

    top = df_exp.head(top_n).copy()
    pos = top[top["contribution"] > 0].copy()
    neg = top[top["contribution"] < 0].copy()

    print("\n================ EXPLICAÇÃO DA DECISÃO ================")

    print("\nFeatures que mais EMPURRARAM para ALTA:")
    if pos.empty:
        print("  Nenhuma entre as top features.")
    else:
        for _, r in pos.iterrows():
            print(
                f"  {r['feature']}: valor={float(r['value']):.6f} | contribuição=+{float(r['contribution']):.6f}"
            )

    print("\nFeatures que mais EMPURRARAM para BAIXA:")
    if neg.empty:
        print("  Nenhuma entre as top features.")
    else:
        for _, r in neg.iterrows():
            print(
                f"  {r['feature']}: valor={float(r['value']):.6f} | contribuição={float(r['contribution']):.6f}"
            )

    news_top = df_exp[df_exp["feature"].str.startswith("news_")].head(8)
    print("\nImpacto das FEATURES DE NOTÍCIA:")
    if news_top.empty:
        print("  Nenhuma feature de notícia apareceu entre as contribuições disponíveis.")
    else:
        for _, r in news_top.iterrows():
            sign = "+" if float(r["contribution"]) >= 0 else ""
            print(
                f"  {r['feature']}: valor={float(r['value']):.6f} | contribuição={sign}{float(r['contribution']):.6f}"
            )

    print("=======================================================\n")


def _predict_dict(
    ticker: str,
    db_path: str,
    models_dir: str,
    range_: str,
    interval: str,
    asof: Optional[str],
    brapi_token: Optional[str],
    brapi_bearer: Optional[str],
) -> Dict[str, float]:
    auth = BrapiAuth(token=brapi_token, bearer=brapi_bearer)

    sector, _ = fetch_sector_yfinance(ticker)
    sector_key = (sector or "UNKNOWN").replace(" ", "_").upper()

    models_dir_p = Path(models_dir)
    model_path = models_dir_p / f"lgbm_{sector_key}.joblib"
    if not model_path.exists():
        model_path = models_dir_p / "lgbm_GLOBAL.joblib"

    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    bundle = load_bundle(str(model_path))

    df_yf = fetch_ohlcv_yfinance(ticker, range_=range_, interval=interval)
    df_br = fetch_ohlcv_brapi(ticker, auth=auth, range_=range_, interval=interval)
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

    with connect(DBConfig(path=Path(db_path))) as con:
        macro = _load_macro(con)

    feat = build_feature_frame(best_df, fundamentals_daily, macro, news_daily)

    if feat.empty:
        raise SystemExit("Feature frame is empty.")

    if asof:
        feat = feat[feat["date"] <= asof].copy()

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

    return {
        "ticker": ticker,
        "model": model_path.name,
        "source_used": src,
        "date": row["date"].iloc[0],
        "entry": entry,
        "prob_up": prob_up,
        "stop_loss_pct": sl_hat,
        "stop_gain_pct": sg_hat,
        "stop_loss": stop_loss,
        "stop_gain": stop_gain,
        "future_vol_logstd": vol_hat,
        "_row_for_explain": row,
        "_bundle_for_explain": bundle,
    }


def decide(
    prob_up: float,
    stop_loss_pct: float,
    stop_gain_pct: float,
    future_vol: float,
    prob_buy: float,
    prob_sell: float,
    rr_min: float,
    max_vol: float,
) -> str:
    loss = abs(stop_loss_pct)
    gain = max(stop_gain_pct, 0.0)
    rr = (gain / loss) if loss > 1e-9 else float("inf")

    if prob_up >= prob_buy and rr >= rr_min and future_vol <= max_vol:
        return "BUY"
    if prob_up <= prob_sell:
        return "SELL"
    return "HOLD"


def _explain_decision_text(
    decision: str,
    prob_up: float,
    rr: float,
    future_vol: float,
    prob_buy: float,
    prob_sell: float,
    rr_min: float,
    max_vol: float,
) -> None:
    print("[Resumo simples da decisão]")

    if decision == "BUY":
        print(
            f"- BUY porque prob_up={prob_up:.4f} ficou acima de {prob_buy:.2f}, "
            f"RR={rr:.4f} ficou acima de {rr_min:.2f} e a vol futura={future_vol:.6f} "
            f"ficou abaixo de {max_vol:.4f}."
        )
    elif decision == "SELL":
        print(
            f"- SELL porque prob_up={prob_up:.4f} ficou abaixo de {prob_sell:.2f}."
        )
    else:
        print(
            f"- HOLD porque não houve força suficiente para BUY "
            f"(prob_up={prob_up:.4f}, RR={rr:.4f}, vol={future_vol:.6f}) "
            f"e também não bateu regra de SELL."
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--db", default="data/market.sqlite3")
    ap.add_argument("--models_dir", default="models")
    ap.add_argument("--range", dest="range_", default="1y")
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--asof", default=None)

    ap.add_argument("--brapi_token", default=None)
    ap.add_argument("--brapi_bearer", default=None)

    ap.add_argument("--prob_buy", type=float, default=0.55)
    ap.add_argument("--prob_sell", type=float, default=0.45)
    ap.add_argument("--rr_min", type=float, default=1.5)
    ap.add_argument("--max_vol", type=float, default=0.03)

    ap.add_argument("--top_features", type=int, default=12, help="Quantidade de features a mostrar na explicação")
    args = ap.parse_args()

    p = _predict_dict(
        ticker=args.ticker.strip().upper(),
        db_path=args.db,
        models_dir=args.models_dir,
        range_=args.range_,
        interval=args.interval,
        asof=args.asof,
        brapi_token=args.brapi_token,
        brapi_bearer=args.brapi_bearer,
    )

    decision = decide(
        prob_up=p["prob_up"],
        stop_loss_pct=p["stop_loss_pct"],
        stop_gain_pct=p["stop_gain_pct"],
        future_vol=p["future_vol_logstd"],
        prob_buy=args.prob_buy,
        prob_sell=args.prob_sell,
        rr_min=args.rr_min,
        max_vol=args.max_vol,
    )

    loss = abs(p["stop_loss_pct"])
    gain = max(p["stop_gain_pct"], 0.0)
    rr = (gain / loss) if loss > 1e-9 else float("inf")

    row = p.pop("_row_for_explain")
    bundle = p.pop("_bundle_for_explain")

    p_out = {
        **p,
        "rr": round(rr, 4),
        "decision": decision,
        "rules": {
            "prob_buy": args.prob_buy,
            "prob_sell": args.prob_sell,
            "rr_min": args.rr_min,
            "max_vol": args.max_vol,
        },
    }
    print(p_out)
    print()
    _explain_decision_text(
        decision=decision,
        prob_up=p["prob_up"],
        rr=rr,
        future_vol=p["future_vol_logstd"],
        prob_buy=args.prob_buy,
        prob_sell=args.prob_sell,
        rr_min=args.rr_min,
        max_vol=args.max_vol,
    )
    _show_feature_contributions(row=row, bundle=bundle, top_n=args.top_features)


if __name__ == "__main__":
    main()