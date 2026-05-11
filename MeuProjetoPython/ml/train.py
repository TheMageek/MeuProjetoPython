from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from ml.db import DBConfig, connect, executemany, init_db, upsert_ticker
from ml.features import build_feature_frame, parse_brapi_fundamentals
from ml.modeling import save_bundle, train_bundle
from ml.sources import (
    BrapiAuth,
    fetch_fundamentals_brapi,
    fetch_macro_yfinance,
    fetch_news_daily,
    fetch_ohlcv_brapi,
    fetch_ohlcv_yfinance,
    fetch_sector_yfinance,
    fetch_sgs_series,
    yf_symbol_b3,
)
from ml.targets import make_targets


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _date_bounds(range_years: int = 6) -> tuple[str, str, str, str]:
    end = datetime.now(UTC).date()
    start = end - timedelta(days=365 * range_years)
    return str(start), str(end), start.strftime("%d/%m/%Y"), end.strftime("%d/%m/%Y")


def _choose_best_source(df_yf: pd.DataFrame, df_br: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if len(df_br) >= len(df_yf) and not df_br.empty:
        return df_br, "brapi"
    if not df_yf.empty:
        return df_yf, "yfinance"
    return pd.DataFrame(), "none"


def _fundamentals_to_daily(payload: Dict[str, Any], dates: pd.Series) -> pd.DataFrame:
    feat = parse_brapi_fundamentals(payload)
    return pd.DataFrame([{"date": str(d), **feat} for d in dates])


def main() -> None:
    ap = argparse.ArgumentParser(description="Treinamento ML com suporte a horizontes variáveis")
    ap.add_argument("--tickers", nargs="*", default=[])
    ap.add_argument("--tickers_file", default=None)
    ap.add_argument("--db", default="data/market.sqlite3")
    ap.add_argument("--models_dir", default="models")
    ap.add_argument("--horizon", type=int, default=10, help="Horizonte de previsão em dias")
    ap.add_argument("--range", dest="range_", default="5y")
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--min_rows", type=int, default=400)
    ap.add_argument("--min_sector_rows", type=int, default=800)
    ap.add_argument("--brapi_token", default=None)
    args = ap.parse_args()

    # Carrega tickers
    tickers = []
    if args.tickers_file:
        with open(args.tickers_file, "r", encoding="utf-8") as f:
            tickers = [line.strip().upper() for line in f if line.strip() and not line.startswith("#")]
    tickers.extend([t.strip().upper() for t in args.tickers])
    tickers = list(dict.fromkeys([t for t in tickers if t]))

    if not tickers:
        raise SystemExit("Informe --tickers ou --tickers_file")

    print(f"\n{'='*90}")
    print(f"TREINAMENTO COM HORIZONTE DE {args.horizon} DIAS")
    print(f"Tickers: {len(tickers)} | Período: {args.range_}")
    print(f"{'='*90}\n")

    db = DBConfig(path=Path(args.db))
    init_db(db)

    auth = BrapiAuth(token=args.brapi_token)
    start_iso, end_iso, _, _ = _date_bounds(range_years=6)

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    with connect(db) as con:
        print("[1/4] Carregando macro...")
        macro = fetch_macro_yfinance(start_iso, end_iso)

        all_sector_frames: Dict[str, List[pd.DataFrame]] = {}

        print("[2/4] Processando dados dos tickers...")
        for i, t in enumerate(tickers, 1):
            print(f"  [{i:02d}/{len(tickers)}] {t}")
            sector, _ = fetch_sector_yfinance(t)

            df_yf = fetch_ohlcv_yfinance(t, range_=args.range_, interval=args.interval)
            df_br = fetch_ohlcv_brapi(t, auth=auth, range_=args.range_, interval=args.interval)
            best_df, src = _choose_best_source(df_yf, df_br)

            if best_df.empty or len(best_df) < args.min_rows:
                print(f"     Pulado: dados insuficientes")
                continue

            fpay = fetch_fundamentals_brapi(t, auth=auth)
            fundamentals_daily = _fundamentals_to_daily(fpay, best_df["date"])
            news_daily = fetch_news_daily(ticker=t, company_name=t, sector=sector)

            feat = build_feature_frame(best_df, fundamentals_daily, macro, news_daily)
            feat = make_targets(feat, horizon=args.horizon)

            if feat.empty:
                print("     Pulado: sem targets")
                continue

            feat["ticker"] = t
            feat["sector"] = sector or "UNKNOWN"
            all_sector_frames.setdefault(feat["sector"].iloc[0], []).append(feat)

            print(f"     Sucesso → {len(feat)} linhas")

        # Treinamento
        print("\n[3/4] Treinando modelos...")
        global_parts = []
        for sec, parts in all_sector_frames.items():
            sec_df = pd.concat(parts).sort_values("date").reset_index(drop=True)
            global_parts.append(sec_df)

            sec_name = sec.replace(" ", "_").upper()
            if len(sec_df) >= args.min_sector_rows:
                print(f"Treinando {sec_name}...")
                feature_cols = [c for c in sec_df.columns if c not in {"ticker","sector","date","y_cls","y_sl","y_sg","y_vol"}]
                bundle, _ = train_bundle(sec_df, feature_cols=feature_cols, model_name=sec_name)
                save_bundle(bundle, str(models_dir / f"lgbm_{sec_name}.joblib"))
            else:
                print(f"[{sec_name}] Dados insuficientes")

        # GLOBAL
        if global_parts:
            global_df = pd.concat(global_parts).sort_values("date").reset_index(drop=True)
            print("Treinando modelo GLOBAL...")
            feature_cols = [c for c in global_df.columns if c not in {"ticker","sector","date","y_cls","y_sl","y_sg","y_vol"}]
            bundle, _ = train_bundle(global_df, feature_cols=feature_cols, model_name="GLOBAL")
            save_bundle(bundle, str(models_dir / "lgbm_GLOBAL.joblib"))

    print(f"\n✅ Treinamento finalizado com horizonte de {args.horizon} dias!")


if __name__ == "__main__":
    main()