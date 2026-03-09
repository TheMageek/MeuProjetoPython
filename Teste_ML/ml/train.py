from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from ml.db import DBConfig, connect, executemany, init_db, upsert_ticker
from ml.features import build_feature_frame, parse_brapi_fundamentals
from ml.modeling import save_bundle, train_bundle
from ml.sources import (
    BrapiAuth,
    SGS,
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
    df_yf = df_yf if df_yf is not None else pd.DataFrame()
    df_br = df_br if df_br is not None else pd.DataFrame()
    if len(df_br) >= len(df_yf) and len(df_br) > 0:
        return df_br, "brapi"
    if len(df_yf) > 0:
        return df_yf, "yfinance"
    return pd.DataFrame(), "none"


def _fundamentals_to_daily(payload: Dict[str, Any], dates: pd.Series) -> pd.DataFrame:
    feat = parse_brapi_fundamentals(payload)
    return pd.DataFrame([{"date": str(d), **feat} for d in dates])


def _macro_daily(start_iso: str, end_iso: str, start_bcb: str, end_bcb: str) -> pd.DataFrame:
    selic = fetch_sgs_series(SGS["selic"], start_bcb, end_bcb).rename(columns={"value": "selic"})
    ipca = fetch_sgs_series(SGS["ipca"], start_bcb, end_bcb).rename(columns={"value": "ipca"})
    usd = fetch_sgs_series(SGS["usd_brl"], start_bcb, end_bcb).rename(columns={"value": "usd_brl"})

    ibov = fetch_ohlcv_yfinance("^BVSP", range_="6y", interval="1d")
    if not ibov.empty:
        ibov = ibov[["date", "close"]].rename(columns={"close": "ibov_close"})
    else:
        ibov = pd.DataFrame(columns=["date", "ibov_close"])

    glob = fetch_macro_yfinance(start_iso, end_iso)

    out = ibov.merge(glob, on="date", how="outer")
    out = out.merge(selic, on="date", how="left")
    out = out.merge(ipca, on="date", how="left")
    out = out.merge(usd, on="date", how="left")

    out = out.sort_values("date").ffill()
    out["source"] = "BCB/SGS+yfinance"
    return out.reset_index(drop=True)


def _persist_prices(con, ticker: str, df: pd.DataFrame, source: str) -> None:
    rows = []
    for _, r in df.iterrows():
        rows.append(
            (
                ticker,
                str(r["date"]),
                float(r["open"]) if pd.notna(r["open"]) else None,
                float(r["high"]) if pd.notna(r["high"]) else None,
                float(r["low"]) if pd.notna(r["low"]) else None,
                float(r["close"]) if pd.notna(r["close"]) else None,
                float(r["volume"]) if pd.notna(r["volume"]) else None,
                source,
            )
        )
    executemany(
        con,
        """
        INSERT OR REPLACE INTO prices (ticker, date, open, high, low, close, volume, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def _persist_macro(con, macro: pd.DataFrame) -> None:
    rows = []
    for _, r in macro.iterrows():
        rows.append(
            (
                str(r["date"]),
                float(r.get("selic")) if pd.notna(r.get("selic")) else None,
                float(r.get("ipca")) if pd.notna(r.get("ipca")) else None,
                float(r.get("usd_brl")) if pd.notna(r.get("usd_brl")) else None,
                float(r.get("ibov_close")) if pd.notna(r.get("ibov_close")) else None,
                float(r.get("brent_close")) if pd.notna(r.get("brent_close")) else None,
                float(r.get("spx_close")) if pd.notna(r.get("spx_close")) else None,
                float(r.get("vix_close")) if pd.notna(r.get("vix_close")) else None,
                str(r.get("source") or "macro"),
            )
        )
    executemany(
        con,
        """
        INSERT OR REPLACE INTO macro (date, selic, ipca, usd_brl, ibov_close, brent_close, spx_close, vix_close, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def _load_macro_from_db(con) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM macro ORDER BY date", con)


def _read_tickers_file(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        t = line.strip().upper()
        if t and not t.startswith("#"):
            out.append(t)
    return out


def _merge_tickers(cli: Sequence[str], file_path: Optional[str]) -> List[str]:
    out = [t.strip().upper() for t in cli if t.strip()]
    if file_path:
        out.extend(_read_tickers_file(file_path))
    seen = set()
    uniq = []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*", default=[])
    ap.add_argument("--tickers_file", default=None, help="Optional file with tickers (one per line)")
    ap.add_argument("--db", default="data/market.sqlite3")
    ap.add_argument("--models_dir", default="models")
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--range", dest="range_", default="5y")
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--min_rows", type=int, default=400)
    ap.add_argument("--min_sector_rows", type=int, default=800, help="Minimum rows to train a sector model")
    ap.add_argument("--brapi_token", default=None)
    ap.add_argument("--brapi_bearer", default=None)
    args = ap.parse_args()

    tickers = _merge_tickers(args.tickers, args.tickers_file)
    if not tickers:
        raise SystemExit("Provide --tickers ... or --tickers_file ...")

    print("\n================ INÍCIO DO TREINO ================")
    print(f"Tickers recebidos: {len(tickers)}")
    print(f"Range: {args.range_} | Interval: {args.interval} | Horizon: {args.horizon}")
    print("Split temporal: 70% treino / 30% teste")
    print("Ao final da validação, o bundle salvo será refeito com 100% dos dados.\n")

    db = DBConfig(path=Path(args.db))
    init_db(db)

    auth = BrapiAuth(token=args.brapi_token, bearer=args.brapi_bearer)
    start_iso, end_iso, start_bcb, end_bcb = _date_bounds(range_years=6)

    print("[1/4] Montando base macro...")
    macro = _macro_daily(start_iso, end_iso, start_bcb, end_bcb)

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    with connect(db) as con:
        _persist_macro(con, macro)
        con.commit()

        all_sector_frames: Dict[str, List[pd.DataFrame]] = {}

        print("[2/4] Coletando e preparando dados por ticker...")
        for i, t in enumerate(tickers, start=1):
            print(f"  -> ({i}/{len(tickers)}) {t}")

            yf_sym = yf_symbol_b3(t)
            sector, industry = fetch_sector_yfinance(t)

            df_yf = fetch_ohlcv_yfinance(t, range_=args.range_, interval=args.interval)
            df_br = fetch_ohlcv_brapi(t, auth=auth, range_=args.range_, interval=args.interval)
            best_df, best_src = _choose_best_source(df_yf, df_br)

            if best_df.empty or len(best_df) < args.min_rows:
                print(f"     pulado: linhas insuficientes ({len(best_df)})")
                continue

            print(f"     fonte escolhida: {best_src} | linhas: {len(best_df)} | setor: {sector or 'UNKNOWN'}")

            _persist_prices(con, t, best_df, best_src)
            con.commit()

            fpay = fetch_fundamentals_brapi(t, auth=auth)
            fundamentals_daily = _fundamentals_to_daily(fpay, best_df["date"])

            news_daily = fetch_news_daily(
                ticker=t,
                company_name=t,
                sector=sector,
                start=start_iso,
                end=end_iso,
            )

            macro_db = _load_macro_from_db(con).drop(columns=["source"], errors="ignore")
            feat = build_feature_frame(best_df, fundamentals_daily, macro_db, news_daily)
            feat = make_targets(feat, horizon=args.horizon)

            if feat.empty:
                print("     pulado: feature frame ficou vazio após targets")
                continue

            feat["ticker"] = t
            feat["sector"] = sector or "UNKNOWN"
            all_sector_frames.setdefault(feat["sector"].iloc[0], []).append(feat)

            upsert_ticker(con, t, yf_sym, sector, industry, _now_iso())
            con.commit()

        def train_and_save(name: str, df: pd.DataFrame) -> None:
            print(f"\n[3/4] Treinando modelo {name}...")
            ignore = {"ticker", "sector", "y_cls", "y_sl", "y_sg", "y_vol"}
            feature_cols = [c for c in df.columns if c not in ignore and c != "date"]

            print(f"[{name}] Features usadas: {len(feature_cols)}")
            news_cols = [c for c in feature_cols if c.startswith("news_")]
            print(f"[{name}] Features de notícia: {len(news_cols)}")

            bundle, metrics = train_bundle(
                df=df,
                feature_cols=feature_cols,
                model_name=name,
                test_ratio=0.30,
            )

            bundle_path = models_dir / f"lgbm_{name}.joblib"
            metrics_path = models_dir / f"lgbm_{name}.metrics.json"
            errors_path = models_dir / f"lgbm_{name}.error_report.json"

            save_bundle(bundle, str(bundle_path))

            metrics_out = dict(metrics)
            error_report = metrics_out.pop("error_report", {})

            metrics_path.write_text(json.dumps(metrics_out, indent=2, ensure_ascii=False), encoding="utf-8")
            errors_path.write_text(json.dumps(error_report, indent=2, ensure_ascii=False), encoding="utf-8")

            print(f"[{name}] Arquivos salvos:")
            print(f"  - {bundle_path}")
            print(f"  - {metrics_path}")
            print(f"  - {errors_path}")

        print("\n[3/4] Treinando modelos setoriais e global...")
        global_parts: List[pd.DataFrame] = []
        for sec, parts in all_sector_frames.items():
            sec_df = pd.concat(parts, axis=0).sort_values("date").reset_index(drop=True)
            global_parts.append(sec_df)

            sec_name = sec.replace(" ", "_").upper()
            if len(sec_df) >= args.min_sector_rows:
                train_and_save(sec_name, sec_df)
            else:
                print(f"[{sec_name}] pulado: poucas linhas para modelo setorial ({len(sec_df)})")

        if global_parts:
            global_df = pd.concat(global_parts, axis=0).sort_values("date").reset_index(drop=True)
            train_and_save("GLOBAL", global_df)

    print("\n[4/4] Processo concluído com validação 70/30 + refit final.")
    print("================ FIM DO TREINO ================\n")


if __name__ == "__main__":
    main()