from __future__ import annotations

import logging
import pandas as pd
import yfinance as yf

from .ticker import ticker_yfinance
from .data_quality import normalize_history_df, add_log_returns, validate_history

logger = logging.getLogger(__name__)


def fetch_history_yf(
    ticker: str,
    period: str = "6mo",
    interval: str = "1d",
    min_rows: int = 60,
) -> tuple[list[dict], dict]:

    yf_ticker = ticker_yfinance(ticker)

    meta = {
        "ticker_in": ticker,
        "ticker_yf": yf_ticker,
        "source": "yfinance",
        "period": period,
        "interval": interval,
        "fallback_used": False,
    }

    def _download(p: str, i: str) -> pd.DataFrame:
        return yf.download(
            yf_ticker,
            period=p,
            interval=i,
            progress=False,
            auto_adjust=False, 
            threads=False,
        )

    raw = _download(period, interval)
    df = normalize_history_df(raw)

    if df.empty:
        meta["fallback_used"] = True
        raw = _download("1y", "1d")
        df = normalize_history_df(raw)

    df = add_log_returns(df)

    ok, err = validate_history(df, min_rows=min_rows)
    meta.update({
        "rows": int(len(df)) if df is not None else 0,
        "min_date": str(df["date"].iloc[0]) if ok else None,
        "max_date": str(df["date"].iloc[-1]) if ok else None,
        "quality_ok": ok,
        "quality_error": err,
    })

    logger.info("history_fetch", extra=meta)

    if not ok:
        return [], meta

    items = [{"date": str(d), "close": float(c)} for d, c in zip(df["date"], df["close"])]
    return items, meta
