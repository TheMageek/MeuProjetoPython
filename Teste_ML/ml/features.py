from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def realized_vol(logret: pd.Series, window: int = 10) -> pd.Series:
    return logret.rolling(window, min_periods=window).std()


def _raw(x: Any) -> Any:
    if isinstance(x, dict) and "raw" in x:
        return x.get("raw")
    return x


def parse_brapi_fundamentals(payload: Dict[str, Any]) -> Dict[str, Optional[float]]:
    res = (payload.get("results") or [{}])[0]

    def get_path(*keys: str) -> Optional[float]:
        cur: Any = res
        for k in keys:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(k)
        cur = _raw(cur)
        try:
            if cur is None:
                return None
            return float(cur)
        except Exception:
            return None

    return {
        "pl": get_path("defaultKeyStatistics", "trailingPE"),
        "pvp": get_path("defaultKeyStatistics", "priceToBook"),
        "ev_ebitda": get_path("defaultKeyStatistics", "enterpriseToEbitda"),
        "roe": get_path("financialData", "returnOnEquity"),
        "roa": get_path("financialData", "returnOnAssets"),
        "margem_ebitda": get_path("financialData", "ebitdaMargins"),
        "margem_liquida": get_path("financialData", "profitMargins"),
        "div_yield": get_path("summaryDetail", "dividendYield"),
        "peg": get_path("defaultKeyStatistics", "pegRatio"),
    }


def build_feature_frame(
    ohlcv: pd.DataFrame,
    fundamentals_daily: pd.DataFrame,
    macro_daily: pd.DataFrame,
    news_daily: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    df = ohlcv.copy().sort_values("date")
    df["date"] = df["date"].astype(str)

    df["logret"] = np.log(df["close"] / df["close"].shift(1))
    df["ret_1"] = df["close"].pct_change(1)
    df["ret_5"] = df["close"].pct_change(5)
    df["ret_10"] = df["close"].pct_change(10)

    df["ma_10"] = df["close"].rolling(10, min_periods=10).mean()
    df["ma_20"] = df["close"].rolling(20, min_periods=20).mean()
    df["ma_60"] = df["close"].rolling(60, min_periods=60).mean()
    df["ma_ratio_10_20"] = df["ma_10"] / df["ma_20"]

    df["rsi_14"] = _rsi(df["close"], 14)
    df["atr_14"] = _atr(df, 14)
    df["atr_pct_14"] = df["atr_14"] / df["close"]

    df["vol_10"] = realized_vol(df["logret"], 10)
    df["vol_20"] = realized_vol(df["logret"], 20)

    df = df.merge(macro_daily, on="date", how="left")
    df = df.merge(fundamentals_daily, on="date", how="left")

    if news_daily is not None and not news_daily.empty:
        news_daily = news_daily.copy()
        if "date" in news_daily.columns:
            news_daily["date"] = news_daily["date"].astype(str)
        df = df.merge(news_daily, on="date", how="left")

    macro_daily = macro_daily.copy()
    if "date" in macro_daily.columns:
        macro_daily["date"] = macro_daily["date"].astype(str)

    fundamentals_daily = fundamentals_daily.copy()
    if "date" in fundamentals_daily.columns:
        fundamentals_daily["date"] = fundamentals_daily["date"].astype(str)

    df = df.merge(macro_daily, on="date", how="left")
    df = df.merge(fundamentals_daily, on="date", how="left")
    df = df.sort_values("date").ffill()

    keep = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "logret",
        "ret_1",
        "ret_5",
        "ret_10",
        "ma_10",
        "ma_20",
        "ma_60",
        "ma_ratio_10_20",
        "rsi_14",
        "atr_pct_14",
        "vol_10",
        "vol_20",
        "selic",
        "ipca",
        "usd_brl",
        "ibov_close",
        "brent_close",
        "spx_close",
        "vix_close",
        "pl",
        "pvp",
        "ev_ebitda",
        "roe",
        "roa",
        "margem_ebitda",
        "margem_liquida",
        "div_yield",
        "peg",
        "news_count",
        "news_sent_mean",
        "news_sent_sum",
        "news_pos_count",
        "news_neg_count",
        "news_burst_3d",
        "news_burst_7d",
        "news_sent_3d",
        "news_sent_7d",
    ]

    out = df[[c for c in keep if c in df.columns]].copy()
    out = out.dropna(subset=["close", "high", "low"]).reset_index(drop=True)

    for c in out.columns:
        if c != "date":
            out[c] = pd.to_numeric(out[c], errors="coerce")

    for c in out.columns:
        if c != "date":
            out[c] = pd.to_numeric(out[c], errors="coerce")

    news_cols = [c for c in out.columns if c.startswith("news_")]
    if news_cols:
        out[news_cols] = out[news_cols].fillna(0.0)

    return out