from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, List
from urllib.parse import quote_plus
import re

import pandas as pd
import requests
import yfinance as yf
import feedparser

BCB_SGS_URL = (
    "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados?formato=json&dataInicial={start}&dataFinal={end}"
)

SGS = {
    "selic": 432,
    "ipca": 433,
    "usd_brl": 1,
}


@dataclass(frozen=True)
class BrapiAuth:
    token: Optional[str] = None
    bearer: Optional[str] = None

    def headers(self) -> Dict[str, str]:
        if self.bearer:
            return {"Authorization": f"Bearer {self.bearer}"}
        return {}


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(str(x).replace(",", "."))
    except Exception:
        return None


def yf_symbol_b3(ticker: str) -> str:
    t = (ticker or "").strip().upper()
    if not t:
        return t
    if t.endswith(".SA"):
        return t
    if len(t) in (5, 6) and t[-1].isdigit():
        return f"{t}.SA"
    return t


def _flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    # yfinance pode retornar MultiIndex ou tuplas: ('Close','^BVSP')
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c[0]) for c in df.columns.to_list()]
    else:
        df.columns = [str(c[0]) if isinstance(c, tuple) and len(c) > 0 else str(c) for c in df.columns.to_list()]
    return df


def fetch_ohlcv_yfinance(ticker: str, range_: str = "5y", interval: str = "1d") -> pd.DataFrame:
    sym = yf_symbol_b3(ticker)
    df = yf.download(
        sym,
        period=range_,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
        group_by="column",
    )
    if df is None or df.empty:
        return pd.DataFrame()

    df = _flatten_yf_columns(df)
    df = df.reset_index()
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]

    if "adj_close" in df.columns and "close" not in df.columns:
        df = df.rename(columns={"adj_close": "close"})

    if "date" not in df.columns:
        if "datetime" in df.columns:
            df = df.rename(columns={"datetime": "date"})
        else:
            df = df.rename(columns={df.columns[0]: "date"})

    needed = ["date", "open", "high", "low", "close", "volume"]
    if any(c not in df.columns for c in needed):
        return pd.DataFrame()

    out = df[needed].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.date.astype(str)
    return out.dropna(subset=["close"]).reset_index(drop=True)


def fetch_ohlcv_brapi(ticker: str, auth: BrapiAuth, range_: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """
    Se brapi negar (401/403) ou rate-limit (429), retorna DF vazio e o pipeline cai pro yfinance.
    """
    url = f"https://brapi.dev/api/quote/{ticker}"
    params: Dict[str, Any] = {"range": range_, "interval": interval}
    if auth.token:
        params["token"] = auth.token

    try:
        r = requests.get(url, params=params, headers=auth.headers(), timeout=30)
        if r.status_code in (401, 403, 429):
            return pd.DataFrame()
        r.raise_for_status()
        data = r.json()
    except Exception:
        return pd.DataFrame()

    res = (data.get("results") or [{}])[0]
    hist = res.get("historicalDataPrice") or res.get("historicalData") or []
    if not hist:
        return pd.DataFrame()

    rows = []
    for it in hist:
        dt = it.get("date")
        if dt is None:
            continue
        try:
            d = pd.to_datetime(dt, unit="s").date()
        except Exception:
            d = pd.to_datetime(dt).date()
        rows.append(
            {
                "date": str(d),
                "open": _safe_float(it.get("open")),
                "high": _safe_float(it.get("high")),
                "low": _safe_float(it.get("low")),
                "close": _safe_float(it.get("close")),
                "volume": _safe_float(it.get("volume")),
            }
        )

    return pd.DataFrame(rows).dropna(subset=["close"]).sort_values("date").reset_index(drop=True)


def fetch_sector_yfinance(ticker: str) -> tuple[Optional[str], Optional[str]]:
    sym = yf_symbol_b3(ticker)
    try:
        info = yf.Ticker(sym).info or {}
        return (info.get("sector"), info.get("industry"))
    except Exception:
        return (None, None)


def fetch_fundamentals_brapi(ticker: str, auth: BrapiAuth) -> Dict[str, Any]:
    """
    Também não derruba o treino se brapi negar.
    """
    url = f"https://brapi.dev/api/quote/{ticker}"
    params: Dict[str, Any] = {
        "modules": "summaryProfile,summaryDetail,defaultKeyStatistics,financialData,incomeStatementHistory,balanceSheetHistory",
    }
    if auth.token:
        params["token"] = auth.token

    try:
        r = requests.get(url, params=params, headers=auth.headers(), timeout=30)
        if r.status_code in (401, 403, 429):
            return {"results": [{}]}
        r.raise_for_status()
        return r.json()
    except Exception:
        return {"results": [{}]}


def fetch_sgs_series(code: int, start_ddmmyyyy: str, end_ddmmyyyy: str) -> pd.DataFrame:
    url = BCB_SGS_URL.format(code=code, start=start_ddmmyyyy, end=end_ddmmyyyy)
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    arr = r.json() or []
    rows = [{"date": pd.to_datetime(it["data"], dayfirst=True).date(), "value": _safe_float(it["valor"])} for it in arr]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = df["date"].astype(str)
    return df.sort_values("date").reset_index(drop=True)


def fetch_macro_yfinance(start: str, end: str) -> pd.DataFrame:
    # proxies globais
    tickers = {"brent_close": "BZ=F", "spx_close": "^GSPC", "vix_close": "^VIX"}
    frames = []

    for col, sym in tickers.items():
        df = yf.download(
            sym,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
            group_by="column",
        )
        if df is None or df.empty:
            continue

        df = _flatten_yf_columns(df)
        df = df.reset_index()
        df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]

        if "date" not in df.columns:
            if "datetime" in df.columns:
                df = df.rename(columns={"datetime": "date"})
            else:
                df = df.rename(columns={df.columns[0]: "date"})

        if "close" not in df.columns and "adj_close" in df.columns:
            df = df.rename(columns={"adj_close": "close"})
        if "close" not in df.columns:
            continue

        df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
        frames.append(df[["date", "close"]].rename(columns={"close": col}))

    if not frames:
        return pd.DataFrame()

    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on="date", how="outer")

    return out.sort_values("date").reset_index(drop=True)

POS_WORDS = {
    "alta", "subida", "lucro", "crescimento", "recorde", "forte", "compra",
    "positivo", "ganho", "valoriza", "valorização", "expansão", "supera",
    "melhora", "avance", "otimista", "aprova", "reduz dívida"
}

NEG_WORDS = {
    "queda", "baixa", "prejuízo", "fraco", "venda", "negativo", "perda",
    "desvalorização", "desaba", "risco", "crise", "piora", "corte",
    "processo", "multa", "dívida", "downgrade", "pressão"
}


def _simple_sentiment_pt(text: str) -> float:
    s = (text or "").lower()
    tokens = re.findall(r"\w+", s, flags=re.UNICODE)
    if not tokens:
        return 0.0

    pos = sum(1 for t in tokens if t in POS_WORDS)
    neg = sum(1 for t in tokens if t in NEG_WORDS)

    score = (pos - neg) / max(len(tokens), 1)
    return float(score)


def _google_news_rss(query: str, lang: str = "pt-BR", country: str = "BR") -> str:
    q = quote_plus(query)
    return f"https://news.google.com/rss/search?q={q}&hl={lang}&gl={country}&ceid={country}:pt-419"


def fetch_news_daily(
    ticker: str,
    company_name: Optional[str] = None,
    sector: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Retorna features diárias de notícia para o ativo.
    """
    queries: List[str] = [ticker]
    if company_name:
        queries.append(company_name)
    if sector:
        queries.append(f"{company_name or ticker} {sector}")

    rows: List[Dict[str, Any]] = []

    for q in queries:
        try:
            feed = feedparser.parse(_google_news_rss(q))
        except Exception:
            continue

        for entry in getattr(feed, "entries", []):
            title = str(getattr(entry, "title", "") or "")
            summary = str(getattr(entry, "summary", "") or "")
            text = f"{title}. {summary}".strip()

            published = getattr(entry, "published", None) or getattr(entry, "updated", None)
            if not published:
                continue

            try:
                d = pd.to_datetime(published, utc=True).date()
            except Exception:
                continue

            ds = str(d)
            if start and ds < start:
                continue
            if end and ds > end:
                continue

            rows.append(
                {
                    "date": ds,
                    "headline": title,
                    "sent_score": _simple_sentiment_pt(text),
                    "is_pos": 1 if _simple_sentiment_pt(text) > 0 else 0,
                    "is_neg": 1 if _simple_sentiment_pt(text) < 0 else 0,
                }
            )

    if not rows:
        return pd.DataFrame(columns=[
            "date",
            "news_count",
            "news_sent_mean",
            "news_sent_sum",
            "news_pos_count",
            "news_neg_count",
            "news_burst_3d",
            "news_burst_7d",
            "news_sent_3d",
            "news_sent_7d",
        ])

    df = pd.DataFrame(rows).drop_duplicates(subset=["date", "headline"]).sort_values("date")

    daily = (
        df.groupby("date", as_index=False)
        .agg(
            news_count=("headline", "count"),
            news_sent_mean=("sent_score", "mean"),
            news_sent_sum=("sent_score", "sum"),
            news_pos_count=("is_pos", "sum"),
            news_neg_count=("is_neg", "sum"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    daily["news_burst_3d"] = daily["news_count"].rolling(3, min_periods=1).sum()
    daily["news_burst_7d"] = daily["news_count"].rolling(7, min_periods=1).sum()
    daily["news_sent_3d"] = daily["news_sent_mean"].rolling(3, min_periods=1).mean()
    daily["news_sent_7d"] = daily["news_sent_mean"].rolling(7, min_periods=1).mean()

    # anti-leakage: usa apenas o passado conhecido
    news_cols = [c for c in daily.columns if c != "date"]
    daily[news_cols] = daily[news_cols].shift(1)

    return daily.fillna(0.0)