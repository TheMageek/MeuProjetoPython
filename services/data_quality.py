# services/data_quality.py
from __future__ import annotations

import numpy as np
import pandas as pd

CANON_COLS = ["date", "open", "high", "low", "close", "volume"]


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        lvl0 = list(df.columns.get_level_values(0))
        lvl1 = list(df.columns.get_level_values(1))

        ohlcv = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
        if any(x in ohlcv for x in lvl0):
            df.columns = df.columns.get_level_values(0)
        else:
            df.columns = df.columns.get_level_values(1)

    return df



def normalize_history_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza histórico vindo do yfinance para o formato canônico:
    date, open, high, low, close, volume

    Robusto contra:
    - MultiIndex
    - nomes diferentes (Close/close, Adj Close/AdjClose)
    - retorno parcial
    """
    # Se vier None/empty, já devolve canônico vazio
    if df is None or getattr(df, "empty", True):
        return pd.DataFrame(columns=CANON_COLS)

    df = df.copy()
    df = _flatten_columns(df)

    # Index -> coluna date
    df = df.reset_index()

    # Descobre coluna de data
    if "Date" in df.columns and "date" not in df.columns:
        df.rename(columns={"Date": "date"}, inplace=True)
    elif "Datetime" in df.columns and "date" not in df.columns:
        df.rename(columns={"Datetime": "date"}, inplace=True)

    # Normaliza nomes (aceita variações)
    rename_map = {
        "Open": "open",
        "open": "open",
        "High": "high",
        "high": "high",
        "Low": "low",
        "low": "low",
        "Close": "close_raw",
        "close": "close_raw",
        "Adj Close": "adj_close",
        "AdjClose": "adj_close",
        "adj close": "adj_close",
        "Volume": "volume",
        "volume": "volume",
    }
    df.rename(columns={c: rename_map.get(c, c) for c in df.columns}, inplace=True)

    # date obrigatório
    if "date" not in df.columns:
        # sem date não dá pra usar
        return pd.DataFrame(columns=CANON_COLS)

    # Decide close final (Aula 7: prioriza adj_close)
    if "adj_close" in df.columns:
        df["close"] = pd.to_numeric(df["adj_close"], errors="coerce")
    elif "close_raw" in df.columns:
        df["close"] = pd.to_numeric(df["close_raw"], errors="coerce")
    else:
        # não veio close de jeito nenhum
        return pd.DataFrame(columns=CANON_COLS)

    # Converte OHLC/volume se existirem
    for c in ["open", "high", "low", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            # garante coluna existir (mesmo vazia) para manter contrato
            df[c] = np.nan

    # Date sem timezone
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.dropna(subset=["date"])

    # Ordena e remove duplicadas
    df = df.sort_values("date")
    df = df.drop_duplicates(subset=["date"], keep="last")

    # Agora sim remove NaN de close (close existe garantido aqui)
    df = df.dropna(subset=["close"])

    # Mantém contrato canônico
    df = df[CANON_COLS]

    return df.reset_index(drop=True)


def add_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    df = df.copy()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df = df.dropna(subset=["log_return"])
    return df.reset_index(drop=True)


def validate_history(df: pd.DataFrame, min_rows: int = 60) -> tuple[bool, str | None]:
    if df is None or df.empty:
        return False, "HISTORY_EMPTY"
    if "close" not in df.columns:
        return False, "HISTORY_NO_CLOSE"
    if df["close"].isna().any():
        return False, "CLOSE_HAS_NAN"
    if len(df) < min_rows:
        return False, "HISTORY_TOO_SHORT"
    return True, None
