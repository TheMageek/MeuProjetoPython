from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class DBConfig:
    path: Path


SCHEMA_SQL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS tickers (
  ticker TEXT PRIMARY KEY,
  yf_symbol TEXT NOT NULL,
  sector TEXT,
  industry TEXT,
  last_updated TEXT
);

CREATE TABLE IF NOT EXISTS prices (
  ticker TEXT NOT NULL,
  date TEXT NOT NULL,
  open REAL,
  high REAL,
  low REAL,
  close REAL,
  volume REAL,
  source TEXT NOT NULL,
  PRIMARY KEY (ticker, date)
);

CREATE TABLE IF NOT EXISTS fundamentals (
  ticker TEXT NOT NULL,
  asof TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  source TEXT NOT NULL,
  PRIMARY KEY (ticker, asof)
);

CREATE TABLE IF NOT EXISTS macro (
  date TEXT PRIMARY KEY,
  selic REAL,
  ipca REAL,
  usd_brl REAL,
  ibov_close REAL,
  brent_close REAL,
  spx_close REAL,
  vix_close REAL,
  source TEXT NOT NULL
);
"""


def connect(db: DBConfig) -> sqlite3.Connection:
    db.path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db.path))
    con.execute("PRAGMA foreign_keys=ON;")
    return con


def init_db(db: DBConfig) -> None:
    con = connect(db)
    try:
        con.executescript(SCHEMA_SQL)
        con.commit()
    finally:
        con.close()


def executemany(con: sqlite3.Connection, sql: str, rows: Iterable[tuple]) -> None:
    con.executemany(sql, list(rows))


def upsert_ticker(
    con: sqlite3.Connection,
    ticker: str,
    yf_symbol: str,
    sector: Optional[str],
    industry: Optional[str],
    last_updated: str,
) -> None:
    con.execute(
        """
        INSERT INTO tickers (ticker, yf_symbol, sector, industry, last_updated)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(ticker) DO UPDATE SET
          yf_symbol=excluded.yf_symbol,
          sector=excluded.sector,
          industry=excluded.industry,
          last_updated=excluded.last_updated
        """,
        (ticker, yf_symbol, sector, industry, last_updated),
    )