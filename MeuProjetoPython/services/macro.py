# services/macro.py
from __future__ import annotations

import time
import requests
from typing import Dict, Any, Optional

BCB_BASE = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados/ultimos/1?formato=json"

# Códigos SGS (BCB)
SGS = {
    "selic_meta": 432,     # Meta Selic (Copom) :contentReference[oaicite:3]{index=3}
    "ipca": 433,           # IPCA :contentReference[oaicite:4]{index=4}
    "pib": 22099,          # PIB :contentReference[oaicite:5]{index=5}
    "desemprego": 24369,   # Taxa de desocupação (PNADC) :contentReference[oaicite:6]{index=6}
    "usd_brl": 1,          # Dólar comercial (venda) :contentReference[oaicite:7]{index=7}
}

_cache: Dict[str, Any] = {"ts": 0, "data": None}

def _safe_float(v: str) -> Optional[float]:
    try:
        return float(str(v).replace(",", "."))
    except Exception:
        return None

def _get_bcb_last(code: int, timeout: int = 15) -> Dict[str, Any]:
    url = BCB_BASE.format(code=code)
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    arr = r.json()
    if not arr:
        return {"value": None, "date": None, "source": "BCB/SGS"}
    item = arr[-1]
    # Ex.: {"data":"23/02/2026","valor":"4.50"}
    return {"value": _safe_float(item.get("valor")), "date": item.get("data"), "source": "BCB/SGS"}

def _get_brapi_quote(ticker: str, token: Optional[str] = None, timeout: int = 15) -> Dict[str, Any]:
    # token opcional; se você já usa BRAPI_KEY, pode reaproveitar.
    url = f"https://brapi.dev/api/quote/{ticker}"
    params = {}
    if token:
        params["token"] = token
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    res = (data.get("results") or [{}])[0]
    return {
        "value": res.get("regularMarketPrice"),
        "date": None,
        "source": "brapi",
        "change_pct": res.get("regularMarketChangePercent"),
    }

def get_macro_cards(brapi_token: Optional[str] = None) -> Dict[str, Any]:
    """
    Retorna um dict pronto pro template.
    Faz cache simples (10 min) pra não bater na API toda hora.
    """
    now = time.time()
    if _cache["data"] is not None and (now - _cache["ts"] < 600):
        return _cache["data"]

    cards = {}

    # BCB
    cards["PIB Brasil"] = _get_bcb_last(SGS["pib"])
    cards["Taxa Selic"] = _get_bcb_last(SGS["selic_meta"])
    cards["Inflação (IPCA)"] = _get_bcb_last(SGS["ipca"])
    cards["Desemprego"] = _get_bcb_last(SGS["desemprego"])
    cards["Dólar (USD/BRL)"] = _get_bcb_last(SGS["usd_brl"])

    # Ibovespa via brapi
    try:
        cards["Ibovespa"] = _get_brapi_quote("^BVSP", token=brapi_token)
    except Exception:
        cards["Ibovespa"] = {"value": None, "date": None, "source": "brapi", "change_pct": None}

    _cache["ts"] = now
    _cache["data"] = cards
    return cards