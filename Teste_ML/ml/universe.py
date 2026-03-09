# File: ml/universe.py
from __future__ import annotations

import argparse
import base64
import json
import re
import time
from typing import List, Optional

import requests

# Página pública da composição do IBOV (fallback por regex no HTML)
B3_IBOV_PAGE = (
    "https://www.b3.com.br/pt_br/market-data-e-indices/indices/indices-amplos/"
    "indice-ibovespa-ibovespa-composicao-da-carteira.htm"
)

# Endpoint “indexCall” (rápido quando está acessível) — pode falhar com 520 em alguns momentos
B3_PORTFOLIO_ENDPOINT = (
    "https://sistemaswebb3-listados.b3.com.br/indexProxy/indexCall/GetPortfolioDay/{payload_b64}"
)

_TICKER_RE = re.compile(r"\b[A-Z]{4}\d{1,2}\b")
_CARTEIRA_RE = re.compile(r"Carteira do Dia", re.IGNORECASE)


def _b64_payload(payload: dict) -> str:
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return base64.b64encode(raw).decode("ascii")


def _headers(referer: Optional[str] = None) -> dict:
    h = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0 Safari/537.36"
        ),
        "Accept": "application/json,text/html;q=0.9,*/*;q=0.8",
        "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
        "Connection": "keep-alive",
    }
    if referer:
        h["Referer"] = referer
        h["Origin"] = "https://www.b3.com.br"
    return h


def _dedupe_preserve(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _fetch_json_endpoint(timeout: int, page_size: int, max_retries: int) -> List[str]:
    tickers: List[str] = []
    page = 1

    while True:
        payload = {
            "language": "pt-br",
            "pageNumber": page,
            "pageSize": page_size,
            "index": "IBOV",
            "segment": "1",
        }
        url = B3_PORTFOLIO_ENDPOINT.format(payload_b64=_b64_payload(payload))

        last_exc: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                r = requests.get(url, timeout=timeout, headers=_headers(referer=B3_IBOV_PAGE))
                # 520/502/503 às vezes acontecem; trata como retry
                if r.status_code in (520, 502, 503, 504, 429, 403):
                    time.sleep(0.8 * (attempt + 1))
                    continue
                r.raise_for_status()
                data = r.json()
                results = data.get("results") or []
                for row in results:
                    cod = str(row.get("cod") or "").strip().upper()
                    if cod:
                        tickers.append(cod)

                page_info = data.get("page") or {}
                total_pages = int(page_info.get("totalPages") or 1)
                if page >= total_pages:
                    return _dedupe_preserve(tickers)

                page += 1
                break  # próxima página
            except Exception as e:
                last_exc = e
                time.sleep(0.8 * (attempt + 1))

        # se estourou retries dessa página, aborta e deixa o fallback agir
        if last_exc:
            return []


def _fetch_from_b3_html(timeout: int) -> List[str]:
    """
    Fallback: baixa a página pública da B3 e extrai tickers por regex.
    O HTML costuma conter um trecho “Carteira do Dia - dd/mm/aa ; AZZA3, ...”
    """
    r = requests.get(B3_IBOV_PAGE, timeout=timeout, headers=_headers())
    r.raise_for_status()
    html = r.text.upper()

    # se não achar “Carteira do Dia”, ainda assim tentamos extrair tickers (às vezes vem em outro trecho)
    if not _CARTEIRA_RE.search(html):
        # Mesmo sem a âncora, tenta regex no HTML inteiro
        found = _TICKER_RE.findall(html)
        return _dedupe_preserve(found)

    # recorte pequeno para reduzir falsos positivos
    idx = html.find("CARTEIRA DO DIA")
    snippet = html[idx : idx + 200000]  # 200 KB a partir do ponto
    found = _TICKER_RE.findall(snippet)
    return _dedupe_preserve(found)


def fetch_ibov_tickers(timeout: int = 30, page_size: int = 200, max_retries: int = 4) -> List[str]:
    """
    Robust strategy:
    1) Try JSON endpoint (fast).
    2) If it fails (520/403/etc.), fallback to parsing B3 page HTML.
    """
    tickers = _fetch_json_endpoint(timeout=timeout, page_size=page_size, max_retries=max_retries)
    if tickers:
        return tickers

    # fallback: HTML scrape
    return _fetch_from_b3_html(timeout=timeout)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ibov", action="store_true", help="Fetch IBOV tickers")
    ap.add_argument("--out", default=None, help="Output file (one ticker per line)")
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--page_size", type=int, default=200)
    ap.add_argument("--retries", type=int, default=4)
    args = ap.parse_args()

    if not args.ibov:
        raise SystemExit("Use --ibov")

    tickers = fetch_ibov_tickers(timeout=args.timeout, page_size=args.page_size, max_retries=args.retries)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for t in tickers:
                f.write(t + "\n")

    print({"count": len(tickers), "tickers_preview": tickers[:25], "out": args.out})


if __name__ == "__main__":
    main()