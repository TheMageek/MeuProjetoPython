from flask import Flask, render_template, request

import requests
import logging
import time

from config import BRAPI_KEY

from analysis.calibration import calibrar_k
from analysis.backtest import backtest_faixa
from analysis.indicators import analisar_indicadores
from analysis.forecast import projetar_faixa
from services.macro import get_macro_cards
from services.yf_history import fetch_history_yf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)

app = Flask(__name__)


@app.get("/")
def index():
    macro = {}
    try:
        macro = get_macro_cards(brapi_token=BRAPI_KEY)  # reaproveita seu token
    except Exception as e:
        print("MACRO ERROR:", e)
        macro = {}

    return render_template("index.html", macro=macro)


def get_stock_price(symbol: str):
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return None

    url = "https://brapi.dev/api/quote/" + symbol
    params = {"token": BRAPI_KEY}

    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        return float(data["results"][0]["regularMarketPrice"])
    except Exception as e:
        print("BRAPI PRICE ERROR:", e)
        return None


@app.post("/analyze")
def analyze():
    ticker = (request.form.get("ticker") or "").strip().upper()
    dias = int(request.form.get("dias", 10))

    price = get_stock_price(ticker)

    history, history_meta = fetch_history_yf(
        ticker=ticker,
        period="2y",
        interval="1d",
        min_rows=60,
    )

    print("AUDIT:", {
        "endpoint": "/analyze",
        "ts": int(time.time()),
        "ticker": ticker,
        "price_source": "brapi",
        "history_source": "yfinance",
        "meta": history_meta
    })

    if not history:
        return render_template(
            "dashboard.html",
            ticker=ticker,
            price=price,
            history=[],
            indicadores={
                "tendencia": "Tendência Indefinida",
                "rsi": None,
                "volatilidade": None,
                "drawdown": None,
                "risco": "Risco Indefinido",
            },
            faixa=None,
            faixa_std=None,
            faixa_ewma=None,
            faixa_rob=None,
            dias=dias,
            error="Histórico insuficiente para análise (min 60 candles). Verifique o ticker.",
            history_meta=history_meta,
            bt_std=None,
            bt_ewma=None,
            bt_rob=None,
            k_otimo=None,
            bt_calibrado=None,
            faixa_calibrada=None,
        )

    indicadores = analisar_indicadores(history)

    # Backtests
    bt_std = backtest_faixa(history, dias=dias, metodo="std")
    bt_ewma = backtest_faixa(history, dias=dias, metodo="ewma")
    bt_rob = backtest_faixa(history, dias=dias, metodo="rob")

    # Calibração k (EWMA)
    calib = calibrar_k(
        historico=history,
        dias=dias,
        metodo="ewma",
        target_coverage=0.8,
    )
    k_otimo = calib.get("k_otimo")
    bt_calibrado = calib.get("resultado")

    # Faixas (baseline)
    faixa = projetar_faixa(price, indicadores["volatilidade"], dias=dias)

    faixa_std = (
        projetar_faixa(price, indicadores["vol_std"], dias=dias)
        if indicadores.get("vol_std") is not None else None
    )
    faixa_ewma = (
        projetar_faixa(price, indicadores["vol_ewma"], dias=dias)
        if indicadores.get("vol_ewma") is not None else None
    )
    faixa_rob = (
        projetar_faixa(price, indicadores["vol_robusta"], dias=dias)
        if indicadores.get("vol_robusta") is not None else None
    )

    faixa_calibrada = (
        projetar_faixa(price, indicadores["volatilidade"], dias=dias, k=k_otimo)
        if k_otimo is not None else None
    )

    return render_template(
        "dashboard.html",
        ticker=ticker,
        price=price,
        history=history,
        indicadores=indicadores,
        faixa=faixa,
        faixa_std=faixa_std,
        faixa_ewma=faixa_ewma,
        faixa_rob=faixa_rob,
        dias=dias,
        error=None,
        history_meta=history_meta,
        bt_std=bt_std,
        bt_ewma=bt_ewma,
        bt_rob=bt_rob,
        k_otimo=k_otimo,
        bt_calibrado=bt_calibrado,
        faixa_calibrada=faixa_calibrada,
    )


if __name__ == "__main__":
    print("INICIANDO FLASK......")
    app.run(debug=True, host="0.0.0.0", port=5000)