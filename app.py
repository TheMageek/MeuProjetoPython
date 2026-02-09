from flask import Flask, render_template, request
import requests
import os
from datetime import datetime
from config import BRAPI_KEY
from analysis.indicators import analisar_indicadores
from analysis.forecast import projetar_faixa



app = Flask(__name__)

@app.get("/")
def index():
    return render_template("index.html")

def get_stock_price(symbol):
    url = "https://brapi.dev/api/quote/" + symbol
    params = {"token": BRAPI_KEY}

    r = requests.get(url, params=params, timeout=20)
    data = r.json()

    try:
        
        return float(data["results"][0]["regularMarketPrice"])
    except Exception as e:
        print("BRAPI PRICE ERROR:", e, data)
        return None

    
def get_stock_history(symbol, limit=120):
    url = "https://brapi.dev/api/quote/" + symbol
    params = {
        "token": BRAPI_KEY,
        "range": "6mo",          
        "interval": "1d"        
    }

    r = requests.get(url, params=params, timeout=20)
    data = r.json()

    try:
        candles = data["results"][0]["historicalDataPrice"]
    except Exception as e:
        print("BRAPI HISTORY ERROR:", e, data)
        return []

    items = []
    for c in candles[-limit:]:
        timestamp = c["date"]
        date_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")

        close_price = float(c["close"])
        items.append({
            "date": date_str,
            "close": close_price
    })

    return items


@app.post("/analyze")
def analyze():
    ticker = (request.form.get("ticker") or "").strip().upper()

    price = get_stock_price(ticker)
    history = get_stock_history(ticker, limit=120)
    indicadores = analisar_indicadores(history)
    dias = int(request.form.get("dias", 10))
    print("DIAS RECEBIDO:", dias, "FORM:", dict(request.form))




    print("HISTORY LEN:", len(history))
    if history:
        print("HISTORY FIRST:", history[0])

    faixa = projetar_faixa(price, indicadores["volatilidade"], dias=dias)

    return render_template(
    "dashboard.html",
    ticker=ticker,
    price=price,
    history=history,
    indicadores=indicadores,
    faixa=faixa,
    dias=dias
)


if __name__ == "__main__":
    print("INICIANDO FLASK......")
    app.run(debug=True, host="0.0.0.0", port=5000)