def ticker_yfinance(ticker: str) -> str:
    t = (ticker or "").strip().upper()
    if not t:
        return t
    if t.endswith(".SA"):
        return t
    if len(t) in (5, 6) and t[-1].isdigit():
        return t + ".SA"
    return t
