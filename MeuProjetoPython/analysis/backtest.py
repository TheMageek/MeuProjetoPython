import math

from analysis.indicators import (
    calcular_volatilidade_log_std,
    calcular_volatilidade_ewma,
    calcular_volatilidade_robusta,
)


def _faixa(preco, vol_pct, dias, k=1.0):
    vol = vol_pct / 100.0
    delta = preco * k * vol * math.sqrt(dias)
    return preco - delta, preco + delta


def backtest_faixa(
    historico,
    dias=10,
    k=1.0,
    metodo="ewma",
    min_hist=60,
):
    """
    Walk-forward backtest.
    historico: lista [{date, close}]
    metodo: 'std' | 'ewma' | 'rob'
    """
    closes = [h["close"] for h in historico]

    acertos = 0
    larguras = []
    total = 0

    for t in range(min_hist, len(closes) - dias):
        passado = closes[:t]
        preco_hoje = closes[t]
        preco_futuro = closes[t + dias]

        if metodo == "std":
            vol = calcular_volatilidade_log_std(passado)
        elif metodo == "rob":
            vol = calcular_volatilidade_robusta(passado)
        else:
            vol = calcular_volatilidade_ewma(passado)

        if vol is None:
            continue

        minimo, maximo = _faixa(preco_hoje, vol, dias, k=k)
        largura = maximo - minimo

        larguras.append(largura)
        total += 1

        if minimo <= preco_futuro <= maximo:
            acertos += 1

    if total == 0:
        return None

    coverage = acertos / total
    largura_media = sum(larguras) / len(larguras)

    return {
        "metodo": metodo,
        "dias": dias,
        "k": k,
        "total_testes": total,
        "coverage": round(coverage * 100, 2),
        "largura_media": round(largura_media, 2),
        "sharpness": round(largura_media / coverage, 2) if coverage > 0 else None,
    }
