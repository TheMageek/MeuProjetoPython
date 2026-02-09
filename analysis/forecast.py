import math

def projetar_faixa(preco_atual, volatilidade_percentual, dias=100, k=1.0):
    """
    Faixa por volatilidade (baseline sem ML).
    - volatilidade_percentual: vol diária em %
    - dias: horizonte
    - k: multiplicador (Aula 10 vai calibrar isso)
    """
    if preco_atual is None or volatilidade_percentual is None:
        return None

    vol = volatilidade_percentual / 100.0
    fator_tempo = math.sqrt(dias)

    variacao = preco_atual * (k * vol) * fator_tempo
    minimo = preco_atual - variacao
    maximo = preco_atual + variacao

    return {
        "min": round(minimo, 2),
        "max": round(maximo, 2),
        "k": k,
        "dias": dias,
        "vol_diaria_pct": float(volatilidade_percentual),
    }
