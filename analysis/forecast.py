import math

def projetar_faixa(preco_atual, volatilidade_percentual, dias=100):

    if preco_atual is None or volatilidade_percentual is None:
        return None

    vol = volatilidade_percentual / 100

    fator_tempo = math.sqrt(dias)

    variacao = preco_atual * vol * fator_tempo

    minimo = preco_atual - variacao
    maximo = preco_atual + variacao

    return {
        "min": round(minimo, 2),
        "max": round(maximo, 2)
    }
