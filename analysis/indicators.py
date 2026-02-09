import statistics
import math


def media_movel(valores, janela):
    if len(valores) < janela:
        return None
    return sum(valores[-janela:]) / janela


def calcular_rsi(precos, periodo=14):
    if len(precos) < periodo + 1:
        return None

    ganhos = []
    perdas = []

    for i in range(-periodo, 0):
        diferenca = precos[i] - precos[i - 1]

        if diferenca >= 0:
            ganhos.append(diferenca)
        else:
            perdas.append(abs(diferenca))

    media_ganho = sum(ganhos) / periodo if ganhos else 0
    media_perda = sum(perdas) / periodo if perdas else 0

    if media_perda == 0:
        return 100

    rs = media_ganho / media_perda
    return round(100 - (100 / (1 + rs)), 2)


def _log_returns(precos):
    """Aula 7/8: base de cálculo = log-return."""
    rets = []
    for i in range(1, len(precos)):
        p0 = precos[i - 1]
        p1 = precos[i]
        if p0 is None or p1 is None or p0 <= 0 or p1 <= 0:
            continue
        rets.append(math.log(p1 / p0))
    return rets


def calcular_volatilidade_log_std(precos):
    """
    Volatilidade simples: desvio padrão dos log-returns (diário) em %.
    (Seu modelo antigo, mas já na base correta da Aula 7.)
    """
    rets = _log_returns(precos)
    if len(rets) < 2:
        return None
    return round(statistics.stdev(rets) * 100, 2)


def calcular_volatilidade_ewma(precos, lam=0.94):
    """
    EWMA (RiskMetrics): variância recursiva.
    lam = 0.94 é comum para dados diários (mais “realista” que std fixo).
    Retorna vol diária em %.
    """
    rets = _log_returns(precos)
    if len(rets) < 2:
        return None

    # inicializa com variância amostral
    var = statistics.pvariance(rets)
    for r in rets[1:]:
        var = lam * var + (1 - lam) * (r * r)

    vol = math.sqrt(var)
    return round(vol * 100, 2)


def calcular_volatilidade_robusta(precos, clip_k=3.0):
    """
    Volatilidade robusta: winsoriza outliers usando MAD.
    - Calcula log-returns
    - Encontra mediana e MAD
    - Clipa retornos fora de mediana ± clip_k * sigma_robusta
    - Usa stdev no conjunto clipado
    Retorna vol diária em %.
    """
    rets = _log_returns(precos)
    if len(rets) < 10:
        return None  # robustez pede um pouco mais de dados

    med = statistics.median(rets)
    abs_dev = [abs(r - med) for r in rets]
    mad = statistics.median(abs_dev)

    # MAD ~ sigma: 1.4826 * MAD (para normal)
    sigma = 1.4826 * mad if mad and mad > 0 else None
    if sigma is None:
        # fallback: stdev normal
        return round(statistics.stdev(rets) * 100, 2)

    low = med - clip_k * sigma
    high = med + clip_k * sigma

    clipped = [min(max(r, low), high) for r in rets]
    if len(clipped) < 2:
        return None

    return round(statistics.stdev(clipped) * 100, 2)


def calcular_drawdonw(precos):
    topo = precos[0]
    maior_queda = 0

    for preco in precos:
        if preco > topo:
            topo = preco

        queda = (topo - preco) / topo
        maior_queda = max(maior_queda, queda)
    return round(maior_queda * 100, 2)


def detectar_tendencia(precos):
    mm20 = media_movel(precos, 20)
    mm60 = media_movel(precos, 60)

    if mm20 is None or mm60 is None:
        return "Tendencia Indefinida"
    if mm20 > mm60:
        return "Alta Tendencia"
    elif mm20 < mm60:
        return "Baixa Tendencia"
    else:
        return "Tendencia Lateral"


def classificar_risco(volatilidade, drawdonw):
    if volatilidade is None or drawdonw is None:
        return "Risco Indefinido"
    if volatilidade < 2 and drawdonw < 10:
        return "Risco Baixo"
    elif volatilidade < 4 and drawdonw < 20:
        return "Risco Medio"
    else:
        return "Risco Alto"


def analisar_indicadores(historico):
    precos = [p["close"] for p in historico if p.get("close") is not None]

    vol_std = calcular_volatilidade_log_std(precos)
    vol_ewma = calcular_volatilidade_ewma(precos, lam=0.94)
    vol_rob = calcular_volatilidade_robusta(precos, clip_k=3.0)

    drawdown = calcular_drawdonw(precos)

    # Escolha padrão da Aula 8: EWMA (se existir), senão std
    vol_usada = vol_ewma if vol_ewma is not None else vol_std

    return {
        "tendencia": detectar_tendencia(precos),
        "rsi": calcular_rsi(precos),
        # mantemos compatibilidade: "volatilidade" continua existindo (agora é a escolhida)
        "volatilidade": vol_usada,
        # expõe as 3 para comparação no dashboard
        "vol_std": vol_std,
        "vol_ewma": vol_ewma,
        "vol_robusta": vol_rob,
        "drawdown": drawdown,
        "risco": classificar_risco(vol_usada, drawdown),
    }
