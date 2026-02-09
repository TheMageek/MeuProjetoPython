import statistics

def media_movel(valores,janela):
    if len(valores) < janela:
        return None
    return sum(valores[-janela:])/janela

def calcular_rsi(precos, periodo=14):
    if len(precos) < periodo +1:
        return None
    
    ganhos = []
    perdas = []
    
    for i in range (-periodo, 0):
        diferenca = precos[i] - precos[i-1]

        if diferenca >= 0:
            ganhos.append(diferenca)
        else:
            perdas.append(abs(diferenca))

    media_ganho = sum(ganhos) / periodo if ganhos else 0
    media_perda = sum(perdas) / periodo if perdas else 0

    if media_perda == 0:
        return 100
    
    rs = media_ganho/ media_perda
    return round(100 -(100/(1 + rs)),2)

def calcular_volatilidade(precos):
    if len(precos) < 2:
        return None
    
    retornos =[]
    for i in range(1, len(precos)):
        retornos.append((precos[i] - precos[i-1]) / precos[i-1])

    return round(statistics.stdev(retornos)*100,2)

def calcular_drawdonw(precos):
    topo = precos[0]
    maior_queda = 0

    for preco in precos:
        if preco>topo:
            topo=preco

        queda = (topo - preco) / topo
        maior_queda = max(maior_queda, queda)
    return round(maior_queda*100,2)

def detectar_tendencia(precos):
    mm20 = media_movel(precos,20)
    mm60 = media_movel(precos,60)

    if mm20 is None or mm60 is None:
        return "Tendencia Indefinida"
    if mm20 > mm60:
        return "Alta Tendencia"
    elif mm20 <mm60:
        return "Baixa Tendencia"
    else:
        return "Tendencia Lateral"
    
def classificar_risco(volatilidade,drawdonw):
    if volatilidade is None or drawdonw is None:
        return "Risco Indefinido"
    if volatilidade <2 and drawdonw <10:
        return "Risco Baixo"
    elif volatilidade <4 and drawdonw < 20:
        return "Risco Medio"
    else:
        return "Risco Alto"
    
def analisar_indicadores(historico):
    precos = [p["close"] for p in historico]

    volatilidade = calcular_volatilidade(precos)
    drawdown = calcular_drawdonw(precos)

    return {
        "tendencia": detectar_tendencia(precos),
        "rsi": calcular_rsi(precos),
        "volatilidade": volatilidade,
        "drawdown": drawdown,
        "risco": classificar_risco(volatilidade, drawdown)
    }