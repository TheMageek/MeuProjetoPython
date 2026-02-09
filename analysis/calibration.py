from analysis.backtest import backtest_faixa


def calibrar_k(
    historico,
    dias=10,
    metodo="ewma",
    target_coverage=0.80,
    k_min=0.6,
    k_max=3.0,
    k_step=0.05,
):
    resultados = []

    k = k_min
    while k <= k_max:
        bt = backtest_faixa(
            historico=historico,
            dias=dias,
            k=round(k, 3),
            metodo=metodo,
        )
        if bt is None:
            k += k_step
            continue

        resultados.append(bt)

        if bt["coverage"] >= target_coverage * 100:
            return {
                "k_otimo": bt["k"],
                "resultado": bt,
                "tentativas": resultados,
            }

        k += k_step

    return {
        "k_otimo": None,
        "resultado": None,
        "tentativas": resultados,
    }
