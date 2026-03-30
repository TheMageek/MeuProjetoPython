import warnings
warnings.filterwarnings("ignore")
import sqlite3         
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import date
from tqdm import tqdm  
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score
)
DB_PATH       = "bolsa_b3.db"   
N_SPLITS      = 10              
MIN_CONF      = 0.52            
MIN_PREGOES   = 200             
TOP_N         = 5              
CSV_SAIDA     = "previsoes_b3.csv"
MACRO_POR_SETOR = {
    "Petróleo e Gás": [
        "brent",   
        "gasoline",
        "natgas",  
        "dolar",   
        "ibov",    
        "vix",     
    ],
    "Mineração e Siderurgia": [
        "minerio", 
        "cobre",   
        "aco",     
        "dolar",
        "ibov",
        "vix",
    ],
    "Financeiro e Bancos": [
        "dolar",
        "ibov",
        "sp500",   
        "vix",
        "t2y",     
        "t10y",    
    ],
    "Energia Elétrica": [
        "ibov",
        "dolar",
        "natgas",  
        "ouro",    
    ],
    "Varejo e Consumo": [
        "ibov",
        "dolar",
        "t2y",     
        "ouro",    
    ],
    "Alimentos e Bebidas": [
        "dolar",
        "ibov",
        "soja",    
        "milho",   
        "acucar",  
        "cafe",    
        "trigo",   
    ],
    "Saúde e Farmácia": [
        "ibov",
        "dolar",
        "euro",    
        "xlv",     
    ],
    "Construção e Imóveis": [
        "ibov",
        "dolar",
        "t10y",    
        "aco",     
        "ouro",    
    ],
    "Telecomunicações": [
        "ibov",
        "dolar",
        "vix",
        "t10y",    
    ],
    "Tecnologia": [
        "ibov",
        "nasdaq",  
        "sp500",
        "dolar",
        "vix",
        "t10y",    
    ],
    "Logística e Transporte": [
        "ibov",
        "dolar",
        "brent",
        "diesel",  
        "bdi",     
    ],
    "Agronegócio": [
        "dolar",
        "soja",
        "milho",
        "ibov",
        "algodao", 
        "cafe",    
        "trigo",   
    ],
    "Papel e Celulose": [
        "dolar",
        "ibov",
        "euro",    
        "yuan",    
    ],
    "Saneamento": [
        "ibov",
        "dolar",
        "t10y",    
        "ouro",    
    ],
    "Shopping e Imóveis Comerciais": [
        "ibov",
        "dolar",
        "t10y",    
        "xlre",    
    ],
}
SCHEMA_PREVISOES = 
def inicializar_tabela_previsoes():
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA_PREVISOES)
    conn.commit()
    conn.close()
def carregar_dados(ticker, setor, conn):
    df_p = pd.read_sql_query(, conn, params=(ticker,))
    if len(df_p) < MIN_PREGOES:
        return None, []
    df_p["data"] = pd.to_datetime(df_p["data"])
    df_p = df_p.set_index("data")
    df_i = pd.read_sql_query(, conn, params=(ticker,))
    df_i["data"] = pd.to_datetime(df_i["data"])
    df_i = df_i.set_index("data")
    df = df_p.join(df_i, how="left")
    close = df["Close"]
    ret   = close.pct_change()
    for lag in [1, 2, 3, 5]:
        df[f"ret_lag{lag}"] = ret.shift(lag)
    df["rsi_9"]  = _rsi(close, 9)
    df["rsi_21"] = _rsi(close, 21)
    df["vol_5"]        = ret.rolling(5).std()
    df["vol_ratio"]    = df["vol_5"] / (df["vol_20"] + 1e-10)
    df["volume_trend"] = df["Volume"].rolling(5).mean() / (df["Volume"].rolling(20).mean() + 1e-10)
    df["hl_ratio"]   = (df["High"] - df["Low"]) / (close + 1e-10)         
    df["body_ratio"] = abs(close - df["Open"]) / (df["High"] - df["Low"] + 1e-10)  
    df["gap"]        = (df["Open"] - close.shift(1)) / (close.shift(1) + 1e-10)    
    df["momentum_20"] = close / (close.shift(20) + 1e-10) - 1
    df["sma_r_5_20"]  = df["sma_5"]  / (df["sma_20"] + 1e-10) - 1
    df["sma_r_10_50"] = (close.rolling(10).mean()) / (df["sma_50"] + 1e-10) - 1
    df["price_sma20"] = close / (df["sma_20"] + 1e-10) - 1
    ativos_macro = MACRO_POR_SETOR.get(setor, ["ibov", "dolar"])
    placeholders = ",".join("?" * len(ativos_macro))   
    df_m = pd.read_sql_query(f, conn, params=ativos_macro)
    if not df_m.empty:
        df_m["data"] = pd.to_datetime(df_m["data"])
        macro_pivot = df_m.pivot(index="data", columns="ativo", values="retorno")
        macro_pivot.columns = [f"macro_{c}" for c in macro_pivot.columns]
        df = df.join(macro_pivot, how="left")
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df = df.iloc[:-1]  
    df = _tratar_dados(df)
    if len(df) < MIN_PREGOES:
        return None, []
    excluir  = {"Open", "High", "Low", "Close", "Volume", "target"}
    features = [c for c in df.columns if c not in excluir
                and df[c].notna().sum() > MIN_PREGOES * 0.8]
    return df, features
def _rsi(series, window):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    return 100 - (100 / (1 + gain / (loss + 1e-10)))
def _tratar_dados(df):
    thresh_col = int(len(df) * 0.60)   
    df = df.dropna(axis=1, thresh=thresh_col)
    df = df.ffill(limit=5)
    skip = {"Open", "High", "Low", "Close", "Volume", "target"}
    num_cols = [c for c in df.select_dtypes(include=np.number).columns if c not in skip]
    for col in num_cols:
        med = df[col].median()
        std = df[col].std()
        if std > 0:
            df[col] = df[col].clip(med - 10 * std, med + 10 * std)
    for col in num_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    df = df.dropna(subset=["Close", "target"])
    return df
def find_threshold(y_true, y_proba):
    best_t, best_f1 = MIN_CONF, 0
    for t in np.arange(MIN_CONF, 0.70, 0.01):
        pred = (y_proba >= t).astype(int)
        if pred.sum() == 0:
            continue  
        f = f1_score(y_true, pred, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return round(best_t, 2)
def treinar_avaliar(df, features):
    X = df[features].values
    y = df["target"].values
    split    = int(len(df) * 0.80)
    X_tr_raw = X[:split]; X_te_raw = X[split:]
    y_tr     = y[:split]; y_te     = y[split:]
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr_raw)
    X_te   = scaler.transform(X_te_raw)
    model = LogisticRegression(max_iter=2000, C=0.05, solver="lbfgs",
                               class_weight="balanced")
    model.fit(X_tr, y_tr)
    tr_proba  = model.predict_proba(X_tr)[:, 1]
    threshold = find_threshold(y_tr, tr_proba)
    te_proba = model.predict_proba(X_te)[:, 1]
    y_pred   = (te_proba >= threshold).astype(int)
    try:
        auc = roc_auc_score(y_te, te_proba)
    except Exception:
        auc = 0.5   
    metrics = {
        "accuracy" : accuracy_score(y_te, y_pred),
        "precision": precision_score(y_te, y_pred, zero_division=0),
        "recall"   : recall_score(y_te, y_pred, zero_division=0),
        "f1"       : f1_score(y_te, y_pred, zero_division=0),
        "auc"      : auc,
        "threshold": threshold,
    }
    X_last    = scaler.transform(X[[-1]])
    prob_alta = model.predict_proba(X_last)[0, 1]
    if prob_alta >= threshold:
        sinal = "COMPRAR"
    elif prob_alta >= threshold - 0.03:
        sinal = "NEUTRO"
    else:
        sinal = "AGUARDAR"
    return metrics, prob_alta, sinal, threshold
def rodar_scanner():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            f"Banco '{DB_PATH}' não encontrado. "
            "Execute bolsa_db.py primeiro."
        )
    inicializar_tabela_previsoes()
    conn_r = sqlite3.connect(DB_PATH)
    acoes  = pd.read_sql_query(
        "SELECT ticker, setor, nome FROM acoes WHERE ativo = 1 ORDER BY setor, ticker",
        conn_r
    )
    conn_r.close()
    hoje       = date.today().strftime("%Y-%m-%d")
    resultados = []   
    erros      = []   
    print(f"\n{'='*60}")
    print(f"  B3 SCANNER v3  —  {len(acoes)} ações  —  {hoje}")
    print(f"{'='*60}\n")
    conn_r = sqlite3.connect(DB_PATH)
    conn_w = sqlite3.connect(DB_PATH)
    for _, row in tqdm(acoes.iterrows(), total=len(acoes), desc="Analisando"):
        ticker = row["ticker"]
        setor  = row["setor"]
        nome   = row.get("nome") or ticker
        try:
            df, features = carregar_dados(ticker, setor, conn_r)
            if df is None or len(features) < 5:
                erros.append({"ticker": ticker, "motivo": "dados insuficientes"})
                continue
            metrics, prob_alta, sinal, threshold = treinar_avaliar(df, features)
            conn_w.execute(, (
                ticker, setor, hoje,
                round(prob_alta, 4),
                sinal,
                round(metrics["accuracy"], 4),
                round(metrics["auc"], 4),
                round(metrics["f1"], 4),
                threshold,
                len(features),
                len(df),
            ))
            conn_w.commit()  
            resultados.append({
                "ticker"    : ticker,
                "nome"      : nome,
                "setor"     : setor,
                "prob_alta" : prob_alta,
                "sinal"     : sinal,
                "acuracia"  : metrics["accuracy"],
                "auc"       : metrics["auc"],
                "f1"        : metrics["f1"],
                "threshold" : threshold,
                "n_pregoes" : len(df),
            })
        except Exception as e:
            erros.append({"ticker": ticker, "motivo": str(e)})
    conn_r.close()
    conn_w.close()
    return pd.DataFrame(resultados), erros
def imprimir_ranking(df_res):
    if df_res.empty:
        print("Nenhum resultado para exibir.")
        return
    df_res["score"] = (
        df_res["prob_alta"] * 0.50 +
        df_res["auc"]       * 0.30 +
        df_res["acuracia"]  * 0.20
    )
    comprar = df_res[df_res["sinal"] == "COMPRAR"].sort_values("score", ascending=False)
    neutro  = df_res[df_res["sinal"] == "NEUTRO"].sort_values("score",  ascending=False)
    aguard  = df_res[df_res["sinal"] == "AGUARDAR"].sort_values("score", ascending=True)
    linha = "─" * 80
    def bloco(titulo, subset):
        if subset.empty:
            return
        print(f"\n{linha}")
        print(f"  {titulo}  ({len(subset)} ações)")
        print(linha)
        print(f"  {'Ticker':<12} {'Nome':<28} {'Prob.Alta':>9} {'AUC':>6} {'Acur.':>6}  Setor")
        print(f"  {'-'*12} {'-'*28} {'-'*9} {'-'*6} {'-'*6}  {'-'*20}")
        for _, r in subset.head(TOP_N).iterrows():
            nome_curto = (r["nome"][:26] + "..") if len(str(r["nome"])) > 28 else r["nome"]
            print(f"  {r['ticker']:<12} {nome_curto:<28} "
                  f"{r['prob_alta']:>8.1%} {r['auc']:>6.3f} {r['acuracia']:>6.1%}  {r['setor']}")
    bloco("COMPRAR  ✓", comprar)
    bloco("NEUTRO   ~", neutro)
    bloco("AGUARDAR ✗", aguard)
    print(f"\n{linha}")
    print(f"  Total processado : {len(df_res)} ações")
    print(f"  COMPRAR          : {len(comprar)}")
    print(f"  NEUTRO           : {len(neutro)}")
    print(f"  AGUARDAR         : {len(aguard)}")
    print(linha)
def salvar_csv(df_res):
    if df_res.empty:
        return
    df_res["score"] = (
        df_res["prob_alta"] * 0.50 +
        df_res["auc"]       * 0.30 +
        df_res["acuracia"]  * 0.20
    )
    df_out = df_res.sort_values("score", ascending=False).reset_index(drop=True)
    df_out.to_csv(CSV_SAIDA, index=False, float_format="%.4f")
    print(f"\n  CSV salvo: {CSV_SAIDA}  ({len(df_out)} linhas)")
def plotar_resultados(df_res):
    if df_res.empty:
        return
    df_res = df_res.copy()
    df_res["score"] = (
        df_res["prob_alta"] * 0.50 +
        df_res["auc"]       * 0.30 +
        df_res["acuracia"]  * 0.20
    )
    df_sorted = df_res.sort_values("score", ascending=False).reset_index(drop=True)
    top    = df_sorted.head(TOP_N)                           
    bottom = df_sorted.tail(TOP_N).sort_values("score")     
    comprar = df_res[df_res["sinal"] == "COMPRAR"]
    neutro  = df_res[df_res["sinal"] == "NEUTRO"]
    aguard  = df_res[df_res["sinal"] == "AGUARDAR"]
    fig = plt.figure(figsize=(18, 20))
    fig.suptitle(
        f"B3 Scanner v3  —  Previsões para {date.today().strftime('%d/%m/%Y')}  "
        f"({len(df_res)} ações analisadas)",
        fontsize=14, fontweight="bold", y=0.99
    )
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.35)
    ax0 = fig.add_subplot(gs[0, :])  
    cores_top = ["
                 else "
    bars = ax0.barh(top["ticker"][::-1], top["prob_alta"][::-1] * 100,
                    color=cores_top[::-1], alpha=0.85)
    ax0.axvline(50, color="gray", linestyle="--", linewidth=0.8, label="50%")
    ax0.set_xlabel("Probabilidade de Alta (%)")
    ax0.set_title(f"Top {TOP_N} ações por probabilidade de alta")
    ax0.set_xlim(0, 100)
    ax0.grid(alpha=0.15, axis="x")
    for bar, (_, r) in zip(bars[::-1], top.iterrows()):
        ax0.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 f"{r['prob_alta']:.1%}  {r['sinal']}",
                 va="center", fontsize=9)
    ax1 = fig.add_subplot(gs[1, 0])
    sizes  = [len(comprar), len(neutro), len(aguard)]
    labels = [f"COMPRAR\n{len(comprar)}", f"NEUTRO\n{len(neutro)}", f"AGUARDAR\n{len(aguard)}"]
    cores_pizza = ["
    wedges, texts, autotexts = ax1.pie(
        sizes, labels=labels, colors=cores_pizza,
        autopct="%1.0f%%", startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5}
    )
    for at in autotexts:
        at.set_fontsize(10)
    ax1.set_title("Distribuição de sinais")
    ax2 = fig.add_subplot(gs[1, 1])
    auc_setor = df_res.groupby("setor")["auc"].mean().sort_values(ascending=True)
    cores_auc = ["
    ax2.barh(auc_setor.index, auc_setor.values, color=cores_auc, alpha=0.85)
    ax2.axvline(0.50, color="gray", linestyle="--", linewidth=0.8, label="Baseline 0.50")
    ax2.axvline(0.52, color="
    ax2.set_xlabel("AUC médio")
    ax2.set_title("AUC médio por setor")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.15, axis="x")
    ax2.set_xlim(0.44, 0.60)
    ax3 = fig.add_subplot(gs[2, 0])
    color_map = {"COMPRAR": "
    for sinal, grp in df_res.groupby("sinal"):
        ax3.scatter(grp["auc"], grp["prob_alta"] * 100,
                    c=color_map.get(sinal, "gray"),
                    label=sinal, alpha=0.75, s=50, edgecolors="none")
    ax3.axhline(50, color="gray", linestyle="--", linewidth=0.8)
    ax3.axvline(0.50, color="gray", linestyle="--", linewidth=0.8)
    ax3.set_xlabel("AUC")
    ax3.set_ylabel("Probabilidade de Alta (%)")
    ax3.set_title("Prob. alta vs AUC  (quadrante superior direito = melhor)")
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.15)
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.barh(bottom["ticker"], bottom["prob_alta"] * 100,
             color="
    ax4.axvline(50, color="gray", linestyle="--", linewidth=0.8)
    ax4.set_xlabel("Probabilidade de Alta (%)")
    ax4.set_title(f"Piores {TOP_N} — menor probabilidade de alta")
    ax4.set_xlim(0, 100)
    ax4.grid(alpha=0.15, axis="x")
    plt.savefig("scanner_v3.png", dpi=140, bbox_inches="tight")
    print(f"  Gráfico salvo: scanner_v3.png")
    plt.show()
def resumo_previsoes():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(, conn)
    top5 = pd.read_sql_query(, conn)
    conn.close()
    print("\n" + "="*54)
    print("  PREVISÕES SALVAS NO BANCO (última rodada)")
    print("="*54)
    print(df.to_string(index=False))
    print("\n  Top 5 COMPRAR:")
    print(top5.to_string(index=False))
    print("="*54)
if __name__ == "__main__":
    print("\nIniciando B3 Scanner v3...")
    print(f"Banco: {os.path.abspath(DB_PATH)}")
    print(f"Mínimo de pregões para treinar: {MIN_PREGOES}")
    print(f"Threshold mínimo de confiança : {MIN_CONF}")
    df_resultados, erros = rodar_scanner()
    imprimir_ranking(df_resultados)
    salvar_csv(df_resultados)
    plotar_resultados(df_resultados)
    resumo_previsoes()
    if erros:
        print(f"\n  {len(erros)} ações ignoradas (dados insuficientes ou erro):")
        for e in erros:
            print(f"    {e['ticker']}: {e['motivo']}")
    print("\nPronto!\n")