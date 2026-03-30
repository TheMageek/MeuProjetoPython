import warnings
warnings.filterwarnings("ignore")
import sqlite3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from datetime import date
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score,
    roc_curve, confusion_matrix,
)
from sklearn.calibration import calibration_curve
DB_PATH     = "bolsa_b3.db"
N_SPLITS    = 10       
MIN_CONF    = 0.52     
MIN_PREGOES = 200      
TOP_N       = 5        
CSV_SAIDA   = "previsoes_b3.csv"
PNG_RANKING = "scanner_v4_ranking.png"
PNG_DIAG    = "scanner_v4_diagnostico.png"
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
        "t10y",     
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
LAGS_POR_SETOR = {
    "Petróleo e Gás"              : [1, 5, 10, 21],
    "Mineração e Siderurgia"      : [1, 5, 10, 21],
    "Agronegócio"                 : [1, 5, 21, 63],   
    "Alimentos e Bebidas"         : [1, 5, 21],
    "Papel e Celulose"            : [1, 5, 21],
    "Logística e Transporte"      : [1, 5, 10],
    "Financeiro e Bancos"         : [1, 2, 5],         
    "Tecnologia"                  : [1, 2, 5],
    "_default"                    : [1, 5],
}
SCHEMA_BASE = 
MIGRACOES_V4 = [
    "ALTER TABLE previsoes ADD COLUMN precision_val REAL",
    "ALTER TABLE previsoes ADD COLUMN recall_val    REAL",
    "ALTER TABLE previsoes ADD COLUMN score         REAL",
    "CREATE INDEX IF NOT EXISTS idx_prev_score ON previsoes(data_previsao, score DESC)",
    "ALTER TABLE macro ADD COLUMN nivel REAL",
]
def inicializar_banco():
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA_BASE)
    for sql in MIGRACOES_V4:
        try:
            conn.execute(sql)
            conn.commit()
        except sqlite3.OperationalError:
            pass  
    conn.close()
def _rsi(series, window):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    return 100 - (100 / (1 + gain / (loss + 1e-10)))
def _zscore_rolling(series, window=252):
    roll = series.rolling(window)
    return (series - roll.mean()) / (roll.std() + 1e-10)
def _tratar_dados(df):
    skip = {"Open", "High", "Low", "Close", "Volume", "target"}
    thresh = int(len(df) * 0.60)
    df = df.dropna(axis=1, thresh=thresh)
    df = df.ffill(limit=5)
    num_cols = [c for c in df.select_dtypes(include=np.number).columns if c not in skip]
    for col in num_cols:
        med = df[col].median()
        std = df[col].std()
        if std > 0:
            df[col] = df[col].clip(med - 10 * std, med + 10 * std)
        if df[col].isna().any():
            df[col] = df[col].fillna(med)
    df = df.dropna(subset=["Close", "target"])
    return df
def find_threshold(y_true, y_proba):
    best_t, best_f1 = MIN_CONF, 0.0
    for t in np.arange(MIN_CONF, 0.71, 0.01):
        pred = (y_proba >= t).astype(int)
        if pred.sum() == 0:
            continue
        f = f1_score(y_true, pred, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return round(best_t, 2)
def _score_composto(prob_alta, auc, acuracia, precision_val, recall_val):
    return (
        prob_alta     * 0.35 +
        auc           * 0.25 +
        precision_val * 0.20 +
        acuracia      * 0.10 +
        recall_val    * 0.10
    )
def carregar_dados(ticker, setor, conn):
    df_p = pd.read_sql_query(, conn, params=(ticker,))
    if len(df_p) < MIN_PREGOES:
        return None, []
    df_p["data"] = pd.to_datetime(df_p["data"])
    df_p = df_p.set_index("data")
    df_i = pd.read_sql_query(, conn, params=(ticker,))
    df_i["data"] = pd.to_datetime(df_i["data"])
    df_i = df_i.set_index("data")
    df   = df_p.join(df_i, how="left")
    close = df["Close"]
    ret   = close.pct_change()
    for lag in [1, 2, 3, 5, 10]:
        df[f"ret_lag{lag}"] = ret.shift(lag)
    df["rsi_9"]  = _rsi(close, 9)
    df["rsi_21"] = _rsi(close, 21)
    df["vol_5"]        = ret.rolling(5).std()
    df["vol_ratio"]    = df["vol_5"] / (df["vol_20"] + 1e-10)
    df["volume_trend"] = (df["Volume"].rolling(5).mean()
                          / (df["Volume"].rolling(20).mean() + 1e-10))
    df["hl_ratio"]   = (df["High"] - df["Low"]) / (close + 1e-10)
    df["body_ratio"] = (abs(close - df["Open"])
                        / (df["High"] - df["Low"] + 1e-10))
    df["gap"]        = ((df["Open"] - close.shift(1))
                        / (close.shift(1) + 1e-10))
    df["momentum_20"] = close / (close.shift(20) + 1e-10) - 1
    df["momentum_60"] = close / (close.shift(60) + 1e-10) - 1
    df["sma_r_5_20"]  = df["sma_5"] / (df["sma_20"] + 1e-10) - 1
    df["sma_r_10_50"] = (close.rolling(10).mean()) / (df["sma_50"] + 1e-10) - 1
    df["price_sma20"] = close / (df["sma_20"] + 1e-10) - 1
    df["price_zscore"] = _zscore_rolling(close, 252)
    ativos_macro = MACRO_POR_SETOR.get(setor, ["ibov", "dolar"])
    lags_macro   = LAGS_POR_SETOR.get(setor, LAGS_POR_SETOR["_default"])
    placeholders = ",".join("?" * len(ativos_macro))
    _cols_macro = [r[1] for r in conn.execute("PRAGMA table_info(macro)").fetchall()]
    _tem_nivel_col = "nivel" in _cols_macro
    _select_nivel = ", nivel" if _tem_nivel_col else ""
    df_m = pd.read_sql_query(f, conn, params=ativos_macro)
    tem_nivel = _tem_nivel_col and "nivel" in df_m.columns and df_m["nivel"].notna().any()
    if not df_m.empty:
        df_m["data"] = pd.to_datetime(df_m["data"])
        ret_pivot = df_m.pivot(index="data", columns="ativo", values="retorno")
        for lag in lags_macro:
            lagged = ret_pivot.shift(lag)
            lagged.columns = [f"macro_{c}_ret{lag}" for c in lagged.columns]
            df = df.join(lagged, how="left")
        if tem_nivel:
            niv_pivot = df_m.pivot(index="data", columns="ativo", values="nivel")
            for col in niv_pivot.columns:
                zscore = _zscore_rolling(niv_pivot[col], 252)
                df = df.join(zscore.rename(f"macro_{col}_nivel"), how="left")
        g  = f"macro_gasoline_ret1"
        b  = f"macro_brent_ret1"
        if g in df.columns and b in df.columns:
            df["spread_crack"] = df[g] - df[b]
        d  = f"macro_diesel_ret1"
        if d in df.columns and b in df.columns:
            df["spread_diesel_brent"] = df[d] - df[b]
        t10 = f"macro_t10y_ret1"
        t2  = f"macro_t2y_ret1"
        if t10 in df.columns and t2 in df.columns:
            df["spread_yield_curve"] = df[t10] - df[t2]
        sj = f"macro_soja_ret1"
        mi = f"macro_milho_ret1"
        if sj in df.columns and mi in df.columns:
            df["spread_soja_milho"] = df[sj] - df[mi]
        vx = f"macro_vix_ret1"
        if b in df.columns and vx in df.columns:
            df["spread_brent_vix"] = df[b] - df[vx]
        yn  = f"macro_yuan_ret1"
        dol = f"macro_dolar_ret1"
        if yn in df.columns and dol in df.columns:
            df["spread_yuan_dolar"] = df[yn] - df[dol]
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df = df.iloc[:-1]   
    df = _tratar_dados(df)
    if len(df) < MIN_PREGOES:
        return None, []
    excluir  = {"Open", "High", "Low", "Close", "Volume", "target"}
    features = [
        c for c in df.columns
        if c not in excluir
        and df[c].notna().sum() > MIN_PREGOES * 0.8
    ]
    return df, features
def treinar_avaliar(df, features):
    X = df[features].values
    y = df["target"].values
    split    = int(len(df) * 0.80)
    X_tr_raw = X[:split];  X_te_raw = X[split:]
    y_tr     = y[:split];  y_te     = y[split:]
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr_raw)
    X_te   = scaler.transform(X_te_raw)
    model = LogisticRegression(
        max_iter=2000, C=0.05, solver="lbfgs",
        class_weight="balanced"
    )
    model.fit(X_tr, y_tr)
    tr_proba  = model.predict_proba(X_tr)[:, 1]
    threshold = find_threshold(y_tr, tr_proba)
    te_proba = model.predict_proba(X_te)[:, 1]
    y_pred   = (te_proba >= threshold).astype(int)
    try:
        auc = roc_auc_score(y_te, te_proba)
    except Exception:
        auc = 0.5
    try:
        frac_pos, mean_pred = calibration_curve(y_te, te_proba, n_bins=8)
        cal_data = (frac_pos, mean_pred)
    except Exception:
        cal_data = None
    metrics = {
        "accuracy"      : accuracy_score(y_te, y_pred),
        "auc"           : auc,
        "f1"            : f1_score(y_te, y_pred, zero_division=0),
        "precision_val" : precision_score(y_te, y_pred, zero_division=0),
        "recall_val"    : recall_score(y_te, y_pred, zero_division=0),
        "threshold"     : threshold,
    }
    X_last    = scaler.transform(X[[-1]])
    prob_alta = model.predict_proba(X_last)[0, 1]
    if prob_alta >= threshold:
        sinal = "COMPRAR"
    elif prob_alta >= threshold - 0.03:
        sinal = "NEUTRO"
    else:
        sinal = "AGUARDAR"
    return {
        "metrics"  : metrics,
        "prob_alta": prob_alta,
        "sinal"    : sinal,
        "cal_data" : cal_data,
        "te_proba" : te_proba,
        "y_te"     : y_te,
    }
def rodar_scanner():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            f"Banco '{DB_PATH}' não encontrado. Execute bolsa_db.py primeiro."
        )
    inicializar_banco()
    conn_r = sqlite3.connect(DB_PATH)
    acoes  = pd.read_sql_query(
        "SELECT ticker, setor, nome FROM acoes WHERE ativo = 1 ORDER BY setor, ticker",
        conn_r
    )
    conn_r.close()
    hoje      = date.today().strftime("%Y-%m-%d")
    resultados = []
    erros      = []
    print(f"\n{'='*64}")
    print(f"  B3 SCANNER v4  —  {len(acoes)} ações  —  {hoje}")
    print(f"{'='*64}\n")
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
            resultado = treinar_avaliar(df, features)
            m         = resultado["metrics"]
            prob_alta = resultado["prob_alta"]
            sinal     = resultado["sinal"]
            score = _score_composto(
                prob_alta, m["auc"], m["accuracy"],
                m["precision_val"], m["recall_val"]
            )
            conn_w.execute(, (
                ticker, setor, hoje,
                round(prob_alta, 4),
                sinal,
                round(m["accuracy"],      4),
                round(m["auc"],           4),
                round(m["f1"],            4),
                round(m["precision_val"], 4),
                round(m["recall_val"],    4),
                m["threshold"],
                len(features),
                len(df),
                round(score, 4),
            ))
            conn_w.commit()
            resultados.append({
                "ticker"       : ticker,
                "nome"         : nome,
                "setor"        : setor,
                "prob_alta"    : prob_alta,
                "sinal"        : sinal,
                "acuracia"     : m["accuracy"],
                "auc"          : m["auc"],
                "f1"           : m["f1"],
                "precision_val": m["precision_val"],
                "recall_val"   : m["recall_val"],
                "threshold"    : m["threshold"],
                "n_features"   : len(features),
                "n_pregoes"    : len(df),
                "score"        : score,
                "_cal_data"    : resultado["cal_data"],
                "_te_proba"    : resultado["te_proba"],
                "_y_te"        : resultado["y_te"],
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
    comprar = df_res[df_res["sinal"] == "COMPRAR"].sort_values("score", ascending=False)
    neutro  = df_res[df_res["sinal"] == "NEUTRO"].sort_values("score",  ascending=False)
    aguard  = df_res[df_res["sinal"] == "AGUARDAR"].sort_values("score", ascending=True)
    linha = "─" * 100
    def bloco(titulo, subset):
        if subset.empty:
            return
        print(f"\n{linha}")
        print(f"  {titulo}  ({len(subset)} ações)")
        print(linha)
        print(f"  {'Ticker':<10} {'P.Alta':>7} {'AUC':>6} {'Acur':>6} "
              f"{'Prec':>6} {'Rec':>6} {'F1':>6} {'Score':>7}  Setor")
        print(f"  {'-'*10} {'-'*7} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*7}  {'-'*22}")
        for _, r in subset.head(TOP_N).iterrows():
            print(
                f"  {r['ticker']:<10} "
                f"{r['prob_alta']:>7.1%} "
                f"{r['auc']:>6.3f} "
                f"{r['acuracia']:>6.1%} "
                f"{r['precision_val']:>6.1%} "
                f"{r['recall_val']:>6.1%} "
                f"{r['f1']:>6.3f} "
                f"{r['score']:>7.4f}  "
                f"{r['setor']}"
            )
    bloco("COMPRAR  ✓", comprar)
    bloco("NEUTRO   ~", neutro)
    bloco("AGUARDAR ✗", aguard)
    print(f"\n{linha}")
    print(f"  Total processado : {len(df_res):>4}  |  "
          f"COMPRAR: {len(comprar):>3}  NEUTRO: {len(neutro):>3}  AGUARDAR: {len(aguard):>3}")
    print(f"  AUC médio geral  : {df_res['auc'].mean():.4f}  |  "
          f"Acurácia média: {df_res['acuracia'].mean():.2%}  |  "
          f"Precision média: {df_res['precision_val'].mean():.2%}")
    pct_auc_ok = (df_res["auc"] >= 0.52).mean()
    pct_acc_ok = (df_res["acuracia"] >= 0.50).mean()
    print(f"  % ações AUC>=0.52: {pct_auc_ok:.1%}  |  "
          f"% ações Acur>=50%: {pct_acc_ok:.1%}")
    print(linha)
def salvar_csv(df_res):
    if df_res.empty:
        return
    cols_export = [c for c in df_res.columns if not c.startswith("_")]
    df_out = df_res[cols_export].sort_values("score", ascending=False).reset_index(drop=True)
    df_out.to_csv(CSV_SAIDA, index=False, float_format="%.4f")
    print(f"\n  CSV salvo: {CSV_SAIDA}  ({len(df_out)} linhas)")
def plotar_ranking(df_res):
    if df_res.empty:
        return
    df_s     = df_res.sort_values("score", ascending=False).reset_index(drop=True)
    top      = df_s.head(TOP_N)
    bottom   = df_s.tail(TOP_N).sort_values("score")
    comprar  = df_res[df_res["sinal"] == "COMPRAR"]
    neutro   = df_res[df_res["sinal"] == "NEUTRO"]
    aguard   = df_res[df_res["sinal"] == "AGUARDAR"]
    COR = {"COMPRAR": "
    fig = plt.figure(figsize=(20, 22))
    fig.suptitle(
        f"B3 Scanner v4  —  Ranking  —  {date.today().strftime('%d/%m/%Y')}  "
        f"({len(df_res)} ações)",
        fontsize=14, fontweight="bold", y=0.995
    )
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.48, wspace=0.32)
    ax0 = fig.add_subplot(gs[0, :])
    cores = [COR.get(s, "gray") for s in top["sinal"][::-1]]
    bars  = ax0.barh(top["ticker"][::-1], top["prob_alta"][::-1] * 100,
                     color=cores, alpha=0.85)
    ax0.axvline(50, color="gray", linestyle="--", linewidth=0.8)
    ax0.set_xlabel("Probabilidade de Alta (%)")
    ax0.set_title(f"Top {TOP_N} — maior probabilidade de alta")
    ax0.set_xlim(0, 105)
    ax0.grid(alpha=0.15, axis="x")
    for bar, (_, r) in zip(bars[::-1], top.iterrows()):
        ax0.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 f"{r['prob_alta']:.1%}  AUC:{r['auc']:.3f}  {r['sinal']}",
                 va="center", fontsize=8.5)
    ax1 = fig.add_subplot(gs[1, 0])
    sizes  = [len(comprar), len(neutro), len(aguard)]
    labels = [f"COMPRAR\n{len(comprar)}", f"NEUTRO\n{len(neutro)}", f"AGUARDAR\n{len(aguard)}"]
    wedges, _, autotexts = ax1.pie(
        sizes, labels=labels,
        colors=[COR["COMPRAR"], COR["NEUTRO"], COR["AGUARDAR"]],
        autopct="%1.0f%%", startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5}
    )
    for at in autotexts:
        at.set_fontsize(10)
    ax1.set_title("Distribuição de sinais")
    ax2 = fig.add_subplot(gs[1, 1])
    auc_s = df_res.groupby("setor")["auc"].mean().sort_values(ascending=True)
    cores_auc = ["
    ax2.barh(auc_s.index, auc_s.values, color=cores_auc, alpha=0.85)
    ax2.axvline(0.50, color="gray",    linestyle="--", linewidth=0.8, label="Baseline 0.50")
    ax2.axvline(0.52, color="
    ax2.set_xlabel("AUC médio")
    ax2.set_title("AUC médio por setor")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.15, axis="x")
    ax2.set_xlim(0.43, 0.62)
    ax3 = fig.add_subplot(gs[2, 0])
    for sinal, grp in df_res.groupby("sinal"):
        ax3.scatter(grp["auc"], grp["prob_alta"] * 100,
                    c=COR.get(sinal, "gray"), label=sinal,
                    alpha=0.70, s=45, edgecolors="none")
    ax3.axhline(50, color="gray", linestyle="--", linewidth=0.8)
    ax3.axvline(0.50, color="gray", linestyle="--", linewidth=0.8)
    ax3.set_xlabel("AUC")
    ax3.set_ylabel("Prob. Alta (%)")
    ax3.set_title("Mapa de qualidade — quadrante ↑ dir. = melhor")
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.15)
    ax4 = fig.add_subplot(gs[2, 1])
    cores_bot = [COR.get(s, "gray") for s in bottom["sinal"]]
    ax4.barh(bottom["ticker"], bottom["prob_alta"] * 100,
             color="
    ax4.axvline(50, color="gray", linestyle="--", linewidth=0.8)
    ax4.set_xlabel("Probabilidade de Alta (%)")
    ax4.set_title(f"Piores {TOP_N} — menor prob. de alta")
    ax4.set_xlim(0, 100)
    ax4.grid(alpha=0.15, axis="x")
    plt.savefig(PNG_RANKING, dpi=140, bbox_inches="tight")
    print(f"  Gráfico de ranking salvo: {PNG_RANKING}")
    plt.show()
def plotar_diagnostico(df_res):
    if df_res.empty:
        return
    all_te_prob, all_y_te = [], []
    for _, row in df_res.iterrows():
        if row.get("_te_proba") is not None and len(row["_te_proba"]) > 0:
            all_te_prob.extend(row["_te_proba"].tolist())
            all_y_te.extend(row["_y_te"].tolist())
    fig = plt.figure(figsize=(22, 26))
    fig.suptitle(
        f"B3 Scanner v4  —  Diagnóstico do Modelo  —  {date.today().strftime('%d/%m/%Y')}",
        fontsize=14, fontweight="bold", y=0.995
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.38)
    VERDE    = "
    VERMELHO = "
    AZUL     = "
    ax0 = fig.add_subplot(gs[0, 0])
    auc_vals = df_res["auc"].dropna()
    pct_ok   = (auc_vals >= 0.52).mean()
    ax0.hist(auc_vals, bins=25, color=AZUL, alpha=0.75, edgecolor="white")
    ax0.axvline(0.50, color="gray",    linestyle="--", linewidth=1, label="Aleatório (0.50)")
    ax0.axvline(0.52, color=VERDE,     linestyle="-",  linewidth=1.2, label=f"Meta (0.52) — {pct_ok:.1%} das ações")
    ax0.axvline(auc_vals.mean(), color=VERMELHO, linestyle=":", linewidth=1.2,
                label=f"Média ({auc_vals.mean():.3f})")
    ax0.set_xlabel("AUC")
    ax0.set_ylabel("Nº de ações")
    ax0.set_title("Distribuição de AUC por ação")
    ax0.legend(fontsize=8)
    ax0.grid(alpha=0.15)
    ax1 = fig.add_subplot(gs[0, 1])
    acc_vals = df_res["acuracia"].dropna()
    pct_acc  = (acc_vals >= 0.50).mean()
    ax1.hist(acc_vals * 100, bins=25, color="
    ax1.axvline(50, color="gray",    linestyle="--", linewidth=1, label="Baseline (50%)")
    ax1.axvline(acc_vals.mean() * 100, color=VERMELHO, linestyle=":", linewidth=1.2,
                label=f"Média ({acc_vals.mean():.1%}) — {pct_acc:.1%} ok")
    ax1.set_xlabel("Acurácia (%)")
    ax1.set_ylabel("Nº de ações")
    ax1.set_title("Distribuição de acurácia por ação")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.15)
    ax2 = fig.add_subplot(gs[0, 2])
    pre_vals = df_res["precision_val"].dropna()
    ax2.hist(pre_vals * 100, bins=25, color="
    ax2.axvline(50, color="gray",    linestyle="--", linewidth=1, label="Baseline (50%)")
    ax2.axvline(pre_vals.mean() * 100, color=VERMELHO, linestyle=":", linewidth=1.2,
                label=f"Média ({pre_vals.mean():.1%})")
    ax2.set_xlabel("Precision (%)")
    ax2.set_ylabel("Nº de ações")
    ax2.set_title("Distribuição de precision por ação\n(dos sinais COMPRAR, % que realmente subiu)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.15)
    ax3 = fig.add_subplot(gs[1, 0])
    sc = ax3.scatter(
        df_res["auc"], df_res["precision_val"] * 100,
        c=df_res["prob_alta"] * 100,
        cmap="RdYlGn", alpha=0.65, s=40, edgecolors="none"
    )
    plt.colorbar(sc, ax=ax3, label="Prob. Alta (%)")
    ax3.axhline(50, color="gray", linestyle="--", linewidth=0.8)
    ax3.axvline(0.50, color="gray", linestyle="--", linewidth=0.8)
    ax3.set_xlabel("AUC")
    ax3.set_ylabel("Precision (%)")
    ax3.set_title("AUC vs Precision\n(cor = prob. de alta)")
    ax3.grid(alpha=0.15)
    ax4 = fig.add_subplot(gs[1, 1])
    setores_ord = (df_res.groupby("setor")["auc"]
                   .median()
                   .sort_values(ascending=True)
                   .index.tolist())
    dados_box = [df_res[df_res["setor"] == s]["auc"].dropna().values
                 for s in setores_ord]
    bp = ax4.boxplot(dados_box, vert=False, patch_artist=True,
                     medianprops={"color": "white", "linewidth": 2})
    for patch, setor in zip(bp["boxes"], setores_ord):
        med = df_res[df_res["setor"] == setor]["auc"].median()
        patch.set_facecolor(VERDE if med >= 0.52 else VERMELHO)
        patch.set_alpha(0.7)
    ax4.set_yticks(range(1, len(setores_ord) + 1))
    ax4.set_yticklabels(setores_ord, fontsize=8)
    ax4.axvline(0.50, color="gray",    linestyle="--", linewidth=0.8)
    ax4.axvline(0.52, color=AZUL,      linestyle=":",  linewidth=0.8)
    ax4.set_xlabel("AUC")
    ax4.set_title("Boxplot de AUC por setor\n(verde = mediana ≥ 0.52)")
    ax4.grid(alpha=0.15, axis="x")
    ax5 = fig.add_subplot(gs[1, 2])
    if len(all_te_prob) > 50:
        try:
            y_arr    = np.array(all_y_te)
            prob_arr = np.array(all_te_prob)
            frac_pos, mean_pred = calibration_curve(y_arr, prob_arr, n_bins=10)
            ax5.plot(mean_pred, frac_pos, "o-", color=AZUL, linewidth=1.8,
                     markersize=5, label="Modelo (agregado)")
            ax5.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Perfeitamente calibrado")
            ax5.fill_between([0, 1], [0, 1], [0, 0.5], alpha=0.05, color="red",
                             label="Sub-confiante")
            ax5.fill_between([0, 1], [0.5, 1], [0, 1], alpha=0.05, color="green",
                             label="Super-confiante")
        except Exception:
            ax5.text(0.5, 0.5, "Dados insuficientes\npara calibração",
                     ha="center", va="center", transform=ax5.transAxes)
    ax5.set_xlabel("Probabilidade prevista")
    ax5.set_ylabel("Frequência real de alta")
    ax5.set_title("Calibração agregada do modelo\n(diagonal = ideal)")
    ax5.legend(fontsize=8)
    ax5.grid(alpha=0.15)
    ax5.set_xlim(0, 1); ax5.set_ylim(0, 1)
    ax6 = fig.add_subplot(gs[2, 0])
    prob_vals = df_res["prob_alta"].dropna()
    ax6.hist(prob_vals * 100, bins=30, color="
    ax6.axvline(50, color="gray",    linestyle="--", linewidth=1, label="50%")
    ax6.axvline(prob_vals.mean() * 100, color=VERMELHO, linestyle=":", linewidth=1.2,
                label=f"Média ({prob_vals.mean():.1%})")
    pct_acima = (prob_vals >= 0.52).mean()
    ax6.set_xlabel("Probabilidade de Alta prevista (%)")
    ax6.set_ylabel("Nº de ações")
    ax6.set_title(f"Distribuição de prob. de alta\n({pct_acima:.1%} das ações acima de 52%)")
    ax6.legend(fontsize=8)
    ax6.grid(alpha=0.15)
    ax7 = fig.add_subplot(gs[2, 1])
    setor_prec = df_res.groupby("setor")["precision_val"].mean()
    setor_rec  = df_res.groupby("setor")["recall_val"].mean()
    setores    = setor_prec.index.tolist()
    cores_s    = [VERDE if setor_prec[s] >= 0.50 and setor_rec[s] >= 0.50
                  else "
                  else VERMELHO
                  for s in setores]
    ax7.scatter(setor_rec * 100, setor_prec * 100, c=cores_s, s=90,
                edgecolors="white", linewidth=0.6, zorder=3)
    for s in setores:
        ax7.annotate(s[:14], (setor_rec[s] * 100, setor_prec[s] * 100),
                     fontsize=7, ha="left", va="bottom",
                     xytext=(3, 3), textcoords="offset points")
    ax7.axhline(50, color="gray", linestyle="--", linewidth=0.8)
    ax7.axvline(50, color="gray", linestyle="--", linewidth=0.8)
    ax7.set_xlabel("Recall médio do setor (%)")
    ax7.set_ylabel("Precision médio do setor (%)")
    ax7.set_title("Precision vs Recall por setor\n(quadrante ↑ dir. = melhor)")
    ax7.grid(alpha=0.15)
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis("off")
    tab_data = (
        df_res.groupby("setor")
        .agg(
            N=("ticker", "count"),
            AUC=("auc", "mean"),
            Acur=("acuracia", "mean"),
            Prec=("precision_val", "mean"),
            Rec=("recall_val", "mean"),
        )
        .round(3)
        .sort_values("AUC", ascending=False)
        .reset_index()
    )
    col_labels = ["Setor", "N", "AUC", "Acur", "Prec", "Rec"]
    cell_text  = []
    for _, r in tab_data.iterrows():
        setor_val   = str(r["setor"])
        setor_curto = setor_val[:18] + ".." if len(setor_val) > 18 else setor_val
        cell_text.append([
            setor_curto,
            int(r["N"]),
            f"{r['AUC']:.3f}",
            f"{r['Acur']:.1%}",
            f"{r['Prec']:.1%}",
            f"{r['Rec']:.1%}",
        ])
    tbl = ax8.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.35)
    for i, row_data in enumerate(cell_text):
        auc_val = float(row_data[2])
        cor = "
        tbl[i + 1, 2].set_facecolor(cor)
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    ax8.set_title("Saúde do modelo por setor", fontweight="bold", pad=12)
    plt.savefig(PNG_DIAG, dpi=140, bbox_inches="tight")
    print(f"  Painel de diagnóstico salvo: {PNG_DIAG}")
    plt.show()
def resumo_banco():
    conn = sqlite3.connect(DB_PATH)
    df_agg = pd.read_sql_query(, conn)
    top_comprar = pd.read_sql_query(, conn)
    piores = pd.read_sql_query(, conn)
    conn.close()
    sep = "=" * 64
    print(f"\n{sep}")
    print("  RESUMO DO BANCO — ÚLTIMA RODADA")
    print(sep)
    print(df_agg.to_string(index=False))
    print(f"\n  TOP 10 COMPRAR (por score composto):")
    print("  " + "-" * 62)
    print(top_comprar.to_string(index=False))
    if not piores.empty:
        print(f"\n  ATENÇÃO — ações com AUC < 0.50 (modelo pior que aleatório):")
        print("  " + "-" * 62)
        print(piores.to_string(index=False))
        print("  → Considere aumentar MIN_PREGOES ou revisar os dados dessas ações.")
    print(sep)
if __name__ == "__main__":
    print("\n" + "=" * 64)
    print("  B3 SCANNER v4 — MACROS EXPANDIDAS + DIAGNÓSTICO COMPLETO")
    print("=" * 64)
    print(f"  Banco           : {os.path.abspath(DB_PATH)}")
    print(f"  Min. pregões    : {MIN_PREGOES}")
    print(f"  Threshold mín.  : {MIN_CONF}")
    print(f"  Folds CV        : {N_SPLITS}")
    print(f"  Top N ranking   : {TOP_N}")
    print("=" * 64 + "\n")
    df_resultados, erros = rodar_scanner()
    if df_resultados.empty:
        print("Nenhuma ação processada. Verifique o banco e os logs de erro.")
    else:
        imprimir_ranking(df_resultados)
        salvar_csv(df_resultados)
        plotar_ranking(df_resultados)
        plotar_diagnostico(df_resultados)
        resumo_banco()
    if erros:
        print(f"\n  {len(erros)} ações ignoradas:")
        for e in erros[:20]:   
            print(f"    {e['ticker']}: {e['motivo']}")
        if len(erros) > 20:
            print(f"    ... e mais {len(erros) - 20} ações. Veja o CSV para o log completo.")
    print("\nPronto!\n")