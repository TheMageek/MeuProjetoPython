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
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    average_precision_score, precision_score, recall_score
)
import lightgbm as lgb
DB_PATH     = "bolsa_b3.db"
MIN_CONF    = 0.52
MIN_PREGOES = 200
TOP_N       = 15
CSV_SAIDA   = "previsoes_b3.csv"
SEED        = 42
MACRO_POR_SETOR = {
    "Petróleo e Gás"              : ["brent", "dolar", "ibov", "vix"],
    "Mineração e Siderurgia"      : ["minerio", "dolar", "ibov", "cobre"],
    "Financeiro e Bancos"         : ["dolar", "ibov", "sp500", "vix"],
    "Energia Elétrica"            : ["ibov", "dolar"],
    "Varejo e Consumo"            : ["ibov", "dolar"],
    "Alimentos e Bebidas"         : ["dolar", "ibov", "soja", "milho"],
    "Saúde e Farmácia"            : ["ibov", "dolar"],
    "Construção e Imóveis"        : ["ibov", "dolar"],
    "Telecomunicações"            : ["ibov", "dolar", "vix"],
    "Tecnologia"                  : ["ibov", "sp500", "dolar", "vix"],
    "Logística e Transporte"      : ["ibov", "dolar", "brent"],
    "Agronegócio"                 : ["dolar", "soja", "milho", "ibov"],
    "Papel e Celulose"            : ["dolar", "ibov"],
    "Saneamento"                  : ["ibov", "dolar"],
    "Shopping e Imóveis Comerciais": ["ibov", "dolar"],
}
SCHEMA_PREVISOES = 
def inicializar_tabela_previsoes():
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA_PREVISOES)
    for col, tipo in [("pr_auc", "REAL"), ("modelo_usado", "TEXT")]:
        try:
            conn.execute(f"ALTER TABLE previsoes ADD COLUMN {col} {tipo}")
        except Exception:
            pass
    conn.commit()
    conn.close()
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
        if df[col].isna().any():
            df[col] = df[col].fillna(med)
    df = df.dropna(subset=["Close", "target"])
    return df
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
    df["rsi_9"]        = _rsi(close, 9)
    df["rsi_21"]       = _rsi(close, 21)
    df["vol_5"]        = ret.rolling(5).std()
    df["vol_ratio"]    = df["vol_5"] / (df["vol_20"] + 1e-10)
    df["volume_trend"] = df["Volume"].rolling(5).mean() / (df["Volume"].rolling(20).mean() + 1e-10)
    df["hl_ratio"]     = (df["High"] - df["Low"]) / (close + 1e-10)
    df["body_ratio"]   = abs(close - df["Open"]) / (df["High"] - df["Low"] + 1e-10)
    df["gap"]          = (df["Open"] - close.shift(1)) / (close.shift(1) + 1e-10)
    df["momentum_20"]  = close / (close.shift(20) + 1e-10) - 1
    df["sma_r_5_20"]   = df["sma_5"]  / (df["sma_20"] + 1e-10) - 1
    df["sma_r_10_50"]  = close.rolling(10).mean() / (df["sma_50"] + 1e-10) - 1
    df["price_sma20"]  = close / (df["sma_20"] + 1e-10) - 1
    ativos_macro = MACRO_POR_SETOR.get(setor, ["ibov", "dolar"])
    placeholders = ",".join("?" * len(ativos_macro))
    df_m = pd.read_sql_query(f, conn, params=ativos_macro)
    if not df_m.empty:
        df_m["data"] = pd.to_datetime(df_m["data"])
        macro_pivot  = df_m.pivot(index="data", columns="ativo", values="retorno")
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
def _avaliar_proba(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    try:
        auc    = roc_auc_score(y_true, y_proba)
        pr_auc = average_precision_score(y_true, y_proba)
    except Exception:
        auc = pr_auc = 0.5
    return {
        "accuracy" : accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall"   : recall_score(y_true, y_pred, zero_division=0),
        "f1"       : f1_score(y_true, y_pred, zero_division=0),
        "auc"      : auc,
        "pr_auc"   : pr_auc,
        "threshold": threshold,
    }
def treinar_logreg(X_tr, y_tr, X_val, y_val, X_te, y_te):
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   LogisticRegression(
            class_weight="balanced", max_iter=2000,
            C=0.05, solver="lbfgs", random_state=SEED
        )),
    ])
    pipe.fit(X_tr, y_tr)
    threshold = find_threshold(y_tr, pipe.predict_proba(X_tr)[:, 1])
    prob_val = pipe.predict_proba(X_val)[:, 1]
    prob_te  = pipe.predict_proba(X_te)[:, 1]
    metrics_val  = _avaliar_proba(y_val, prob_val, threshold)
    metrics_test = _avaliar_proba(y_te,  prob_te,  threshold)
    return pipe, prob_val, prob_te, metrics_val, metrics_test, threshold
def treinar_lightgbm(X_tr, y_tr, X_val, y_val, X_te, y_te, threshold_lr):
    if len(X_val) < 20:
        return None, None, None, None, None, threshold_lr
    params = {
        "objective"        : "binary",
        "metric"           : ["auc", "average_precision"],
        "n_estimators"     : 1000,
        "learning_rate"    : 0.05,
        "num_leaves"       : 15,          
        "min_child_samples": 15,
        "subsample"        : 0.8,
        "subsample_freq"   : 1,
        "colsample_bytree" : 0.8,
        "reg_alpha"        : 0.1,
        "reg_lambda"       : 1.0,
        "is_unbalance"     : True,
        "random_state"     : SEED,
        "verbose"          : -1,
    }
    try:
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=30, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )
    except Exception:
        return None, None, None, None, None, threshold_lr
    prob_val = model.predict_proba(X_val)[:, 1]
    prob_te  = model.predict_proba(X_te)[:, 1]
    threshold = find_threshold(y_tr, model.predict_proba(X_tr)[:, 1])
    metrics_val  = _avaliar_proba(y_val, prob_val, threshold)
    metrics_test = _avaliar_proba(y_te,  prob_te,  threshold)
    return model, prob_val, prob_te, metrics_val, metrics_test, threshold
def treinar_avaliar(df, features):
    X = df[features].fillna(0).values
    y = df["target"].values
    n      = len(X)
    n_tr   = int(n * 0.70)
    n_val  = int(n * 0.85)   
    X_tr,  y_tr  = X[:n_tr],       y[:n_tr]
    X_val, y_val = X[n_tr:n_val],  y[n_tr:n_val]
    X_te,  y_te  = X[n_val:],      y[n_val:]
    base_pr = float(y_te.mean()) if len(y_te) > 0 else 0.5
    (pipe_lr,
     prob_val_lr, prob_te_lr,
     mval_lr, mte_lr, thr_lr) = treinar_logreg(
        X_tr, y_tr, X_val, y_val, X_te, y_te)
    (model_lgb,
     prob_val_lgb, prob_te_lgb,
     mval_lgb, mte_lgb, thr_lgb) = treinar_lightgbm(
        X_tr, y_tr, X_val, y_val, X_te, y_te, thr_lr)
    auc_lgb = mte_lgb["auc"] if mte_lgb else 0.0
    if model_lgb is not None and auc_lgb >= mte_lr["auc"]:
        best_model    = model_lgb
        best_metrics  = mte_lgb
        best_thr      = thr_lgb
        modelo_usado  = "LightGBM"
        scaler_lgb    = StandardScaler()
        scaler_lgb.fit(X_tr)
        X_last = scaler_lgb.transform(X[[-1]])
        prob_alta = best_model.predict_proba(X_last)[0, 1]
    else:
        best_metrics  = mte_lr
        best_thr      = thr_lr
        modelo_usado  = "LogReg"
        prob_alta     = pipe_lr.predict_proba(X[[-1]])[0, 1]
    if prob_alta >= best_thr:
        sinal = "COMPRAR"
    elif prob_alta >= best_thr - 0.03:
        sinal = "NEUTRO"
    else:
        sinal = "AGUARDAR"
    best_metrics["base_pr"]     = base_pr
    best_metrics["modelo_usado"] = modelo_usado
    best_metrics["auc_lr"]    = mte_lr["auc"]
    best_metrics["pr_auc_lr"] = mte_lr["pr_auc"]
    best_metrics["auc_lgb"]   = auc_lgb
    best_metrics["pr_auc_lgb"]= mte_lgb["pr_auc"] if mte_lgb else 0.0
    return best_metrics, prob_alta, sinal, best_thr, modelo_usado
def rodar_scanner():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            f"Banco '{DB_PATH}' não encontrado. Execute bolsa_db.py primeiro."
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
    print(f"\n{'='*62}")
    print(f"  B3 SCANNER v3.1  —  {len(acoes)} ações  —  {hoje}")
    print(f"  Split: 70% treino / 15% validação / 15% teste")
    print(f"  Modelos: LogReg (Pipeline) vs LightGBM (early stopping)")
    print(f"{'='*62}\n")
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
            metrics, prob_alta, sinal, threshold, modelo_usado = treinar_avaliar(
                df, features)
            conn_w.execute(, (
                ticker, setor, hoje,
                round(prob_alta, 4), sinal,
                round(metrics["accuracy"], 4),
                round(metrics["auc"], 4),
                round(metrics["pr_auc"], 4),
                round(metrics["f1"], 4),
                threshold,
                modelo_usado,
                len(features), len(df),
            ))
            conn_w.commit()
            resultados.append({
                "ticker"      : ticker,
                "nome"        : nome,
                "setor"       : setor,
                "prob_alta"   : prob_alta,
                "sinal"       : sinal,
                "acuracia"    : metrics["accuracy"],
                "auc"         : metrics["auc"],
                "pr_auc"      : metrics["pr_auc"],
                "f1"          : metrics["f1"],
                "threshold"   : threshold,
                "modelo_usado": modelo_usado,
                "base_pr"     : metrics["base_pr"],
                "auc_lr"      : metrics["auc_lr"],
                "pr_auc_lr"   : metrics["pr_auc_lr"],
                "auc_lgb"     : metrics["auc_lgb"],
                "pr_auc_lgb"  : metrics["pr_auc_lgb"],
                "n_pregoes"   : len(df),
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
        df_res["prob_alta"] * 0.40 +
        df_res["auc"]       * 0.35 +
        df_res["pr_auc"]    * 0.25
    )
    comprar = df_res[df_res["sinal"] == "COMPRAR"].sort_values("score", ascending=False)
    neutro  = df_res[df_res["sinal"] == "NEUTRO"].sort_values("score",  ascending=False)
    aguard  = df_res[df_res["sinal"] == "AGUARDAR"].sort_values("score", ascending=True)
    linha = "─" * 90
    def bloco(titulo, subset):
        if subset.empty:
            return
        print(f"\n{linha}")
        print(f"  {titulo}  ({len(subset)} ações)")
        print(linha)
        print(f"  {'Ticker':<12} {'Nome':<26} {'Prob':>6} {'AUC':>6} {'PR-AUC':>7} "
              f"{'Modelo':<10}  Setor")
        print(f"  {'-'*12} {'-'*26} {'-'*6} {'-'*6} {'-'*7} {'-'*10}  {'-'*20}")
        for _, r in subset.head(TOP_N).iterrows():
            nome_c = (r["nome"][:24] + "..") if len(str(r["nome"])) > 26 else r["nome"]
            print(f"  {r['ticker']:<12} {nome_c:<26} "
                  f"{r['prob_alta']:>5.1%} {r['auc']:>6.3f} {r['pr_auc']:>7.3f} "
                  f"{r['modelo_usado']:<10}  {r['setor']}")
    bloco("COMPRAR  ✓", comprar)
    bloco("NEUTRO   ~", neutro)
    bloco("AGUARDAR ✗", aguard)
    print(f"\n{linha}")
    print(f"  Total processado : {len(df_res)} ações")
    print(f"  COMPRAR          : {len(comprar)}  "
          f"(LightGBM: {(comprar['modelo_usado']=='LightGBM').sum()}  "
          f"LogReg: {(comprar['modelo_usado']=='LogReg').sum()})")
    print(f"  NEUTRO           : {len(neutro)}")
    print(f"  AGUARDAR         : {len(aguard)}")
    print(f"\n  Referência aleatória → ROC-AUC~0.500 | "
          f"PR-AUC~{df_res['base_pr'].mean():.3f}")
    print(linha)
def salvar_csv(df_res):
    if df_res.empty:
        return
    df_res["score"] = (
        df_res["prob_alta"] * 0.40 +
        df_res["auc"]       * 0.35 +
        df_res["pr_auc"]    * 0.25
    )
    df_out = df_res.sort_values("score", ascending=False).reset_index(drop=True)
    df_out.to_csv(CSV_SAIDA, index=False, float_format="%.4f")
    print(f"\n  CSV salvo: {CSV_SAIDA}  ({len(df_out)} linhas)")
def plotar_resultados(df_res):
    if df_res.empty:
        return
    df_res = df_res.copy()
    df_res["score"] = (
        df_res["prob_alta"] * 0.40 +
        df_res["auc"]       * 0.35 +
        df_res["pr_auc"]    * 0.25
    )
    df_sorted = df_res.sort_values("score", ascending=False).reset_index(drop=True)
    top    = df_sorted.head(TOP_N)
    bottom = df_sorted.tail(TOP_N).sort_values("score")
    comprar = df_res[df_res["sinal"] == "COMPRAR"]
    neutro  = df_res[df_res["sinal"] == "NEUTRO"]
    aguard  = df_res[df_res["sinal"] == "AGUARDAR"]
    fig = plt.figure(figsize=(18, 24))
    fig.suptitle(
        f"B3 Scanner v3.1  —  {date.today().strftime('%d/%m/%Y')}  "
        f"({len(df_res)} ações)  |  LogReg vs LightGBM",
        fontsize=13, fontweight="bold", y=0.99
    )
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.50, wspace=0.35)
    ax0 = fig.add_subplot(gs[0, :])
    cores_top = ["
                 else "
    bars = ax0.barh(top["ticker"][::-1], top["prob_alta"][::-1] * 100,
                    color=cores_top[::-1], alpha=0.85)
    ax0.axvline(50, color="gray", linestyle="--", linewidth=0.8)
    ax0.set_xlabel("Probabilidade de Alta (%)")
    ax0.set_title(f"Top {TOP_N} ações — score composto (prob × 0.40 + AUC × 0.35 + PR-AUC × 0.25)")
    ax0.set_xlim(0, 105)
    ax0.grid(alpha=0.15, axis="x")
    for bar, (_, r) in zip(bars[::-1], top.iterrows()):
        ax0.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 f"{r['prob_alta']:.1%}  {r['modelo_usado']}",
                 va="center", fontsize=8)
    ax1 = fig.add_subplot(gs[1, 0])
    sizes  = [len(comprar), len(neutro), len(aguard)]
    labels = [f"COMPRAR\n{len(comprar)}", f"NEUTRO\n{len(neutro)}", f"AGUARDAR\n{len(aguard)}"]
    wedges, texts, autotexts = ax1.pie(
        sizes, labels=labels,
        colors=["
        autopct="%1.0f%%", startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5}
    )
    for at in autotexts:
        at.set_fontsize(10)
    ax1.set_title("Distribuição de sinais")
    ax2 = fig.add_subplot(gs[1, 1])
    auc_setor = df_res.groupby("setor")["auc"].mean().sort_values(ascending=True)
    ax2.barh(auc_setor.index, auc_setor.values,
             color=["
             alpha=0.85)
    ax2.axvline(0.50, color="gray", linestyle="--", linewidth=0.8, label="Baseline")
    ax2.axvline(0.52, color="
    ax2.set_xlabel("ROC-AUC médio")
    ax2.set_title("ROC-AUC médio por setor")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.15, axis="x")
    ax2.set_xlim(0.44, 0.62)
    ax3 = fig.add_subplot(gs[2, 0])
    color_map = {"COMPRAR": "
    for sinal_g, grp in df_res.groupby("sinal"):
        ax3.scatter(grp["auc"], grp["prob_alta"] * 100,
                    c=color_map.get(sinal_g, "gray"),
                    label=sinal_g, alpha=0.75, s=50, edgecolors="none")
    ax3.axhline(50, color="gray", linestyle="--", linewidth=0.8)
    ax3.axvline(0.50, color="gray", linestyle="--", linewidth=0.8)
    ax3.set_xlabel("ROC-AUC")
    ax3.set_ylabel("Probabilidade de Alta (%)")
    ax3.set_title("Prob. alta vs AUC  (quadrante sup. dir. = melhor)")
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.15)
    ax4 = fig.add_subplot(gs[2, 1])
    df_comp = df_res[df_res["auc_lgb"] > 0].copy()
    if not df_comp.empty:
        ganho = df_comp["auc_lgb"] - df_comp["auc_lr"]
        cores_g = ["
        ax4.barh(df_comp["ticker"], ganho, color=cores_g, alpha=0.85)
        ax4.axvline(0, color="black", linewidth=0.8)
        ax4.set_xlabel("Δ AUC  (LightGBM − LogReg)")
        ax4.set_title("Ganho do LightGBM vs LogReg por ação\n(verde = LGB ganhou)")
        ax4.grid(alpha=0.15, axis="x")
    else:
        ax4.text(0.5, 0.5, "LightGBM não rodou\n(poucos dados de validação)",
                 ha="center", va="center", transform=ax4.transAxes, fontsize=11)
        ax4.set_title("Ganho LightGBM vs LogReg")
    ax5 = fig.add_subplot(gs[3, :])
    ax5.barh(bottom["ticker"], bottom["prob_alta"] * 100,
             color="
    ax5.axvline(50, color="gray", linestyle="--", linewidth=0.8)
    ax5.set_xlabel("Probabilidade de Alta (%)")
    ax5.set_title(f"Piores {TOP_N} — menor probabilidade de alta (candidatos a evitar)")
    ax5.set_xlim(0, 100)
    ax5.grid(alpha=0.15, axis="x")
    for i, (_, r) in enumerate(bottom.iterrows()):
        ax5.text(r["prob_alta"] * 100 + 0.5, i,
                 f"{r['prob_alta']:.1%}  AUC {r['auc']:.3f}",
                 va="center", fontsize=8)
    plt.savefig("scanner_v3_1.png", dpi=140, bbox_inches="tight")
    print("  Gráfico salvo: scanner_v3_1.png")
    plt.show()
def resumo_previsoes():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(, conn)
    top5 = pd.read_sql_query(, conn)
    conn.close()
    print("\n" + "=" * 62)
    print("  PREVISÕES SALVAS NO BANCO (última rodada)")
    print("=" * 62)
    print(df.to_string(index=False))
    print("\n  Top 5 COMPRAR (ordenados por AUC + prob_alta):")
    print(top5.to_string(index=False))
    print("=" * 62)
if __name__ == "__main__":
    print("\nIniciando B3 Scanner v3.1...")
    print(f"Banco         : {os.path.abspath(DB_PATH)}")
    print(f"Mín. pregões  : {MIN_PREGOES}")
    print(f"Threshold mín.: {MIN_CONF}")
    print(f"Split         : 70% treino / 15% validação / 15% teste")
    df_resultados, erros = rodar_scanner()
    imprimir_ranking(df_resultados)
    salvar_csv(df_resultados)
    plotar_resultados(df_resultados)
    resumo_previsoes()
    if erros:
        print(f"\n  {len(erros)} ações ignoradas:")
        for e in erros:
            print(f"    {e['ticker']}: {e['motivo']}")
    print("\nPronto!\n")