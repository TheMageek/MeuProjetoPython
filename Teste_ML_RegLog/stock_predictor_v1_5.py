import warnings
warnings.filterwarnings("ignore")
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, roc_auc_score,
    average_precision_score, roc_curve, precision_recall_curve
)
import lightgbm as lgb
TICKER     = "PETR4.SA"
START_DATE = "2020-01-01"
END_DATE   = "2024-12-31"
N_SPLITS   = 5
SEED       = 42
FEATURE_COLS = [
    "return_lag1", "return_lag2", "return_lag3", "return_lag5",
    "rsi", "macd", "macd_signal", "macd_hist",
    "sma_ratio_5_20", "sma_ratio_5_50",
    "bb_pct", "volatility_5", "volatility_20",
    "volume_ratio", "hl_ratio"
]
def download_data(ticker, start, end):
    print(f"\n[1/5] Baixando dados de {ticker}...")
    df = yf.download(ticker, start=start, end=end,
                     auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"Nenhum dado para {ticker}.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)
    print(f"      {len(df)} pregões carregados ({start} → {end})")
    return df
def add_features(df):
    print("[2/5] Calculando indicadores técnicos...")
    for lag in [1, 2, 3, 5]:
        df[f"return_lag{lag}"] = df["Close"].pct_change(lag)
    def rsi(series, w=14):
        d = series.diff()
        g = d.clip(lower=0).rolling(w).mean()
        l = (-d.clip(upper=0)).rolling(w).mean()
        return 100 - (100 / (1 + g / (l + 1e-10)))
    df["rsi"] = rsi(df["Close"])
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]
    df["sma_5"]  = df["Close"].rolling(5).mean()
    df["sma_20"] = df["Close"].rolling(20).mean()
    df["sma_50"] = df["Close"].rolling(50).mean()
    df["sma_ratio_5_20"] = df["sma_5"] / df["sma_20"] - 1
    df["sma_ratio_5_50"] = df["sma_5"] / df["sma_50"] - 1
    roll = df["Close"].rolling(20)
    bb_mid = roll.mean(); bb_std = roll.std()
    df["bb_upper"] = bb_mid + 2 * bb_std
    df["bb_lower"] = bb_mid - 2 * bb_std
    df["bb_pct"]   = (df["Close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-10)
    ret = df["Close"].pct_change()
    df["volatility_5"]  = ret.rolling(5).std()
    df["volatility_20"] = ret.rolling(20).std()
    df["volume_ratio"]  = df["Volume"] / df["Volume"].rolling(20).mean()
    df["hl_ratio"]      = (df["High"] - df["Low"]) / df["Close"]
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df = df.iloc[:-1].dropna()
    print(f"      {len(df)} dias com indicadores completos.")
    return df
def split_temporal(df):
    print("[3/5] Dividindo dados (treino 70% / validação 15% / teste 15%)...")
    n       = len(df)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.85)          
    df_train = df.iloc[:n_train]
    df_val   = df.iloc[n_train:n_val]
    df_test  = df.iloc[n_val:]
    X_train = df_train[FEATURE_COLS].fillna(0)
    y_train = df_train["target"]
    X_val   = df_val[FEATURE_COLS].fillna(0)
    y_val   = df_val["target"]
    X_test  = df_test[FEATURE_COLS].fillna(0)
    y_test  = df_test["target"]
    print(f"      Treino    : {len(X_train):,} dias  ({y_train.mean()*100:.1f}% alta)")
    print(f"      Validação : {len(X_val):,} dias  ({y_val.mean()*100:.1f}% alta)")
    print(f"      Teste     : {len(X_test):,} dias  ({y_test.mean()*100:.1f}% alta)")
    print(f"      Referência aleatória → ROC-AUC~0.5000 | PR-AUC~{y_test.mean():.4f}")
    return (X_train, y_train, X_val, y_val, X_test, y_test,
            df_train, df_val, df_test)
def treinar_logistica(X_train, y_train, X_val, y_val, X_test, y_test):
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            C=0.1,
            random_state=SEED
        ))
    ])
    pipe.fit(X_train, y_train)
    prob_val  = pipe.predict_proba(X_val)[:, 1]
    prob_test = pipe.predict_proba(X_test)[:, 1]
    auc_val  = roc_auc_score(y_val,  prob_val)
    prc_val  = average_precision_score(y_val,  prob_val)
    auc_test = roc_auc_score(y_test, prob_test)
    prc_test = average_precision_score(y_test, prob_test)
    print("\n  ── Regressão Logística (Pipeline) ──")
    print(f"     Validação : ROC-AUC={auc_val:.4f} | PR-AUC={prc_val:.4f}")
    print(f"     Teste     : ROC-AUC={auc_test:.4f} | PR-AUC={prc_test:.4f}")
    return pipe, prob_val, prob_test, auc_val, prc_val, auc_test, prc_test
def treinar_lightgbm(X_train, y_train, X_val, y_val, X_test, y_test):
    params = {
        "objective"         : "binary",
        "metric"            : ["auc", "average_precision"],
        "n_estimators"      : 2000,
        "learning_rate"     : 0.05,
        "num_leaves"        : 31,
        "min_child_samples" : 20,
        "subsample"         : 0.8,
        "subsample_freq"    : 1,
        "colsample_bytree"  : 0.8,
        "reg_alpha"         : 0.1,
        "reg_lambda"        : 1.0,
        "is_unbalance"      : True,
        "random_state"      : SEED,
        "verbose"           : -1,
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=-1),
        ]
    )
    prob_val  = model.predict_proba(X_val)[:, 1]
    prob_test = model.predict_proba(X_test)[:, 1]
    auc_val  = roc_auc_score(y_val,  prob_val)
    prc_val  = average_precision_score(y_val,  prob_val)
    auc_test = roc_auc_score(y_test, prob_test)
    prc_test = average_precision_score(y_test, prob_test)
    print(f"\n  ── LightGBM ({model.best_iteration_} árvores via early stopping) ──")
    print(f"     Validação : ROC-AUC={auc_val:.4f} | PR-AUC={prc_val:.4f}")
    print(f"     Teste     : ROC-AUC={auc_test:.4f} | PR-AUC={prc_test:.4f}")
    return model, prob_val, prob_test, auc_val, prc_val, auc_test, prc_test
def cross_validate_lr(df):
    print(f"\n[4/5] Validação cruzada temporal ({N_SPLITS} folds) — LogReg...")
    X = df[FEATURE_COLS].fillna(0).values
    y = df["target"].values
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    fold_aucs = []
    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X), 1):
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("model",   LogisticRegression(
                class_weight="balanced", max_iter=1000,
                C=0.1, random_state=SEED
            ))
        ])
        pipe.fit(X[tr_idx], y[tr_idx])
        prob = pipe.predict_proba(X[te_idx])[:, 1]
        try:
            auc = roc_auc_score(y[te_idx], prob)
        except Exception:
            auc = 0.5
        fold_aucs.append(auc)
        print(f"      Fold {fold}: ROC-AUC = {auc:.4f}")
    print(f"      Média: {np.mean(fold_aucs):.4f} | Desvio: {np.std(fold_aucs):.4f}")
    return fold_aucs
def predict_tomorrow(df, pipe_lr, model_lgb,
                     auc_test_lr, auc_test_lgb):
    print("\n[5/5] Gerando previsão para o próximo pregão...")
    last = df[FEATURE_COLS].iloc[[-1]].fillna(0)
    prob_lr  = pipe_lr.predict_proba(last)[0, 1]
    prob_lgb = model_lgb.predict_proba(last)[0, 1]
    if auc_test_lgb >= auc_test_lr:
        prob_final = prob_lgb
        modelo_usado = "LightGBM"
    else:
        prob_final = prob_lr
        modelo_usado = "Regressão Logística"
    sinal = "ALTA (comprar)" if prob_final >= 0.50 else "BAIXA (aguardar)"
    print(f"      Modelo selecionado : {modelo_usado} (maior AUC no teste)")
    print(f"      LogReg prob. alta  : {prob_lr:.2%}")
    print(f"      LightGBM prob. alta: {prob_lgb:.2%}")
    print(f"      Previsão final     : {sinal}  ({prob_final:.2%})")
    return prob_lr, prob_lgb, prob_final, sinal, modelo_usado
def plot_results(df, df_test, y_test,
                 prob_test_lr, prob_test_lgb,
                 fold_aucs, pipe_lr,
                 auc_test_lr, prc_test_lr,
                 auc_test_lgb, prc_test_lgb,
                 prob_lr, prob_lgb, sinal):
    fig = plt.figure(figsize=(18, 16))
    fig.suptitle(
        f"Previsão de Ações v1.5 — {TICKER}  |  LogReg vs LightGBM",
        fontsize=13, fontweight="bold", y=0.99
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(df.index, df["Close"], color="
    ax0.axvspan(df_test.index[0], df.index[-1],
                alpha=0.10, color="
    ax0.set_title("Preço de fechamento  —  fundo laranja = conjunto de teste")
    ax0.set_ylabel("Preço (R$)")
    ax0.legend(fontsize=9)
    ax0.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax0.grid(alpha=0.15)
    ax1 = fig.add_subplot(gs[1, 0])
    for prob, label, color, auc in [
        (prob_test_lr,  f"LogReg  AUC={auc_test_lr:.3f}",  "
        (prob_test_lgb, f"LightGBM AUC={auc_test_lgb:.3f}", "
    ]:
        fpr, tpr, _ = roc_curve(y_test, prob)
        ax1.plot(fpr, tpr, color=color, linewidth=2, label=label)
    ax1.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Aleatório")
    ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR")
    ax1.set_title("Curva ROC")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.15)
    ax2 = fig.add_subplot(gs[1, 1])
    base_pr = y_test.mean()
    for prob, label, color, prc in [
        (prob_test_lr,  f"LogReg  PR-AUC={prc_test_lr:.3f}",  "
        (prob_test_lgb, f"LightGBM PR-AUC={prc_test_lgb:.3f}", "
    ]:
        prec, rec, _ = precision_recall_curve(y_test, prob)
        ax2.plot(rec, prec, color=color, linewidth=2, label=label)
    ax2.axhline(base_pr, color="gray", linestyle="--",
                linewidth=0.8, label=f"Aleatório ({base_pr:.2f})")
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precisão")
    ax2.set_title("Curva Precision-Recall")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.15)
    ax3 = fig.add_subplot(gs[1, 2])
    metricas  = ["ROC-AUC", "PR-AUC"]
    vals_lr   = [auc_test_lr,  prc_test_lr]
    vals_lgb  = [auc_test_lgb, prc_test_lgb]
    x = np.arange(len(metricas))
    ax3.bar(x - 0.2, vals_lr,  0.35, label="LogReg",   color="
    ax3.bar(x + 0.2, vals_lgb, 0.35, label="LightGBM", color="
    ax3.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Baseline ROC")
    ax3.axhline(base_pr, color="gray", linestyle=":",
                linewidth=0.8, label=f"Baseline PR ({base_pr:.2f})")
    ax3.set_xticks(x); ax3.set_xticklabels(metricas)
    ax3.set_ylim(0.3, 0.8); ax3.set_title("Comparativo de métricas (teste)")
    ax3.legend(fontsize=8); ax3.grid(alpha=0.15, axis="y")
    for xi, (vl, vg) in enumerate(zip(vals_lr, vals_lgb)):
        ax3.text(xi - 0.2, vl + 0.005, f"{vl:.3f}", ha="center", fontsize=8)
        ax3.text(xi + 0.2, vg + 0.005, f"{vg:.3f}", ha="center", fontsize=8)
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.bar([f"F{i+1}" for i in range(len(fold_aucs))],
            fold_aucs,
            color=["
            alpha=0.85)
    ax4.axhline(0.50, color="gray", linestyle="--", linewidth=0.8, label="Baseline 0.50")
    ax4.axhline(np.mean(fold_aucs), color="
                label=f"Média {np.mean(fold_aucs):.3f}")
    ax4.set_ylim(0.40, 0.65); ax4.set_ylabel("ROC-AUC")
    ax4.set_title("ROC-AUC por fold — CV temporal")
    ax4.legend(fontsize=8); ax4.grid(alpha=0.15, axis="y")
    ax5 = fig.add_subplot(gs[2, 1])
    imp = pd.Series(
        model_lgb_global.feature_importances_,
        index=FEATURE_COLS
    ).sort_values(ascending=True)
    imp.plot(kind="barh", ax=ax5, color="
    ax5.set_title("Importância de features — LightGBM")
    ax5.set_xlabel("Importância (gain)")
    ax5.grid(alpha=0.15, axis="x")
    ax6 = fig.add_subplot(gs[2, 2])
    lr_model = pipe_lr.named_steps["model"]
    coefs    = lr_model.coef_[0]
    sorted_i = np.argsort(coefs)
    ax6.barh([FEATURE_COLS[i] for i in sorted_i],
             [coefs[i] for i in sorted_i],
             color=["
             alpha=0.85)
    ax6.axvline(0, color="black", linewidth=0.8)
    ax6.set_title("Coeficientes — Regressão Logística\n(verde = favorece ALTA)")
    ax6.set_xlabel("Coeficiente padronizado")
    ax6.grid(alpha=0.15, axis="x")
    plt.savefig("resultado_v1_5.png", dpi=140, bbox_inches="tight")
    print("\n  Gráfico salvo: resultado_v1_5.png")
    plt.show()
def print_summary(auc_val_lr, prc_val_lr, auc_test_lr, prc_test_lr,
                  auc_val_lgb, prc_val_lgb, auc_test_lgb, prc_test_lgb,
                  base_pr, prob_lr, prob_lgb, sinal, modelo_usado):
    linha = "=" * 58
    print(f"\n{linha}")
    print("  RESUMO FINAL — v1.5  (LogReg vs LightGBM)")
    print(linha)
    print(f"  {'Modelo':<18} {'Val ROC':>8} {'Val PR':>7} {'Tst ROC':>8} {'Tst PR':>7}")
    print(f"  {'-'*18} {'-'*8} {'-'*7} {'-'*8} {'-'*7}")
    print(f"  {'LogReg':<18} {auc_val_lr:>8.4f} {prc_val_lr:>7.4f} "
          f"{auc_test_lr:>8.4f} {prc_test_lr:>7.4f}")
    print(f"  {'LightGBM':<18} {auc_val_lgb:>8.4f} {prc_val_lgb:>7.4f} "
          f"{auc_test_lgb:>8.4f} {prc_test_lgb:>7.4f}")
    print(f"  {'Aleatório':<18} {'~0.5000':>8} {base_pr:>7.4f} "
          f"{'~0.5000':>8} {base_pr:>7.4f}")
    print(f"{'-'*58}")
    print(f"  Modelo selecionado : {modelo_usado}")
    print(f"  LogReg  prob. alta : {prob_lr:.2%}")
    print(f"  LightGBM prob. alta: {prob_lgb:.2%}")
    print(f"  Previsão amanhã    : {sinal}")
    print(linha)
    print("\n  AVISO: modelo educacional. Não é recomendação")
    print("  de investimento. Consulte um profissional.\n")
model_lgb_global = None   
if __name__ == "__main__":
    df_raw = download_data(TICKER, START_DATE, END_DATE)
    df     = add_features(df_raw)
    (X_train, y_train, X_val, y_val, X_test, y_test,
     df_train, df_val, df_test) = split_temporal(df)
    print(f"\n[3/5] Treinando modelos...")
    (pipe_lr,
     prob_val_lr, prob_test_lr,
     auc_val_lr,  prc_val_lr,
     auc_test_lr, prc_test_lr) = treinar_logistica(
        X_train, y_train, X_val, y_val, X_test, y_test)
    (model_lgb,
     prob_val_lgb, prob_test_lgb,
     auc_val_lgb,  prc_val_lgb,
     auc_test_lgb, prc_test_lgb) = treinar_lightgbm(
        X_train, y_train, X_val, y_val, X_test, y_test)
    model_lgb_global = model_lgb   
    print(f"\n  Ganho LightGBM vs LogReg:")
    print(f"    ROC-AUC +{auc_test_lgb - auc_test_lr:+.4f} | "
          f"PR-AUC +{prc_test_lgb - prc_test_lr:+.4f}")
    fold_aucs = cross_validate_lr(df)
    (prob_lr, prob_lgb, prob_final,
     sinal, modelo_usado) = predict_tomorrow(
        df, pipe_lr, model_lgb, auc_test_lr, auc_test_lgb)
    base_pr = float(y_test.mean())
    plot_results(
        df, df_test, y_test,
        prob_test_lr, prob_test_lgb,
        fold_aucs, pipe_lr,
        auc_test_lr, prc_test_lr,
        auc_test_lgb, prc_test_lgb,
        prob_lr, prob_lgb, sinal
    )
    print_summary(
        auc_val_lr, prc_val_lr, auc_test_lr, prc_test_lr,
        auc_val_lgb, prc_val_lgb, auc_test_lgb, prc_test_lgb,
        base_pr, prob_lr, prob_lgb, sinal, modelo_usado
    )