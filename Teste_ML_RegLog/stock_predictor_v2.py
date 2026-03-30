import warnings
warnings.filterwarnings("ignore")
import requests          
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec  
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve, roc_auc_score  
)
TICKER      = "PETR4.SA"
START_DATE  = "2015-01-01"
END_DATE    = "2024-12-31"
N_SPLITS    = 10
MIN_CONF    = 0.52   
MACRO_TICKERS = {
    "brent"  : "BZ=F",      
    "dolar"  : "USDBRL=X",  
    "ibov"   : "^BVSP",     
    "sp500"  : "^GSPC",     
    "vix"    : "^VIX",      
    "ouro"   : "GC=F",      
}
def flatten_cols(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df
def download_main(ticker, start, end):
    print(f"\n[1/5] Baixando dados de {ticker}...")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"Nenhum dado para {ticker}. Verifique o ticker.")
    df = flatten_cols(df)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)
    print(f"      {len(df)} pregões carregados.")
    return df
def download_macro(start, end, index):
    print("[2/5] Baixando features macroeconômicas...")
    macro_frames = {}
    for name, sym in MACRO_TICKERS.items():
        try:
            raw = yf.download(sym, start=start, end=end,
                              auto_adjust=True, progress=False)
            raw = flatten_cols(raw)
            if raw.empty:
                print(f"      ⚠ {name} ({sym}): sem dados, ignorado.")
                continue
            close = raw["Close"].squeeze()
            macro_frames[f"macro_{name}_ret1"]  = close.pct_change(1)   
            macro_frames[f"macro_{name}_ret5"]  = close.pct_change(5)   
            macro_frames[f"macro_{name}_level"] = close                  
            print(f"      ✓ {name} ({sym})")
        except Exception as e:
            print(f"      ✗ {name} ({sym}): {e}")
    if not macro_frames:
        print("      Nenhum dado macro disponível, continuando sem eles.")
        return pd.DataFrame(index=index)
    macro_df = pd.DataFrame(macro_frames)
    macro_df = macro_df.reindex(index).ffill().bfill()
    for col in [c for c in macro_df.columns if c.endswith("_level")]:
        roll = macro_df[col].rolling(252)
        macro_df[col] = (macro_df[col] - roll.mean()) / (roll.std() + 1e-10)
    return macro_df
def download_fear_greed(index):
    print("[3/5] Baixando índice Fear & Greed (sentimento)...")
    url = "https://api.alternative.me/fng/?limit=2000&format=json"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json().get("data", [])
        if not data:
            raise ValueError("Resposta vazia")
        fg_df = pd.DataFrame(data)
        fg_df["timestamp"] = pd.to_datetime(fg_df["timestamp"].astype(int), unit="s")
        fg_df = fg_df.set_index("timestamp").sort_index()
        fg_df.index = fg_df.index.normalize()  
        fg_df["fg_value"] = pd.to_numeric(fg_df["value"], errors="coerce")
        fg_df["fg_ma7"]    = fg_df["fg_value"].rolling(7).mean()
        fg_df["fg_change"] = fg_df["fg_value"].diff(1)
        fg_df["fg_zone"] = pd.cut(
            fg_df["fg_value"],
            bins=[0, 25, 45, 55, 75, 100],
            labels=["extreme_fear", "fear", "neutral", "greed", "extreme_greed"]
        ).cat.codes
        result = fg_df[["fg_value", "fg_ma7", "fg_change", "fg_zone"]]
        result = result.reindex(index).ffill().bfill()
        print(f"      ✓ Fear & Greed carregado ({len(fg_df)} dias).")
        return result
    except Exception as e:
        print(f"      ⚠ Fear & Greed indisponível ({e}), usando neutro.")
        return pd.DataFrame(
            {"fg_value": 50, "fg_ma7": 50, "fg_change": 0, "fg_zone": 2},
            index=index
        )
def compute_rsi(series, window=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()     
    loss  = (-delta.clip(upper=0)).rolling(window).mean()  
    rs    = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))
def add_technical_features(df):
    print("[4/5] Calculando indicadores técnicos...")
    for lag in [1, 2, 3, 5, 10]:
        df[f"return_lag{lag}"] = df["Close"].pct_change(lag)
    df["rsi_9"]  = compute_rsi(df["Close"], 9)
    df["rsi_14"] = compute_rsi(df["Close"], 14)
    df["rsi_21"] = compute_rsi(df["Close"], 21)
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]
    for w in [5, 10, 20, 50]:
        df[f"sma_{w}"] = df["Close"].rolling(w).mean()
    df["sma_ratio_5_20"]  = df["sma_5"]  / df["sma_20"]  - 1
    df["sma_ratio_10_50"] = df["sma_10"] / df["sma_50"]  - 1
    df["price_vs_sma20"]  = df["Close"]  / df["sma_20"]  - 1
    roll20 = df["Close"].rolling(20)
    bb_mid = roll20.mean()
    bb_std = roll20.std()
    df["bb_upper"] = bb_mid + 2 * bb_std
    df["bb_lower"] = bb_mid - 2 * bb_std
    df["bb_pct"]   = (df["Close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-10)
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (bb_mid + 1e-10)  
    ret = df["Close"].pct_change()
    df["vol_5"]     = ret.rolling(5).std()
    df["vol_20"]    = ret.rolling(20).std()
    df["vol_ratio"] = df["vol_5"] / (df["vol_20"] + 1e-10)
    df["volume_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
    df["volume_trend"] = df["Volume"].rolling(5).mean() / df["Volume"].rolling(20).mean()
    df["hl_ratio"]   = (df["High"] - df["Low"]) / (df["Close"] + 1e-10)
    df["body_ratio"] = abs(df["Close"] - df["Open"]) / (df["High"] - df["Low"] + 1e-10)
    df["gap"]        = (df["Open"] - df["Close"].shift(1)) / (df["Close"].shift(1) + 1e-10)
    df["momentum_10"] = df["Close"] / df["Close"].shift(10) - 1
    df["momentum_20"] = df["Close"] / df["Close"].shift(20) - 1
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df = df.iloc[:-1]   
    df.dropna(inplace=True)
    print(f"      {len(df)} pregões com indicadores completos.")
    return df
TECH_FEATURES = [
    "return_lag1", "return_lag2", "return_lag3", "return_lag5", "return_lag10",
    "rsi_9", "rsi_14", "rsi_21",
    "macd", "macd_signal", "macd_hist",
    "sma_ratio_5_20", "sma_ratio_10_50", "price_vs_sma20",
    "bb_pct", "bb_width",
    "vol_5", "vol_20", "vol_ratio",
    "volume_ratio", "volume_trend",
    "hl_ratio", "body_ratio", "gap",
    "momentum_10", "momentum_20",
]
def build_dataset(ticker, start, end):
    df_main  = download_main(ticker, start, end)
    df_tech  = add_technical_features(df_main.copy())
    macro_df = download_macro(start, end, df_tech.index)
    fg_df    = download_fear_greed(df_tech.index)
    df_full = df_tech.join(macro_df, how="left").join(fg_df, how="left")
    df_full.ffill(inplace=True)   
    df_full.dropna(inplace=True)
    macro_cols   = [c for c in macro_df.columns if c in df_full.columns]
    fg_cols      = [c for c in fg_df.columns    if c in df_full.columns]
    all_features = [f for f in TECH_FEATURES + macro_cols + fg_cols if f in df_full.columns]
    print(f"\n      Total de features: {len(all_features)}")
    print(f"      Técnicas: {len(TECH_FEATURES)} | Macro: {len(macro_cols)} | Sentimento: {len(fg_cols)}")
    return df_full, all_features
def find_best_threshold(y_true, y_proba):
    best_t, best_f1 = MIN_CONF, 0
    for t in np.arange(MIN_CONF, 0.71, 0.01):
        pred = (y_proba >= t).astype(int)
        if pred.sum() == 0:
            continue  
        f = f1_score(y_true, pred, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return round(best_t, 2), round(best_f1, 4)
def train_model(df, features):
    print(f"\n[5/5] Treinando Regressão Logística ({N_SPLITS} folds)...")
    X = df[features].values
    y = df["target"].values
    scaler = StandardScaler()
    tscv   = TimeSeriesSplit(n_splits=N_SPLITS)
    fold_scores = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        m = LogisticRegression(max_iter=2000, C=0.05, solver="lbfgs")
        m.fit(X_tr_s, y_tr)
        acc = accuracy_score(y_te, m.predict(X_te_s))
        fold_scores.append(acc)
        print(f"      Fold {fold}: acurácia = {acc:.2%}")
    print(f"      Média: {np.mean(fold_scores):.2%} | Desvio: {np.std(fold_scores):.2%}")
    X_s = scaler.fit_transform(X)
    final = LogisticRegression(max_iter=2000, C=0.05, solver="lbfgs")
    final.fit(X_s, y)
    return final, scaler, fold_scores
def evaluate_test(df, features):
    split  = int(len(df) * 0.80)
    tr_df  = df.iloc[:split]
    te_df  = df.iloc[split:]
    scaler_eval = StandardScaler()
    X_tr = scaler_eval.fit_transform(tr_df[features].values)
    X_te = scaler_eval.transform(te_df[features].values)
    y_tr, y_te = tr_df["target"].values, te_df["target"].values
    model = LogisticRegression(max_iter=2000, C=0.05, solver="lbfgs")
    model.fit(X_tr, y_tr)
    proba    = model.predict_proba(X_te)[:, 1]    
    tr_proba = model.predict_proba(X_tr)[:, 1]    
    threshold, best_f1_tr = find_best_threshold(y_tr, tr_proba)
    print(f"\n      Threshold automático: {threshold:.2f}  (F1 no treino: {best_f1_tr:.2%})")
    y_pred = (proba >= threshold).astype(int)
    print("\n  ── Relatório de classificação ──")
    print(classification_report(y_te, y_pred, target_names=["BAIXA", "ALTA"]))
    metrics = {
        "accuracy" : accuracy_score(y_te, y_pred),
        "precision": precision_score(y_te, y_pred, zero_division=0),
        "recall"   : recall_score(y_te, y_pred, zero_division=0),
        "f1"       : f1_score(y_te, y_pred, zero_division=0),
        "auc"      : roc_auc_score(y_te, proba),  
        "threshold": threshold,
    }
    return metrics, y_te, y_pred, proba, te_df, model, scaler_eval
def predict_tomorrow(df, features, model, scaler_eval, threshold):
    last_s  = scaler_eval.transform(df[features].iloc[[-1]])
    prob_up = model.predict_proba(last_s)[0, 1]
    if prob_up >= threshold:
        signal = "COMPRAR"
    elif prob_up >= threshold - 0.03:
        signal = "NEUTRO"
    else:
        signal = "VENDER/AGUARDAR"
    return prob_up, signal
def plot_all(df, te_df, y_te, y_pred, proba, metrics,
             fold_scores, features, model, scaler_eval, prob_tomorrow, signal):
    fig = plt.figure(figsize=(16, 16))
    fig.suptitle(f"Previsão de Ações v2 — {TICKER}  |  Regressão Logística",
                 fontsize=13, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.32)
    ax0 = fig.add_subplot(gs[0, :])  
    ax0.plot(df.index, df["Close"], color="
    ax0.axvspan(te_df.index[0], df.index[-1], alpha=0.10, color="
                label="Conjunto de teste")
    ax0.set_title("Preço de fechamento (laranja = conjunto de teste)")
    ax0.set_ylabel("Preço (R$)")
    ax0.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax0.legend(fontsize=9)
    ax0.grid(alpha=0.15)
    ax1 = fig.add_subplot(gs[1, :])
    colors = ["
    ax1.bar(te_df.index, proba, color=colors, width=1.0, alpha=0.75)
    ax1.axhline(metrics["threshold"], color="
                linewidth=1, label=f"Threshold: {metrics['threshold']:.2f}")
    ax1.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, label="50%")
    ax1.set_title("Probabilidade de ALTA — conjunto de teste  (verde = previu ALTA)")
    ax1.set_ylim(0, 1)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.15)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
    ax2 = fig.add_subplot(gs[2, 0])
    ConfusionMatrixDisplay(confusion_matrix(y_te, y_pred),
                           display_labels=["BAIXA", "ALTA"]).plot(
        ax=ax2, colorbar=False, cmap="Blues")
    ax2.set_title("Matriz de confusão")
    ax3 = fig.add_subplot(gs[2, 1])
    fpr, tpr, _ = roc_curve(y_te, proba)
    ax3.plot(fpr, tpr, color="
             label=f"AUC = {metrics['auc']:.3f}")
    ax3.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Aleatório")
    ax3.set_xlabel("FPR")
    ax3.set_ylabel("TPR")
    ax3.set_title("Curva ROC")
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.15)
    ax4 = fig.add_subplot(gs[3, 0])
    bar_c = ["
    ax4.bar([f"Fold {i+1}" for i in range(N_SPLITS)],
            [s * 100 for s in fold_scores], color=bar_c, alpha=0.85)
    ax4.axhline(50, color="gray", linestyle="--", linewidth=0.8, label="Baseline 50%")
    ax4.axhline(np.mean(fold_scores) * 100, color="
                label=f"Média {np.mean(fold_scores):.1%}")
    ax4.set_ylim(40, 70)
    ax4.set_ylabel("Acurácia (%)")
    ax4.set_title("Acurácia por fold")
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.15, axis="y")
    ax5 = fig.add_subplot(gs[3, 1])
    coefs   = model.coef_[0]
    top_idx = np.argsort(np.abs(coefs))[-15:]
    feat_names = [features[i] for i in top_idx]
    feat_vals  = [coefs[i]    for i in top_idx]
    bar_c2 = ["
    ax5.barh(feat_names, feat_vals, color=bar_c2, alpha=0.85)
    ax5.axvline(0, color="black", linewidth=0.8)
    ax5.set_title("Top 15 coeficientes\n(verde = favorece ALTA)")
    ax5.set_xlabel("Coeficiente padronizado")
    ax5.grid(alpha=0.15, axis="x")
    color_map = {"COMPRAR": "
    bg = color_map.get(signal, "
    fig.text(0.73, 0.955, f"Amanhã: {signal}  ({prob_tomorrow:.1%})",
             ha="center", va="center", fontsize=12, fontweight="bold",
             color="white", bbox=dict(facecolor=bg, boxstyle="round,pad=0.5",
                                      edgecolor="none"))
    plt.savefig("resultado_v2.png", dpi=140, bbox_inches="tight")
    print("  Gráfico salvo: resultado_v2.png")
    plt.show()
def print_summary(metrics, prob_tomorrow, signal):
    line = "=" * 54
    print(f"\n{line}")
    print("  RESUMO FINAL — Regressão Logística v2")
    print(line)
    print(f"  Threshold usado : {metrics['threshold']:.2f}")   
    print(f"  AUC-ROC         : {metrics['auc']:.4f}")         
    print(f"  Acurácia        : {metrics['accuracy']:.2%}")
    print(f"  Precisão        : {metrics['precision']:.2%}")
    print(f"  Recall          : {metrics['recall']:.2%}")
    print(f"  F1-Score        : {metrics['f1']:.2%}")
    print(f"{'-'*54}")
    print(f"  Previsão amanhã : {signal}")
    print(f"  Probabilidade   : {prob_tomorrow:.2%}")
    print(line)
if __name__ == "__main__":
    df, features = build_dataset(TICKER, START_DATE, END_DATE)
    model, scaler, fold_scores = train_model(df, features)
    metrics, y_te, y_pred, proba, te_df, eval_model, scaler_eval = evaluate_test(df, features)
    threshold = metrics["threshold"]
    prob_tomorrow, signal = predict_tomorrow(df, features, eval_model, scaler_eval, threshold)
    print(f"\n  Probabilidade de ALTA amanhã : {prob_tomorrow:.2%}")
    print(f"  Sinal                        : {signal}")
    plot_all(df, te_df, y_te, y_pred, proba, metrics,
             fold_scores, features, eval_model, scaler_eval, prob_tomorrow, signal)
    print_summary(metrics, prob_tomorrow, signal)