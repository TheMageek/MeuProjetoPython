import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)

TICKER       = "PETR4.SA"   
START_DATE   = "2015-01-01"
END_DATE     = "2024-12-31"
THRESHOLD    = 0.52         
N_SPLITS     = 10           

def download_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)

    if df.empty:
        raise ValueError(f"Erro: Ticker {ticker} não encontrado.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)
    return df

def add_features(df):
    for lag in [1, 2, 3, 5]:
        df[f"return_lag{lag}"] = df["Close"].pct_change(lag)

    def compute_rsi(series, window=14):
        delta = series.diff()                                     
        gain  = delta.clip(lower=0).rolling(window).mean()        
        loss  = (-delta.clip(upper=0)).rolling(window).mean()     
        rs    = gain / (loss + 1e-10)                             
        return 100 - (100 / (1 + rs))                             

    df["rsi"] = compute_rsi(df["Close"])

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    df["sma_5"]  = df["Close"].rolling(5).mean()
    df["sma_20"] = df["Close"].rolling(20).mean()
    df["sma_50"] = df["Close"].rolling(50).mean()

    df["sma_ratio_5_20"]  = df["sma_5"]  / df["sma_20"]  - 1
    df["sma_ratio_5_50"]  = df["sma_5"]  / df["sma_50"]  - 1

    rolling = df["Close"].rolling(20)
    df["bb_mid"]   = rolling.mean()
    df["bb_std"]   = rolling.std()
    df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]
    df["bb_pct"]   = (df["Close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-10)

    df["volatility_5"]  = df["Close"].pct_change().rolling(5).std()
    df["volatility_20"] = df["Close"].pct_change().rolling(20).std()

    df["volume_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
    df["hl_ratio"] = (df["High"] - df["Low"]) / df["Close"]

    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    df = df.iloc[:-1]
    df.dropna(inplace=True)
    return df

FEATURE_COLS = [
    "return_lag1", "return_lag2", "return_lag3", "return_lag5",
    "rsi", "macd", "macd_signal", "macd_hist",
    "sma_ratio_5_20", "sma_ratio_5_50",
    "bb_pct", "volatility_5", "volatility_20",
    "volume_ratio", "hl_ratio"
]

def train_and_evaluate(df):
    X = df[FEATURE_COLS].values
    y = df["target"].values

    scaler = StandardScaler()
    tscv   = TimeSeriesSplit(n_splits=N_SPLITS)

    fold_scores = []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        model = LogisticRegression(max_iter=1000, C=0.1, solver="lbfgs")
        model.fit(X_train_s, y_train)

        y_pred = model.predict(X_test_s)
        fold_scores.append(accuracy_score(y_test, y_pred))

    X_scaled = scaler.fit_transform(X)
    final_model = LogisticRegression(max_iter=1000, C=0.1, solver="lbfgs")
    final_model.fit(X_scaled, y)

    return final_model, scaler, fold_scores

def evaluate_test_set(df, model, scaler):
    split    = int(len(df) * 0.80)
    train_df = df.iloc[:split]
    test_df  = df.iloc[split:]

    X_train = scaler.fit_transform(train_df[FEATURE_COLS].values)
    X_test  = scaler.transform(test_df[FEATURE_COLS].values)
    y_train = train_df["target"].values
    y_test  = test_df["target"].values

    m = LogisticRegression(max_iter=1000, C=0.1, solver="lbfgs")
    m.fit(X_train, y_train)

    proba  = m.predict_proba(X_test)[:, 1]
    y_pred = (proba >= THRESHOLD).astype(int)

    metrics = {
        "accuracy" : accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall"   : recall_score(y_test, y_pred),
        "f1"       : f1_score(y_test, y_pred),
    }

    return metrics, y_test, y_pred, proba, test_df

def predict_tomorrow(df, model, scaler):
    last_row = df[FEATURE_COLS].iloc[[-1]]
    last_scaled = scaler.transform(last_row)
    prob_up = model.predict_proba(last_scaled)[0, 1]
    direction = "ALTA (comprar)" if prob_up >= THRESHOLD else "BAIXA (aguardar/vender)"
    return prob_up, direction

def plot_results(df, test_df, y_test, y_pred, proba, metrics, fold_scores, prob_tomorrow):
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(f"Análise de Previsão — {TICKER}", fontsize=14, fontweight="bold", y=0.98)

    ax1 = fig.add_subplot(3, 2, (1, 2))
    split_date = test_df.index[0]
    ax1.plot(df.index, df["Close"], color="#444", linewidth=0.8, label="Preço")
    ax1.axvspan(split_date, df.index[-1], alpha=0.08, color="orange", label="Teste")
    ax1.legend()
    ax1.grid(alpha=0.2)

    ax2 = fig.add_subplot(3, 2, 3)
    colors = ["#e74c3c" if p < THRESHOLD else "#27ae60" for p in proba]
    ax2.bar(test_df.index, proba, color=colors, width=1.0, alpha=0.7)
    ax2.axhline(THRESHOLD, color="black", linestyle="--")
    ax2.set_ylim(0, 1)
    ax2.grid(alpha=0.2)

    ax3 = fig.add_subplot(3, 2, 4)
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["BAIXA", "ALTA"]).plot(ax=ax3, cmap="Blues", colorbar=False)

    ax4 = fig.add_subplot(3, 2, 5)
    folds = [f"F{i+1}" for i in range(len(fold_scores))]
    ax4.bar(folds, [s * 100 for s in fold_scores], color="#3498db")
    ax4.axhline(50, color="gray", linestyle="--")
    ax4.set_ylim(40, 70)
    ax4.grid(alpha=0.2)

    ax5 = fig.add_subplot(3, 2, 6)
    m_final = LogisticRegression(max_iter=1000, C=0.1).fit(scaler.fit_transform(df[FEATURE_COLS]), df["target"])
    coefs = m_final.coef_[0]
    sorted_idx = np.argsort(np.abs(coefs))
    ax5.barh([FEATURE_COLS[i] for i in sorted_idx], [coefs[i] for i in sorted_idx], color="#27ae60")
    ax5.axvline(0, color="black")
    ax5.grid(alpha=0.2)

    plt.tight_layout()
    plt.show()

def print_summary(metrics, prob_tomorrow, direction):
    print("\n" + "=" * 30)
    print(f"Acurácia : {metrics['accuracy']:.2%}")
    print(f"Precisão : {metrics['precision']:.2%}")
    print(f"F1-Score : {metrics['f1']:.2%}")
    print("-" * 30)
    print(f"Previsão : {direction}")
    print(f"Probab.  : {prob_tomorrow:.1%}")
    print("=" * 30)

if __name__ == "__main__":
    df_raw = download_data(TICKER, START_DATE, END_DATE)
    df = add_features(df_raw)
    model, scaler, fold_scores = train_and_evaluate(df)
    metrics, y_test, y_pred, proba, test_df = evaluate_test_set(df, model, scaler)
    prob_tomorrow, direction = predict_tomorrow(df, model, scaler)
    plot_results(df, test_df, y_test, y_pred, proba, metrics, fold_scores, prob_tomorrow)
    print_summary(metrics, prob_tomorrow, direction)