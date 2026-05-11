from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import lightgbm


@dataclass(frozen=True)
class ModelBundle:
    clf: lightgbm.LGBMClassifier
    reg_sl: lightgbm.LGBMRegressor
    reg_sg: lightgbm.LGBMRegressor
    reg_vol: lightgbm.LGBMRegressor
    feature_cols: List[str]


def _split_time(df: pd.DataFrame, test_ratio: float = 0.30) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    cut = max(10, int(n * (1.0 - test_ratio)))
    cut = min(cut, n - 1)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    accuracy = _safe_div(tp + tn, len(y_true))
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean(np.abs(y_pred - y_true)))


def _build_lgbm_classifier() -> lightgbm.LGBMClassifier:
    return lightgbm.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        min_child_samples=40,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.5,
        reg_lambda=1.0,
        random_state=42,
        verbosity=-1,
    )


def _build_lgbm_regressor() -> lightgbm.LGBMRegressor:
    return lightgbm.LGBMRegressor(
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        min_child_samples=40,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.5,
        reg_lambda=1.0,
        random_state=42,
        verbosity=-1,
    )


def _error_examples(
    valid_df: pd.DataFrame,
    feature_cols: List[str],
    prob: np.ndarray,
    y_pred: np.ndarray,
    sl_pred: np.ndarray,
    sg_pred: np.ndarray,
    vol_pred: np.ndarray,
    top_k: int = 8,
) -> Dict[str, Any]:
    rep = valid_df[["date", "ticker", "sector", "close", "y_cls", "y_sl", "y_sg", "y_vol"]].copy()
    rep["prob_up"] = prob
    rep["pred_cls"] = y_pred
    rep["sl_pred"] = sl_pred
    rep["sg_pred"] = sg_pred
    rep["vol_pred"] = vol_pred

    rep["cls_error"] = (rep["pred_cls"] != rep["y_cls"]).astype(int)
    rep["confidence"] = np.where(rep["pred_cls"] == 1, rep["prob_up"], 1.0 - rep["prob_up"])
    rep["sl_abs_err"] = np.abs(rep["sl_pred"] - rep["y_sl"])
    rep["sg_abs_err"] = np.abs(rep["sg_pred"] - rep["y_sg"])
    rep["vol_abs_err"] = np.abs(rep["vol_pred"] - rep["y_vol"])

    fp = rep[(rep["y_cls"] == 0) & (rep["pred_cls"] == 1)].sort_values("confidence", ascending=False).head(top_k)
    fn = rep[(rep["y_cls"] == 1) & (rep["pred_cls"] == 0)].sort_values("confidence", ascending=False).head(top_k)

    worst_sl = rep.sort_values("sl_abs_err", ascending=False).head(top_k)
    worst_sg = rep.sort_values("sg_abs_err", ascending=False).head(top_k)
    worst_vol = rep.sort_values("vol_abs_err", ascending=False).head(top_k)

    preview_cols = [
        "date", "ticker", "sector", "close", "y_cls", "pred_cls", "prob_up",
        "y_sl", "sl_pred", "y_sg", "sg_pred", "y_vol", "vol_pred",
        "sl_abs_err", "sg_abs_err", "vol_abs_err",
    ]

    return {
        "false_positives": fp[preview_cols].to_dict(orient="records"),
        "false_negatives": fn[preview_cols].to_dict(orient="records"),
        "worst_sl_errors": worst_sl[preview_cols].to_dict(orient="records"),
        "worst_sg_errors": worst_sg[preview_cols].to_dict(orient="records"),
        "worst_vol_errors": worst_vol[preview_cols].to_dict(orient="records"),
    }


def _print_console_summary(
    name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cls_metrics: Dict[str, float],
    reg_metrics: Dict[str, float],
    error_report: Dict[str, Any],
) -> None:
    print(f"\n{'=' * 80}")
    print(f"[{name}] INÍCIO DA VALIDAÇÃO 70/30")
    print(f"Treino: {len(train_df)} linhas | Teste: {len(test_df)} linhas")

    if "ticker" in train_df.columns:
        print(f"Tickers treino: {train_df['ticker'].nunique()} | Tickers teste: {test_df['ticker'].nunique()}")
    if "sector" in train_df.columns:
        print(f"Setor principal: {train_df['sector'].iloc[0] if len(train_df) else 'N/A'}")

    print("\n[Classificação - direção]")
    print(
        f"accuracy={cls_metrics['accuracy']:.4f} | "
        f"precision={cls_metrics['precision']:.4f} | "
        f"recall={cls_metrics['recall']:.4f} | "
        f"f1={cls_metrics['f1']:.4f}"
    )
    print(
        f"TP={cls_metrics['tp']} | TN={cls_metrics['tn']} | "
        f"FP={cls_metrics['fp']} | FN={cls_metrics['fn']}"
    )

    print("\n[Regressões]")
    print(
        f"SL -> rmse={reg_metrics['sl_rmse']:.6f} | mae={reg_metrics['sl_mae']:.6f}\n"
        f"SG -> rmse={reg_metrics['sg_rmse']:.6f} | mae={reg_metrics['sg_mae']:.6f}\n"
        f"VOL -> rmse={reg_metrics['vol_rmse']:.6f} | mae={reg_metrics['vol_mae']:.6f}"
    )

    fp = error_report["false_positives"][:3]
    fn = error_report["false_negatives"][:3]

    print("\n[Erros de classificação - resumo simples]")
    if fp:
        print("Falsos positivos mais confiantes (modelo achou que subiria, mas não subiu):")
        for r in fp:
            print(
                f"  {r['date']} {r.get('ticker', '-')}: prob_up={r['prob_up']:.4f} | "
                f"real={int(r['y_cls'])} | pred={int(r['pred_cls'])}"
            )
    else:
        print("Nenhum falso positivo relevante no topo.")

    if fn:
        print("Falsos negativos mais confiantes (modelo achou que não subiria, mas subiu):")
        for r in fn:
            print(
                f"  {r['date']} {r.get('ticker', '-')}: prob_up={r['prob_up']:.4f} | "
                f"real={int(r['y_cls'])} | pred={int(r['pred_cls'])}"
            )
    else:
        print("Nenhum falso negativo relevante no topo.")

    print("\n[Leitura simples do que aconteceu]")
    if cls_metrics["fp"] > cls_metrics["fn"]:
        print("- O modelo errou mais comprando cedo demais do que deixando passar altas.")
    elif cls_metrics["fn"] > cls_metrics["fp"]:
        print("- O modelo errou mais ficando conservador demais e perdendo altas.")
    else:
        print("- Os erros ficaram equilibrados entre falsos positivos e falsos negativos.")

    print("- Depois dessa validação, o bundle final será refeito com 100% dos dados.")
    print(f"{'=' * 80}\n")


def train_bundle(
    df: pd.DataFrame,
    feature_cols: List[str],
    model_name: str = "MODEL",
    test_ratio: float = 0.30,
) -> tuple[ModelBundle, Dict[str, Any]]:
    df = df.sort_values("date").reset_index(drop=True)
    train_df, test_df = _split_time(df, test_ratio=test_ratio)

    if len(train_df) < 20 or len(test_df) < 10:
        raise ValueError(
            f"Dataset muito pequeno para split 70/30 com segurança. "
            f"train={len(train_df)} test={len(test_df)}"
        )

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]

    y_train_cls = train_df["y_cls"].astype(int).values
    y_test_cls = test_df["y_cls"].astype(int).values

    y_train_sl = train_df["y_sl"].astype(float).values
    y_test_sl = test_df["y_sl"].astype(float).values

    y_train_sg = train_df["y_sg"].astype(float).values
    y_test_sg = test_df["y_sg"].astype(float).values

    y_train_vol = train_df["y_vol"].astype(float).values
    y_test_vol = test_df["y_vol"].astype(float).values

    print(f"\n[{model_name}] Etapa 1/4 - treinando classificador...")
    clf = _build_lgbm_classifier()
    clf.fit(X_train, y_train_cls)

    print(f"[{model_name}] Etapa 2/4 - treinando regressor de stop-loss...")
    reg_sl = _build_lgbm_regressor()
    reg_sl.fit(X_train, y_train_sl)

    print(f"[{model_name}] Etapa 3/4 - treinando regressor de stop-gain...")
    reg_sg = _build_lgbm_regressor()
    reg_sg.fit(X_train, y_train_sg)

    print(f"[{model_name}] Etapa 4/4 - treinando regressor de volatilidade...")
    reg_vol = _build_lgbm_regressor()
    reg_vol.fit(X_train, y_train_vol)

    print(f"[{model_name}] Validando no bloco final de 30%...")
    prob = clf.predict_proba(X_test)[:, 1]
    y_pred_cls = (prob >= 0.5).astype(int)

    sl_pred = reg_sl.predict(X_test)
    sg_pred = reg_sg.predict(X_test)
    vol_pred = reg_vol.predict(X_test)

    cls_metrics = _binary_metrics(y_test_cls, y_pred_cls)
    reg_metrics = {
        "sl_rmse": _rmse(y_test_sl, sl_pred),
        "sl_mae": _mae(y_test_sl, sl_pred),
        "sg_rmse": _rmse(y_test_sg, sg_pred),
        "sg_mae": _mae(y_test_sg, sg_pred),
        "vol_rmse": _rmse(y_test_vol, vol_pred),
        "vol_mae": _mae(y_test_vol, vol_pred),
    }

    error_report = _error_examples(
        valid_df=test_df,
        feature_cols=feature_cols,
        prob=prob,
        y_pred=y_pred_cls,
        sl_pred=sl_pred,
        sg_pred=sg_pred,
        vol_pred=vol_pred,
        top_k=8,
    )

    _print_console_summary(
        name=model_name,
        train_df=train_df,
        test_df=test_df,
        cls_metrics=cls_metrics,
        reg_metrics=reg_metrics,
        error_report=error_report,
    )

    print(f"[{model_name}] Refit final com 100% dos dados para salvar o bundle de produção...")
    X_full = df[feature_cols]

    clf_final = _build_lgbm_classifier()
    clf_final.fit(X_full, df["y_cls"].astype(int).values)

    reg_sl_final = _build_lgbm_regressor()
    reg_sl_final.fit(X_full, df["y_sl"].astype(float).values)

    reg_sg_final = _build_lgbm_regressor()
    reg_sg_final.fit(X_full, df["y_sg"].astype(float).values)

    reg_vol_final = _build_lgbm_regressor()
    reg_vol_final.fit(X_full, df["y_vol"].astype(float).values)

    bundle = ModelBundle(
        clf=clf_final,
        reg_sl=reg_sl_final,
        reg_sg=reg_sg_final,
        reg_vol=reg_vol_final,
        feature_cols=feature_cols,
    )

    metrics: Dict[str, Any] = {
        "split": {"train_ratio": 0.70, "test_ratio": 0.30},
        "rows": {"total": len(df), "train": len(train_df), "test": len(test_df)},
        "classification": cls_metrics,
        "regression": reg_metrics,
        "error_report": error_report,
    }
    return bundle, metrics


def save_bundle(bundle: ModelBundle, path: str) -> None:
    joblib.dump(bundle, path)


def load_bundle(path: str) -> ModelBundle:
    return joblib.load(path)