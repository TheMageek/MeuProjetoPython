from __future__ import annotations

import numpy as np
import pandas as pd


def make_targets(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    out = df.copy()
    close = out["close"].astype(float).values
    high = out["high"].astype(float).values
    low = out["low"].astype(float).values

    n = len(out)
    y_cls = np.full(n, np.nan)
    y_sl = np.full(n, np.nan)
    y_sg = np.full(n, np.nan)
    y_vol = np.full(n, np.nan)

    logret = np.log(out["close"] / out["close"].shift(1))

    for i in range(n - horizon - 1):
        entry = close[i]
        fut_slice = slice(i + 1, i + 1 + horizon)

        fut_close = close[i + horizon]
        y_cls[i] = 1.0 if (fut_close / entry - 1.0) > 0 else 0.0

        fut_min_low = np.nanmin(low[fut_slice])
        fut_max_high = np.nanmax(high[fut_slice])

        y_sl[i] = (fut_min_low / entry) - 1.0
        y_sg[i] = (fut_max_high / entry) - 1.0

        fut_lr = logret.iloc[i + 1 : i + 1 + horizon].dropna().values
        if len(fut_lr) >= 2:
            y_vol[i] = float(np.std(fut_lr, ddof=1))

    out["y_cls"] = y_cls
    out["y_sl"] = y_sl
    out["y_sg"] = y_sg
    out["y_vol"] = y_vol

    return out.dropna(subset=["y_cls", "y_sl", "y_sg", "y_vol"]).reset_index(drop=True)