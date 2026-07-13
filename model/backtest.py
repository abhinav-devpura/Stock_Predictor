from __future__ import annotations

import numpy as np
import pandas as pd

from config import FORWARD_HORIZON

_PERIODS_PER_YEAR = 252 / FORWARD_HORIZON  # rebalances/yr, tracks the horizon


def rebalance_backtest(
    panel_df: pd.DataFrame,
    score_col: str = "Score",
    horizon: int = 21,
    top_frac: float = 0.1,
    min_names: int = 30,
) -> pd.DataFrame:
    """Non-overlapping rebalanced portfolio backtest.

    Every `horizon` trading days: rank names by `score_col`, go equal-weight
    long the top `top_frac` and (for the long-short leg) short the bottom
    `top_frac`, hold `horizon` days. Because we rebalance every `horizon` days
    and the return is the `horizon`-day forward return, periods don't overlap —
    they chain cleanly and there is no look-ahead (Score at t uses only data
    <= t; the realized return is fwd_return[t] = t -> t+horizon).

    Returns one row per rebalance date with the top-decile, bottom-decile,
    long-short, and universe-mean forward returns, plus long-side turnover.
    """
    df = panel_df.dropna(subset=[score_col, "fwd_return"])
    # Rebalance on a common grid derived from the FULL panel's trading calendar
    # (not the score-filtered subset) so different factors' backtests land on the
    # same dates and are directly comparable / correlatable.
    dates = np.sort(panel_df["Date"].unique())[::horizon]

    rows, prev_top, prev_bottom = [], set(), set()
    for d in dates:
        day = df[df["Date"] == d]
        if len(day) < min_names:
            continue
        n = max(1, int(round(len(day) * top_frac)))
        top = day.nlargest(n, score_col)
        bottom = day.nsmallest(n, score_col)
        top_names, bottom_names = set(top["Ticker"]), set(bottom["Ticker"])
        top_turn = 1.0 - len(prev_top & top_names) / n if prev_top else 1.0
        bottom_turn = 1.0 - len(prev_bottom & bottom_names) / n if prev_bottom else 1.0

        rows.append({
            "Date": d,
            "top": top["fwd_return"].mean(),
            "bottom": bottom["fwd_return"].mean(),
            "longshort": top["fwd_return"].mean() - bottom["fwd_return"].mean(),
            "universe": day["fwd_return"].mean(),
            "turnover": top_turn,                       # long-only (top-decile) book
            "ls_turnover": top_turn + bottom_turn,      # long-short trades BOTH legs
            "n_names": n,
        })
        prev_top, prev_bottom = top_names, bottom_names

    return pd.DataFrame(rows)


def summarize(returns: pd.Series, turnover: pd.Series | None = None,
              cost_bps: float = 0.0, periods_per_year: float = _PERIODS_PER_YEAR) -> dict:
    """Portfolio stats for a per-rebalance return series, optionally net of a
    per-rebalance transaction cost (turnover x cost_bps, charged on entry)."""
    r = returns.copy()
    if turnover is not None and cost_bps:
        r = r - turnover.values * (cost_bps / 1e4)
    r = r.dropna()
    if len(r) < 2:
        return {}

    cum = (1 + r).prod() - 1
    ann_return = (1 + cum) ** (periods_per_year / len(r)) - 1
    sharpe = (r.mean() / r.std(ddof=1)) * np.sqrt(periods_per_year) if r.std(ddof=1) > 0 else float("nan")
    curve = (1 + r).cumprod()
    max_dd = (curve / curve.cummax() - 1).min()
    return {
        "n_periods": len(r),
        "cum_return": cum,
        "ann_return": ann_return,
        "sharpe": sharpe,
        "hit_rate": float((r > 0).mean()),
        "max_drawdown": float(max_dd),
        "avg_turnover": float(turnover.mean()) if turnover is not None else float("nan"),
    }
