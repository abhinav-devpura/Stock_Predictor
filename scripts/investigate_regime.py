"""Regime investigation: is there a real ~2021 break where classic factors
decayed, and do they all break together (common macro shift) or separately
(factor-specific decay)?

Computes a per-date cross-sectional rank Information Coefficient (IC) — the
Spearman correlation between a factor and the forward return across stocks on
each date — then aggregates by quarter to show each factor's trajectory over
time. IC > 0 means the factor ranks stocks in the expected direction that
period; a sign change or fade is a regime shift.

Note: fwd_return is overlapping (21-day), which affects the *autocorrelation*
of the IC series but not the per-date point estimates — fine for reading the
trajectory. Run: python scripts/investigate_regime.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config import FORWARD_HORIZON, UNIVERSE_NAME
from data.prices import add_forward_return, add_returns, load_price_panel
from data.universe import get_universe
from factors import pead
from model.panel import prepare_features

MIN_NAMES = 20  # skip dates with too few stocks to rank


def daily_ic(df: pd.DataFrame, factor: str, ret: str = "fwd_return") -> pd.Series:
    d = df.dropna(subset=[factor, ret])
    def ic(g):
        if len(g) < MIN_NAMES:
            return np.nan
        return g[factor].corr(g[ret], method="spearman")
    out = d.groupby("Date").apply(ic)
    out.index = pd.to_datetime(out.index)
    return out


def quarterly(ic: pd.Series) -> pd.Series:
    return ic.resample("QE").mean()


def main():
    tickers = get_universe(UNIVERSE_NAME)
    panel = add_returns(load_price_panel(tickers))
    panel = prepare_features(panel)
    panel = add_forward_return(panel, horizon=FORWARD_HORIZON)

    # Short-term reversal (5-day): positive IC means recent losers rebound.
    panel = panel.sort_values(["Ticker", "Date"])
    panel["Reversal5"] = -(panel["Close"] / panel.groupby("Ticker")["Close"].shift(5) - 1)

    factors = {
        "Momentum(raw)": panel,
        "Momentum(sectorN)": panel,
        "PEAD": panel[pead._in_drift_window(panel)],
        "Reversal5d": panel,
    }
    factor_cols = {
        "Momentum(raw)": "Momentum12_1",
        "Momentum(sectorN)": "Momentum12_1_SectorNeutral",
        "PEAD": "PEAD_Surprise",
        "Reversal5d": "Reversal5",
    }

    ics = {name: quarterly(daily_ic(df, factor_cols[name])) for name, df in factors.items()}
    table = pd.DataFrame(ics)
    table.index = table.index.to_period("Q").astype(str)

    pd.set_option("display.float_format", lambda x: f"{x:+.3f}")
    print("Quarterly cross-sectional rank IC by factor (IC>0 = factor ranks correctly):\n")
    print(table.to_string())

    print("\nMean IC by half (split at 2021-01-01):")
    dt = pd.PeriodIndex(table.index, freq="Q").to_timestamp()
    early = table[dt < "2021-01-01"].mean()
    late = table[dt >= "2021-01-01"].mean()
    summary = pd.DataFrame({"pre-2021": early, "2021+": late, "delta": late - early})
    print(summary.to_string())


if __name__ == "__main__":
    main()
