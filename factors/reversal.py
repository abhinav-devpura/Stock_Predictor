from __future__ import annotations

import pandas as pd

from validation.event_study import EventStudyResult, bucket_test, stability_check

# Short-term reversal (Lehmann 1990, Jegadeesh 1990): recent losers tend to
# bounce. The signal is the *negated* trailing 5-day return, so a high value =
# a stock that recently fell hard = expected to rebound (expected_sign=+1).
# It surfaced as the most persistently positive-IC signal in the regime
# investigation (scripts/investigate_regime.py), so it's worth a formal gate.
REVERSAL_WINDOW = 5

# Raw and sector-neutral variants, gated independently — same sector-conflation
# concern as momentum (a sector selling off makes all its names look like
# "losers", so raw reversal is partly "my sector dropped").
FEATURE_COLUMNS = {
    "reversal_raw": ["Reversal5"],
    "reversal_sector_neutral": ["Reversal5_SectorNeutral"],
}


def add_features(panel_df: pd.DataFrame) -> pd.DataFrame:
    """Adds Reversal5 = -(trailing REVERSAL_WINDOW-day return) per ticker
    (trailing-only, no look-ahead; NaN for the first REVERSAL_WINDOW rows) and
    Reversal5_SectorNeutral (that minus the same-date same-Sector mean).

    Requires a `Sector` column (from data.universe.attach_sector) — raises if
    absent, same contract as momentum.add_features."""
    if "Sector" not in panel_df.columns:
        raise ValueError(
            "reversal.add_features requires a 'Sector' column — call "
            "data.universe.attach_sector(panel) before assembling the panel."
        )

    panel_df = panel_df.sort_values(["Ticker", "Date"]).copy()
    close = panel_df.groupby("Ticker")["Close"]
    panel_df["Reversal5"] = -(panel_df["Close"] / close.shift(REVERSAL_WINDOW) - 1)

    sector_mean = panel_df.groupby(["Date", "Sector"])["Reversal5"].transform("mean")
    panel_df["Reversal5_SectorNeutral"] = panel_df["Reversal5"] - sector_mean
    return panel_df


def run_event_study(panel_df: pd.DataFrame) -> dict[str, EventStudyResult]:
    """Cross-sectional test for both reversal variants: do recent losers
    (high Reversal5) forward-outperform? expected_sign=+1. Pass a
    non-overlapping subsample so the multi-day forward returns don't overlap.

    panel_df must have the reversal columns (add_features) and fwd_return
    (data.prices.add_forward_return).
    """
    return {
        "reversal_raw": bucket_test(
            panel_df, "Reversal5", "fwd_return", "reversal_5d_raw", expected_sign=1
        ),
        "reversal_sector_neutral": bucket_test(
            panel_df, "Reversal5_SectorNeutral", "fwd_return",
            "reversal_5d_sector_neutral", expected_sign=1,
        ),
    }


def run_reversal_stability_check(
    panel_df: pd.DataFrame, feature_col: str
) -> tuple[EventStudyResult, EventStudyResult]:
    """Split-sample robustness check for one reversal variant (standing gate:
    event-study PASS *and* stable across both halves). expected_sign=None so
    each half is scored on significance alone; use is_stable() on the pair."""
    def reversal_bucket_test(half_df: pd.DataFrame) -> EventStudyResult:
        return bucket_test(
            half_df, feature_col, "fwd_return", f"{feature_col}_half", expected_sign=None
        )

    return stability_check(reversal_bucket_test, panel_df)
