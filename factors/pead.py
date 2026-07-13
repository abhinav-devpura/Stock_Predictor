from __future__ import annotations

import numpy as np
import pandas as pd

from validation.event_study import EventStudyResult, bucket_test, stability_check

# Post-earnings-announcement drift persists ~60 trading days; the signal is
# "live" only within the quarter after an announcement, and 0 outside it.
DRIFT_WINDOW = 63  # ~3 trading months

FEATURE_COLUMNS = {"pead": ["PEAD_Surprise"]}


def _in_drift_window(panel_df: pd.DataFrame) -> pd.Series:
    # days == 0 is the effective (first tradeable) date: the surprise is already
    # known (announced the prior session, usually after close) and fwd_return
    # measures forward from this date, so it's a valid, leakage-safe drift day.
    days = panel_df["DaysSinceEarnings"]
    return days.notna() & (days >= 0) & (days <= DRIFT_WINDOW)


def add_features(panel_df: pd.DataFrame) -> pd.DataFrame:
    """PEAD_Surprise = the most recent earnings Surprise(%) if the row is within
    the post-earnings drift window (0..DRIFT_WINDOW trading days after the
    announcement's tradeable effective date), else 0. Continuous cross-sectional
    feature defined on every stock-day (mostly 0; non-zero in the post-earnings
    quarter).

    Requires MostRecentSurprise / DaysSinceEarnings columns (from
    data.earnings.attach_earnings) — raises if absent, same contract as
    momentum.add_features requiring Sector."""
    for col in ("MostRecentSurprise", "DaysSinceEarnings"):
        if col not in panel_df.columns:
            raise ValueError(
                f"pead.add_features requires a '{col}' column — call "
                "data.earnings.attach_earnings(panel) before assembling the panel."
            )

    panel_df = panel_df.copy()
    panel_df["PEAD_Surprise"] = np.where(
        _in_drift_window(panel_df), panel_df["MostRecentSurprise"], 0.0
    )
    return panel_df


def run_event_study(panel_df: pd.DataFrame) -> dict[str, EventStudyResult]:
    """Cross-sectional PEAD test, restricted to in-drift-window rows (where the
    signal is live). Buckets by the earnings surprise vs forward return,
    expected_sign=+1 (bigger positive surprise -> more forward outperformance).
    Restricting to live rows avoids dumping all no-earnings (surprise=0) rows
    into one qcut bucket.

    panel_df must have MostRecentSurprise / DaysSinceEarnings (attach_earnings)
    and fwd_return (data.prices.add_forward_return).
    """
    live = panel_df[_in_drift_window(panel_df)]
    return {
        "pead": bucket_test(
            live, "MostRecentSurprise", "fwd_return", "pead_surprise", expected_sign=1
        )
    }


def run_pead_stability_check(panel_df: pd.DataFrame) -> tuple[EventStudyResult, EventStudyResult]:
    """Split-sample robustness check for PEAD (standing gate: event-study PASS
    *and* stable across both halves). expected_sign=None so each half is scored
    on significance alone; use validation.event_study.is_stable() on the pair.
    Filters to in-window rows inside each half (PEAD isn't clustered into one
    period, so a plain full-timeline split is fine)."""
    def pead_bucket_test(half_df: pd.DataFrame) -> EventStudyResult:
        live = half_df[_in_drift_window(half_df)]
        return bucket_test(
            live, "MostRecentSurprise", "fwd_return", "pead_surprise_half", expected_sign=None
        )

    return stability_check(pead_bucket_test, panel_df)
