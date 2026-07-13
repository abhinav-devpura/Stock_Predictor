from __future__ import annotations

import pandas as pd

from validation.event_study import EventStudyResult, bucket_test, stability_check

# 12-1 month momentum (Jegadeesh & Titman 1993): cumulative return over the
# trailing ~12 months, skipping the most recent ~1 month. The skip matters —
# the most recent month is dominated by short-term *reversal*, the opposite
# sign of momentum, so including it contaminates the signal.
FORMATION_WINDOW = 252  # ~12 trading months
SKIP_WINDOW = 21  # ~1 trading month

# Two candidate features, gated independently:
#  - Momentum12_1: raw price momentum. Conflates pure trend, earnings drift,
#    and sector/macro-regime moves.
#  - Momentum12_1_SectorNeutral: raw momentum minus its same-date same-sector
#    cross-sectional mean, stripping out "the whole sector rallied." Targets
#    the macro/sector-regime component that drives momentum crashes; documented
#    to be more robust than raw momentum. (The earnings-driven component is
#    handled separately as PEAD, Phase 5.)
FEATURE_COLUMNS = {
    "momentum_raw": ["Momentum12_1"],
    "momentum_sector_neutral": ["Momentum12_1_SectorNeutral"],
}


def add_features(panel_df: pd.DataFrame) -> pd.DataFrame:
    """Adds Momentum12_1 (per-ticker cumulative return from t-252 to t-21,
    trailing-only so no look-ahead; NaN for the first FORMATION_WINDOW rows per
    ticker) and Momentum12_1_SectorNeutral (that minus the same-date,
    same-Sector cross-sectional mean momentum).

    Requires a `Sector` column (from data.universe.attach_sector) — raises if
    absent rather than silently skipping the sector-neutral variant, so a
    forgotten attach_sector can't quietly degrade the signal."""
    if "Sector" not in panel_df.columns:
        raise ValueError(
            "momentum.add_features requires a 'Sector' column — call "
            "data.universe.attach_sector(panel, universe_name) before assembling the panel."
        )

    panel_df = panel_df.sort_values(["Ticker", "Date"]).copy()
    close = panel_df.groupby("Ticker")["Close"]
    panel_df["Momentum12_1"] = close.shift(SKIP_WINDOW) / close.shift(FORMATION_WINDOW) - 1

    sector_mean = panel_df.groupby(["Date", "Sector"])["Momentum12_1"].transform("mean")
    panel_df["Momentum12_1_SectorNeutral"] = panel_df["Momentum12_1"] - sector_mean
    return panel_df


def run_event_study(panel_df: pd.DataFrame) -> dict[str, EventStudyResult]:
    """Cross-sectional test for BOTH momentum variants: do high-momentum stocks
    forward-outperform low-momentum ones? expected_sign=+1 (winners keep
    winning). Pass a non-overlapping subsample (see
    data.prices.select_non_overlapping_dates) so the multi-day forward returns
    don't overlap and inflate significance.

    panel_df must already have the momentum columns (add_features) and
    fwd_return (data.prices.add_forward_return) present.
    """
    return {
        "momentum_raw": bucket_test(
            panel_df, "Momentum12_1", "fwd_return", "momentum_12_1_raw", expected_sign=1
        ),
        "momentum_sector_neutral": bucket_test(
            panel_df, "Momentum12_1_SectorNeutral", "fwd_return",
            "momentum_12_1_sector_neutral", expected_sign=1,
        ),
    }


def run_momentum_stability_check(
    panel_df: pd.DataFrame, feature_col: str
) -> tuple[EventStudyResult, EventStudyResult]:
    """Split-sample robustness check for one momentum variant. Momentum is
    famously regime-dependent ("momentum crashes"), so a pooled result agreeing
    across an early and late half is exactly the reassurance we want. Uses
    expected_sign=None so each half is scored on significance alone; use
    validation.event_study.is_stable() on the pair.

    Momentum isn't periodic (unlike the January effect), so a plain
    full-timeline split is correct — no filter-then-split subtlety here.
    """
    def momentum_bucket_test(half_df: pd.DataFrame) -> EventStudyResult:
        return bucket_test(
            half_df, feature_col, "fwd_return", f"{feature_col}_half", expected_sign=None
        )

    return stability_check(momentum_bucket_test, panel_df)
