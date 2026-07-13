from __future__ import annotations

import pandas as pd

from validation.event_study import EventStudyResult, bucket_test, stability_check

# Value: cheap stocks outperform (the value premium). BookToMarket = book equity
# / market cap (high = cheap); EarningsYield = trailing annual earnings / market
# cap. Both from EDGAR XBRL fundamentals (data.fundamentals_edgar). Fundamentals
# have severe sector tilts (tech is never "cheap", banks always high B/M), so
# the sector-neutral variant is the meaningful form — test both.
FEATURE_COLUMNS = {
    "book_to_market": ["BookToMarket"],
    "book_to_market_sectorN": ["BookToMarket_SectorNeutral"],
    "earnings_yield": ["EarningsYield"],
    "earnings_yield_sectorN": ["EarningsYield_SectorNeutral"],
}

_BASE = ["BookToMarket", "EarningsYield"]


def add_features(panel_df: pd.DataFrame) -> pd.DataFrame:
    """Adds the value factors + their same-date same-sector-neutral variants.
    Requires the BookToMarket/EarningsYield columns (from attach_fundamentals)
    and a Sector column (attach_sector); fails loud if absent."""
    for col in _BASE + ["Sector"]:
        if col not in panel_df.columns:
            raise ValueError(
                f"value.add_features requires '{col}' — call attach_fundamentals + "
                "attach_sector before assembling the panel."
            )
    panel_df = panel_df.copy()
    for col in _BASE:
        sector_mean = panel_df.groupby(["Date", "Sector"])[col].transform("mean")
        panel_df[f"{col}_SectorNeutral"] = panel_df[col] - sector_mean
    return panel_df


def run_event_study(panel_df: pd.DataFrame) -> dict[str, EventStudyResult]:
    """Cross-sectional test for each value variant: do cheaper stocks (higher
    B/M, higher earnings yield) forward-outperform? expected_sign=+1. Pass a
    non-overlapping subsample so multi-day forward returns don't overlap."""
    return {
        name: bucket_test(panel_df, cols[0], "fwd_return", name, expected_sign=1)
        for name, cols in FEATURE_COLUMNS.items()
    }


def run_value_stability_check(panel_df: pd.DataFrame, feature_col: str):
    """Split-sample robustness check for one value variant (standing gate:
    event-study PASS + halves agree). expected_sign=None (significance only)."""
    def value_bucket_test(half_df: pd.DataFrame) -> EventStudyResult:
        return bucket_test(half_df, feature_col, "fwd_return", f"{feature_col}_half", expected_sign=None)

    return stability_check(value_bucket_test, panel_df)
