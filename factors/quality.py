from __future__ import annotations

import pandas as pd

from validation.event_study import EventStudyResult, bucket_test, stability_check

# Quality: profitable firms outperform. ROE = trailing annual net income / book
# equity; GrossProfitability = gross profit / assets (Novy-Marx 2013, the most
# robust quality measure). From EDGAR XBRL fundamentals. Sector-neutral variant
# matters (profitability differs structurally by sector) — test both.
FEATURE_COLUMNS = {
    "roe": ["ROE"],
    "roe_sectorN": ["ROE_SectorNeutral"],
    "gross_profitability": ["GrossProfitability"],
    "gross_profitability_sectorN": ["GrossProfitability_SectorNeutral"],
}

_BASE = ["ROE", "GrossProfitability"]


def add_features(panel_df: pd.DataFrame) -> pd.DataFrame:
    """Adds the quality factors + their same-date same-sector-neutral variants.
    Requires ROE/GrossProfitability (from attach_fundamentals) and Sector
    (attach_sector); fails loud if absent."""
    for col in _BASE + ["Sector"]:
        if col not in panel_df.columns:
            raise ValueError(
                f"quality.add_features requires '{col}' — call attach_fundamentals + "
                "attach_sector before assembling the panel."
            )
    panel_df = panel_df.copy()
    for col in _BASE:
        sector_mean = panel_df.groupby(["Date", "Sector"])[col].transform("mean")
        panel_df[f"{col}_SectorNeutral"] = panel_df[col] - sector_mean
    return panel_df


def run_event_study(panel_df: pd.DataFrame) -> dict[str, EventStudyResult]:
    """Cross-sectional test for each quality variant: do more profitable stocks
    forward-outperform? expected_sign=+1. Pass a non-overlapping subsample."""
    return {
        name: bucket_test(panel_df, cols[0], "fwd_return", name, expected_sign=1)
        for name, cols in FEATURE_COLUMNS.items()
    }


def run_quality_stability_check(panel_df: pd.DataFrame, feature_col: str):
    """Split-sample robustness check for one quality variant."""
    def quality_bucket_test(half_df: pd.DataFrame) -> EventStudyResult:
        return bucket_test(half_df, feature_col, "fwd_return", f"{feature_col}_half", expected_sign=None)

    return stability_check(quality_bucket_test, panel_df)
