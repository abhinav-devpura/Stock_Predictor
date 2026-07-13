from __future__ import annotations

import pandas as pd

from validation.event_study import EventStudyResult, bucket_test, stability_check

# Disclosure factors from 10-K/10-Q text (data.filings_text):
#  - TextSimilarity: cosine similarity to the prior same-form filing. Low
#    similarity = big rewrite = bearish ("Lazy Prices", Cohen-Malloy-Nguyen
#    2020) -> expected_sign +1 (stable filings outperform).
#  - NegTone / UncTone: Loughran-McDonald negative / uncertainty word ratios ->
#    expected_sign -1 (more negative/uncertain language -> underperform).
# Each with a sector-neutral variant.
_EXPECTED = {
    "text_similarity": ("TextSimilarity", 1),
    "text_similarity_sectorN": ("TextSimilarity_SectorNeutral", 1),
    "section_similarity": ("SectionSimilarity", 1),            # Risk Factors + MD&A only
    "section_similarity_sectorN": ("SectionSimilarity_SectorNeutral", 1),
    "neg_tone": ("NegTone", -1),
    "neg_tone_sectorN": ("NegTone_SectorNeutral", -1),
    "unc_tone": ("UncTone", -1),
    "unc_tone_sectorN": ("UncTone_SectorNeutral", -1),
}
FEATURE_COLUMNS = {k: [v[0]] for k, v in _EXPECTED.items()}

_BASE = ["TextSimilarity", "SectionSimilarity", "NegTone", "UncTone"]


def add_features(panel_df: pd.DataFrame) -> pd.DataFrame:
    """Adds the disclosure factors + their same-date same-sector-neutral
    variants. Requires the attached text columns (attach_filing_text) and Sector
    (attach_sector); fails loud if absent."""
    for col in _BASE + ["Sector"]:
        if col not in panel_df.columns:
            raise ValueError(
                f"disclosure.add_features requires '{col}' — call attach_filing_text + "
                "attach_sector before assembling the panel."
            )
    panel_df = panel_df.copy()
    for col in _BASE:
        sector_mean = panel_df.groupby(["Date", "Sector"])[col].transform("mean")
        panel_df[f"{col}_SectorNeutral"] = panel_df[col] - sector_mean
    return panel_df


def run_event_study(panel_df: pd.DataFrame) -> dict[str, EventStudyResult]:
    """Cross-sectional test for each disclosure variant with its expected sign.
    Pass a non-overlapping subsample so multi-day forward returns don't overlap."""
    return {
        name: bucket_test(panel_df, col, "fwd_return", name, expected_sign=sign)
        for name, (col, sign) in _EXPECTED.items()
    }


def run_disclosure_stability_check(panel_df: pd.DataFrame, feature_col: str):
    """Split-sample robustness check for one disclosure variant (significance
    only; expected_sign=None)."""
    def disclosure_bucket_test(half_df: pd.DataFrame) -> EventStudyResult:
        return bucket_test(half_df, feature_col, "fwd_return", f"{feature_col}_half", expected_sign=None)

    return stability_check(disclosure_bucket_test, panel_df)
