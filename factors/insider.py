from __future__ import annotations

import pandas as pd

from validation.event_study import EventStudyResult, bucket_test, stability_check

# Two size-neutral insider signals, gated independently (see data.insider_edgar
# for construction). NetInsiderRatio tests net buying; InsiderBuyIntensity tests
# purchases only — if buys-only passes while net doesn't, that's evidence for
# the literature claim that insider *sales* are noise (liquidity/taxes) while
# *purchases* carry the signal (Lakonishok-Lee; Cohen-Malloy-Pomorski).
FEATURE_COLUMNS = {
    "insider_net": ["NetInsiderRatio"],
    "insider_buys": ["InsiderBuyIntensity"],
}

# Columns produced by data.insider_edgar.attach_insider.
_REQUIRED = ["NetInsiderRatio", "InsiderBuyIntensity", "InsiderActive"]


def add_features(panel_df: pd.DataFrame) -> pd.DataFrame:
    """Insider features are computed in data.insider_edgar.attach_insider (they
    need the raw transaction stream + a rolling window, not just panel columns).
    This just validates their presence so the factor obeys the same
    add_features contract as the others; fail loud if attach_insider wasn't run."""
    for col in _REQUIRED:
        if col not in panel_df.columns:
            raise ValueError(
                f"insider.add_features requires a '{col}' column — call "
                "data.insider_edgar.attach_insider(panel) before assembling the panel."
            )
    return panel_df


def run_event_study(panel_df: pd.DataFrame) -> dict[str, EventStudyResult]:
    """Cross-sectional test for both insider signals, restricted to rows with
    recent insider activity (InsiderActive) — where the signal is live.
    expected_sign=+1 (net buying / purchase intensity -> forward outperformance).
    Pass a non-overlapping subsample so the multi-day forward returns don't
    overlap and inflate significance.

    panel_df must have the insider columns (attach_insider) and fwd_return
    (data.prices.add_forward_return).
    """
    live = panel_df[panel_df["InsiderActive"]]
    return {
        "insider_net": bucket_test(
            live, "NetInsiderRatio", "fwd_return", "insider_net_ratio", expected_sign=1
        ),
        "insider_buys": bucket_test(
            live, "InsiderBuyIntensity", "fwd_return", "insider_buy_intensity", expected_sign=1
        ),
    }


def run_insider_stability_check(
    panel_df: pd.DataFrame, feature_col: str
) -> tuple[EventStudyResult, EventStudyResult]:
    """Split-sample robustness check for one insider variant (standing gate:
    event-study PASS *and* stable across both halves). expected_sign=None so
    each half is scored on significance alone; use is_stable() on the pair.
    Restricts to active rows inside each half."""
    def insider_bucket_test(half_df: pd.DataFrame) -> EventStudyResult:
        live = half_df[half_df["InsiderActive"]]
        return bucket_test(
            live, feature_col, "fwd_return", f"{feature_col}_half", expected_sign=None
        )

    return stability_check(insider_bucket_test, panel_df)
