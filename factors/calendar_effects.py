from __future__ import annotations

import numpy as np
import pandas as pd

from validation.event_study import EventStudyResult, bucket_test, event_window_test, stability_check

# Two classes of calendar effect, and only one can feed a cross-sectional model:
#
# - Market-timing (date-level): turn-of-month, September. Identical for every
#   ticker on a given day, so they carry ZERO information about the model's
#   "beat today's cross-sectional median" target — verified empirically:
#   P(beat | dummy=1) == P(beat | dummy=0) == ~0.497. They ARE validated
#   market-wide in run_event_study() (research/market-timing interest), but
#   they are deliberately NOT model-feature candidates and are absent below.
#
# - Stock-selection (cross-sectional): the January effect, but only via its
#   documented small-cap concentration (Keim 1983). IsJanuary alone is also
#   date-level/useless; bundled with SizeProxy (which varies by ticker) a tree
#   can learn the "January AND small -> outperform" interaction, so the pair is
#   the unit gated into the model.
#
# Keyed by the same names used in run_january_size_study()'s result — lets
# callers decide the model's feature list data-driven from verdicts rather
# than hardcoding column names against a PASS/FAIL judgment call.
EFFECT_FEATURE_COLUMNS = {
    "january_size": ["IsJanuary", "SizeProxy"],
}

_SIZE_PROXY_WINDOW = 60
_SIZE_PROXY_MIN_PERIODS = 30


def turn_of_month_dates(dates: list[str]) -> set[str]:
    """Ariel (1987) turn-of-month definition: the last trading day of each
    month through the 3rd trading day of the following month."""
    idx = pd.DatetimeIndex(sorted(set(dates)))
    df = pd.DataFrame({"date": idx})
    df["month_period"] = df["date"].dt.to_period("M")
    last_day_of_month = df.groupby("month_period")["date"].max()
    first_three_of_month = df.groupby("month_period")["date"].apply(lambda s: s.nsmallest(3))
    event_dates = set(last_day_of_month.dt.strftime("%Y-%m-%d")) | set(
        first_three_of_month.dt.strftime("%Y-%m-%d")
    )
    return event_dates


def september_dates(dates: list[str]) -> set[str]:
    idx = pd.DatetimeIndex(sorted(set(dates)))
    return set(idx[idx.month == 9].strftime("%Y-%m-%d"))


def january_dates(dates: list[str]) -> set[str]:
    idx = pd.DatetimeIndex(sorted(set(dates)))
    return set(idx[idx.month == 1].strftime("%Y-%m-%d"))


def _add_size_proxy(panel_df: pd.DataFrame) -> pd.DataFrame:
    """SizeProxy: log1p of trailing 60-day average dollar volume per ticker,
    a stand-in for market cap (real market cap needs Phase 7's
    shares-outstanding data). Exists specifically so the January effect's
    documented small-cap concentration (Keim 1983) is testable/usable —
    unlike the pure calendar dummies, this varies by ticker within a date."""
    panel_df = panel_df.sort_values(["Ticker", "Date"]).copy()
    panel_df["_dollar_volume"] = panel_df["Close"] * panel_df["Volume"]
    rolling_dollar_volume = panel_df.groupby("Ticker")["_dollar_volume"].transform(
        lambda s: s.rolling(_SIZE_PROXY_WINDOW, min_periods=_SIZE_PROXY_MIN_PERIODS).mean()
    )
    panel_df["SizeProxy"] = np.log1p(rolling_dollar_volume)
    return panel_df.drop(columns=["_dollar_volume"])


def add_features(panel_df: pd.DataFrame) -> pd.DataFrame:
    """Adds the cross-sectional model-feature candidates for calendar effects:
    IsJanuary (date-level dummy) + SizeProxy (per-ticker, continuous). These
    are a bundle — IsJanuary alone can't inform the "beat today's median"
    target, but paired with SizeProxy a tree can learn the January small-cap
    interaction (see EFFECT_FEATURE_COLUMNS).

    Deliberately does NOT add IsTurnOfMonth / IsSeptember: those are pure
    date-level market-timing effects, identical for every ticker on a date, so
    they carry zero cross-sectional signal (P(beat|=1) == P(beat|=0)). They're
    validated market-wide in run_event_study() but never enter the model.
    RandomForest is scale-invariant so no normalization is applied here."""
    panel_df = panel_df.copy()
    dates = panel_df["Date"].unique().tolist()
    panel_df["IsJanuary"] = panel_df["Date"].isin(january_dates(dates)).astype(int)
    panel_df = _add_size_proxy(panel_df)
    return panel_df


def run_event_study(daily_returns: pd.DataFrame) -> dict[str, EventStudyResult]:
    """Market-wide test for all 3 effects against a single return series
    (pass the benchmark's daily returns) — turn-of-month, September, and
    January-in-aggregate. See run_january_size_study() for the cross-sectional
    (actual literature) version of the January effect."""
    dates = daily_returns["Date"].tolist()
    return {
        "turn_of_month": event_window_test(
            daily_returns, turn_of_month_dates(dates), "turn_of_month", window=(0, 0), expected_sign=1
        ),
        "september": event_window_test(
            daily_returns, september_dates(dates), "september", window=(0, 0), expected_sign=-1
        ),
        "january": event_window_test(
            daily_returns, january_dates(dates), "january", window=(0, 0), expected_sign=1
        ),
    }


def run_january_size_study(panel_df: pd.DataFrame) -> EventStudyResult:
    """Cross-sectional test of the January effect: within January trading
    days, do smaller-size stocks (low SizeProxy) forward-outperform larger
    ones? This is the literature's actual claim (Keim 1983); run_event_study's
    market-wide January test is a much weaker/diluted proxy for the same claim
    and is expected to often fail even when this one passes.

    panel_df must already have SizeProxy (from add_features) and fwd_return
    (from data.prices.add_forward_return) columns.
    """
    dates = panel_df["Date"].unique().tolist()
    january_rows = panel_df[panel_df["Date"].isin(january_dates(dates))]
    return bucket_test(january_rows, "SizeProxy", "fwd_return", "january_size_effect", expected_sign=-1)


def run_january_size_stability_check(panel_df: pd.DataFrame) -> tuple[EventStudyResult, EventStudyResult]:
    """Split-sample robustness check for the January-size effect: does
    whatever pattern the data shows hold consistently across an early half
    and a late half, independent of the literature's predicted sign? Use
    validation.event_study.is_stable() on the returned pair to decide.

    Uses expected_sign=None so each half is scored on significance alone,
    not against the (already known to fail) literature-predicted direction —
    this checks internal replicability, a different and stricter question
    than run_january_size_study's literature-match verdict.

    Splits the *full* panel timeline first, then filters to January within
    each half — not the other way around. January is a clustered/periodic
    subset (~21 trading days/year out of ~2600), so splitting the
    January-only date list at its own midpoint can land mid-January in the
    boundary year and fragment that year's data across both halves instead
    of giving each half whole, comparable years.
    """
    def january_bucket_test(half_df: pd.DataFrame) -> EventStudyResult:
        half_dates = half_df["Date"].unique().tolist()
        half_january_rows = half_df[half_df["Date"].isin(january_dates(half_dates))]
        return bucket_test(
            half_january_rows,
            "SizeProxy",
            "fwd_return",
            "january_size_effect_half",
            expected_sign=None,
        )

    return stability_check(january_bucket_test, panel_df)
