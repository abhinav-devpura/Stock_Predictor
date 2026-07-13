from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

# Fixed here so no factor module can silently redefine what "significant" means.
P_VALUE_THRESHOLD = 0.05
T_STAT_THRESHOLD = 2.0


@dataclass
class EventStudyResult:
    factor_name: str
    mean_ar: float
    t_stat: float
    p_value: float
    n_obs: int
    ci_low: float
    ci_high: float
    verdict: str  # "PASS" or "FAIL"
    placebo_p_value: float | None = None  # event_window_test only
    monotonic: bool | None = None  # bucket_test only

    def __str__(self) -> str:
        placebo = f", placebo_p={self.placebo_p_value:.4f}" if self.placebo_p_value is not None else ""
        mono = f", monotonic={self.monotonic}" if self.monotonic is not None else ""
        return (
            f"[{self.verdict}] {self.factor_name}: mean_ar={self.mean_ar:.5f}, "
            f"t={self.t_stat:.2f}, p={self.p_value:.4f}{placebo}, n={self.n_obs}, "
            f"95% CI=({self.ci_low:.5f}, {self.ci_high:.5f}){mono}"
        )


def abnormal_return(
    panel_df: pd.DataFrame,
    mode: str = "cross_sectional",
    benchmark_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add an `abnormal_return` column to a (Date, Ticker, Return) panel.

    mode="cross_sectional": stock's return minus the cross-sectional mean
      return on that date. Use for stock-specific factors (momentum, PEAD,
      buybacks, insider) where the question is "does this stock outperform
      its peers," not "does the whole market move."
    mode="benchmark": stock's return minus the benchmark return on that
      date. Use for market-wide factors (calendar effects, FOMC) where the
      question is about the market/regime, not which stock wins.
    """
    panel_df = panel_df.copy()
    if mode == "cross_sectional":
        cs_mean = panel_df.groupby("Date")["Return"].transform("mean")
        panel_df["abnormal_return"] = panel_df["Return"] - cs_mean
    elif mode == "benchmark":
        if benchmark_df is None:
            raise ValueError("benchmark_df is required when mode='benchmark'")
        bench_return = benchmark_df.set_index("Date")["Return"]
        panel_df["abnormal_return"] = panel_df["Return"] - panel_df["Date"].map(bench_return)
    else:
        raise ValueError(f"Unknown mode: {mode!r}, expected 'cross_sectional' or 'benchmark'")
    return panel_df


def _bootstrap_ci(values: np.ndarray, n_boot: int = 2000, random_state: int = 10) -> tuple[float, float]:
    """95% bootstrap CI for the mean of `values` (vectorized resample)."""
    rng = np.random.default_rng(random_state)
    n = len(values)
    boot_means = rng.choice(values, size=(n_boot, n), replace=True).mean(axis=1)
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def _bootstrap_ci_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 2000,
                       random_state: int = 10) -> tuple[float, float]:
    """95% bootstrap CI for the difference of means (mean(a) - mean(b)),
    resampling BOTH groups independently — the correct two-sample interval for
    the top-minus-bottom spread. Resampling only `a` against a fixed mean(b)
    understates the width by ~sqrt(2)."""
    rng = np.random.default_rng(random_state)
    na, nb = len(a), len(b)
    diffs = (rng.choice(a, size=(n_boot, na), replace=True).mean(axis=1)
             - rng.choice(b, size=(n_boot, nb), replace=True).mean(axis=1))
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def _normalize_date(d) -> str:
    """Coerce a date-like value (str, Timestamp, datetime, date) to the
    'YYYY-MM-DD' string format this codebase keys panels on, so callers
    aren't required to know the internal string convention — a Timestamp
    that fails to match silently produces zero events rather than an error,
    which is far worse than the small cost of normalizing here."""
    if isinstance(d, str):
        return d
    return pd.Timestamp(d).date().isoformat()


def _verdict(t_stat: float, p_value: float, expected_sign: int | None) -> str:
    if np.isnan(p_value) or p_value >= P_VALUE_THRESHOLD or abs(t_stat) < T_STAT_THRESHOLD:
        return "FAIL"
    if expected_sign is not None and np.sign(t_stat) != np.sign(expected_sign):
        return "FAIL"
    return "PASS"


def _summarize(
    values: np.ndarray, factor_name: str, expected_sign: int | None, random_state: int = 10
) -> EventStudyResult:
    values = values[~np.isnan(values)]
    n_obs = len(values)
    if n_obs < 2:
        return EventStudyResult(factor_name, float("nan"), float("nan"), float("nan"), n_obs, float("nan"), float("nan"), "FAIL")

    mean_val = float(np.mean(values))
    t_stat, p_value = stats.ttest_1samp(values, popmean=0.0)
    ci_low, ci_high = _bootstrap_ci(values, random_state=random_state)

    return EventStudyResult(
        factor_name=factor_name,
        mean_ar=mean_val,
        t_stat=float(t_stat),
        p_value=float(p_value),
        n_obs=n_obs,
        ci_low=ci_low,
        ci_high=ci_high,
        verdict=_verdict(float(t_stat), float(p_value), expected_sign),
    )


def event_window_test(
    daily_returns: pd.DataFrame,
    event_dates: set[str] | list[str],
    factor_name: str,
    window: tuple[int, int] = (-1, 1),
    placebo_n: int = 1000,
    expected_sign: int | None = None,
    random_state: int = 10,
) -> EventStudyResult:
    """Test whether returns around `event_dates` differ from zero and from a
    placebo distribution of random non-event windows of the same length.

    daily_returns: one row per date, columns Date, Return — a market-wide or
      benchmark series (turn-of-month, September, FOMC are regime effects,
      not stock-picking signals, so this operates on a single return series
      rather than a cross-sectional panel).
    event_dates: str ('YYYY-MM-DD'), Timestamp, or datetime/date values —
      normalized internally, so callers don't need to match this codebase's
      string date convention exactly (e.g. yfinance earnings dates work as-is).
    window: (start_offset, end_offset) in trading days relative to each
      event date, inclusive. (-1, 1) = day before through day after.

    Reports both a parametric t-test (H0: mean return in window = 0) and an
    empirical placebo p-value (fraction of random same-length windows with
    |mean| >= the observed |mean|). The verdict only passes if both agree —
    guards against autocorrelated returns making the parametric test alone
    over-confident.
    """
    series = daily_returns.sort_values("Date").reset_index(drop=True)
    dates = series["Date"].tolist()
    date_to_idx = {d: i for i, d in enumerate(dates)}
    returns = series["Return"].to_numpy()

    start_off, end_off = window

    def window_mean(center_idx: int) -> float | None:
        lo, hi = center_idx + start_off, center_idx + end_off
        if lo < 0 or hi >= len(returns):
            return None
        segment = returns[lo : hi + 1]
        if np.isnan(segment).any():
            return None
        return float(segment.mean())

    normalized_event_dates = {_normalize_date(d) for d in event_dates}
    event_idx = {date_to_idx[d] for d in normalized_event_dates if d in date_to_idx}
    event_means = np.array([m for m in (window_mean(i) for i in event_idx) if m is not None])

    result = _summarize(event_means, factor_name, expected_sign, random_state=random_state)

    # Null distribution for the *aggregate* statistic: repeatedly draw a
    # same-size (k = number of events) sample of non-event window means and
    # average it, so the placebo distribution is comparable to result.mean_ar
    # (also a mean over k windows) rather than to individual daily draws.
    rng = np.random.default_rng(random_state)
    valid_centers = [i for i in range(-start_off, len(returns) - end_off) if i not in event_idx]
    placebo_pool = np.array([m for m in (window_mean(c) for c in valid_centers) if m is not None])

    k = len(event_means)
    if k > 0 and len(placebo_pool) >= k and not np.isnan(result.mean_ar):
        placebo_means = np.array(
            [rng.choice(placebo_pool, size=k, replace=False).mean() for _ in range(placebo_n)]
        )
        result.placebo_p_value = float(np.mean(np.abs(placebo_means) >= abs(result.mean_ar)))
        if result.verdict == "PASS" and result.placebo_p_value >= P_VALUE_THRESHOLD:
            result.verdict = "FAIL"

    return result


def bucket_test(
    panel_df: pd.DataFrame,
    factor_col: str,
    forward_return_col: str,
    factor_name: str,
    n_buckets: int = 5,
    expected_sign: int = 1,
    random_state: int = 10,
) -> EventStudyResult:
    """Sort stock-days into quantile buckets by factor_col, and t-test the
    top-minus-bottom bucket spread in forward_return_col (Welch's t-test,
    unequal variance). Reports whether bucket means are monotonic in
    factor_col as a sanity flag (not a formal test).

    Use for continuous cross-sectional factors: momentum, buyback %-change,
    insider net-buying, PEAD surprise magnitude.
    """
    df = panel_df.dropna(subset=[factor_col, forward_return_col]).copy()
    df["_bucket"] = pd.qcut(df[factor_col], n_buckets, labels=False, duplicates="drop")

    bucket_means = df.groupby("_bucket")[forward_return_col].mean().sort_index()
    monotonic = bool(bucket_means.is_monotonic_increasing or bucket_means.is_monotonic_decreasing)

    top_bucket, bottom_bucket = bucket_means.index.max(), bucket_means.index.min()
    top = df.loc[df["_bucket"] == top_bucket, forward_return_col].to_numpy()
    bottom = df.loc[df["_bucket"] == bottom_bucket, forward_return_col].to_numpy()

    # qcut(duplicates="drop") can collapse to one bucket for a sparse/heavily
    # duplicated factor (e.g. a mostly-zero binary flag), making top and
    # bottom the same rows — that's not a real comparison, so fail loudly
    # via n_obs/verdict rather than silently reporting a spurious FAIL from
    # comparing a sample to itself. Also guard the plain too-few-rows case
    # scipy would otherwise emit NaN/warnings for.
    if top_bucket == bottom_bucket or len(top) < 2 or len(bottom) < 2:
        return EventStudyResult(
            factor_name=factor_name,
            mean_ar=float("nan"),
            t_stat=float("nan"),
            p_value=float("nan"),
            n_obs=len(df),
            ci_low=float("nan"),
            ci_high=float("nan"),
            verdict="FAIL",
            monotonic=monotonic,
        )

    t_stat, p_value = stats.ttest_ind(top, bottom, equal_var=False)
    # Two-sample bootstrap CI for the top-minus-bottom spread (resample both
    # buckets), so the interval width matches the Welch t-test above.
    ci_low, ci_high = _bootstrap_ci_diff(top, bottom, random_state=random_state)

    result = EventStudyResult(
        factor_name=factor_name,
        mean_ar=float(np.mean(top) - np.mean(bottom)),
        t_stat=float(t_stat),
        p_value=float(p_value),
        n_obs=len(top) + len(bottom),
        ci_low=ci_low,
        ci_high=ci_high,
        verdict=_verdict(float(t_stat), float(p_value), expected_sign),
        monotonic=monotonic,
    )
    return result


def stability_check(test_fn, data: pd.DataFrame, split_frac: float = 0.5):
    """Runs test_fn independently on the early and late halves (by Date) of
    data, and returns (early_result, late_result). A pooled full-sample
    result can be driven entirely by one period — this is exactly what
    happened with the January-size effect (significant and clean 2016-2021,
    faded to non-significant 2021-2026 in the pooled full-sample test) — so
    a factor should only be trusted as a stable, ongoing pattern if it
    replicates independently in both halves. See is_stable().

    test_fn: a callable taking a single DataFrame (the date-sliced half) and
    returning an EventStudyResult — bind any other fixed arguments
    (event_dates, factor_col, expected_sign, ...) with functools.partial
    before passing in. Pass expected_sign=None to the underlying test if you
    want to check replicability of whatever direction the data shows,
    independent of a literature-predicted sign (see is_stable).
    """
    dates = np.sort(data["Date"].unique())
    split_at = dates[int(len(dates) * split_frac)]
    early = data[data["Date"] < split_at]
    late = data[data["Date"] >= split_at]
    return test_fn(early), test_fn(late)


def is_stable(early: EventStudyResult, late: EventStudyResult) -> bool:
    """True only if both halves independently show significance (by
    P_VALUE_THRESHOLD/T_STAT_THRESHOLD, regardless of any expected_sign the
    individual results were scored against) AND agree with each other on
    sign. Deliberately decoupled from EventStudyResult.verdict, which checks
    against a literature-predicted sign — this checks internal replicability
    of whatever direction the data actually shows.
    """
    def significant(result: EventStudyResult) -> bool:
        return (
            not np.isnan(result.p_value)
            and result.p_value < P_VALUE_THRESHOLD
            and abs(result.t_stat) >= T_STAT_THRESHOLD
        )

    if not (significant(early) and significant(late)):
        return False
    return np.sign(early.t_stat) == np.sign(late.t_stat)


def halves_agree(early: EventStudyResult, late: EventStudyResult) -> bool:
    """Relaxed stability: the two sample halves point the same direction (no
    sign reversal), without requiring each half to *independently* clear
    significance. The stricter is_stable() requires both halves significant,
    which throws away power by splitting the sample in ~half (SE grows ~sqrt2,
    t shrinks ~30%) and so unfairly rejects real-but-modest stable effects —
    see NOTES gate-sensitivity + the 161->565 universe experiment, where this
    rule (paired with adequate breadth) is what surfaced PEAD and sector-neutral
    reversal. Same-sign already precludes a *significant* reversal."""
    if np.isnan(early.t_stat) or np.isnan(late.t_stat):
        return False
    return np.sign(early.t_stat) == np.sign(late.t_stat)


def passes_gate(full: EventStudyResult, early: EventStudyResult, late: EventStudyResult) -> bool:
    """The adopted model-inclusion gate: the full-sample event study PASSES
    (significant in the a-priori expected direction — the anti-p-hacking sign
    guard lives in EventStudyResult.verdict) AND the two halves don't reverse
    (halves_agree). Replaces the old `verdict == PASS and is_stable(...)`."""
    return full.verdict == "PASS" and halves_agree(early, late)
