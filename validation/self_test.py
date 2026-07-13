"""Self-test for the event-study harness, using synthetic data with a known
ground truth. Must pass before any real factor is validated with this
harness — a broken statistical test that always says PASS (or always FAIL)
would silently poison every phase built on top of it.

Run: python -m validation.self_test
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from validation.event_study import abnormal_return, bucket_test, event_window_test

RANDOM_STATE = 10
N_DAYS = 2500
N_EVENTS = 150
N_STOCK_DAYS = 5000

checks = []


def check(name: str, condition: bool, detail: str) -> None:
    checks.append((name, condition, detail))
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}: {detail}")


def synthetic_daily_returns(rng: np.random.Generator) -> pd.DataFrame:
    dates = pd.bdate_range("2016-01-04", periods=N_DAYS).strftime("%Y-%m-%d")
    returns = rng.normal(loc=0.0, scale=0.01, size=N_DAYS)
    return pd.DataFrame({"Date": dates, "Return": returns})


def test_event_window_injected_effect(rng: np.random.Generator) -> None:
    daily = synthetic_daily_returns(rng)
    event_dates = set(rng.choice(daily["Date"], size=N_EVENTS, replace=False))

    injected = daily.copy()
    injected.loc[injected["Date"].isin(event_dates), "Return"] += 0.005

    result = event_window_test(
        injected, event_dates, "synthetic_injected_effect", window=(0, 0), random_state=RANDOM_STATE
    )
    print(f"  {result}")
    check(
        "event_window_test detects injected effect",
        result.verdict == "PASS" and result.p_value < 0.05 and result.mean_ar > 0,
        f"verdict={result.verdict}, p={result.p_value:.4f}, mean_ar={result.mean_ar:.5f}",
    )


def test_event_window_pure_noise(rng: np.random.Generator) -> None:
    daily = synthetic_daily_returns(rng)
    event_dates = set(rng.choice(daily["Date"], size=N_EVENTS, replace=False))

    result = event_window_test(
        daily, event_dates, "synthetic_pure_noise", window=(0, 0), random_state=RANDOM_STATE
    )
    print(f"  {result}")
    check(
        "event_window_test rejects pure noise",
        result.verdict == "FAIL",
        f"verdict={result.verdict}, p={result.p_value:.4f}, placebo_p={result.placebo_p_value}",
    )


def synthetic_cross_sectional_panel(rng: np.random.Generator, correlated: bool) -> pd.DataFrame:
    factor = rng.normal(0.0, 1.0, size=N_STOCK_DAYS)
    noise = rng.normal(0.0, 0.02, size=N_STOCK_DAYS)
    if correlated:
        forward_return = 0.01 * factor + noise
    else:
        forward_return = noise
    return pd.DataFrame({"factor": factor, "forward_return": forward_return})


def test_bucket_test_real_signal(rng: np.random.Generator) -> None:
    panel = synthetic_cross_sectional_panel(rng, correlated=True)
    result = bucket_test(panel, "factor", "forward_return", "synthetic_real_signal", random_state=RANDOM_STATE)
    print(f"  {result}")
    check(
        "bucket_test detects real cross-sectional signal",
        result.verdict == "PASS" and result.monotonic and result.mean_ar > 0,
        f"verdict={result.verdict}, p={result.p_value:.4f}, monotonic={result.monotonic}, spread={result.mean_ar:.5f}",
    )


def test_bucket_test_pure_noise(rng: np.random.Generator) -> None:
    panel = synthetic_cross_sectional_panel(rng, correlated=False)
    result = bucket_test(panel, "factor", "forward_return", "synthetic_noise_signal", random_state=RANDOM_STATE)
    print(f"  {result}")
    check(
        "bucket_test rejects pure noise",
        result.verdict == "FAIL",
        f"verdict={result.verdict}, p={result.p_value:.4f}",
    )


def test_abnormal_return_cross_sectional_demeans(rng: np.random.Generator) -> None:
    dates = pd.bdate_range("2016-01-04", periods=50).strftime("%Y-%m-%d")
    tickers = [f"T{i}" for i in range(20)]
    rows = [
        {"Date": d, "Ticker": t, "Return": rng.normal(0.001, 0.02)}
        for d in dates
        for t in tickers
    ]
    panel = pd.DataFrame(rows)
    result = abnormal_return(panel, mode="cross_sectional")
    per_date_mean = result.groupby("Date")["abnormal_return"].mean().abs().max()
    check(
        "abnormal_return (cross_sectional) demeans each date to ~0",
        per_date_mean < 1e-9,
        f"max |per-date mean abnormal_return| = {per_date_mean:.2e}",
    )


def test_abnormal_return_benchmark_mode(rng: np.random.Generator) -> None:
    # Regression check for a real bug: benchmark_df must have a Return
    # column, or this raises KeyError. Only catches it if benchmark_df is
    # actually shaped correctly here, not if the caller forgets to add it —
    # data/benchmark.py's load_benchmark() is what guarantees that in practice.
    dates = pd.bdate_range("2016-01-04", periods=10).strftime("%Y-%m-%d")
    panel = pd.DataFrame({"Date": list(dates) * 2, "Ticker": ["A"] * 10 + ["B"] * 10, "Return": rng.normal(0, 0.01, 20)})
    benchmark = pd.DataFrame({"Date": dates, "Return": rng.normal(0, 0.005, 10)})
    try:
        result = abnormal_return(panel, mode="benchmark", benchmark_df=benchmark)
        ok = "abnormal_return" in result.columns and result["abnormal_return"].notna().all()
        detail = f"abnormal_return column present, {result['abnormal_return'].notna().sum()}/{len(result)} non-null"
    except KeyError as e:
        ok = False
        detail = f"KeyError: {e} (benchmark_df is missing a required column)"
    check("abnormal_return (benchmark mode) runs without KeyError", ok, detail)


def test_bucket_test_degenerate_single_bucket(rng: np.random.Generator) -> None:
    # Regression check: a factor with no variation collapses qcut to one
    # bucket, which used to silently compare a sample to itself.
    panel = pd.DataFrame({"factor": [1] * 50, "forward_return": rng.normal(0, 0.01, 50)})
    result = bucket_test(panel, "factor", "forward_return", "degenerate_single_bucket", random_state=RANDOM_STATE)
    print(f"  {result}")
    check(
        "bucket_test handles a degenerate single bucket without crashing",
        result.verdict == "FAIL" and np.isnan(result.t_stat),
        f"verdict={result.verdict}, t_stat={result.t_stat}",
    )


def test_event_window_test_accepts_timestamps(rng: np.random.Generator) -> None:
    # Regression check: event_dates as Timestamps used to silently match
    # zero events (str/Timestamp equality mismatch) instead of erroring or
    # matching correctly.
    daily = synthetic_daily_returns(rng)
    event_dates_str = set(rng.choice(daily["Date"], size=N_EVENTS, replace=False))
    injected = daily.copy()
    injected.loc[injected["Date"].isin(event_dates_str), "Return"] += 0.005
    event_dates_ts = {pd.Timestamp(d) for d in event_dates_str}

    result = event_window_test(
        injected, event_dates_ts, "timestamp_event_dates", window=(0, 0), random_state=RANDOM_STATE
    )
    check(
        "event_window_test matches Timestamp event_dates (not just str)",
        result.n_obs == N_EVENTS,
        f"n_obs={result.n_obs} (expected {N_EVENTS})",
    )


def main() -> int:
    rng = np.random.default_rng(RANDOM_STATE)

    print("--- event_window_test ---")
    test_event_window_injected_effect(rng)
    test_event_window_pure_noise(rng)

    print("\n--- bucket_test ---")
    test_bucket_test_real_signal(rng)
    test_bucket_test_pure_noise(rng)

    print("\n--- abnormal_return ---")
    test_abnormal_return_cross_sectional_demeans(rng)
    test_abnormal_return_benchmark_mode(rng)

    print("\n--- regression checks (bugs found by code review) ---")
    test_bucket_test_degenerate_single_bucket(rng)
    test_event_window_test_accepts_timestamps(rng)

    print("\n--- summary ---")
    failed = [name for name, ok, _ in checks if not ok]
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}: {name}")

    if failed:
        print(f"\n{len(failed)}/{len(checks)} checks FAILED. Harness is not trustworthy — fix before validating any real factor.")
        return 1

    print(f"\nAll {len(checks)} checks passed. Harness self-test OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
