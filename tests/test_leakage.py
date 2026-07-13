"""Leakage-safety unit tests — the invariants a factor pipeline must never break.

These assert, on tiny synthetic panels, that:
  - the forward return (the label) is NaN where no future price exists and never
    bleeds across ticker boundaries,
  - daily returns don't bleed across tickers,
  - validation sampling produces non-overlapping forward windows,
  - the cross-sectional score at date t depends ONLY on date-t rows (no other
    date — past or future — can change a stock's rank).

A green run is evidence the pipeline can't peek at the future. Run: `pytest -q`.
"""
import numpy as np
import pandas as pd
from pytest import approx

from data.prices import add_forward_return, add_returns, select_non_overlapping_dates
from model.score import add_score


def _panel(closes_by_ticker: dict[str, list[float]], dates: list[str]) -> pd.DataFrame:
    rows = []
    for tkr, closes in closes_by_ticker.items():
        for d, c in zip(dates, closes):
            rows.append({"Ticker": tkr, "Date": d, "Close": c})
    return pd.DataFrame(rows)


DATES = ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07", "2020-01-08"]


def test_forward_return_matches_definition_and_is_nan_at_the_edge():
    # A rises 10/11/12/13/14; B is flat. horizon=2.
    panel = _panel({"A": [10, 11, 12, 13, 14], "B": [20, 20, 20, 20, 20]}, DATES)
    out = add_forward_return(panel, horizon=2).sort_values(["Ticker", "Date"])

    a = out[out["Ticker"] == "A"].reset_index(drop=True)
    # fwd_return[t] = Close[t+2]/Close[t] - 1
    assert a.loc[0, "fwd_return"] == approx(12 / 10 - 1)
    assert a.loc[1, "fwd_return"] == approx(13 / 11 - 1)
    assert a.loc[2, "fwd_return"] == approx(14 / 12 - 1)
    # last `horizon` rows have no future price -> NaN, never fabricated
    assert np.isnan(a.loc[3, "fwd_return"])
    assert np.isnan(a.loc[4, "fwd_return"])


def test_forward_return_does_not_bleed_across_tickers():
    # Each ticker's last `horizon` rows must be NaN independently — A's tail must
    # not borrow B's early prices (which a wrong global shift would do).
    panel = _panel({"A": [1, 2, 3, 4, 5], "B": [100, 200, 300, 400, 500]}, DATES)
    out = add_forward_return(panel, horizon=2)
    for tkr in ("A", "B"):
        tail = out[out["Ticker"] == tkr].sort_values("Date").tail(2)
        assert tail["fwd_return"].isna().all()


def test_daily_return_is_nan_on_each_tickers_first_row():
    panel = _panel({"A": [10, 11, 12], "B": [5, 6, 7]}, DATES[:3])
    out = add_returns(panel)
    firsts = out.sort_values(["Ticker", "Date"]).groupby("Ticker").head(1)
    assert firsts["Return"].isna().all()  # no return bleeds in from the prior ticker


def test_non_overlapping_dates_are_at_least_horizon_apart():
    dates = [f"2020-02-{d:02d}" for d in range(1, 13)]  # 12 unique dates
    panel = _panel({"A": list(range(12)), "B": list(range(12))}, dates)
    horizon = 3
    kept = sorted(select_non_overlapping_dates(panel, horizon)["Date"].unique())
    idx = [dates.index(d) for d in kept]
    gaps = np.diff(idx)
    # every retained pair is >= horizon apart, so their forward windows don't overlap
    assert (gaps >= horizon).all()


def test_score_is_cross_sectional_within_date_only():
    # A stock's score on a given date must be identical whether computed on the
    # full multi-date panel or on that date's slice alone — proving no other date
    # (future or past) leaks into the cross-sectional z-score.
    rng = np.random.default_rng(0)
    frames = []
    for d in ["2020-03-02", "2020-03-03", "2020-03-04"]:
        for i in range(6):
            frames.append({"Ticker": f"T{i}", "Date": d, "F": float(rng.normal())})
    panel = pd.DataFrame(frames)

    full = add_score(panel, ["F"]).sort_values(["Date", "Ticker"]).reset_index(drop=True)
    d = "2020-03-03"
    slice_only = add_score(panel[panel["Date"] == d].copy(), ["F"]).sort_values("Ticker")
    full_d = full[full["Date"] == d].sort_values("Ticker")
    assert np.allclose(full_d["Score"].to_numpy(), slice_only["Score"].to_numpy())
