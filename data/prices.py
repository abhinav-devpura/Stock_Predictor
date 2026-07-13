from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from config import CACHE_DIR, END_DATE, START_DATE
from data.cache import read_parquet_cache, write_parquet_cache

PANEL_COLUMNS = ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]


def _price_panel_cache_path(tickers: list[str], start: str, end: str | None) -> Path:
    # Hash the ticker list (can be ~100 names) rather than putting it directly
    # in the filename; start/end stay readable since they're short.
    tickers_key = hashlib.sha1("|".join(sorted(tickers)).encode()).hexdigest()[:12]
    return CACHE_DIR / f"price_panel_{tickers_key}_{start}_{end}.parquet"


def load_price_panel(
    tickers: list[str],
    start: str = START_DATE,
    end: str | None = END_DATE,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Bulk-download daily OHLCV for tickers and return a long-format panel.

    One row per (Date, Ticker). Cached to parquet since a full-universe pull
    is slow and the underlying historical prices don't change. The cache
    file is keyed on (tickers, start, end) so a different request never
    silently returns another request's cached data.
    """
    cache_path = _price_panel_cache_path(tickers, start, end)
    if use_cache:
        cached = read_parquet_cache(cache_path)
        if cached is not None:
            return cached

    raw = yf.download(
        tickers, start=start, end=end, auto_adjust=True, group_by="ticker", threads=True
    )

    frames = []
    available_tickers = set(raw.columns.get_level_values(0))
    for ticker in tickers:
        if ticker not in available_tickers:
            print(f"WARNING: no price data returned for {ticker}, skipping")
            continue
        ticker_df = raw[ticker].dropna(how="all").reset_index()
        if ticker_df.empty:
            print(f"WARNING: empty price history for {ticker}, skipping")
            continue
        ticker_df["Ticker"] = ticker
        frames.append(ticker_df)

    if not frames:
        raise RuntimeError(
            f"No price data could be loaded for any of {len(tickers)} requested tickers "
            f"({tickers[:5]}{'...' if len(tickers) > 5 else ''}). Check network access and "
            "ticker symbols before retrying."
        )

    panel = pd.concat(frames, ignore_index=True)
    panel["Date"] = pd.to_datetime(panel["Date"]).dt.date.astype(str)
    panel = panel[PANEL_COLUMNS].sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # Write after a fetch so a forced re-download (use_cache=False, e.g.
    # `--refresh`) actually *updates* the cache instead of silently not persisting
    # — but only when every requested ticker came back. A partial panel (some
    # tickers failed to download) must not clobber a complete cached panel; the
    # caller still gets the partial result in-memory for this run.
    if set(panel["Ticker"].unique()) >= set(tickers):
        write_parquet_cache(panel, cache_path)
    else:
        missing = len(set(tickers) - set(panel["Ticker"].unique()))
        print(f"WARNING: {missing}/{len(tickers)} tickers missing from fetch; "
              "not overwriting cache with a partial panel")
    return panel


def add_returns(panel_df: pd.DataFrame) -> pd.DataFrame:
    """Add a per-ticker daily Return column (pct_change grouped by Ticker
    so returns never bleed across ticker boundaries)."""
    panel_df = panel_df.sort_values(["Ticker", "Date"]).copy()
    panel_df["Return"] = panel_df.groupby("Ticker")["Close"].pct_change()
    return panel_df


def add_forward_return(panel_df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Add a per-ticker forward return over `horizon` trading days (fwd_return
    = Close[t+horizon]/Close[t] - 1), the shared basis for both validation
    bucket tests and the model's Target — one implementation rather than two
    copies that could drift. NaN on each ticker's last `horizon` rows (no
    future price that far out). Column name stays `fwd_return`: the horizon is
    a config concern, not part of the column name."""
    panel_df = panel_df.sort_values(["Ticker", "Date"]).copy()
    panel_df["fwd_return"] = panel_df.groupby("Ticker")["Close"].shift(-horizon) / panel_df["Close"] - 1
    return panel_df


def select_non_overlapping_dates(panel_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Return the rows whose Date falls on every `horizon`-th unique trading
    date. Used ONLY for validation tests: a `horizon`-day forward return
    computed on consecutive dates overlaps ~(horizon-1)/horizon with its
    neighbor, so adjacent observations are heavily autocorrelated — which
    violates the independence assumption of the event-study t-tests and
    understates p-values. Sampling every `horizon`-th date makes the forward
    windows non-overlapping and the t-tests valid. The MODEL keeps all rows
    (RandomForest doesn't assume iid); only the validation gate needs this."""
    dates = np.sort(panel_df["Date"].unique())
    keep = set(dates[::horizon])
    return panel_df[panel_df["Date"].isin(keep)]


def coverage_report(panel_df: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker row counts and date ranges, to surface gaps or missing
    tickers before anything downstream is built on top of the panel."""
    report = panel_df.groupby("Ticker")["Date"].agg(rows="count", start="min", end="max")
    return report.sort_values("rows")
