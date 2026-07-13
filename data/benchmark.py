from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

from config import BENCHMARK_TICKER, CACHE_DIR, END_DATE, START_DATE
from data.cache import read_parquet_cache, write_parquet_cache
from data.prices import PANEL_COLUMNS, add_returns


def _benchmark_cache_path(ticker: str, start: str, end: str | None) -> Path:
    return CACHE_DIR / f"benchmark_{ticker}_{start}_{end}.parquet"


def load_benchmark(
    ticker: str = BENCHMARK_TICKER,
    start: str = START_DATE,
    end: str | None = END_DATE,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load daily OHLCV + Return for the benchmark ticker, same shape as the
    price panel (Date, Ticker, Open, High, Low, Close, Volume, Return). Used
    by later phases for benchmark-relative abnormal-return calculations —
    the Return column is added here (not left to the caller) so every
    consumer of load_benchmark() can rely on it being present, matching what
    validation.event_study.abnormal_return(mode="benchmark") expects."""
    cache_path = _benchmark_cache_path(ticker, start, end)
    if use_cache:
        cached = read_parquet_cache(cache_path)
        if cached is not None:
            return cached

    raw = yf.download(ticker, start=start, end=end, auto_adjust=True)
    raw.columns = raw.columns.get_level_values(0)
    benchmark = raw.reset_index()
    benchmark["Date"] = pd.to_datetime(benchmark["Date"]).dt.date.astype(str)
    benchmark["Ticker"] = ticker
    benchmark = benchmark[PANEL_COLUMNS].sort_values("Date").reset_index(drop=True)
    benchmark = add_returns(benchmark)

    # Always write after a fetch (see data/prices.py) so a forced re-download
    # updates the cache instead of silently not persisting.
    write_parquet_cache(benchmark, cache_path)
    return benchmark
