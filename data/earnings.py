from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from config import CACHE_DIR
from data.cache import read_parquet_cache, write_parquet_cache


def _earnings_cache_path(tickers: list[str], limit: int) -> Path:
    tickers_key = hashlib.sha1("|".join(sorted(tickers)).encode()).hexdigest()[:12]
    return CACHE_DIR / f"earnings_{tickers_key}_lim{limit}.parquet"


def load_earnings(tickers: list[str], limit: int = 48, use_cache: bool = True) -> pd.DataFrame:
    """Long-format earnings history: one row per (Ticker, past earnings
    announcement) with the announcement date and Surprise(%). One
    get_earnings_dates() call per ticker (slow) → cached to parquet keyed on
    the ticker set. Keeps only rows that are actual reported earnings (non-null
    Reported EPS and Surprise). Warns and skips tickers that error/return empty,
    mirroring load_price_panel's partial-failure handling."""
    cache_path = _earnings_cache_path(tickers, limit)
    if use_cache:
        cached = read_parquet_cache(cache_path)
        if cached is not None:
            return cached

    frames = []
    for ticker in tickers:
        try:
            raw = yf.Ticker(ticker).get_earnings_dates(limit=limit)
        except Exception as e:  # noqa: BLE001 - yfinance raises assorted network/parse errors
            print(f"WARNING: earnings fetch failed for {ticker} ({e}), skipping")
            continue
        if raw is None or raw.empty:
            print(f"WARNING: no earnings data for {ticker}, skipping")
            continue

        df = raw.reset_index()
        df = df.rename(columns={"Earnings Date": "AnnouncementTs", "Surprise(%)": "Surprise"})
        df = df[df["Reported EPS"].notna() & df["Surprise"].notna()]
        if df.empty:
            continue
        # AnnouncementTs is tz-aware; keep the calendar date of the announcement.
        df["AnnouncementDate"] = pd.to_datetime(df["AnnouncementTs"]).dt.tz_localize(None).dt.date.astype(str)
        df["Ticker"] = ticker
        frames.append(df[["Ticker", "AnnouncementDate", "Surprise"]])

    if not frames:
        raise RuntimeError(
            f"No earnings data could be loaded for any of {len(tickers)} tickers. "
            "Check network access before retrying."
        )

    earnings = pd.concat(frames, ignore_index=True).sort_values(["Ticker", "AnnouncementDate"])
    earnings = earnings.reset_index(drop=True)

    if use_cache:
        write_parquet_cache(earnings, cache_path)

    return earnings


def attach_earnings(panel_df: pd.DataFrame) -> pd.DataFrame:
    """Attach MostRecentSurprise and DaysSinceEarnings (in trading days) to
    every stock-day via a leakage-safe backward as-of merge.

    Leakage rule: an announcement on calendar date D is only tradeable on the
    next panel trading day strictly after D (its "effective date"), since the
    surprise is unknown until the announcement and most firms report after the
    close. So the feature at date t only ever uses an effective date <= t.
    """
    tickers = panel_df["Ticker"].unique().tolist()
    earnings = load_earnings(tickers)

    # Trading-day ordinal for every unique panel date (exact trading-day gaps).
    trading_dates = np.sort(panel_df["Date"].unique())
    ordinal = {d: i for i, d in enumerate(trading_dates)}

    # Effective date = first panel trading day strictly after the announcement.
    date_pos = np.searchsorted(trading_dates, earnings["AnnouncementDate"].to_numpy(), side="right")
    valid = date_pos < len(trading_dates)  # announcements after the last panel date have no effect
    eff = earnings.loc[valid].copy()
    eff["EffectiveDate"] = trading_dates[date_pos[valid]]
    eff["_eff_ord"] = eff["EffectiveDate"].map(ordinal)
    # merge_asof requires each frame globally sorted by its on-key (the `by`
    # grouping is handled internally).
    eff = eff.sort_values("_eff_ord")

    panel_df = panel_df.copy()
    panel_df["_ord"] = panel_df["Date"].map(ordinal)
    panel_df = panel_df.sort_values("_ord")

    # Backward as-of within each ticker: most recent earnings whose effective
    # date ordinal is <= the row's ordinal.
    merged = pd.merge_asof(
        panel_df,
        eff[["Ticker", "_eff_ord", "Surprise"]].rename(columns={"Surprise": "MostRecentSurprise"}),
        left_on="_ord",
        right_on="_eff_ord",
        by="Ticker",
        direction="backward",
    )
    merged["DaysSinceEarnings"] = merged["_ord"] - merged["_eff_ord"]
    merged = merged.drop(columns=["_ord", "_eff_ord"])
    return merged
