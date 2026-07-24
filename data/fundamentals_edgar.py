from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from config import CACHE_DIR, START_DATE
from data.cache import read_parquet_cache, write_parquet_cache
from data.edgar import MAX_WORKERS, load_cik_map, throttled_get

# Concepts to pull from XBRL company facts, mapped to a short name.
# "instant" concepts are balance-sheet point-in-time values; "annual" income
# concepts are taken from ~12-month periods (10-K TTM proxy — clean, robust;
# quarterly-TTM assembly is deferred, see NOTES).
_CONCEPTS = {
    ("us-gaap", "StockholdersEquity", "USD", "instant"): "Equity",
    ("us-gaap", "Assets", "USD", "instant"): "Assets",
    ("us-gaap", "NetIncomeLoss", "USD", "annual"): "NetIncome",
    ("us-gaap", "GrossProfit", "USD", "annual"): "GrossProfit",
    ("dei", "EntityCommonStockSharesOutstanding", "shares", "instant"): "Shares",
}


def _extract(facts: dict, taxonomy: str, concept: str, unit: str, kind: str) -> list[dict]:
    entries = facts.get(taxonomy, {}).get(concept, {}).get("units", {}).get(unit, [])
    out = []
    for e in entries:
        val, filed, end = e.get("val"), e.get("filed"), e.get("end")
        if val is None or filed is None or end is None:
            continue
        if kind == "annual":
            start = e.get("start")
            if start is None:
                continue
            days = (pd.Timestamp(end) - pd.Timestamp(start)).days
            if not (330 <= days <= 400):  # ~annual only
                continue
        out.append({"period_end": end, "filed": filed, "val": float(val)})
    return out


def _fundamentals_cache_path(ticker: str, start: str) -> Path:
    return CACHE_DIR / f"fundamentals_{ticker}_{start}.parquet"


def _fetch_ticker(ticker: str, cik: str) -> pd.DataFrame:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    try:
        resp = throttled_get(url)
        if resp.status_code != 200:
            return pd.DataFrame()
        facts = resp.json().get("facts", {})
    except Exception:  # noqa: BLE001 - network/JSON error: skip ticker
        return pd.DataFrame()

    rows = []
    for (tax, concept, unit, kind), short in _CONCEPTS.items():
        for r in _extract(facts, tax, concept, unit, kind):
            r["concept"] = short
            r["Ticker"] = ticker
            rows.append(r)
    return pd.DataFrame(rows, columns=["Ticker", "concept", "period_end", "filed", "val"])


def load_fundamentals(tickers: list[str], start: str = START_DATE,
                      use_cache: bool = True, incremental: bool = False) -> pd.DataFrame:
    """Long-format point-in-time fundamentals: one row per (Ticker, concept,
    reported value) with its period_end and filing date. Cached per ticker.
    Concurrent + rate-limited via data.edgar. Tickers with no XBRL are skipped.

    incremental=True: refetch each cached ticker's companyfacts (one request) but
    only rewrite/use the new data when it contains a filing newer than the cache,
    otherwise keep the cached version untouched (captures new reports, no churn).
    use_cache=False forces a full refetch."""
    cik_map = load_cik_map()

    to_fetch, frames = [], []
    for ticker in tickers:
        path = _fundamentals_cache_path(ticker, start)
        cached = read_parquet_cache(path) if (use_cache or incremental) else None
        if cached is not None and not incremental:
            frames.append(cached)
            continue
        cik = cik_map.get(ticker)
        if cik is None:
            if cached is not None:
                frames.append(cached)  # keep what we have
            else:
                print(f"WARNING: no CIK for {ticker}, skipping fundamentals")
            continue
        to_fetch.append((ticker, cik, cached))

    def work(tc):
        ticker, cik, cached = tc
        fresh = _fetch_ticker(ticker, cik)
        if incremental and cached is not None and not cached.empty:
            new_max = fresh["filed"].max() if not fresh.empty else None
            if new_max is None or new_max <= cached["filed"].max():
                return ticker, cached, False  # nothing newer than the cache
            return ticker, fresh, True
        return ticker, fresh, True

    if to_fetch:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            for ticker, df, changed in pool.map(work, to_fetch):
                n_concepts = df["concept"].nunique() if not df.empty else 0
                tag = " (+new)" if changed and incremental else ""
                print(f"  {ticker}: {len(df)} fundamental facts, {n_concepts}/5 concepts{tag}")
                # Persist changed results (even on a forced refresh), but never
                # clobber a good cache with an empty/failed pull, and skip rewrites
                # when an incremental refetch found nothing new (avoids churn).
                cpath = _fundamentals_cache_path(ticker, start)
                if changed and (not df.empty or not cpath.exists()):
                    write_parquet_cache(df, cpath)
                if not df.empty:
                    frames.append(df)

    frames = [f for f in frames if not f.empty]
    if not frames:
        raise RuntimeError(f"No fundamentals loaded for any of {len(tickers)} tickers.")
    return pd.concat(frames, ignore_index=True)


def attach_fundamentals(panel_df: pd.DataFrame, use_cache: bool = True,
                        incremental: bool = False) -> pd.DataFrame:
    """Attach BookToMarket, EarningsYield, ROE, GrossProfitability to every
    stock-day via a leakage-safe as-of merge (each fundamental usable only on
    the next panel trading day strictly after its filing date — same rule as
    data.earnings/insider). Ratios with a non-positive denominator -> NaN.

    incremental=True fetches newly-filed reports since the cache (cheap)."""
    fund = load_fundamentals(panel_df["Ticker"].unique().tolist(),
                             use_cache=use_cache, incremental=incremental)
    trading_dates = np.sort(panel_df["Date"].unique())

    pos = np.searchsorted(trading_dates, fund["filed"].to_numpy(), side="right")
    valid = pos < len(trading_dates)
    fund = fund.loc[valid].copy()
    fund["eff_ord"] = pos[valid]

    panel_df = panel_df.sort_values(["Ticker", "Date"]).copy()
    ordinal = {d: i for i, d in enumerate(trading_dates)}
    panel_df["_ord"] = panel_df["Date"].map(ordinal)
    panel_df = panel_df.sort_values("_ord")

    for short in ["Equity", "Assets", "NetIncome", "GrossProfit", "Shares"]:
        rows = fund[fund["concept"] == short]
        # one value per (Ticker, effective date): the most recently reported period
        rows = (rows.sort_values(["Ticker", "eff_ord", "period_end"])
                    .drop_duplicates(["Ticker", "eff_ord"], keep="last"))
        rows = rows[["Ticker", "eff_ord", "val"]].rename(columns={"val": short}).sort_values("eff_ord")
        panel_df = pd.merge_asof(
            panel_df, rows, left_on="_ord", right_on="eff_ord", by="Ticker", direction="backward"
        ).drop(columns=["eff_ord"])

    mktcap = panel_df["Close"] * panel_df["Shares"]
    def safe(num, den):
        return np.where((den.notna()) & (den > 0) & num.notna(), num / den.where(den != 0, np.nan), np.nan)

    panel_df["MarketCap"] = np.where(mktcap > 0, mktcap, np.nan)
    mc = panel_df["MarketCap"]
    panel_df["BookToMarket"] = safe(panel_df["Equity"], mc)
    panel_df["EarningsYield"] = np.where(mc > 0, panel_df["NetIncome"] / mc, np.nan)
    panel_df["ROE"] = safe(panel_df["NetIncome"], panel_df["Equity"])
    panel_df["GrossProfitability"] = safe(panel_df["GrossProfit"], panel_df["Assets"])

    # Data-error guard: a mis-reported share count (e.g. BRK-B reporting only one
    # class) yields an absurd market cap and B/M in the hundreds. Real B/M rarely
    # exceeds ~5; NaN anything above 10 so a data glitch can't top the screener.
    # Also NaN B/M <= 0: negative-equity firms produce a negative B/M that would
    # otherwise land in the short bucket as if it were extreme-growth, not distress.
    panel_df.loc[(panel_df["BookToMarket"] > 10) | (panel_df["BookToMarket"] <= 0),
                 "BookToMarket"] = np.nan
    return panel_df.drop(columns=["_ord"])
