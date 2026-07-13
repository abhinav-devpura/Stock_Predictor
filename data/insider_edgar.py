from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from lxml import etree

from config import CACHE_DIR, INSIDER_WINDOW, START_DATE
from data.cache import read_parquet_cache, write_parquet_cache
from data.edgar import MAX_WORKERS, list_filings, load_cik_map, throttled_get

# Open-market codes only: P = purchase (bullish), S = sale. Everything else
# (grants A, option exercises M, gifts G, tax-withholding F, …) is noise.
_OPEN_MARKET = {"P", "S"}


def _form4_filings(cik: str, start: str) -> list[tuple[str, str, str]]:
    """(accessionNumber, filingDate, primaryDocument) for Form 4 filings >= start."""
    return [(acc, fd, doc) for acc, fd, doc, _form in list_filings(cik, {"4"}, start)]


def _parse_form4(cik: str, accession: str, doc: str) -> list[dict]:
    """Fetch and parse one Form 4 XML into open-market P/S transactions by
    officers/directors. Returns [] on any parse failure (robust — a single bad
    filing shouldn't kill the whole ticker)."""
    accnd = accession.replace("-", "")
    raw_doc = doc.split("/")[-1]  # strip any xslF345…/ viewer prefix
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accnd}/{raw_doc}"
    try:
        resp = throttled_get(url)
        if resp.status_code != 200:
            return []
        root = etree.fromstring(resp.content)
    except Exception:  # noqa: BLE001 - network/XML errors: skip this filing
        return []

    def flag(path: str) -> bool:
        e = root.find(path)
        return e is not None and (e.text or "").strip() in ("1", "true")

    is_officer = flag(".//reportingOwnerRelationship/isOfficer")
    is_director = flag(".//reportingOwnerRelationship/isDirector")
    if not (is_officer or is_director):
        return []  # exclude pure 10%-owners (often institutions, noisier signal)
    role = "officer" if is_officer else "director"

    rows = []
    for tx in root.findall(".//nonDerivativeTransaction"):
        code_e = tx.find(".//transactionCoding/transactionCode")
        code = (code_e.text or "").strip() if code_e is not None else ""
        if code not in _OPEN_MARKET:
            continue
        sh_e = tx.find(".//transactionAmounts/transactionShares/value")
        pr_e = tx.find(".//transactionAmounts/transactionPricePerShare/value")
        try:
            shares = float(sh_e.text)
            price = float(pr_e.text)
        except (TypeError, ValueError, AttributeError):
            continue
        if shares <= 0 or price <= 0:
            continue
        rows.append({"code": code, "shares": shares, "price": price,
                     "value": shares * price, "role": role})
    return rows


def _insider_cache_path(ticker: str, start: str) -> Path:
    return CACHE_DIR / f"insider_{ticker}_{start}.parquet"


def load_insider(tickers: list[str], start: str = START_DATE, use_cache: bool = True) -> pd.DataFrame:
    """Long-format open-market insider transactions: one row per (Ticker,
    FilingDate, P/S transaction). Cached per ticker to parquet (filings never
    change). Warns and skips tickers that error, like load_price_panel."""
    cik_map = load_cik_map()
    frames = []
    for ticker in tickers:
        path = _insider_cache_path(ticker, start)
        if use_cache:
            cached = read_parquet_cache(path)
            if cached is not None:
                frames.append(cached)
                continue

        cik = cik_map.get(ticker)
        if cik is None:
            print(f"WARNING: no CIK for {ticker}, skipping insider data")
            continue
        try:
            filings = _form4_filings(cik, start)
        except Exception as e:  # noqa: BLE001
            print(f"WARNING: EDGAR submissions failed for {ticker} ({e}), skipping")
            continue

        # Fetch+parse this ticker's Form 4 XMLs concurrently (the shared
        # _acquire_slot limiter keeps aggregate < 10/s across all workers).
        def fetch_one(f):
            accession, fdate, doc = f
            return [{**tx, "Ticker": ticker, "FilingDate": fdate}
                    for tx in _parse_form4(cik, accession, doc)]

        rows = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            for txs in pool.map(fetch_one, filings):
                rows.extend(txs)
        tdf = pd.DataFrame(rows, columns=["Ticker", "FilingDate", "code", "shares", "price", "value", "role"])
        print(f"  {ticker}: {len(filings)} Form 4 filings -> {len(tdf)} open-market P/S transactions")
        if use_cache:
            write_parquet_cache(tdf, path)
        frames.append(tdf)

    if not frames:
        raise RuntimeError(f"No insider data loaded for any of {len(tickers)} tickers.")
    return pd.concat(frames, ignore_index=True)


def attach_insider(panel_df: pd.DataFrame, window: int = INSIDER_WINDOW) -> pd.DataFrame:
    """Attach NetInsiderRatio, InsiderBuyIntensity and InsiderActive to every
    stock-day via a leakage-safe trailing-window aggregation.

    Leakage rule: a transaction becomes usable on the next panel trading day
    strictly after its FilingDate (the filing date is when it becomes public;
    next-trading-day is the conservative, look-ahead-free choice — same as
    data.earnings.attach_earnings).
    """
    insider = load_insider(panel_df["Ticker"].unique().tolist())
    trading_dates = np.sort(panel_df["Date"].unique())

    # Effective (first tradeable) date for each transaction.
    pos = np.searchsorted(trading_dates, insider["FilingDate"].to_numpy(), side="right")
    valid = pos < len(trading_dates)
    ins = insider.loc[valid].copy()
    ins["EffectiveDate"] = trading_dates[pos[valid]]
    ins["buy"] = np.where(ins["code"] == "P", ins["value"], 0.0)
    ins["sell"] = np.where(ins["code"] == "S", ins["value"], 0.0)

    # Daily per-ticker buy$/sell$ on the effective date, merged onto the panel.
    daily = ins.groupby(["Ticker", "EffectiveDate"])[["buy", "sell"]].sum().reset_index()
    panel_df = panel_df.sort_values(["Ticker", "Date"]).copy()
    panel_df = panel_df.merge(
        daily.rename(columns={"EffectiveDate": "Date"}), on=["Ticker", "Date"], how="left"
    )
    panel_df[["buy", "sell"]] = panel_df[["buy", "sell"]].fillna(0.0)

    g = panel_df.groupby("Ticker")
    trail_buy = g["buy"].transform(lambda s: s.rolling(window, min_periods=1).sum())
    trail_sell = g["sell"].transform(lambda s: s.rolling(window, min_periods=1).sum())
    total = trail_buy + trail_sell

    panel_df["NetInsiderRatio"] = np.where(total > 0, (trail_buy - trail_sell) / total, 0.0)
    dollar_vol = panel_df["Close"] * panel_df["Volume"]
    avg_dv = panel_df.assign(_dv=dollar_vol).groupby("Ticker")["_dv"].transform(
        lambda s: s.rolling(63, min_periods=20).mean()
    )
    panel_df["InsiderBuyIntensity"] = np.where(avg_dv > 0, trail_buy / avg_dv, 0.0)
    panel_df["InsiderActive"] = total > 0
    return panel_df.drop(columns=["buy", "sell"])
