from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import CACHE_DIR, START_DATE
from data.cache import read_parquet_cache, write_parquet_cache
from data.edgar import MAX_WORKERS, fetch_filing_text, list_filings, load_cik_map
from data.loughran_mcdonald import NEGATIVE_WORDS, UNCERTAINTY_WORDS

_WORD = re.compile(r"[a-z]+")
_MIN_WORDS = 500  # skip tiny/failed full-doc extractions
_MIN_SECTION_WORDS = 300  # section-similarity is NaN below this (parse failed)

# "Lazy Prices" (Cohen-Malloy-Nguyen 2020) finds the signal is strongest in the
# Risk Factors (Item 1A) and MD&A (Item 7 / 10-Q Item 2) sections. Anchor the
# section START on its title (disambiguates from the TOC/cross-refs) and the END
# on the *next* section's title (a bare "Item N" also matches mid-sentence
# cross-references, which truncates the section — see NOTES).
_RF_START = re.compile(r"item\s+1a[\.\s]{1,4}risk\s+factors", re.I)
_RF_END = re.compile(r"item\s+(1b[\.\s]{1,4}unresolved|2[\.\s]{1,4}propert|3[\.\s]{1,4}legal)", re.I)
_MDA_START = re.compile(r"item\s+\d+a?[\.\s]{1,4}management.{0,3}s\s+discussion\s+and\s+analysis", re.I)
_MDA_END = re.compile(
    r"item\s+(7a[\.\s]{1,4}quantitat|8[\.\s]{1,4}financial|3[\.\s]{1,4}quantitat|4[\.\s]{1,4}controls)", re.I
)


def _tone(text: str) -> tuple[float, float]:
    words = _WORD.findall(text.lower())
    n = len(words)
    if n == 0:
        return np.nan, np.nan
    neg = sum(1 for w in words if w in NEGATIVE_WORDS)
    unc = sum(1 for w in words if w in UNCERTAINTY_WORDS)
    return neg / n, unc / n


def _one_section(text: str, start_re: re.Pattern, end_re: re.Pattern) -> str:
    starts = list(start_re.finditer(text))
    if not starts:
        return ""
    s = starts[-1].start()  # body occurrence, past the table of contents
    en = end_re.search(text, starts[-1].end())
    e = en.start() if en else min(len(text), s + 120000)
    return text[s:e]


def _sections(text: str) -> str:
    """Risk Factors + MD&A concatenated; '' if neither parses (caller NaNs it)."""
    rf = _one_section(text, _RF_START, _RF_END)
    mda = _one_section(text, _MDA_START, _MDA_END)
    return (rf + " " + mda).strip()


def _cosine(a: str, b: str) -> float:
    try:
        tf = TfidfVectorizer(stop_words="english", max_features=20000).fit_transform([a, b])
        return float(cosine_similarity(tf[0], tf[1])[0, 0])
    except ValueError:
        return np.nan


_TEXT_COLS = ["FilingDate", "form", "TextSimilarity", "SectionSimilarity", "NegTone", "UncTone"]
_LOOKBACK_DAYS = 450  # re-fetch window for incremental updates: long enough to
# re-include the prior same-form filing (annual 10-K / quarterly 10-Q) as the
# TF-IDF similarity baseline for any newly-filed report.


def _features_from_filings(cik: str, filings: list) -> pd.DataFrame:
    """Compute per-filing features for a given filing list (fetches each doc's
    text and diffs it against the prior same-form filing in the list)."""
    filings = sorted(filings, key=lambda f: f[1])  # by filing date
    texts = [fetch_filing_text(cik, acc, doc) for acc, fd, doc, form in filings]

    rows = []
    prev_full: dict[str, str] = {}
    prev_sec: dict[str, str] = {}
    for (acc, fdate, doc, form), text in zip(filings, texts):
        if len(text.split()) < _MIN_WORDS:
            continue
        neg, unc = _tone(text)
        sec = _sections(text)
        sec_ok = len(sec.split()) >= _MIN_SECTION_WORDS

        full_sim = _cosine(prev_full[form], text) if form in prev_full else np.nan
        p_sec = prev_sec.get(form)
        sec_sim = _cosine(p_sec, sec) if (sec_ok and p_sec and len(p_sec.split()) >= _MIN_SECTION_WORDS) else np.nan

        prev_full[form] = text
        if sec_ok:
            prev_sec[form] = sec
        rows.append({"FilingDate": fdate, "form": form, "TextSimilarity": full_sim,
                     "SectionSimilarity": sec_sim, "NegTone": neg, "UncTone": unc})
    return pd.DataFrame(rows, columns=_TEXT_COLS)


def _text_features_for_ticker(cik: str, start: str) -> pd.DataFrame:
    """Per-filing full-doc + section-level TextSimilarity (vs prior same-form
    filing) + LM tone, over all 10-K/10-Q filings since `start`."""
    return _features_from_filings(cik, list_filings(cik, {"10-K", "10-Q"}, start))


def _incremental_text_features(cik: str, cached: pd.DataFrame) -> pd.DataFrame:
    """Append only filings NEWER than what's cached. Lists filings in a lookback
    window (cheap, one request); if none are newer, returns the cache untouched
    with NO document fetches. Otherwise fetches just the window (so the new
    filing's prior same-form baseline is present), keeps the genuinely-new rows,
    and merges them into the cache."""
    latest = str(cached["FilingDate"].max())
    lookback = (pd.Timestamp(latest) - pd.Timedelta(days=_LOOKBACK_DAYS)).date().isoformat()
    filings = list_filings(cik, {"10-K", "10-Q"}, lookback)
    if not any(f[1] > latest for f in filings):
        return cached  # nothing new — no text downloads
    fresh = _features_from_filings(cik, filings)
    new_rows = fresh[fresh["FilingDate"] > latest]
    if new_rows.empty:
        return cached
    out = pd.concat([cached, new_rows], ignore_index=True)
    return out.drop_duplicates(["FilingDate", "form"], keep="last").sort_values("FilingDate").reset_index(drop=True)


def _cache_path(ticker: str, start: str) -> Path:
    return CACHE_DIR / f"filingtext_v2_{ticker}_{start}.parquet"  # v2: + SectionSimilarity


def load_text_features(tickers: list[str], start: str = START_DATE,
                       use_cache: bool = True, incremental: bool = False) -> pd.DataFrame:
    """Per-filing disclosure features [Ticker, FilingDate, form, TextSimilarity,
    SectionSimilarity, NegTone, UncTone]. Cached per ticker, fetched concurrently
    and rate-limited via data.edgar. Warns+skips tickers that error.

    incremental=True: for cached tickers, fetch only filings newer than the cache
    and append them (cheap — most runs download nothing). Without a cache for a
    ticker it falls back to a full fetch. use_cache=False forces a full refetch."""
    cik_map = load_cik_map()
    to_fetch, frames = [], []
    for ticker in tickers:
        path = _cache_path(ticker, start)
        cached = read_parquet_cache(path) if (use_cache or incremental) else None
        if cached is not None and not incremental:
            frames.append(cached.assign(Ticker=ticker))
            continue
        cik = cik_map.get(ticker)
        if cik is None:
            if cached is not None:
                frames.append(cached.assign(Ticker=ticker))  # keep what we have
            else:
                print(f"WARNING: no CIK for {ticker}, skipping filing text")
            continue
        to_fetch.append((ticker, cik, cached))

    def work(tc):
        ticker, cik, cached = tc
        try:
            if incremental and cached is not None and not cached.empty:
                df = _incremental_text_features(cik, cached)
                changed = len(df) != len(cached)
            else:
                df = _text_features_for_ticker(cik, start)
                changed = True
        except Exception as e:  # noqa: BLE001
            print(f"WARNING: filing-text failed for {ticker} ({e})")
            df = cached if cached is not None else pd.DataFrame(columns=_TEXT_COLS)
            changed = False
        return ticker, df, changed

    if to_fetch:
        # Modest worker count: each task itself fans out many document fetches.
        with ThreadPoolExecutor(max_workers=max(2, MAX_WORKERS // 2)) as pool:
            for ticker, df, changed in pool.map(work, to_fetch):
                n_sim = int(df["TextSimilarity"].notna().sum()) if not df.empty else 0
                tag = " (+new)" if changed and incremental else ""
                print(f"  {ticker}: {len(df)} filings, {n_sim} with similarity{tag}")
                # Persist changed results (even on a forced refresh), but never
                # clobber a good cache with an empty/failed pull, and skip rewrites
                # when an incremental update found nothing new (avoids churn).
                cpath = _cache_path(ticker, start)
                if changed and (not df.empty or not cpath.exists()):
                    write_parquet_cache(df, cpath)
                if not df.empty:
                    frames.append(df.assign(Ticker=ticker))

    frames = [f for f in frames if not f.empty]
    if not frames:
        raise RuntimeError(f"No filing text loaded for any of {len(tickers)} tickers.")
    return pd.concat(frames, ignore_index=True)


def attach_filing_text(panel_df: pd.DataFrame, use_cache: bool = True,
                       incremental: bool = False) -> pd.DataFrame:
    """Attach TextSimilarity, NegTone, UncTone to every stock-day via a
    leakage-safe as-of merge (each filing usable only on the next panel trading
    day strictly after its FilingDate — same rule as data.earnings/insider).

    incremental=True fetches only newly-filed 10-K/10-Q since the cache (cheap)."""
    feat = load_text_features(panel_df["Ticker"].unique().tolist(),
                              use_cache=use_cache, incremental=incremental)
    trading_dates = np.sort(panel_df["Date"].unique())

    pos = np.searchsorted(trading_dates, feat["FilingDate"].to_numpy(), side="right")
    valid = pos < len(trading_dates)
    feat = feat.loc[valid].copy()
    feat["eff_ord"] = pos[valid]
    feat = (feat.sort_values(["Ticker", "eff_ord"])
                .drop_duplicates(["Ticker", "eff_ord"], keep="last"))

    ordinal = {d: i for i, d in enumerate(trading_dates)}
    panel_df = panel_df.sort_values(["Ticker", "Date"]).copy()
    panel_df["_ord"] = panel_df["Date"].map(ordinal)
    panel_df = panel_df.sort_values("_ord")

    cols = ["TextSimilarity", "SectionSimilarity", "NegTone", "UncTone"]
    rows = feat[["Ticker", "eff_ord"] + cols].sort_values("eff_ord")
    panel_df = pd.merge_asof(
        panel_df, rows, left_on="_ord", right_on="eff_ord", by="Ticker", direction="backward"
    ).drop(columns=["_ord", "eff_ord"])
    return panel_df
