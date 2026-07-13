from __future__ import annotations

import re
import threading
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

from config import CACHE_DIR, EDGAR_USER_AGENT
from data.cache import read_parquet_cache, write_parquet_cache

# Shared SEC EDGAR primitives, used by every EDGAR consumer (insider Form 4,
# fundamentals). SEC allows up to 10 req/s; latency (~0.85s/request) makes
# serial fetching the bottleneck, so consumers fetch concurrently while a
# single *global* token-slot limiter here caps the aggregate rate below 10/s —
# the limit is per-requester, so it must be shared across all consumers.
HEADERS = {"User-Agent": EDGAR_USER_AGENT}
MAX_WORKERS = 8
_MIN_INTERVAL = 0.11  # ~9 req/s aggregate
_rate_lock = threading.Lock()
_next_slot = [0.0]


def acquire_slot() -> None:
    """Hand out a staggered send-time slot (thread-safe), then sleep until it
    *outside* the lock so workers wait concurrently rather than serializing."""
    with _rate_lock:
        slot = max(time.time(), _next_slot[0])
        _next_slot[0] = slot + _MIN_INTERVAL
    delay = slot - time.time()
    if delay > 0:
        time.sleep(delay)


def throttled_get(url: str) -> requests.Response:
    acquire_slot()
    return requests.get(url, headers=HEADERS, timeout=45)


# SEC's company_tickers.json resolves a few tickers to the wrong CIK (ambiguous
# ticker reuse). Pin the correct CIK for known-bad ones so filings actually load.
_CIK_OVERRIDES = {"XOM": "0000034088"}  # Exxon Mobil


def load_cik_map() -> dict[str, str]:
    """ticker -> zero-padded 10-digit CIK, cached to a small parquet."""
    path = CACHE_DIR / "edgar_cik_map.parquet"
    cached = read_parquet_cache(path)
    if cached is not None:
        return {**dict(zip(cached["ticker"], cached["cik"])), **_CIK_OVERRIDES}
    data = throttled_get("https://www.sec.gov/files/company_tickers.json").json()
    rows = [(v["ticker"], str(v["cik_str"]).zfill(10)) for v in data.values()]
    df = pd.DataFrame(rows, columns=["ticker", "cik"])
    write_parquet_cache(df, path)
    return {**dict(zip(df["ticker"], df["cik"])), **_CIK_OVERRIDES}


def list_filings(cik: str, forms: set[str], start: str) -> list[tuple[str, str, str, str]]:
    """(accessionNumber, filingDate, primaryDocument, form) for filings whose
    form is in `forms` and filingDate >= start, across the recent page and any
    older-year pages of the submissions API."""
    out = []

    def collect(block: dict) -> None:
        block_forms = block.get("form", [])
        for i, f in enumerate(block_forms):
            if f not in forms:
                continue
            fdate = block["filingDate"][i]
            if fdate < start:
                continue
            out.append((block["accessionNumber"][i], fdate, block["primaryDocument"][i], f))

    sub = throttled_get(f"https://data.sec.gov/submissions/CIK{cik}.json").json()
    collect(sub["filings"]["recent"])
    for extra in sub["filings"].get("files", []):
        if extra.get("filingTo", "9999") < start:  # older page can't reach our range
            continue
        collect(throttled_get(f"https://data.sec.gov/submissions/{extra['name']}").json())
    return out


def fetch_filing_text(cik: str, accession: str, doc: str) -> str:
    """Fetch a filing's primary document and return whitespace-normalized
    plaintext (HTML stripped via BeautifulSoup). Empty string on any failure."""
    accnd = accession.replace("-", "")
    raw_doc = doc.split("/")[-1]
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accnd}/{raw_doc}"
    try:
        resp = throttled_get(url)
        if resp.status_code != 200:
            return ""
        text = BeautifulSoup(resp.content, "html.parser").get_text(" ")
        return re.sub(r"\s+", " ", text).strip()
    except Exception:  # noqa: BLE001 - network/parse errors: skip this filing
        return ""
