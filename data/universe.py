from __future__ import annotations

import pandas as pd
import requests

from config import CACHE_DIR

SNAPSHOT_PATH = CACHE_DIR / "universe_snapshot.csv"


def get_universe(name: str | list[str] = "core100") -> list[str]:
    """Return the ticker list for one or more named universe tags from the
    committed snapshot, e.g. get_universe(["core100", "small_cap"]) to pull
    the union of both tags (a ticker appears once even if tagged in more
    than one requested universe).

    Reads a static CSV rather than fetching live so the panel is reproducible
    run-to-run. Refresh the snapshot with fetch_sp500_from_wikipedia() when
    the universe needs updating.
    """
    if not SNAPSHOT_PATH.exists():
        raise FileNotFoundError(
            f"No universe snapshot at {SNAPSHOT_PATH}. "
            "Commit data_cache/universe_snapshot.csv (see fetch_sp500_from_wikipedia() "
            "for a way to generate a fresh one)."
        )
    names = [name] if isinstance(name, str) else name
    snapshot = pd.read_csv(SNAPSHOT_PATH)
    tickers = snapshot.loc[snapshot["universe"].isin(names), "ticker"].drop_duplicates().tolist()
    if not tickers:
        raise ValueError(f"No tickers found for universe(s) {names} in {SNAPSHOT_PATH}")
    return tickers


def attach_sector(panel_df: pd.DataFrame) -> pd.DataFrame:
    """Merge a `Sector` column onto a panel from the committed universe
    snapshot (the same CSV get_universe reads), mapping by ticker across the
    whole snapshot regardless of universe tag. Used by sector-neutral momentum.
    Raises if any ticker in the panel has no sector mapping, so a stale
    snapshot can't silently drop names to NaN sector. Shape is `panel -> panel`
    so it composes as a model.panel INPUT_STEP."""
    snapshot = pd.read_csv(SNAPSHOT_PATH).drop_duplicates("ticker")
    sector_map = snapshot.set_index("ticker")["sector"]

    merged = panel_df.copy()
    merged["Sector"] = merged["Ticker"].map(sector_map)
    missing = sorted(merged.loc[merged["Sector"].isna(), "Ticker"].unique())
    if missing:
        raise ValueError(
            f"No sector mapping in {SNAPSHOT_PATH} for tickers: {missing}. "
            "Update the snapshot."
        )
    return merged


# GICS sector labels (Wikipedia) → this project's Sector convention, so
# sector-neutral factors group consistently across the snapshot.
_GICS_TO_SECTOR = {
    "Information Technology": "Technology",
    "Health Care": "Healthcare",
    "Financials": "Financials",
    "Consumer Discretionary": "Consumer Discretionary",
    "Consumer Staples": "Consumer Staples",
    "Communication Services": "Communication Services",
    "Industrials": "Industrials",
    "Energy": "Energy",
    "Utilities": "Utilities",
    "Materials": "Materials",
    "Real Estate": "Real Estate",
}


def fetch_sp500_from_wikipedia() -> pd.DataFrame:
    """Opt-in maintenance helper to pull current S&P 500 constituents as a
    DataFrame [ticker, name, sector, universe='sp500'].

    Not called by the default pipeline — use it to regenerate/expand the
    committed snapshot. Fetches with a descriptive User-Agent (Wikipedia 403s
    pandas' default UA) then parses with pd.read_html. Known limitation: this
    is *today's* constituents, so a panel built from it has survivorship bias
    when projected back over a 10y window.
    """
    import io

    from config import EDGAR_USER_AGENT

    html = requests.get(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        headers={"User-Agent": EDGAR_USER_AGENT},
        timeout=30,
    ).text
    table = pd.read_html(io.StringIO(html))[0]
    sp500 = table[["Symbol", "Security", "GICS Sector"]].rename(
        columns={"Symbol": "ticker", "Security": "name", "GICS Sector": "sector"}
    )
    sp500["ticker"] = sp500["ticker"].str.replace(".", "-", regex=False)
    sp500["sector"] = sp500["sector"].map(_GICS_TO_SECTOR).fillna(sp500["sector"])
    sp500["universe"] = "sp500"
    return sp500
