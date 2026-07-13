import os
from pathlib import Path

try:  # load .env so EDGAR_USER_AGENT / API keys are available via os.getenv
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_ROOT / "data_cache"

# 10y lookback window for the price panel. END_DATE=None resolves to "today"
# at fetch time (see data/prices.py) so the panel always covers the trailing
# 10 years rather than a fixed historical window.
START_DATE = "2016-01-01"
END_DATE = None

BENCHMARK_TICKER = "SPY"
# Prediction horizon in trading days: the model predicts whether a stock beats
# the cross-sectional median over the NEXT FORWARD_HORIZON days. ~21 = 1 month.
# Slow factors (momentum) need this; a next-day horizon is dominated by noise
# and untradeable after costs. Also used as the walk-forward embargo gap and
# the non-overlapping stride for validation sampling.
FORWARD_HORIZON = 21
# sp500: full current S&P 500 (~503). small_cap: 62 curated small-caps for size
# dispersion. Together ~565 names — expanded from the original core100+small_cap
# (161) to test whether factor signals were underpowered by a thin universe
# (power vs. real-decay experiment). core100 (subset of sp500) retained in the
# snapshot for provenance. Survivorship caveat grows with the expansion — see NOTES.
UNIVERSE_NAME = ["sp500", "small_cap"]

# --- Insider Form 4 (Phase 8) ---
# SEC fair-access policy requires a descriptive, identifying User-Agent with a
# contact address on every request (https://www.sec.gov/os/webmaster-faq#developers).
# Set EDGAR_USER_AGENT in your .env to your own name/email before running EDGAR
# pulls — SEC may throttle a shared/placeholder UA.
EDGAR_USER_AGENT = os.getenv(
    "EDGAR_USER_AGENT", "stock-predictor research example@example.com"
)
# Trailing window (trading days) over which insider buying/selling is aggregated.
INSIDER_WINDOW = 126  # ~6 months
# Phase 8a validates on this ~25-ticker sub-universe before paying the full-
# universe EDGAR fetch. Weighted toward smaller-caps and financials/industrials,
# where insiders actually make open-market purchases (mega-cap insiders mostly
# receive grants and sell), spanning sectors and both size tiers.
INSIDER_SUBUNIVERSE = [
    "JPM", "GM", "XOM", "GILD",
    "FFIN", "WSBC", "UBSI", "HOMB", "FCF", "INDB", "TOWN", "SAFT",
    "MATW", "FSS", "ROCK", "TRS", "ALG",
    "SCVL", "CHEF", "BGS", "JJSF",
    "MTRN", "KALU", "CENX",
    "PRGS",
]

# ~40-ticker sub-universe for the disclosure (filing text-change / tone) factor
# — full-text is the heaviest EDGAR pull, so validate here first. Spans sectors
# and both size tiers; excludes the CIK-broken names (XOM/TOWN/SCVL).
DISCLOSURE_SUBUNIVERSE = [
    "AAPL", "MSFT", "NVDA", "CRUS", "DIOD",
    "JPM", "FFIN", "WSBC", "UBSI", "HOMB", "FCF",
    "JNJ", "PFE", "GILD", "SUPN", "ATRC", "OMCL",
    "PG", "WMT", "JJSF", "CHEF", "BGS",
    "HD", "GM", "DIS",
    "CAT", "BA", "MATW", "FSS", "ROCK", "TRS", "ALG", "SXI",
    "MTRN", "KALU", "CENX", "NGVT",
    "DUK", "NEE",
    "PRGS",
]
