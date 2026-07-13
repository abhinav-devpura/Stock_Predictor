"""Frontier — filing text-change ("Lazy Prices") + Loughran-McDonald tone.
Sub-universe validation: does a disclosure signal (big 10-K/10-Q rewrites, or
negative/uncertain tone) show a beta-neutral long-short premium? Validate each
variant with the adopted gate, then the long-short backtest (early/late + costs).

Run: python scripts/run_disclosure.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DISCLOSURE_SUBUNIVERSE, FORWARD_HORIZON
from data.filings_text import attach_filing_text
from data.prices import add_forward_return, add_returns, load_price_panel, select_non_overlapping_dates
from data.universe import attach_sector, get_universe
from factors import disclosure
from model.backtest import rebalance_backtest, summarize
from validation.event_study import passes_gate

SPLIT = "2021-01-01"
_BROKEN_CIK = {"XOM", "TOWN", "SCVL"}  # mis-mapped / no CIK in SEC's file
TARGET_N = 150  # scaled test (power) — the 40 sub-universe were underpowered


def _build_universe() -> list[str]:
    """~150 tickers: the cached 40 sub-universe + all small-caps (where Lazy
    Prices is strongest) + a slice of the S&P 500, deduped, broken CIKs removed."""
    uni = list(dict.fromkeys(DISCLOSURE_SUBUNIVERSE + get_universe(["small_cap"]) + get_universe(["sp500"])))
    uni = [t for t in uni if t not in _BROKEN_CIK]
    return uni[:TARGET_N]


def main():
    tickers = _build_universe()
    print(f"Disclosure factor (section-level) — {len(tickers)}-ticker universe, "
          f"horizon {FORWARD_HORIZON}d.\n")

    price = add_returns(load_price_panel(tickers))
    print("Loading 10-K/10-Q text (first run fetches + caches; slow)...")
    panel = attach_sector(price)
    panel = attach_filing_text(panel)
    panel = disclosure.add_features(panel)
    panel = add_forward_return(panel, horizon=FORWARD_HORIZON)

    cov = panel.dropna(subset=["TextSimilarity"])["Ticker"].nunique()
    print(f"\nCoverage: {cov}/{panel['Ticker'].nunique()} tickers have TextSimilarity\n")

    vp = select_non_overlapping_dates(panel, FORWARD_HORIZON)
    print("--- Validation (adopted gate) + long-short backtest ---")
    for variant, full in disclosure.run_event_study(vp).items():
        col = disclosure.FEATURE_COLUMNS[variant][0]
        early, late = disclosure.run_disclosure_stability_check(vp, col)
        ok = passes_gate(full, early, late)
        print(f"\n  {variant:<26} full t={full.t_stat:+.2f} p={full.p_value:.3f}  "
              f"early={early.t_stat:+.2f} late={late.t_stat:+.2f}  -> {'PASS' if ok else 'excluded'}")
        # long-short: sign the score by expected direction so the LS is always
        # "long the good side" (bucket_test's expected_sign encodes direction).
        sign = disclosure._EXPECTED[variant][1]
        scored = panel.assign(_score=panel[col] * sign)
        bt = rebalance_backtest(scored, score_col="_score", horizon=FORWARD_HORIZON, top_frac=0.1)
        for tag, sub in [("full", bt), ("early", bt[bt["Date"] < SPLIT]), ("late", bt[bt["Date"] >= SPLIT])]:
            s = summarize(sub["longshort"], sub["ls_turnover"], cost_bps=0)
            s10 = summarize(sub["longshort"], sub["ls_turnover"], cost_bps=10)
            if s:
                print(f"      LS {tag:<5} annRet={s['ann_return']:+.1%} Sharpe={s['sharpe']:+.2f} "
                      f"hit={s['hit_rate']:.0%} | @10bps={s10['ann_return']:+.1%}")


if __name__ == "__main__":
    main()
