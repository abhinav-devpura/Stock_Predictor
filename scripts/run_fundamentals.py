"""Frontier — fundamental value + quality factors via EDGAR XBRL. Validate each
variant with the adopted gate, then (crucially) the beta-neutral long-short
backtest with early/late split — the real test of whether a fundamental factor
shows a market-neutral premium where price/event factors didn't.

Run: python scripts/run_fundamentals.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import FORWARD_HORIZON, UNIVERSE_NAME
from data.fundamentals_edgar import attach_fundamentals
from data.prices import add_forward_return, add_returns, load_price_panel, select_non_overlapping_dates
from data.universe import attach_sector, get_universe
from factors import quality, value
from model.backtest import rebalance_backtest, summarize
from validation.event_study import is_stable, passes_gate

SPLIT = "2021-01-01"


def main():
    print(f"Universe: {UNIVERSE_NAME}  |  horizon: {FORWARD_HORIZON}d  |  fundamentals via EDGAR XBRL\n")

    price = add_returns(load_price_panel(get_universe(UNIVERSE_NAME)))
    print("Loading fundamentals (first run fetches + caches EDGAR companyfacts; slow)...")
    panel = attach_sector(price)
    panel = attach_fundamentals(panel)
    panel = value.add_features(panel)
    panel = quality.add_features(panel)
    panel = add_forward_return(panel, horizon=FORWARD_HORIZON)

    cov = panel.dropna(subset=["BookToMarket"])["Ticker"].nunique()
    print(f"\nFundamentals coverage: {cov}/{panel['Ticker'].nunique()} tickers have BookToMarket\n")

    vp = select_non_overlapping_dates(panel, FORWARD_HORIZON)

    print("--- Value / quality validation (adopted gate) + long-short backtest ---")
    for mod, stab in [(value, value.run_value_stability_check), (quality, quality.run_quality_stability_check)]:
        for variant, full in mod.run_event_study(vp).items():
            col = mod.FEATURE_COLUMNS[variant][0]
            early, late = stab(vp, col)
            ok = passes_gate(full, early, late)
            print(f"\n  {variant:<28} full t={full.t_stat:+.2f} p={full.p_value:.3f}  "
                  f"early={early.t_stat:+.2f} late={late.t_stat:+.2f}  -> {'PASS' if ok else 'excluded'}")
            # long-short backtest for every variant (report regardless — the gate is
            # about the extreme spread; the LS return is the tradeable truth)
            bt = rebalance_backtest(panel, score_col=col, horizon=FORWARD_HORIZON, top_frac=0.1)
            for tag, sub in [("full", bt), ("early", bt[bt["Date"] < SPLIT]), ("late", bt[bt["Date"] >= SPLIT])]:
                s = summarize(sub["longshort"], sub["ls_turnover"], cost_bps=0)
                s10 = summarize(sub["longshort"], sub["ls_turnover"], cost_bps=10)
                if s:
                    print(f"      LS {tag:<5} annRet={s['ann_return']:+.1%} Sharpe={s['sharpe']:+.2f} "
                          f"hit={s['hit_rate']:.0%} | @10bps={s10['ann_return']:+.1%}")


if __name__ == "__main__":
    main()
