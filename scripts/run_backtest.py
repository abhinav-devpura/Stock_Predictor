"""Phase 9 — the screener, measured as a ranked portfolio (the correct metric
for an extremes-concentrated edge). Ranks stocks by combined factor Score,
backtests a rebalanced top-decile + long-short portfolio, and prints today's
top picks. Reports full / early / late splits and transaction-cost sensitivity,
because we know the signal faded post-2021 and the edge is small.

Run: python scripts/run_backtest.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import FORWARD_HORIZON, UNIVERSE_NAME
from data.prices import add_forward_return, add_returns, load_price_panel
from data.universe import get_universe
from model.backtest import rebalance_backtest, summarize
from model.panel import prepare_features
from model.score import add_score

# The factors that cleared the adopted gate (see NOTES). PEAD is the clean
# signal; reversal_sectorN is survivorship-caveated with a ~0 recent half.
VALIDATED = ["PEAD_Surprise", "Reversal5_SectorNeutral"]
SPLIT = "2021-01-01"


def report(name, bt, leg):
    def line(tag, sub):
        s0 = summarize(sub[leg], sub["turnover"], cost_bps=0)
        s10 = summarize(sub[leg], sub["turnover"], cost_bps=10)
        if not s0:
            print(f"    {tag:<6} (too few periods)")
            return
        print(f"    {tag:<6} n={s0['n_periods']:<3} annRet={s0['ann_return']:+.1%} "
              f"Sharpe={s0['sharpe']:+.2f} hit={s0['hit_rate']:.0%} "
              f"maxDD={s0['max_drawdown']:.0%} turn={s0['avg_turnover']:.0%} | "
              f"annRet@10bps={s10['ann_return']:+.1%}")
    print(f"\n{name} [{leg}]:")
    line("full", bt)
    line("early", bt[bt["Date"] < SPLIT])
    line("late", bt[bt["Date"] >= SPLIT])


def main():
    print(f"Universe: {UNIVERSE_NAME}  |  horizon: {FORWARD_HORIZON}d  |  "
          f"rebalanced top-decile portfolio\n")

    price_panel = add_returns(load_price_panel(get_universe(UNIVERSE_NAME)))
    panel = prepare_features(price_panel)
    panel = add_forward_return(panel, horizon=FORWARD_HORIZON)

    for label, feats in [("PEAD only", ["PEAD_Surprise"]), ("PEAD + reversal", VALIDATED)]:
        scored = add_score(panel, feats)
        bt = rebalance_backtest(scored, "Score", horizon=FORWARD_HORIZON, top_frac=0.1)
        # universe (equal-weight) benchmark over the same rebalance periods
        report(f"=== {label} — top decile vs universe ===", bt, "top")
        report(f"=== {label} — long-short (top minus bottom) ===", bt, "longshort")
        uni = summarize(bt["universe"])
        print(f"    universe  annRet={uni.get('ann_return', float('nan')):+.1%} "
              f"Sharpe={uni.get('sharpe', float('nan')):+.2f}  (equal-weight benchmark)")

    # Today's top picks (latest date), by the clean PEAD signal
    scored = add_score(panel, ["PEAD_Surprise"])
    latest = scored["Date"].max()
    today = scored[(scored["Date"] == latest) & scored["Score"].notna()]
    top = today.nlargest(10, "Score")[["Ticker", "Sector", "Score", "PEAD_Surprise"]]
    print(f"\n=== Today's top 10 by PEAD score ({latest}) ===")
    print(top.to_string(index=False))


if __name__ == "__main__":
    main()
