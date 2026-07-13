"""The screener — combine the two validated edges (value = sector-neutral
book-to-market; text-change = sector-neutral section-level filing similarity)
into one score, backtest the combined beta-neutral long-short vs each factor
alone, check whether they diversify, and print today's top picks.

Run: python scripts/run_combined.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DISCLOSURE_SUBUNIVERSE, FORWARD_HORIZON
from data.fundamentals_edgar import attach_fundamentals
from data.filings_text import attach_filing_text
from data.prices import add_forward_return, add_returns, load_price_panel
from data.universe import attach_sector, get_universe
from factors import disclosure, value
from model.backtest import rebalance_backtest, summarize
from model.score import add_score

SPLIT = "2021-01-01"
_BROKEN = {"XOM", "TOWN", "SCVL"}
VALUE = "BookToMarket_SectorNeutral"
TEXT = "SectionSimilarity_SectorNeutral"
CLIP = 3.0  # winsorize combined score so one extreme-B/M name can't dominate the rank


def _universe(n=150):
    uni = list(dict.fromkeys(DISCLOSURE_SUBUNIVERSE + get_universe(["small_cap"]) + get_universe(["sp500"])))
    return [t for t in uni if t not in _BROKEN][:n]


def _ls_line(tag, sub):
    s = summarize(sub["longshort"], sub["ls_turnover"], cost_bps=0)
    s10 = summarize(sub["longshort"], sub["ls_turnover"], cost_bps=10)
    if s:
        print(f"      {tag:<6} annRet={s['ann_return']:+.1%} Sharpe={s['sharpe']:+.2f} "
              f"hit={s['hit_rate']:.0%} | @10bps={s10['ann_return']:+.1%}")


def main():
    tickers = _universe()
    print(f"Combined screener — {len(tickers)} tickers, horizon {FORWARD_HORIZON}d "
          f"(value + section text-change; both cached)\n")

    panel = add_returns(load_price_panel(tickers))
    panel = attach_sector(panel)
    panel = attach_fundamentals(panel)
    panel = attach_filing_text(panel)
    panel = value.add_features(panel)
    panel = disclosure.add_features(panel)
    panel = add_forward_return(panel, horizon=FORWARD_HORIZON)

    bts = {}
    for name, feats, req in [("Value only", [VALUE], False), ("Text-change only", [TEXT], False),
                             ("Combined", [VALUE, TEXT], True)]:  # combined = both factors present
        scored = add_score(panel, feats, require_all=req, clip=CLIP)
        bt = rebalance_backtest(scored, "Score", horizon=FORWARD_HORIZON, top_frac=0.1)
        bts[name] = bt
        print(f"--- {name} — beta-neutral long-short ---")
        _ls_line("full", bt)
        _ls_line("early", bt[bt["Date"] < SPLIT])
        _ls_line("late", bt[bt["Date"] >= SPLIT])
        print()

    # Diversification: correlation of the two single-factor long-short series
    merged = bts["Value only"][["Date", "longshort"]].merge(
        bts["Text-change only"][["Date", "longshort"]], on="Date", suffixes=("_v", "_t")).sort_values("Date")
    corr = merged["longshort_v"].corr(merged["longshort_t"])
    print(f"Value vs Text-change long-short correlation: {corr:+.2f} over {len(merged)} shared "
          f"rebalances (low => they diversify)\n")

    # --- Weighting: equal vs WALK-FORWARD Sharpe-proportional (no look-ahead) ---
    import numpy as np
    ppy = 252 / FORWARD_HORIZON
    def sharpe(x):
        x = np.asarray(x, float)
        return x.mean() / x.std(ddof=1) * np.sqrt(ppy) if len(x) > 1 and x.std(ddof=1) > 0 else 0.0
    rv, rt = merged["longshort_v"].to_numpy(), merged["longshort_t"].to_numpy()
    WARM = 12  # need history to estimate trailing Sharpe before weighting
    eq, sw, wv_last, wt_last = [], [], 0.5, 0.5
    for i in range(WARM, len(merged)):
        sv, st = max(0.0, sharpe(rv[:i])), max(0.0, sharpe(rt[:i]))  # PAST data only
        wv, wt = (0.5, 0.5) if sv + st == 0 else (sv / (sv + st), st / (sv + st))
        wv_last, wt_last = wv, wt
        eq.append(0.5 * rv[i] + 0.5 * rt[i])
        sw.append(wv * rv[i] + wt * rt[i])
    print("Combining the two factor long-shorts, compared over the same post-warmup window:")
    print(f"    value alone        Sharpe={sharpe(rv[WARM:]):+.2f}")
    print(f"    text-change alone  Sharpe={sharpe(rt[WARM:]):+.2f}")
    print(f"    equal weight       Sharpe={sharpe(eq):+.2f}")
    print(f"    walk-fwd Sharpe-wt Sharpe={sharpe(sw):+.2f}  (latest weights: value {wv_last:.0%} / text {wt_last:.0%})\n")

    # Today's top picks by the combined score (both factors required, winsorized)
    scored = add_score(panel, [VALUE, TEXT], require_all=True, clip=CLIP)
    latest = scored["Date"].max()
    today = scored[(scored["Date"] == latest) & scored["Score"].notna()]
    cols = ["Ticker", "Sector", "Score", "BookToMarket", "SectionSimilarity"]
    print(f"=== Today's top 10 by combined score ({latest}) ===")
    print(today.nlargest(10, "Score")[cols].to_string(index=False))


if __name__ == "__main__":
    main()
