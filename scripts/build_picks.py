"""Offline compute step for the front end.

Runs the validated combined screener (value = sector-neutral book-to-market;
text-change = sector-neutral section-level filing similarity), then writes a
small self-contained JSON snapshot to frontend/picks.json:

  - today's ranked picks (combined score + per-factor breakdown),
  - the beta-neutral long-short equity curves (combined / value / text),
  - headline stats per edge and the diversification correlation,
  - methodology + caveats.

The front ends (Streamlit app.py, the shareable HTML) read ONLY this JSON, so
they render instantly with no EDGAR/price refetch. Commit picks.json so anyone
cloning the repo sees the dashboard without a multi-hour first run.

Run: python scripts/build_picks.py
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

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
# Winsorize the combined score: both factors are continuous (no tie problem) and
# this stops a single extreme-B/M name from dominating the cross-sectional rank.
CLIP = 3.0
OUT_PATH = Path(__file__).resolve().parent.parent / "frontend" / "picks.json"


def _universe(n=150):
    uni = list(dict.fromkeys(DISCLOSURE_SUBUNIVERSE + get_universe(["small_cap"]) + get_universe(["sp500"])))
    return [t for t in uni if t not in _BROKEN][:n]


def _edge_stats(bt):
    """Full-sample long-short stats (gross and net of 10bps), rounded for JSON."""
    gross = summarize(bt["longshort"], bt["ls_turnover"], cost_bps=0)
    net = summarize(bt["longshort"], bt["ls_turnover"], cost_bps=10)
    if not gross:
        return None
    return {
        "ann_return": round(gross["ann_return"], 4),
        "ann_return_net": round(net["ann_return"], 4),
        "sharpe": round(gross["sharpe"], 3),
        "hit_rate": round(gross["hit_rate"], 4),
        "max_drawdown": round(gross["max_drawdown"], 4),
        "n_rebalances": gross["n_periods"],
    }


def _equity_curve(bt, col="longshort"):
    """Cumulative growth of $1 in a per-rebalance return column (default the
    long-short spread; also used for the top / bottom / universe legs)."""
    r = bt.dropna(subset=[col]).sort_values("Date")
    curve = (1.0 + r[col]).cumprod()
    return [{"date": str(d), "equity": round(float(v), 4)}
            for d, v in zip(r["Date"], curve)]


def main(refresh: bool = False, refresh_edgar: bool = False):
    tickers = _universe()
    print(f"Building picks snapshot — {len(tickers)} tickers, horizon {FORWARD_HORIZON}d"
          f"{'  [refresh prices]' if refresh else ''}{'  [refresh EDGAR]' if refresh_edgar else ''}")

    panel = add_returns(load_price_panel(tickers, use_cache=not refresh))
    panel = attach_sector(panel)
    panel = attach_fundamentals(panel, use_cache=not refresh_edgar)
    panel = attach_filing_text(panel, use_cache=not refresh_edgar)
    panel = value.add_features(panel)
    panel = disclosure.add_features(panel)
    panel = add_forward_return(panel, horizon=FORWARD_HORIZON)

    # --- Backtests: combined + each factor alone (beta-neutral long-short) ---
    edges, curves = {}, {}
    combined_bt = None
    for name, feats, req in [("value", [VALUE], False), ("text", [TEXT], False),
                             ("combined", [VALUE, TEXT], True)]:
        scored = add_score(panel, feats, require_all=req, clip=CLIP)
        bt = rebalance_backtest(scored, "Score", horizon=FORWARD_HORIZON, top_frac=0.1)
        edges[name] = _edge_stats(bt)
        curves[name] = _equity_curve(bt)
        if name == "combined":
            combined_bt = bt

    # Decompose the combined edge into its legs: growth of $1 if you held the
    # top decile long, the bottom decile long (what the strategy shorts — it
    # should lag), and the whole universe (the market). The long-short curve
    # above is exactly top minus bottom; these show where that spread comes from.
    decile_curves = {
        "top": _equity_curve(combined_bt, "top"),
        "bottom": _equity_curve(combined_bt, "bottom"),
        "universe": _equity_curve(combined_bt, "universe"),
    }

    # Diversification: correlation of the two single-factor long-short series
    v_bt = rebalance_backtest(add_score(panel, [VALUE], clip=CLIP), "Score", horizon=FORWARD_HORIZON, top_frac=0.1)
    t_bt = rebalance_backtest(add_score(panel, [TEXT], clip=CLIP), "Score", horizon=FORWARD_HORIZON, top_frac=0.1)
    merged = v_bt[["Date", "longshort"]].merge(t_bt[["Date", "longshort"]], on="Date", suffixes=("_v", "_t"))
    corr = float(merged["longshort_v"].corr(merged["longshort_t"])) if len(merged) > 2 else float("nan")

    # --- Today's ranked picks by the combined score (both factors required) ---
    scored = add_score(panel, [VALUE, TEXT], require_all=True, clip=CLIP)
    as_of = scored["Date"].max()
    today = scored[(scored["Date"] == as_of) & scored["Score"].notna()].copy()
    today = today.sort_values("Score", ascending=False).reset_index(drop=True)

    def _num(x):
        return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)

    picks = [{
        "rank": i + 1,
        "ticker": row["Ticker"],
        "sector": row.get("Sector") if not (isinstance(row.get("Sector"), float) and np.isnan(row.get("Sector"))) else None,
        "score": _num(row["Score"]),
        "value_z": _num(row.get(f"{VALUE}_z")),
        "text_z": _num(row.get(f"{TEXT}_z")),
        "book_to_market": _num(row.get("BookToMarket")),
        "section_similarity": _num(row.get("SectionSimilarity")),
        "close": _num(row.get("Close")),
    } for i, row in today.iterrows()]

    # Bottom decile = the lowest-scoring 10% on the as-of date — the names the
    # long-short strategy SHORTS (same top_frac=0.1 the backtest uses). Present
    # them worst-first so the most-shorted name is on top.
    decile_n = max(1, round(len(picks) * 0.1))
    bottom_picks = list(reversed(picks[-decile_n:]))

    snapshot = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "as_of_date": str(as_of),
        "universe_size": len(tickers),
        "scored_names": len(picks),
        "horizon_days": FORWARD_HORIZON,
        "decile_size": decile_n,
        "edges": edges,
        "curves": curves,
        "decile_curves": decile_curves,
        "value_text_correlation": round(corr, 3) if not np.isnan(corr) else None,
        "picks": picks,
        "bottom_picks": bottom_picks,
        "methodology": (
            "Cross-sectional combined score: each stock is z-scored within its sector on "
            "book-to-market (value) and on year-over-year 10-K/10-Q section-level text "
            "similarity (the 'Lazy Prices' text-change signal — stable filings outperform). "
            "The two z-scores are averaged; a stock is only scored when both factors are "
            "present. Ranking is purely cross-sectional; go long the highest scores. "
            "Every edge is measured as a beta-neutral long-short (top decile minus bottom "
            f"decile) rebalanced every {FORWARD_HORIZON} trading days, net of transaction costs."
        ),
        "caveats": [
            "Research prototype, not investment advice. Past long-short backtest performance "
            "does not predict future returns.",
            "The S&P 500 universe is survivorship-biased (today's members), which flatters "
            "the value edge; the small-cap/text universe is more conservative.",
            "Both factors are slow (monthly-horizon) and update only when a firm files a new "
            "10-K/10-Q or reports fundamentals — picks change gradually, not daily.",
        ],
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(snapshot, indent=2))
    print(f"Wrote {OUT_PATH} — as_of {as_of}, {len(picks)} scored names")
    print(f"  edges: " + ", ".join(
        f"{k} Sharpe={v['sharpe']:+.2f}" for k, v in edges.items() if v))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build the picks snapshot -> frontend/picks.json")
    ap.add_argument("--refresh", action="store_true",
                    help="refetch prices from yfinance (ignore the price cache)")
    ap.add_argument("--refresh-edgar", action="store_true",
                    help="refetch EDGAR fundamentals + 10-K/10-Q text (slow; picks up new filings)")
    args = ap.parse_args()
    main(refresh=args.refresh, refresh_edgar=args.refresh_edgar)
