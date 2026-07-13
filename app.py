"""Daily best-picks dashboard (Streamlit).

Reads the pre-computed snapshot at frontend/picks.json (produced by
`python scripts/build_picks.py`) and renders the ranked picks, per-factor
breakdown, and the beta-neutral long-short backtest curves. It does NOT run the
research pipeline — so it starts instantly and works for anyone who clones the
repo, as long as frontend/picks.json is present (it is committed).

Run:  streamlit run app.py
Refresh the data first with:  python scripts/build_picks.py
"""
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

PICKS_PATH = Path(__file__).resolve().parent / "frontend" / "picks.json"

st.set_page_config(page_title="Daily Best Picks", page_icon="📈", layout="wide")


@st.cache_data
def load_snapshot(mtime: float) -> dict:
    # mtime in the signature busts the cache when the file is regenerated.
    return json.loads(PICKS_PATH.read_text())


if not PICKS_PATH.exists():
    st.error(
        "No snapshot found at `frontend/picks.json`.\n\n"
        "Generate it first:\n\n```\npython scripts/build_picks.py\n```"
    )
    st.stop()

snap = load_snapshot(PICKS_PATH.stat().st_mtime)
picks = pd.DataFrame(snap["picks"])

# ---- Header -----------------------------------------------------------------
st.title("📈 Daily Best Picks")
gen = datetime.fromisoformat(snap["generated_at"]).strftime("%Y-%m-%d %H:%M UTC")
st.caption(
    f"As of **{snap['as_of_date']}** · {snap['scored_names']} scored names of "
    f"{snap['universe_size']} · {snap['horizon_days']}-day horizon · snapshot built {gen}"
)

# ---- Edge stat cards --------------------------------------------------------
st.subheader("The two edges, measured beta-neutral (long-short)")
edge_cols = st.columns(3)
_LABELS = {"combined": "Combined score", "value": "Value (book/market)",
           "text": "Text-change (Lazy Prices)"}
for col, key in zip(edge_cols, ["combined", "value", "text"]):
    e = snap["edges"].get(key)
    with col:
        st.markdown(f"**{_LABELS[key]}**")
        if not e:
            st.write("—")
            continue
        st.metric("Ann. return (net 10bps)", f"{e['ann_return_net']:+.1%}",
                  help=f"Gross {e['ann_return']:+.1%}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Sharpe", f"{e['sharpe']:.2f}")
        c2.metric("Hit rate", f"{e['hit_rate']:.0%}")
        c3.metric("Max DD", f"{e['max_drawdown']:.0%}")

corr = snap.get("value_text_correlation")
if corr is not None:
    st.caption(
        f"Value vs. text-change long-short correlation **{corr:+.2f}** — near zero, so they "
        "diversify. (The text edge is the stronger of the two; combining reduces drawdown.)"
    )

# ---- Equity curves ----------------------------------------------------------
st.subheader("Backtest — growth of $1 in each long-short book")
curve_frames = []
for key in ["combined", "value", "text"]:
    c = pd.DataFrame(snap["curves"].get(key, []))
    if not c.empty:
        c = c.rename(columns={"equity": _LABELS[key]}).set_index("date")[[_LABELS[key]]]
        curve_frames.append(c)
if curve_frames:
    curves = pd.concat(curve_frames, axis=1).sort_index()
    curves.index = pd.to_datetime(curves.index)
    st.line_chart(curves, height=340)
    st.caption(
        "Non-overlapping rebalances every ~21 trading days, top decile minus bottom decile. "
        "This is the market-neutral spread, not a total-return strategy."
    )

# ---- Where the edge comes from: the three legs ------------------------------
dc = snap.get("decile_curves", {})
_LEG = {"top": "Top decile (bought)", "bottom": "Bottom decile (shorted)", "universe": "Market (avg)"}
leg_frames = []
for key in ["top", "bottom", "universe"]:
    c = pd.DataFrame(dc.get(key, []))
    if not c.empty:
        c = c.rename(columns={"equity": _LEG[key]}).set_index("date")[[_LEG[key]]]
        leg_frames.append(c)
if leg_frames:
    st.subheader("Where the edge comes from — the three legs")
    legs = pd.concat(leg_frames, axis=1).sort_index()
    legs.index = pd.to_datetime(legs.index)
    # green = bought top decile, red = shorted bottom decile, grey = market
    st.line_chart(legs, height=320, color=["#0f8a4f", "#c53434", "#86857f"])
    st.caption(
        "Growth of $1 held **long** in each book. The strategy **buys the top decile and shorts "
        "the bottom decile**, so the gap between top (green) and bottom (red) — above the market "
        "(grey) — is the edge. The long-short curve above is exactly top minus bottom."
    )

# ---- Ranked picks -----------------------------------------------------------
st.subheader("Ranked picks")
left, right = st.columns([1, 3])
with left:
    sectors = sorted(s for s in picks["sector"].dropna().unique())
    chosen = st.multiselect("Sector", sectors, default=sectors)
    top_n = st.slider("Show top N", 5, min(50, len(picks)), min(20, len(picks)))
    tilt = st.radio("Rank by", ["Combined score", "Value only", "Text-change only"], index=0)

view = picks[picks["sector"].isin(chosen) | picks["sector"].isna()].copy()
_SORT = {"Combined score": "score", "Value only": "value_z", "Text-change only": "text_z"}
view = view.sort_values(_SORT[tilt], ascending=False).head(top_n)

display = view.rename(columns={
    "rank": "Rank (combined)", "ticker": "Ticker", "sector": "Sector", "score": "Score",
    "value_z": "Value z", "text_z": "Text z", "book_to_market": "Book/Mkt",
    "section_similarity": "Filing sim.", "close": "Close",
})
with right:
    st.dataframe(
        display.set_index("Ticker"),
        use_container_width=True,
        column_config={
            "Score": st.column_config.NumberColumn(format="%.2f"),
            "Value z": st.column_config.NumberColumn(format="%.2f"),
            "Text z": st.column_config.NumberColumn(format="%.2f"),
            "Book/Mkt": st.column_config.NumberColumn(format="%.2f"),
            "Filing sim.": st.column_config.NumberColumn(format="%.3f"),
            "Close": st.column_config.NumberColumn(format="$%.2f"),
        },
    )
    st.caption(
        "Higher **Value z** = cheaper vs. its sector (higher book/market). Higher **Text z** = "
        "more stable filings vs. a year ago (high filing similarity = bullish, per Lazy Prices)."
    )

# ---- Bottom decile (shorted names) ------------------------------------------
bp = pd.DataFrame(snap.get("bottom_picks", []))
if not bp.empty:
    st.subheader(f"Bottom decile — the {snap.get('decile_size', len(bp))} shorted names")
    sd = bp.rename(columns={
        "rank": "Rank", "ticker": "Ticker", "sector": "Sector", "score": "Score",
        "value_z": "Value z", "text_z": "Text z", "close": "Close",
    })[["Rank", "Ticker", "Sector", "Score", "Value z", "Text z", "Close"]]
    st.dataframe(
        sd.set_index("Ticker"),
        use_container_width=True,
        column_config={
            "Score": st.column_config.NumberColumn(format="%.2f"),
            "Value z": st.column_config.NumberColumn(format="%.2f"),
            "Text z": st.column_config.NumberColumn(format="%.2f"),
            "Close": st.column_config.NumberColumn(format="$%.2f"),
        },
    )
    st.caption(
        "The lowest-scoring names — what the long-short **sells short**. A short profits when these "
        "**underperform**. Mostly big filing-rewriters (very negative Text z) or expensive names "
        "(negative Value z)."
    )

# ---- Methodology + caveats --------------------------------------------------
with st.expander("Methodology & caveats"):
    st.markdown(snap["methodology"])
    st.markdown("**Caveats**")
    for c in snap["caveats"]:
        st.markdown(f"- {c}")
