# Stock Predictor — cross-sectional factor research + daily picks dashboard

A behavioral-finance quant-research pipeline that ranks stocks each day by a
**combined cross-sectional score**, built from two edges that survived a strict
event-study + beta-neutral long-short validation gate:

- **Value** — sector-neutral book-to-market (cheap stocks outperform).
- **Text-change ("Lazy Prices")** — year-over-year 10-K/10-Q section-level text
  similarity; firms that materially rewrite their disclosures subsequently
  underperform, so *stable* filings score higher.

Every factor family we tested is measured as a **beta-neutral long-short** (top
decile minus bottom decile, rebalanced every ~21 trading days, net of costs), so
the reported edge is alpha rather than market beta. See [NOTES.md](NOTES.md) for
the full research log, including the factors that *failed* the gate.

> ⚠️ Research prototype for educational use — **not investment advice.**

## The dashboard

**Live:** https://abhinav-devpura.github.io/Stock_Predictor/ *(via GitHub Pages — see below)*

Two front ends, both reading a single pre-computed snapshot
(`frontend/picks.json`) so they render instantly with no data refetch:

| | What | Run |
|---|---|---|
| **Interactive app** | Streamlit — ranked picks with sector/factor filters, per-factor breakdown, backtest curves | `streamlit run app.py` |
| **Standalone page** | Self-contained HTML, opens with no server | open `frontend/dashboard.html` |

The committed `frontend/picks.json` and `frontend/dashboard.html` mean a fresh
clone shows the dashboard **immediately** — no hours-long first-run fetch.

**Publishing it (GitHub Pages):** `scripts/build_dashboard.py` also writes
`docs/index.html`. In the repo's **Settings → Pages**, set *Deploy from a branch*
→ `main` → `/docs`. Your dashboard goes live at the Live URL above — a *project
site*, independent of any `username.github.io` personal site you may already have.
Re-run the build and commit `docs/` to refresh it.

## Quick start

```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # then set EDGAR_USER_AGENT

streamlit run app.py                                  # open the dashboard
```

To rebuild the snapshot from your own freshly-fetched data:

```bash
python scripts/build_picks.py                   # recompute from cache (fast, same as last time)
python scripts/build_picks.py --refresh         # + refetch prices (new as-of date, updated book/market)
python scripts/build_picks.py --refresh-edgar   # + refetch SEC filings/fundamentals (slow; picks up new 10-K/10-Q)
python scripts/build_dashboard.py               # regenerate the HTML pages + docs/
```

Everything is cached under fixed keys, so a plain `build_picks.py` reproduces the
last snapshot — use `--refresh` / `--refresh-edgar` to actually pull new data.

## Keeping it fresh automatically

`.github/workflows/refresh-picks.yml` rebuilds and commits the snapshot on a
schedule, so GitHub Pages redeploys with fresh data — no manual step:

- **Monthly** (1st): `--refresh` (prices only) — keeps book/market and the as-of date current.
- **Quarterly** (mid Feb/May/Aug/Nov, after earnings season): `--refresh --refresh-edgar` — pulls newly-filed 10-K/10-Q and fundamentals, which is when the picks actually move.
- Or trigger it by hand from the repo's **Actions** tab (with an optional EDGAR refetch).

**One-time setup:** add your SEC user-agent as a repository secret —
**Settings → Secrets and variables → Actions → New repository secret**, name
`EDGAR_USER_AGENT`, value like `your-name research you@example.com`. The workflow
persists `data_cache/` between runs so only new data is fetched. *(GitHub pauses
scheduled workflows after 60 days of repo inactivity — a push or manual run
re-arms them.)*

## First-run data note

The `data_cache/` directory (EDGAR fundamentals, filing text, price panels) is
**git-ignored** — it's large and derived. The dashboard doesn't need it because
`picks.json` is committed. But if you rerun `scripts/build_picks.py` on a clean
clone, it will re-download prices (yfinance) and re-fetch SEC EDGAR filings,
which takes a while (SEC is rate-limited to <10 req/s) and requires
`EDGAR_USER_AGENT` set in `.env`. Runs are cached, so subsequent builds are fast.

## Layout

```
app.py                     Streamlit dashboard (reads frontend/picks.json)
config.py                  universe, horizon, EDGAR user-agent (env-configurable)
data/                      prices, EDGAR (fundamentals / filings / insider), universe, sectors
factors/                   value, quality, disclosure, momentum, reversal, calendar, pead
model/                     score (cross-sectional z-score) + backtest (beta-neutral long-short)
validation/                event_study gate (bucket_test, stability, passes_gate) + self_test
scripts/build_picks.py     compute the snapshot -> frontend/picks.json
scripts/build_dashboard.py render the snapshot -> frontend/dashboard.html
scripts/run_*.py           per-phase research runners (see NOTES.md)
frontend/picks.json        committed snapshot the front ends read
```

Run the validation harness any time with `python -m validation.self_test`.
