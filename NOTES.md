# Project Notes

## Goal

A cross-sectional, multi-stock predictor built on behavioral-finance anomalies, each independently
validated via an event-study test before being added as a model feature. North star: a multi-factor
screener that ranks the universe and surfaces "stocks to buy" (a dashboard is a future step, not
current scope).

## Roadmap

| Phase | Status | Scope |
|---|---|---|
| 0 | **Done** | Data infra: universe + price panel + benchmark |
| 1 | **Done** | Event-study validation harness (self-tested on synthetic data) |
| 2 | **Done** | Calendar effects (turn-of-month, September, January) + first end-to-end model run |
| 3 | **Done** | Momentum (12-1 month, raw + sector-neutral) — both FAILED validation (see below) |
| 3.5 | **Done** | Switched prediction horizon to 21 days; added overlapping-return + leakage guards |
| 4 | **Up next** | Reddit sentiment, rebuilt at universe scale (from today's single-ticker `predict_stock.py`) |
| 5 | **Done** | PEAD (post-earnings drift) — FAILED stability (early PASS, late faded); distinct from momentum (corr ≈ 0) |
| 6 | Not started | FOMC/Fed announcement drift |
| 7 | Not started | Buybacks / equity issuance (via shares-outstanding deltas) |
| 8a | **Done** | Insider Form 4 (SEC EDGAR, 25-ticker sub-universe) — both variants FAILED (early-strong/late-fade); **8b no-go** (do not scale) |
| — | **Done (bonus)** | Short-term reversal (5-day) — FAILED (huge early t=7.4, gone late) |
| 9 | Not started | Combine validated factors into a per-date ranking/score (`model/score.py`) — blocked: still 0 validated factors |

**Deferred (documented, not planned for implementation):**
- **Index rebalancing/inclusion** — thin single-event signal for any one ticker.
- **Distressed mutual fund selling** — no free fund-flow data source.
- **Siamese/twin stocks** — most tickers in a standard universe have no dual-listed twin.
- **Stub arbitrage** — trading a parent company's implied "stub" value after netting out its stake
  in a publicly-traded subsidiary/spinoff. Needs a curated list of parent/subsidiary ownership
  pairs (not derivable generically from price data); revisit as a special-situations addendum.
- **Trump/Truth Social tweets** — can't know tomorrow's tweet before it happens, so this can't be a
  genuine forward-looking feature, only a retrospective event-study curiosity.

## Validation Harness (Phase 1)

`validation/event_study.py` — every future factor gets gated through this before it's allowed
into the model:
- `abnormal_return(panel_df, mode="cross_sectional"|"benchmark")` — adds an `abnormal_return`
  column. `cross_sectional` mode (stock return minus same-day cross-sectional mean) for
  stock-picking factors; `benchmark` mode (stock return minus benchmark return) for market-wide
  regime factors.
- `event_window_test(daily_returns, event_dates, ...)` — for date-based effects (calendar, FOMC).
  Runs a parametric t-test *and* an empirical placebo test (null distribution built by resampling
  same-size sets of non-event windows); verdict only passes if both agree.
- `bucket_test(panel_df, factor_col, forward_return_col, ...)` — for continuous cross-sectional
  factors (momentum, buybacks, insider, PEAD). Quantile-buckets by the factor, Welch's t-test on
  the top-minus-bottom spread, plus a monotonicity sanity flag.
- `stability_check(test_fn, data, split_frac=0.5)` / `is_stable(early, late)` (added Phase 2) — runs
  any test independently on an early and late half of the data by date, and checks both halves
  agree on significance and sign. A strong pooled result can be driven entirely by one period; this
  catches that before a factor is trusted (see Phase 2's January-size writeup below for the case
  that motivated adding this — and a bug caught while building it, worth reading before writing a
  stability check for a periodic/seasonal factor: always split the *full* timeline, not a
  pre-filtered periodic subset of it, or the split can fragment a single occurrence of the period
  across both halves).
- Verdict threshold (`p < 0.05` and `|t| > 2`, both fixed in `event_study.py` so no factor module
  can quietly redefine "significant") lives in one place: `P_VALUE_THRESHOLD`, `T_STAT_THRESHOLD`.

`validation/self_test.py` (`python -m validation.self_test`) proves the harness itself is
trustworthy before trusting it on real data: injects a known effect into synthetic data and checks
it's detected, and checks pure noise is correctly rejected, for both `event_window_test` and
`bucket_test`. All 8 checks pass as of Phase 1. Caught one real bug during development: the first
placebo implementation compared an *aggregated* event mean (over k event days) against
*individual* random daily returns instead of against means of same-size random samples — an
apples-to-oranges comparison that made a genuine injected effect look statistically insignificant.
Fixed by building the placebo null distribution from same-size (k-sample) resampled means. Re-run
the self-test after any change to `event_study.py`.

**Post-hoc code review (2026-07-09) found and fixed 6 more issues**, 3 of which had zero test
coverage and are now regression-tested in `self_test.py`:
- `load_benchmark()` never added a `Return` column, so `abnormal_return(mode="benchmark")` —
  documented as the intended usage — raised `KeyError: 'Return'`. Fixed: `load_benchmark()` now
  calls `add_returns()` before returning/caching. Regression test added.
- Both parquet caches (`data/prices.py`, `data/benchmark.py`) were keyed by a fixed filename that
  ignored the actual tickers/start/end arguments, so a cache hit could silently return data from a
  *different* request. Fixed: cache filenames now encode the request parameters (hashed ticker
  list + date range for prices, ticker + date range for benchmark).
- `bucket_test` silently degenerated into comparing a sample to itself when `qcut` collapsed to one
  bucket (sparse/duplicate-heavy factor, e.g. a mostly-zero insider-buying flag) — reported a clean
  FAIL indistinguishable from a genuine null result. Fixed with an explicit guard. Regression test
  added.
- `event_window_test` assumed `event_dates` were already `"YYYY-MM-DD"` strings; passing
  `Timestamp`/`datetime` values (the natural return type of e.g. `yfinance.get_earnings_dates()`,
  relevant for Phase 5 PEAD) silently matched zero events instead of erroring. Fixed with a
  `_normalize_date()` coercion step. Regression test added.
- `load_price_panel` crashed with an opaque `pd.concat([])` error if every ticker in a request
  failed to download. Fixed with an explicit check and a clear `RuntimeError`.
- `predict_stock.py`'s Reddit credential loading raised a bare `KeyError` with no guidance if
  `.env` was missing/incomplete. Fixed with a `_require_env()` helper that raises an actionable
  error message.

## Calendar Effects + First Model Run (Phase 2)

`factors/calendar_effects.py` tests turn-of-month, September, and January two ways: turn-of-month
and September are market-wide only (`run_event_study`, tested against SPY); January is tested both
market-wide *and* cross-sectionally (`run_january_size_study`) since the real literature effect
(Keim 1983) is concentrated in small-cap stocks, not the market overall — a date-level dummy alone
can't discriminate between stocks, so a `SizeProxy` (trailing 60-day log dollar volume, a stand-in
for market cap until Phase 7 adds real shares-outstanding data) was added specifically to make the
January effect testable/usable the way the literature describes it.

**Post-Phase-2 review fix (market-timing vs stock-selection effects):** a review caught that the
original `EFFECT_FEATURE_COLUMNS` listed `turn_of_month` and `september` as cross-sectional
model-feature candidates — but a pure date-level dummy is identical for every ticker on a given day,
so it carries *zero* information about the "beat today's cross-sectional median" target. Verified
empirically: `P(beat median | IsTurnOfMonth=1)` = `P(beat median | IsTurnOfMonth=0)` = 0.497 (same to
4 decimals for all three dummies; `SizeProxy`, which varies by ticker, shows a real 0.494 vs 0.498
spread by contrast). So even if turn-of-month or September had *passed* their market-wide event
study, the gating would have added a provably useless feature. Fixed: `add_features` no longer emits
`IsTurnOfMonth`/`IsSeptember`, and `EFFECT_FEATURE_COLUMNS` now contains only the `january_size`
bundle (`IsJanuary` + `SizeProxy`). Turn-of-month and September are still validated market-wide in
`run_event_study` for research — they're market-*timing* effects, not stock-*selection* features, and
this cross-sectional model can only use the latter. `IsJanuary` is also date-level/useless alone, but
is correctly kept because bundled with per-ticker `SizeProxy` a tree can learn the January×size
interaction.

Minor known imprecision (not fixed — negligible): `turn_of_month_dates` marks the data's first/last
partial-month boundaries as event days even when they're just where the 10y window was truncated, not
real month starts/ends — a handful of spurious days out of 507, nowhere near enough to change the
turn-of-month FAIL (p=0.15).

**First run, core100 (large-cap only), 2016–2026** — all 4 effects FAILED, including January-by-size
at p=0.065. Suspected this was partly an artifact of the universe being 100% large-cap with no real
size dispersion for the bucket test to work with, so extended the universe (see "Universe expansion"
below) and re-ran.

**Second run, core100 + small_cap (161 tickers), 2016–2026** (`python scripts/run_phase2.py`):

| Effect | Test | p-value | Verdict |
|---|---|---|---|
| Turn-of-month | market-wide vs. SPY | 0.152 (placebo 0.427) | FAIL |
| September | market-wide vs. SPY | 0.347 (placebo 0.619) | FAIL |
| January | market-wide vs. SPY | 0.211 (placebo 0.442) | FAIL |
| January (size) | cross-sectional bucket test | **0.0001**, t=3.82, n=13,074 | FAIL — significant, but wrong sign |

The market-wide tests are essentially unchanged (they're benchmarked against SPY, not our stock
universe, so adding small-caps doesn't move them — expected). The interesting result is
January-by-size: with real size dispersion in the universe, the pooled effect became **highly
significant** (p=0.0001, up from p=0.065) — but in the **opposite direction** from the classic 1983
finding. Large-caps outperformed small-caps in January in this pooled sample, not the reverse.

**Before trusting that pooled result, we asked: is it stable, or is it an artifact of one period?**
A strong pooled p-value can be driven entirely by one part of the sample, especially when the
factor's inclusion is being decided using the same 10 years the model will later be walk-forward
validated on — a subtler form of look-ahead than a code bug, but the same category of problem. Added
`validation.event_study.stability_check()`/`is_stable()`: split the sample in half by date, re-run
the same test on each half independently, and only treat the pattern as trustworthy if both halves
agree on significance and sign.

Result: **not stable.**

| Half | p-value | t-stat | monotonic |
|---|---|---|---|
| Early (2016–2021) | 0.0032 | 2.95 | True |
| Late (2021–2026) | 0.193 | 1.30 | False |

The reversed effect is significant and clean in the early half, but fades to non-significant and
non-monotonic in the half closer to now — the pooled p=0.0001 was disproportionately driven by
2016–2021. That's a better-grounded reason to exclude this factor than the original sign mismatch:
it's not just "doesn't match old literature," it's "isn't even holding up within our own data."

**Bug caught while building the stability check**: the first implementation of
`run_january_size_stability_check` filtered to January rows *before* splitting in half, so it split
the ~231-date January-only list at its own midpoint — which landed mid-January-2021, fragmenting
that year's January data across both halves instead of giving each half whole years. That produced a
false "stable, both halves significant" result. Fixed by splitting the *full* panel timeline first,
then filtering to January within each half — confirmed the fix reproduces the manual calculation
exactly. Worth remembering for future periodic/seasonal factors (FOMC, PEAD around earnings
clusters): always split the full underlying timeline, not a pre-filtered periodic subset of it.

Both runs together, plus the stability check, are useful: the null result on a homogeneous universe,
followed by a real-but-unstable opposite-signed effect once dispersion existed, is exactly the kind
of finding the event-study gate (and now the stability check) is supposed to surface — and
specifically *not* something a model would have caught if calendar dummies had been fed in blindly
without this validation step.

**Universe expansion**: added a `small_cap` tag to `data_cache/universe_snapshot.csv` (62 tickers,
diversified across sectors — Technology, Healthcare, Industrials, Financials, Consumer, Materials,
Energy, Real Estate, Utilities), each verified to have full 2016–2026 `yfinance` history before
being added (6 of 68 candidates were dropped for insufficient/missing data: `CSGS`, `PRFT`, `PDCO`,
`CIR`, `ESGR`, `LANC` — likely renamed/acquired/delisted since these were picked from memory).
Confirmed real size separation before committing: sampled small-caps trade at ~$3–40M/day average
dollar volume vs. ~$8.6B–$40B/day for `AAPL`/`MSFT`/`NVDA` — roughly 3 orders of magnitude apart.
`data/universe.py`'s `get_universe()` now accepts a list of universe tags (`config.py`'s
`UNIVERSE_NAME` is now `["core100", "small_cap"]`) and returns their union.

Since nothing passed (in either run), **no calendar features were added to the model** —
`run_walk_forward` and `run_final_holdout` both hit the "no validated feature columns" path and
skipped the actual `RandomForestClassifier` fit, which is the correct, designed-for behavior
(sklearn can't fit zero features anyway). What this run *did* verify end-to-end: `model/panel.py`'s
panel assembly, `model/target.py`'s cross-sectional target (class balance came out 50.3%/49.7% on
the 161-ticker universe, matching the median-split construction), and `model/splits.py`'s date-based
walk-forward splitter all work correctly on real ~425k-row panel data, not just Phase 1's synthetic
self-test data. Phase 3 (momentum) is the first factor that varies by ticker on its own (no
size-proxy workaround needed) and will be the first real test of whether the trained model beats
its baseline.

## Momentum + Horizon Switch (Phase 3 / 3.5)

12-1 month momentum (Jegadeesh–Titman: trailing ~12-month return skipping the most recent month to
avoid short-term reversal). `factors/momentum.py`.

**First tested against the next-day target — failed, informatively.** Pooled event study ≈ zero
(p=0.645), but the stability check revealed *why*: early half t=−3.14 (short-term **reversal**
dominates), late half t=+2.10 (**momentum**) — opposite significant signs that cancel in aggregate.

**Root cause = horizon mismatch, so we switched the model to a 21-day horizon** ("beat the median
over the next ~month" instead of tomorrow). Momentum is a slow signal; a daily horizon is dominated
by reversal/noise and is untradeable after costs anyway. This required two statistical guards
(`data/prices.py`, `model/splits.py`, `model/train.py`):
- **Overlapping-return guard**: a 21-day forward return on consecutive dates overlaps ~95% with its
  neighbor → autocorrelation inflates event-study significance. Fix: validation tests run on a
  **non-overlapping subsample** (`select_non_overlapping_dates`, every 21st date). The model keeps
  all rows (RF doesn't assume iid).
- **Leakage embargo**: a 21-day label at the train/test boundary peeks into the test window. Fix:
  `panel_walk_forward_splits(embargo=horizon)` (and the same in the holdout) drops `horizon` dates
  between train and test. Verified: every fold has a ≥22-date gap.

Also added, per user feedback that raw momentum conflates pure trend / earnings drift / sector-macro
moves: a **sector-neutral momentum** variant (`Momentum12_1_SectorNeutral` = raw minus same-date
same-sector mean, using `data/universe.py:attach_sector`). Earnings-driven momentum deferred to
Phase 5 (PEAD).

**Result at the 21-day horizon (`python scripts/run_phase3.py`) — both variants still FAIL:**

| Variant | Pooled p | Early half | Late half | Stable? |
|---|---|---|---|---|
| Raw momentum | 0.581 | t=−3.30 (reversal) | t=+2.25 (momentum) | No |
| Sector-neutral | 0.693 | t=−1.61 (n.s.) | t=+1.27 (n.s.) | No |

Two real takeaways:
1. **The regime sign-flip is not a horizon artifact** — raw momentum flips sign between the 2016–2021
   and 2021–2026 halves at *both* daily and monthly horizons (consistent with the well-documented
   ~2021 momentum crash / value rotation). It is genuinely not a stable cross-sectional signal in
   this universe/period, and the gate correctly refuses it.
2. **Sector-neutralizing confirms a large part of raw momentum here was sector/macro** — stripping
   sectors shrinks *both* regime effects toward noise (early t −3.30→−1.61, late +2.25→+1.27). The
   residual stock-specific momentum is too weak to detect. This validates the concern that "general"
   momentum was substantially "my sector moved."

So still **no validated features → the model still doesn't fit** (correctly — the gate is doing its
job, not being tuned until something passes). Everything else is now built and verified: 21-day
horizon, non-overlapping validation, embargo, sector attach, two-variant gating. Next real shot at a
model fit is a factor that survives the gate — sentiment (Phase 4) or PEAD (Phase 5).

## PEAD (Phase 5)

Post-earnings-announcement drift: positive earnings surprises → drift up over the following quarter.
`data/earnings.py` (loader + `attach_earnings`) and `factors/pead.py`. Earnings from
`yfinance.get_earnings_dates(limit=48)` — **~10–12y of history with Surprise(%) on nearly all rows**
(the earlier "~5 years / sparse" caveat was wrong at the default limit=20; corrected). Feature
`PEAD_Surprise` = most-recent surprise within a 63-trading-day drift window, else 0.

**Leakage handling (verified):** an announcement on date D is only tradeable on the next trading day
strictly after D (its "effective date"), since surprises are typically released after the close.
Spot-checked NVDA: announced 2025-05-28 (16:20, after close) → effective 2025-05-29 → drift day 1 =
05-30; no negative `DaysSinceEarnings`. `fwd_return` measures forward from the effective date, so the
announcement-day reaction is never inside the label.

**Result at the 21-day non-overlapping horizon (`python scripts/run_phase5.py`) — FAILED stability:**

| Test | mean_ar | t | p | Verdict |
|---|---|---|---|---|
| Pooled | +0.0031 | 1.24 | 0.214 | FAIL |
| Early half (2016–2021) | +0.0083 | **2.37** | **0.018** | PASS |
| Late half (2021–2026) | −0.0034 | −0.92 | 0.358 | FAIL |

The direction is *correct* (positive surprise → outperformance) and the effect is real and
significant in the early half — but it **faded to nothing in 2021–2026**, so it fails the stability
gate. Same "worked-then-decayed" shape as the January-size effect.

**Momentum-vs-PEAD (user's earlier question):** correlation of the two signals on drift-window rows
is **−0.012 ≈ 0** — they are essentially orthogonal. Momentum's (failed) edge is *not* just PEAD in
disguise; they're distinct signals that both happen to fail the gate for different reasons.

### Regime investigation — the "2021 break" was overstated (corrected)

After three factors failed, the working story was "classic anomalies all decayed around 2021"
(anomaly-decay / crowding). A rolling **quarterly cross-sectional rank IC** (`scripts/investigate_regime.py`,
also visualized as an artifact) **walks that back** — there is *no* clean common 2021 break:

- **Momentum** took **one** catastrophic quarter (2020Q4–2021Q1 momentum crash, IC ≈ −0.27) and then
  **recovered** — positive IC in 2022/2023/2024 and strongly positive into 2026 (mean IC pre-2021
  +0.010, 2021+ **+0.019**, i.e. slightly *better* after). Its two-half "sign-flip → unstable"
  verdict was largely an artifact of the 50/50 split boundary landing right on that short-lived crash.
- **PEAD** never had a stable regime to break — it flickers just above/below zero the whole decade
  (pre +0.020, post −0.008). "Early-half pass" was a couple of good years, not a durable edge.
- **Short-term reversal (5-day)** is the quiet standout: weakly but **persistently positive** IC in
  most quarters across *both* halves (pre +0.017, post +0.014), and it never inverted. It has **not**
  yet been run through the full gate — that's the natural next test.

**Takeaways:**
1. The gate refusing every factor was still *correct* — none is a stable standalone edge.
2. **Methodological**: a 50/50 stability split is **fragile near outlier periods** — one violent
   quarter straddling the boundary can flip the early-vs-late verdict (this is what happened to
   momentum, whose one crash quarter sat on the split). Rolling/annual IC is a useful *complementary*
   lens (`investigate_regime.py`), run alongside the two-half gate.

### Reversal — tested next, and it corrected the rolling-IC read (`scripts/run_reversal.py`)

The rolling IC suggested short-term (5-day) reversal was the "quiet persistent" signal, so it was
built (`factors/reversal.py`, raw + sector-neutral) and run through the strict gate:

| Variant | Pooled | Early half | Late half | Stable? |
|---|---|---|---|---|
| reversal_raw | PASS (t=4.01) | **t=7.37** (huge) | t=−0.61 (gone) | No |
| reversal_sector_neutral | FAIL (t=1.23) | t=2.04 | t=−0.03 | No |

**This walks back the artifact's "reversal quietly outlasted both" sub-claim.** The soft all-dates IC
lens showed a weak positive late-half (+0.014 mean), but the strict non-overlapping significance test
shows reversal's late-half edge is **gone** — it faded early-strong/late-weak like the others (and
its early half was *dramatically* strong, t=7.37). The two lenses genuinely disagree here: a small
average IC across many days can coexist with an insignificant top-vs-bottom quintile spread on the
sparse non-overlapping sample, and the non-overlapping test is the leakage-safe one that governs
model inclusion. Lesson reinforced: the rolling IC is a good *why* diagnostic but is a softer bar
than the gate; don't let it overturn a gate verdict.

## Insider Form 4 (Phase 8a)

The "less-crowded data" bet: disclosed insider buying (SEC EDGAR Form 4). `data/insider_edgar.py`
(concurrent, rate-limited EDGAR fetch/parse + leakage-safe `attach_insider`), `factors/insider.py`
(two size-neutral signals: `NetInsiderRatio` = value-weighted (buy$−sell$)/(buy$+sell$) over 126
trading days; `InsiderBuyIntensity` = buys-only ÷ trailing dollar volume). Validated on a 25-ticker
sub-universe first (`scripts/run_insider.py`) — deliberately NOT wired into the global pipeline, so
the huge fetch stays gated behind a positive result.

**Engineering notes:** serial EDGAR fetching was ~0.85s/request (latency-bound), ~13 min for 2
tickers. Fixed with concurrent fetching (`ThreadPoolExecutor`) + a thread-safe aggregate rate limiter
(`_acquire_slot`) holding < 10 req/s per SEC policy → **7.8× speedup** (816s → 105s for the same 2
tickers). Per-ticker parsed transactions cached to parquet. Leakage verified: effective date is
strictly after the filing date for every transaction (surprise/trade known only once filed).

**Result — both variants FAILED the gate:**

| Variant | Pooled t | Early half | Late half | Stable? |
|---|---|---|---|---|
| insider_net | 0.89 | **t=2.01** (monotonic, +) | t=−1.07 | No |
| insider_buys | 0.59 | t=1.20 | t=−1.03 | No |

The net-ratio early half is a *real, clean* signal — significant, monotonic, correct direction (net
insider buying → forward outperformance), consistent with insiders trading on genuine information —
but it fades/reverses post-2021 like everything else. Note: buys-only was *weaker* than net here, so
the literature claim that "insider sales are noise, purchases carry the signal" is **not** supported
in this sample. **Phase 8b (scale to full universe) is a no-go** — the gate correctly saved us the
expensive full fetch.

Data-quality caveats: 2 of 25 tickers (`SCVL`, `TOWN`) had no CIK in SEC's `company_tickers.json`
(that file omits some smaller names); `XOM` returned 0 Form 4s (a real bug to investigate before any
8b — likely a submissions-pagination edge case, since Exxon obviously has insider filings). 23
tickers had good data (3,519 transactions), enough for the validation.

### Synthesis (5 factor families tested, 0 passed) — the pattern is now the finding

- **January-size, PEAD, reversal, insider**: significant(ish) early → faded/reversed late.
- **Momentum**: never a real edge (noisy ≈ 0), one 2021 crash.

**Every single factor's late half is weaker than or opposite-signed to its early half.** Five
unrelated signals (price seasonality, earnings, price trend, short-term reversal, insider trading)
do not independently decay at the same split by coincidence — this points at something *structural
about the 2021+ period in this universe* (cross-sectional stock-picking signals broadly stopped
working — plausibly a macro-dominated, high-correlation regime) and/or the known **fragility of the
50/50 split** (documented in the regime investigation: momentum's "fade" was largely the split
straddling the 2021 crash). The honest read: **no standalone cross-sectional factor clears the gate
in this universe/period**, and we've now exhausted the obvious families (price, earnings, insider).
This is a genuine crossroads — see the strategy question raised to the user (regime conditioning,
universe/horizon rethink, or accept the null result) rather than mechanically grinding Phases 6–7,
which are likely to show the same pattern.

## Reframe experiment — universe expansion 161 → 565 (the turning point)

Hypothesis: the universal early-strong/late-fade was partly an **underpowered small universe** (161
names). Test: expand to `sp500 ∪ small_cap` (~565 names, ~3.4×) via a fixed
`fetch_sp500_from_wikipedia` (UA + GICS→our-sector normalization), rebuild, re-run all price/earnings
factors (`scripts/gate_sensitivity.py`). Insider not re-run (EDGAR fetch too large at 565).

**Result — the hypothesis was partly right, and it unlocked the first real factors.** t-stats grew
with N (as power predicts), and two factors crossed into significance in the correct direction:

| factor | full t (161→565) | early→late t @565 | strict gate | relaxed gate |
|---|---|---|---|---|
| momentum_raw | −0.55 → −1.89 | −7.8 → +4.0 | fail | fail (flips) |
| momentum_sectorN | −0.40 → −0.04 | −2.7 → +2.5 | fail | fail (flips) |
| reversal_raw | +4.0 → +7.2 | +12.8 → −0.6 | fail | fail (flips) |
| **reversal_sectorN** | +1.2 → **+3.0** | +4.9 → +0.05 | fail | **PASS** |
| **PEAD** | +1.2 → **+2.8** (p=0.005) | +2.9 → +1.1 | fail | **PASS** |
| january_size | +2.5 → +1.7 | +2.8 → +2.5 | (stable, wrong sign) | fail (sign) |

Readings:
- **PEAD and sector-neutral reversal now pass the *relaxed* gate** (full-sample significant + correct
  a-priori sign + no *significant* reversal). PEAD's late half went from −0.9 (reversed) at 161 to
  +1.1 (consistent, underpowered) at 565 — the clearest power signature. **First factors to survive.**
- They still **fail the ultra-strict** "both-halves-independently-significant" gate (late halves are
  same-sign but not individually significant) — the exact power-throwing-away flaw flagged in the
  gate-sensitivity analysis. **Neither the expansion nor the relaxed rule alone sufficed; together
  they work.**
- **Raw reversal & momentum still flip/fade even with power** → genuine regime dependence, not a
  sample-size artifact. But *sector-neutral* reversal is stable, so the instability was the
  macro/sector component; stripping it leaves a smaller real residual. (Consistent with the momentum
  finding.)
- **january_size** is now stable (both halves significant) but still the *large-cap* direction —
  opposite the literature's small-cap prediction — so the a-priori sign guard correctly excludes it
  (using it would be fitting the sign to the data).

**Caveats.** (1) Expanding to *current* S&P 500 adds 404 survivor large-caps, so survivorship bias
grew. PEAD is fairly robust to this (within-stock surprise → relative return); **short-term reversal
is more exposed** (losers that bounced survived; losers that kept falling delisted and are absent),
so reversal_sectorN's pass carries a survivorship asterisk — PEAD is the cleaner win. (2) These pass
the *relaxed* gate, which we have not yet formally adopted — doing so is the decision that would let
them into the model. (3) ~5 tickers dropped for missing yfinance/earnings data (FDXF, HONA, MRSH,
PSKY, Q — recent/renamed symbols); acceptable.

**Bottom line:** the null result was, in part, a small-universe + overly-strict-gate artifact. With
adequate breadth and a statistically sound stability rule, **PEAD (cleanest) and sector-neutral
reversal (survivorship-caveated) are real, correct-direction, non-reversing signals** — enough to
finally fit the model and reach Phase 9 (the screener). Also surfaced a bug: `load_price_panel`'s
`--refresh` skips the cache *write* (uses `use_cache=False` for both read and write) — a refresh
re-fetches but never persists; fix so `--refresh` re-downloads *and* overwrites cache.

## First model fit — and why it's a coin flip (the metric mismatch)

Adopted the relaxed gate (`validation/event_study.py:passes_gate` = full-sample event study PASS in
the expected direction + `halves_agree`, replacing the power-throwing "both halves significant"
rule). `scripts/run_model.py` validates every factor with it, assembles the panel from what passes,
and fits the walk-forward RandomForest. On the 565-universe, **`PEAD_Surprise` + `Reversal5_SectorNeutral`
passed** and the model fit for the first time — on 1.43M rows, 562 tickers.

**Result: the model does NOT beat baseline.** Walk-forward model 0.501 vs majority-class 0.500;
holdout 0.5001 vs 0.5006 (baseline actually higher). A coin flip. Zero classification edge.

**Why — a real measurement mismatch, not a bug:** the validation gate (`bucket_test`) measures the
**top-vs-bottom quintile spread** — do the *extremes* differ? The model target measures something
much harder: can we predict, for **every** stock, whether it beats the **median**? A factor can have
a real extreme-quintile spread while carrying almost no monotonic signal across the full
cross-section. And the effect sizes are *tiny*: PEAD's t=2.8 comes from a ~0.7% 21-day spread between
extreme quintiles on huge N — statistically real, economically marginal. That translates to ~0.1%
classification accuracy, i.e. noise. **Binary median-classification is the wrong tool for a small
edge concentrated in the ranked extremes.**

**Implication (important):** the edge, such as it is, lives where `bucket_test` found it — in the
*ranking extremes*, not in per-stock median-classification. The right way to measure/exploit it is a
**ranked long-short or top-decile portfolio** (go long the highest factor scores), evaluated as
**portfolio return / Sharpe / hit-rate**, not classifier accuracy. That is exactly what Phase 9
(`model/score.py`) should be — and it should be scored as a portfolio, not as accuracy. The
RandomForest binary classifier was the wrong final metric all along; it was fine as a plumbing test,
but the deliverable and its evaluation should be portfolio-based.

**Honest caveats before over-investing in Phase 9:** the PEAD spread is small (~0.7%/21d, extremes
only) and may not survive transaction costs; `Reversal5_SectorNeutral` carries the survivorship
asterisk and its late-half signal is ≈0 (passed the gate on a razor-thin +0.05). So a top-decile
PEAD portfolio is the one worth measuring; expectations should be modest.

## Phase 9 — ranked portfolio backtest, and the honest bottom line

`model/score.py` (cross-sectional z-score → combined `Score`), `model/backtest.py` (non-overlapping
rebalanced top-decile + long-short, with cost sensitivity and early/late split), `scripts/run_backtest.py`.
This is the *correct* measurement for an extremes-concentrated edge — and it delivers the project's
final, honest verdict.

**Results (565 universe, 21-day rebalance, 2016–2026):**

| Portfolio | full annRet | Sharpe | early | late |
|---|---|---|---|---|
| PEAD top-decile (long-only) | +19.4% | 1.05 | +22.4% | +16.7% |
| **PEAD long-short** (beta-neutral) | **−0.1%** | 0.02 | +0.6% | −0.8% |
| PEAD+reversal top-decile | +22.9% | 1.02 | +25.9% | +20.2% |
| **PEAD+reversal long-short** | +2.4% | 0.23 | **+10.3%** | **−4.4%** |
| **equal-weight universe (benchmark)** | **+17.8%** | **1.11** | — | — |

**The top-decile "19–23%" returns are market beta, not alpha — the key trap, caught.** The
equal-weight universe returned +17.8% at a *higher* Sharpe (1.11) than any top-decile portfolio. A
long-only book in the 2016–2026 bull market looks great while adding nothing; the honest test is the
**beta-neutral long-short**, and there:
- **PEAD long-short ≈ 0** (−0.1%/yr, Sharpe 0.02). The statistically-significant `bucket_test` spread
  does *not* survive as a tradeable market-neutral return — it's too small and washes out once you
  rank the whole universe (PEAD is 0 for most stock-days).
- **PEAD+reversal long-short works early (+10.3%) and reverses late (−4.4%)**, is driven by the
  survivorship-caveated reversal factor, has 81% turnover and a −43% drawdown, and is negative net of
  10 bps costs late. Not robust.

**Screener output is also degenerate today:** "top 10 by PEAD" on 2026-07-10 is all Score=0 because
no name has a recent earnings surprise in the drift window (yfinance earnings lag ~1 year at the
panel tail). The screener only produces picks right after earnings season, and even then the edge is
the non-robust one above.

### Final verdict
After 5 factor families, a universe expansion, a gate-methodology fix, a model fit, and a portfolio
backtest: **there is no robust, tradeable, market-neutral edge in these factors in this universe.**
The one statistically-real signal (PEAD) is tiny and extremes-only and vanishes as a long-short; the
one with a tradeable early long-short (reversal) is survivorship-caveated and reverses out of sample.
This is the same conclusion most honest factor searches reach — and arriving at it *without shipping a
beta portfolio dressed up as alpha* is the actual accomplishment. The value delivered is the
**rigorous, reusable pipeline and the discipline** (event-study gate, stability check, non-overlapping
validation, leakage embargo, long-short beta decomposition) that caught every trap along the way,
including this last one.

## Frontier — fundamental value + quality via EDGAR XBRL (the first real edge)

Survivorship-free universe was probed and **blocked** (yfinance has no delisted-name prices; recycled
tickers inject wrong-company data). yfinance fundamentals too shallow (~5 quarters). So fundamentals
were built from **SEC EDGAR XBRL `companyfacts`** (`data/edgar.py` shared primitives refactored out
of `insider_edgar.py`; `data/fundamentals_edgar.py`), point-in-time via filing dates. Factors:
`factors/value.py` (BookToMarket, EarningsYield) + `factors/quality.py` (ROE, GrossProfitability),
each raw + sector-neutral. Coverage: **520/565** tickers (XOM mis-mapped in SEC's `company_tickers.json`
to a spurious CIK — real CIK 34088; affects insider too; TOWN/SCVL have no CIK; banks lack GrossProfit).

**Result (`scripts/run_fundamentals.py`) — VALUE is the first factor with a genuine edge:**

| Factor | full t | early / late t | gate | LS annRet @10bps | LS Sharpe |
|---|---|---|---|---|---|
| **book_to_market** | +4.78 | +2.14 / +3.14 | **PASS (strict!)** | +4.6% | 0.38 |
| **book_to_market_sectorN** | +3.98 | +2.51 / +3.10 | **PASS (strict!)** | **+8.9%** | **0.75** |
| earnings_yield | +1.65 | +0.4 / +1.2 | excluded | −2.0% | — |
| earnings_yield_sectorN | −0.33 | −1.1 / +2.1 | excluded (unstable) | +6.8% | 0.60 |
| roe | −2.25 | −1.5 / −1.7 | excluded (**negative**) | −8.3% | −0.54 |
| gross_profitability(_sectorN) | ~0 | +ve early / −ve late | excluded (fades) | −2 to −5% | <0 |

Why book-to-market is categorically different from every prior factor:
1. **Passes the *strict* gate** (both halves individually significant + same sign) — the only factor
   ever to do so.
2. **Beta-neutral long-short is positive** (+8.9%/yr sector-neutral, net of 10 bps) — unlike the
   Phase 9 top-decile mirage, this is *not* market beta.
3. **Does NOT fade — strengthens late** (LS early +7.5% → late +10.5%, hit-rate 65% both halves),
   breaking the universal early-strong/late-fade pattern — matches the documented value winter
   (2016–20) → rebound (2021+).
4. **Survives transaction costs**, Sharpe 0.75.

Quality specifically *failed*: raw **ROE is significantly negative** (−8.3%/yr LS — high-ROE growth
names underperformed in the 2022 growth crash); gross-profitability fades late. So the edge is
**value, not quality**.

**The critical caveat (must not be lost): value is the factor MOST inflated by our survivorship bias.**
High book-to-market ("cheap") stocks are cheap partly for distress risk; the ones that went bankrupt
delisted and are absent, so our "cheap" bucket = the *survivors* among cheap stocks → the value
premium here is **upward-biased**. The true magnitude is likely well below +9% (survivorship-free
estimates of the value premium are materially lower). The *qualitative* result stands — value behaves
fundamentally differently (robust, non-fading, beta-neutral) from every price/event factor — but the
*magnitude* should be read with heavy skepticism until testable on a survivorship-free universe (paid
data). This is the one factor whose result would most change under a proper universe.

## Frontier — filing text-change ("Lazy Prices") + LM tone (null, but a weak test)

Disclosure signals from 10-K/10-Q text (`data/edgar.py` gained `list_filings`/`fetch_filing_text`;
`data/filings_text.py` = TF-IDF cosine similarity to prior same-form filing + Loughran-McDonald
negative/uncertainty tone; `data/loughran_mcdonald.py` = curated LM subset; `factors/disclosure.py`;
`scripts/run_disclosure.py`). Idea (user's): big filing rewrites signal trouble — the "Lazy Prices"
factor (Cohen–Malloy–Nguyen 2020). Validated on a 40-ticker sub-universe (full text is the heaviest
EDGAR pull); 40/40 coverage, ~42 filings each.

**Result — nothing cleared the gate:**
- **`text_similarity`** (the core idea): full t=**−2.05** — significant in the *wrong* direction
  (big-rewriters marginally *outperformed*, opposite of Lazy Prices), but weak in both halves →
  no reliable signal, possibly a hint of post-publication reversal.
- **`neg_tone` / `unc_tone`**: the familiar early-strong/late-fade (`unc_tone` early t=+2.26 → late
  −0.16; its sector-neutral long-short made +8.8%/yr early, vanished to −1.7% late).

**This is a *weak* test, not a clean refutation** — three real limitations: (1) **underpowered** —
40 tickers vs value's 565, so weak t-stats are partly low power; (2) **full-document is a proxy** —
Lazy Prices' strongest results use *section-level* (Item 1A Risk Factors / Item 7 MD&A) similarity,
which we deferred; (3) **post-publication decay** (published 2020) is consistent with the effect
being weak/reversed recently. Per discipline (no scaling the expensive text fetch without a positive
sub-universe signal), this stays at sub-universe scale. Refinements if revisited: section-level
similarity, the full LM dictionary, same-fiscal-quarter-prior-year comparison, and a larger universe.

### Section-level refinement — the sign flips to the *correct* direction (promising, underpowered)

Restricting the similarity to **Risk Factors (Item 1A) + MD&A (Item 7 / 10-Q Item 2)** — where Lazy
Prices says the signal lives — instead of the whole document (dominated by boilerplate financial
statements). Section extraction is title-anchored on both the section start *and* the next section's
title (a bare "Item N" also matches mid-sentence cross-references, which truncates the section);
`_MIN_SECTION_WORDS=300` NaNs failed parses. Coverage is imperfect (banks' MD&A parse poorly).

**Result — the refinement moved the signal into the right direction:**

| Signal | full t | early LS | late LS |
|---|---|---|---|
| full-document | −1.81 | −8.2% | −3.1% (wrong direction) |
| **section (RF+MD&A)** | −0.65 | **+7.1%** | −1.2% |
| **section, sector-neutral** | +0.86 | **+12.8%** (Sharpe 0.73, hit 60%) | **+2.8%** |

Full-document stays reversed/null (as before), but **section-level flips to the correct Lazy-Prices
direction** — long the stable filers, short the big-rewriters — and the sector-neutral long-short is
**+7.2%/yr, positive in *both* halves** (early +12.8%, late +2.8%), surviving 10 bps costs. That's
the user's exact mechanism, surfaced by confining the comparison to the sections that carry meaning.

At 40 tickers this did **not** pass the strict gate (section_similarity_sectorN t=+0.86, insignificant)
— directionally right but underpowered. Since it cleared the "positive sub-universe signal" bar
(correct direction, positive both halves, cost-surviving), we scaled it up.

### Scaled to 150 tickers — it PASSES: the SECOND validated edge (`scripts/run_disclosure.py`)

Both significance *and* effect size grew with N, exactly as the underpowered read predicted:

| variant | full t (40→150) | early/late t @150 | gate | LS annRet @10bps | Sharpe |
|---|---|---|---|---|---|
| **section_similarity_sectorN** | +0.86 → **+2.45** (p=0.014) | +1.58 / +2.16 | **PASS** | **+13.9%** | 0.85 |
| section_similarity (raw) | → +0.63 | −0.05 / +1.05 | excluded (sig) | +12.9% | **0.89** |
| text_similarity_sectorN (full-doc) | → +2.53 (p=0.012) | +0.19 / +2.23 | PASS (late-driven) | +5.6% | 0.37 |

**`section_similarity_sectorN` clears the gate** — full-sample significant in the a-priori direction,
both halves positive (late individually significant), beta-neutral long-short **+13.9%/yr net of
costs, Sharpe 0.85, positive in both halves and STRONGER late** (early +8.7% → late +18.4%). This is
the user's idea (big filing rewrites → underperformance) validated at scale — long stable filers,
short big-rewriters — and, like value, it does *not* fade. The raw section long-short is even
higher-Sharpe (~1.0 early) but misses the bucket-test significance; the sector-neutral one is the
clean pass.

**Why this is credible (more than most):** (1) **survivorship works in its favor** — the short side
is big-rewriters; those that went bankrupt *delisted* and are absent, so we can't short the worst
outcomes → the premium is **understated**, the opposite of value's upward bias. (2) The direction
was set **a priori** (`expected_sign=+1`) and passed on that sign — not a post-hoc fit; 2 of 8
variants passed, both sector-neutral, both correct-direction. Caveat: full-document text-change is
still null/reversed; only the *section-level* (Risk Factors + MD&A) comparison works, consistent with
the paper. Section extraction is imperfect (banks' MD&A parse poorly), so coverage is partial.

**Two validated edges now: value (book-to-market) and section-level filing text-change** — both
fundamental/disclosure, both non-fading, the two that survived where every price/event factor failed.

## Combined screener (`scripts/run_combined.py`, `model/score.py`)

Combine the two edges into one cross-sectional `Score` (z-score each within date, **winsorized to
±3σ**, averaged) and backtest the beta-neutral long-short vs each alone, on the 150-ticker universe
where both are cached.

| | full Sharpe | early | late | LS annRet @10bps |
|---|---|---|---|---|
| Value only (this universe) | 0.21 | 0.09 | 0.30 | +2.0% |
| **Text-change only** | **0.91** | 1.11 | 0.84 | **+14.3%** |
| Combined (equal-weight) | 0.61 | 0.37 | 0.75 | +9.1% |

**Key finding — uncorrelated, but naive combining *hurts*.** The two edges' long-short series
correlate **−0.13** (118 shared rebalances) — genuinely uncorrelated, so diversification is *possible*.
But equal-weight combining underperforms text-change alone (0.61 vs 0.91) because value is far weaker
*on this small-cap-heavy 150-subset* (0.21 vs 0.91) — equal-weighting a strong + weak signal dilutes
the strong one. The fix is weighting: for uncorrelated factors optimal weight ∝ Sharpe, which would
give ≈0.93 (marginally > text-alone). Practical read: on this universe **text-change dominates**;
value's premium is universe-dependent (strong on the full 565 large-cap set, weak here). The screener
produces sensible **top picks** — cheap (high B/M) *and* stable-disclosure (high section similarity)
names (HAIN, WSBC, ARE, BIIB…).

Bugs fixed in the process (all real): (1) `rebalance_backtest` chose rebalance dates from each
score's *filtered* rows → different factors landed on disjoint dates (correlation showed "0 shared");
now uses a **common date grid** from the full panel. (2) `BRK-B` had `BookToMarket=1564` (mis-reported
A/B share count) and topped the screener; `attach_fundamentals` now **NaNs B/M > 10** as a data-error
guard. (3) `add_score` gained **winsorization** (±3σ) + a `require_all` flag for clean multi-factor
comparison.

### Proportional (Sharpe) weighting — done walk-forward, and it still doesn't beat text alone

Followed the theory that for uncorrelated factors the optimal weight ∝ Sharpe. **Critically done
walk-forward** — at each rebalance, weight each factor by its *trailing* long-short Sharpe (past data
only); full-sample-Sharpe weights would be look-ahead bias (the earlier in-sample "≈0.93" estimate was
exactly that trap). Comparing the two factor long-shorts over the same post-warmup window:

| strategy | Sharpe |
|---|---|
| value alone | +0.07 |
| **text-change alone** | **+1.02** |
| equal weight | +0.86 |
| walk-forward Sharpe-weighted | +0.86 (latest weights: value 14% / text 86%) |

The weighting **worked as designed** — it learned to put 86% on text-change, 14% on value — but the
combo (0.86) **still lost to text-change alone (1.02)**. Lesson: *uncorrelated isn't enough — both
factors must be strong.* Value at Sharpe 0.07 on this small-cap-heavy universe is ≈ noise, so any
allocation to it dilutes more than the −0.13 correlation helps. The two edges are strong on
**different universes** (value on the full 565 large-caps; text-change on the 150), so realizing a
real diversification benefit would require both strong on the *same* names — i.e. running text-change
on the full 565 (a large fetch), the natural next step if pursuing the combination.

## Known Limitations

- **Survivorship bias**: `data_cache/universe_snapshot.csv` is *today's* ~161 constituents (100
  large-cap `core100` + 62 small-cap `small_cap`, added in Phase 2) projected back over the lookback
  window. Tickers that were delisted, acquired, or removed during that window are absent. This likely
  understates small-cap-specific effects even more than large-cap ones, since small caps are more
  prone to acquisition/delisting. Point-in-time constituent history would fix this but requires a
  paid dataset — out of scope for now. `scripts/build_phase0_panel.py` prints a reminder banner on
  every run.
- **SizeProxy is a proxy, not real market cap**: `factors/calendar_effects.py`'s `SizeProxy`
  (trailing 60-day log dollar volume) correlates with size but isn't the same thing — a real
  market-cap-based size measure needs Phase 7's shares-outstanding data. Revisit the January-size
  finding once that's available.
- **Seasonal effects are data-starved at the 21-day horizon**: with the non-overlapping validation
  subsample (needed to avoid overlapping-return significance inflation), an annual effect like
  January-size drops to ~1 obs/stock/year — n fell from ~13k to ~644 when the horizon moved to 21
  days. That's the honest, correct sample for a once-a-year effect tested this way; it still comes
  out unstable (early-half significant, late-half not). Low power on annual seasonals is a true fact
  of 10 years of data, not a fixable bug.
- **Earnings data depth**: `yfinance.get_earnings_dates(limit=48)` returns ~10–12y of reported
  quarters with Surprise(%) on nearly all — plenty for the 2016–2026 panel. (The earlier "~5 years"
  note reflected the default `limit=20`; resolved by loading with `limit=48` in `data/earnings.py`.)
  The most recent quarter or two may lag (Yahoo updates), so the tail of the panel can miss the
  newest earnings — acceptable.
- **yfinance flakiness**: on the first Phase 0 run, `MMC` returned no `history()` data from Yahoo
  (`fast_info` metadata loaded fine, so it isn't actually delisted — looks like a transient Yahoo
  API glitch for that symbol). The loader logged a `WARNING` and continued with 99/100 tickers.
  This is expected/handled behavior — re-run `scripts/build_phase0_panel.py --refresh` if a ticker
  drops out and check the coverage report before trusting the panel.
- **Modularization principle**: every factor/data source gets its own file (`factors/<name>.py`,
  `data/<name>.py`), never a shared "all features" module. Each factor module exposes exactly one
  `add_features(panel_df) -> panel_df` (and, once validated, one `run_event_study(panel_df)`).
  `model/panel.py` (later phases) composes them as a plain list of function references.
