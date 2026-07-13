import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import BENCHMARK_TICKER, FORWARD_HORIZON, UNIVERSE_NAME
from data.benchmark import load_benchmark
from data.prices import add_forward_return, add_returns, load_price_panel, select_non_overlapping_dates
from data.universe import get_universe
from factors import calendar_effects
from model.panel import assemble_panel, prepare_features
from model.train import run_final_holdout, run_walk_forward
from validation.event_study import is_stable


def main():
    print(
        "NOTE: universe = current constituents projected back over the lookback "
        "window, so removed/delisted/acquired tickers are absent (survivorship bias). "
        "See NOTES.md."
    )
    print(f"Prediction horizon: {FORWARD_HORIZON} trading days (~1 month).")

    tickers = get_universe(UNIVERSE_NAME)
    price_panel = add_returns(load_price_panel(tickers))
    benchmark = load_benchmark(BENCHMARK_TICKER)
    print(f"Price panel: {len(price_panel)} rows, {price_panel['Ticker'].nunique()} tickers")
    print(f"Benchmark ({BENCHMARK_TICKER}): {len(benchmark)} rows")

    print("\n--- Calendar effect validation (market-wide, vs benchmark) ---")
    market_wide = calendar_effects.run_event_study(benchmark)
    for result in market_wide.values():
        print(f"  {result}")

    print("\n--- January effect validation (cross-sectional, by size proxy) ---")
    # Non-overlapping subsample at the horizon so the multi-day forward windows
    # don't overlap. NB: January only has ~21 trading days/year, so this leaves
    # very few January observations — see NOTES.md (data-starved by construction).
    validation_panel = prepare_features(price_panel)
    validation_panel = add_forward_return(validation_panel, horizon=FORWARD_HORIZON)
    validation_panel = select_non_overlapping_dates(validation_panel, FORWARD_HORIZON)
    january_size = calendar_effects.run_january_size_study(validation_panel)
    print(f"  {january_size}")

    if january_size.verdict == "FAIL" and january_size.p_value < 0.05:
        print("  Pooled result is significant but doesn't match the literature-predicted sign —"
              " running a split-sample stability check before trusting it either way:")
        early, late = calendar_effects.run_january_size_stability_check(validation_panel)
        print(f"    early half: {early}")
        print(f"    late half:  {late}")
        print(f"    stable (both halves independently significant + agree on sign): {is_stable(early, late)}")

    verdicts = {**market_wide, "january_size": january_size}

    feature_columns = []
    for effect_name, columns in calendar_effects.EFFECT_FEATURE_COLUMNS.items():
        if verdicts[effect_name].verdict == "PASS":
            feature_columns.extend(columns)
    print(f"\nValidated feature columns (PASS only): {feature_columns or '(none)'}")

    panel_df = assemble_panel(price_panel, feature_columns=feature_columns, horizon=FORWARD_HORIZON)
    print(f"Model panel: {len(panel_df)} rows after dropping incomplete rows")

    print("\n--- Walk-forward CV (embargo = horizon) ---")
    run_walk_forward(panel_df, feature_columns, embargo=FORWARD_HORIZON)

    print("\n--- Final holdout ---")
    run_final_holdout(panel_df, feature_columns, embargo=FORWARD_HORIZON)


if __name__ == "__main__":
    main()
