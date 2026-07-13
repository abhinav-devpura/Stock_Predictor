import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import FORWARD_HORIZON, UNIVERSE_NAME
from data.prices import add_forward_return, add_returns, load_price_panel, select_non_overlapping_dates
from data.universe import get_universe
from factors import pead
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
    print(f"Price panel: {len(price_panel)} rows, {price_panel['Ticker'].nunique()} tickers")

    # All factor columns (incl. earnings-derived) + horizon forward return,
    # then a non-overlapping date subsample for the event-study t-tests.
    features_panel = prepare_features(price_panel)
    features_panel = add_forward_return(features_panel, horizon=FORWARD_HORIZON)
    validation_panel = select_non_overlapping_dates(features_panel, FORWARD_HORIZON)

    live = validation_panel[pead._in_drift_window(validation_panel)]
    print(f"Non-overlapping validation sample: {len(validation_panel)} rows; "
          f"{len(live)} in the post-earnings drift window (<= {pead.DRIFT_WINDOW} trading days)")

    print("\n--- PEAD validation (cross-sectional, earnings surprise) ---")
    event_result = pead.run_event_study(validation_panel)["pead"]
    early, late = pead.run_pead_stability_check(validation_panel)
    stable = is_stable(early, late)
    passed = event_result.verdict == "PASS" and stable
    print(f"  event study: {event_result}")
    print(f"  stability early: {early}")
    print(f"  stability late:  {late}")
    print(f"  stable: {stable}  ->  {'INCLUDED' if passed else 'excluded'}")

    # Momentum-vs-PEAD diagnostic: how distinct are the two signals? (correlation
    # on the drift-window rows where PEAD is live and momentum is defined).
    diag = live.dropna(subset=["Momentum12_1", "PEAD_Surprise"])
    if len(diag) > 2:
        corr = diag["Momentum12_1"].corr(diag["PEAD_Surprise"])
        print(f"\nMomentum-vs-PEAD correlation (drift-window rows): {corr:.3f} "
              f"(near 0 => distinct signals)")

    feature_columns = list(pead.FEATURE_COLUMNS["pead"]) if passed else []
    print(f"\nValidated feature columns (event study PASS + stable): {feature_columns or '(none)'}")

    panel_df = assemble_panel(price_panel, feature_columns=feature_columns, horizon=FORWARD_HORIZON)
    print(f"Model panel: {len(panel_df)} rows after dropping incomplete rows")

    print("\n--- Walk-forward CV (embargo = horizon) ---")
    run_walk_forward(panel_df, feature_columns, embargo=FORWARD_HORIZON)

    print("\n--- Final holdout ---")
    run_final_holdout(panel_df, feature_columns, embargo=FORWARD_HORIZON)


if __name__ == "__main__":
    main()
