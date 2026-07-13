import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import FORWARD_HORIZON, INSIDER_SUBUNIVERSE
from data.insider_edgar import attach_insider
from data.prices import add_forward_return, add_returns, load_price_panel, select_non_overlapping_dates
from factors import insider
from validation.event_study import is_stable


def main():
    print("Phase 8a — insider Form 4 validation on a sub-universe "
          f"({len(INSIDER_SUBUNIVERSE)} tickers).")
    print(f"Prediction horizon: {FORWARD_HORIZON} trading days.\n")

    price_panel = add_returns(load_price_panel(INSIDER_SUBUNIVERSE))
    print(f"Price panel: {len(price_panel)} rows, {price_panel['Ticker'].nunique()} tickers\n")

    print("Loading insider Form 4 data (first run fetches + caches EDGAR; slow)...")
    panel = attach_insider(price_panel)
    panel = insider.add_features(panel)
    panel = add_forward_return(panel, horizon=FORWARD_HORIZON)

    active_frac = panel["InsiderActive"].mean()
    print(f"\nActive rows (recent insider trade in trailing window): {active_frac:.1%}")

    validation_panel = select_non_overlapping_dates(panel, FORWARD_HORIZON)
    live_n = int(validation_panel["InsiderActive"].sum())
    print(f"Non-overlapping validation sample: {len(validation_panel)} rows, {live_n} active\n")

    event_results = insider.run_event_study(validation_panel)
    any_pass = False
    for variant, result in event_results.items():
        feature_col = insider.FEATURE_COLUMNS[variant][0]
        early, late = insider.run_insider_stability_check(validation_panel, feature_col)
        stable = is_stable(early, late)
        passed = result.verdict == "PASS" and stable
        any_pass = any_pass or passed
        print(f"--- {variant} ---")
        print(f"  event study: {result}")
        print(f"  stability early: {early}")
        print(f"  stability late:  {late}")
        print(f"  stable: {stable}  ->  {'PASS THE GATE' if passed else 'excluded'}\n")

    print("=" * 70)
    if any_pass:
        print("At least one insider variant cleared the gate on the sub-universe.")
        print("Phase 8b decision: scale attach_insider to the full universe and wire")
        print("it into INPUT_STEPS / FACTOR_STEPS for the model.")
    else:
        print("No insider variant cleared the gate on the sub-universe — do NOT pay")
        print("for the full-universe fetch. Document and reconsider.")


if __name__ == "__main__":
    main()
