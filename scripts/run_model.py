"""The model runner: validate every cross-sectional factor with the adopted
relaxed gate (full-sample significant in the expected direction + no sign
reversal across halves), assemble the panel from whatever passes, and fit the
walk-forward RandomForest. This is the first run that can actually fit a model.

Run: python scripts/run_model.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import FORWARD_HORIZON, UNIVERSE_NAME
from data.prices import add_forward_return, add_returns, load_price_panel, select_non_overlapping_dates
from data.universe import get_universe
from factors import calendar_effects, momentum, pead, reversal
from model.panel import assemble_panel, prepare_features
from model.train import run_final_holdout, run_walk_forward
from validation.event_study import passes_gate


def _collect(vp):
    """Return [(factor_label, feature_col, full, early, late)] for every
    cross-sectional factor variant, evaluated on the non-overlapping panel vp."""
    out = []

    for mod, stab in [(momentum, momentum.run_momentum_stability_check),
                      (reversal, reversal.run_reversal_stability_check)]:
        ev = mod.run_event_study(vp)
        for variant, full in ev.items():
            col = mod.FEATURE_COLUMNS[variant][0]
            early, late = stab(vp, col)
            out.append((variant, col, full, early, late))

    full = pead.run_event_study(vp)["pead"]
    early, late = pead.run_pead_stability_check(vp)
    out.append(("pead", pead.FEATURE_COLUMNS["pead"][0], full, early, late))

    full = calendar_effects.run_january_size_study(vp)
    early, late = calendar_effects.run_january_size_stability_check(vp)
    out.append(("january_size", "SizeProxy", full, early, late))
    return out


def main():
    print(f"Universe: {UNIVERSE_NAME}  |  horizon: {FORWARD_HORIZON} trading days\n")

    price_panel = add_returns(load_price_panel(get_universe(UNIVERSE_NAME)))
    features = prepare_features(price_panel)
    features = add_forward_return(features, horizon=FORWARD_HORIZON)
    vp = select_non_overlapping_dates(features, FORWARD_HORIZON)

    print("--- Factor validation (adopted relaxed gate: full-sample PASS + no sign reversal) ---")
    feature_columns = []
    for label, col, full, early, late in _collect(vp):
        ok = passes_gate(full, early, late)
        # january_size's feature is the IsJanuary+SizeProxy bundle
        cols = calendar_effects.EFFECT_FEATURE_COLUMNS["january_size"] if label == "january_size" else [col]
        print(f"  {label:<26} full t={full.t_stat:+.2f} p={full.p_value:.3f}  "
              f"early={early.t_stat:+.2f} late={late.t_stat:+.2f}  -> {'INCLUDED' if ok else 'excluded'}")
        if ok:
            feature_columns.extend(c for c in cols if c not in feature_columns)

    print(f"\nValidated feature columns: {feature_columns or '(none)'}")
    if not feature_columns:
        print("Nothing passed — no model to fit.")
        return

    panel_df = assemble_panel(price_panel, feature_columns=feature_columns, horizon=FORWARD_HORIZON)
    print(f"Model panel: {len(panel_df)} rows, {panel_df['Ticker'].nunique()} tickers\n")

    print("--- Walk-forward CV (embargo = horizon) ---")
    run_walk_forward(panel_df, feature_columns, embargo=FORWARD_HORIZON)
    print("\n--- Final holdout ---")
    run_final_holdout(panel_df, feature_columns, embargo=FORWARD_HORIZON)


if __name__ == "__main__":
    main()
