"""Gate sensitivity: is the pass threshold too strict? For every factor
variant, report the full-sample and both-half test stats, then show which
would enter the model under the STRICT gate (both halves independently
significant + sign-consistent) vs a RELAXED gate (full-sample significant +
halves merely sign-consistent, i.e. no significant reversal). If the two gates
admit the same factors, strictness isn't the bottleneck.

Run: python scripts/gate_sensitivity.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from config import FORWARD_HORIZON, UNIVERSE_NAME
from data.prices import add_forward_return, add_returns, load_price_panel, select_non_overlapping_dates
from data.universe import get_universe
from factors import calendar_effects, momentum, pead, reversal
from model.panel import prepare_features
from validation.event_study import P_VALUE_THRESHOLD, T_STAT_THRESHOLD


def significant(r):
    return (not np.isnan(r.p_value)) and r.p_value < P_VALUE_THRESHOLD and abs(r.t_stat) >= T_STAT_THRESHOLD


def strict_pass(full, early, late):
    return significant(early) and significant(late) and np.sign(early.t_stat) == np.sign(late.t_stat)


def relaxed_pass(full, early, late):
    # Full-sample significant, and halves don't reverse (same sign, neither
    # half significantly opposite). Direction-agnostic — asks "is there a
    # consistent tradeable signal," not "does it match the literature sign."
    if not significant(full):
        return False
    same_sign = np.sign(early.t_stat) == np.sign(late.t_stat)
    return same_sign


def main():
    price = add_returns(load_price_panel(get_universe(UNIVERSE_NAME)))
    vp = add_forward_return(prepare_features(price), horizon=FORWARD_HORIZON)
    vp = select_non_overlapping_dates(vp, FORWARD_HORIZON)

    rows = []

    # momentum + reversal variants
    for mod, name in [(momentum, "momentum"), (reversal, "reversal")]:
        ev = mod.run_event_study(vp)
        for variant, full in ev.items():
            col = mod.FEATURE_COLUMNS[variant][0]
            check = mod.run_momentum_stability_check if mod is momentum else mod.run_reversal_stability_check
            early, late = check(vp, col)
            rows.append((variant, full, early, late))

    # PEAD
    full = pead.run_event_study(vp)["pead"]
    early, late = pead.run_pead_stability_check(vp)
    rows.append(("pead", full, early, late))

    # January-size
    full = calendar_effects.run_january_size_study(vp)
    early, late = calendar_effects.run_january_size_stability_check(vp)
    rows.append(("january_size", full, early, late))

    hdr = f"{'factor':<26}{'full t':>9}{'full p':>9}{'early t':>9}{'late t':>9}   {'STRICT':>7} {'RELAXED':>8}"
    print(hdr)
    print("-" * len(hdr))
    for name, full, early, late in rows:
        s = "PASS" if strict_pass(full, early, late) else "fail"
        r = "PASS" if relaxed_pass(full, early, late) else "fail"
        print(f"{name:<26}{full.t_stat:>+9.2f}{full.p_value:>9.3f}"
              f"{early.t_stat:>+9.2f}{late.t_stat:>+9.2f}   {s:>7} {r:>8}")

    print("\nSTRICT  = both halves independently significant (p<0.05 & |t|>2) + same sign  [current gate]")
    print("RELAXED = full-sample significant + halves same sign (no significant reversal)")


if __name__ == "__main__":
    main()
