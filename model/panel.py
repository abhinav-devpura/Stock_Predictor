from __future__ import annotations

import pandas as pd

from data.earnings import attach_earnings
from data.prices import add_forward_return
from data.universe import attach_sector
from factors import calendar_effects, momentum, pead, reversal
from model.target import add_target

# External-data attachers (each panel -> panel, deriving tickers from the
# panel). Adding a data source in a later phase is one entry here.
INPUT_STEPS = [attach_sector, attach_earnings]

# Factor feature builders, plain list per the modularization principle —
# adding a validated factor family is one import + one line.
FACTOR_STEPS = [
    calendar_effects.add_features,
    momentum.add_features,
    pead.add_features,
    reversal.add_features,
]


def prepare_features(price_panel: pd.DataFrame) -> pd.DataFrame:
    """Attach all external inputs (INPUT_STEPS) then compute all factor columns
    (FACTOR_STEPS). Shared by assemble_panel (model) and the run scripts'
    validation panels, so external-data attachment lives in exactly one place."""
    panel_df = price_panel.copy()
    for attach in INPUT_STEPS:
        panel_df = attach(panel_df)
    for step in FACTOR_STEPS:
        panel_df = step(panel_df)
    return panel_df


def assemble_panel(
    price_panel: pd.DataFrame, feature_columns: list[str] | None = None, horizon: int = 1
) -> pd.DataFrame:
    """Builds the model panel: prepare_features (inputs + factors), the
    `horizon`-day forward return, and the cross-sectional target, then drops
    rows missing Target or any of feature_columns.

    feature_columns: the validated subset of factor columns the caller
    intends to train on (decided from event-study verdicts) — rows missing
    these are dropped so the model never sees NaN features. If None, only
    Target is required (all computed factor columns are kept, unfiltered).
    horizon: forward-return horizon in trading days (config.FORWARD_HORIZON).
    """
    panel_df = prepare_features(price_panel)
    panel_df = add_forward_return(panel_df, horizon=horizon)
    panel_df = add_target(panel_df)

    required = ["Target"] + list(feature_columns or [])
    panel_df = panel_df.dropna(subset=required).reset_index(drop=True)
    return panel_df
