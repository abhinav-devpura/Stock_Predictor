from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


def panel_walk_forward_splits(panel_df: pd.DataFrame, n_splits: int = 5, embargo: int = 0):
    """Generalizes TimeSeriesSplit to panel data: splits on the sorted array
    of unique dates, not row index, so every ticker's row for a given date
    lands entirely in train or entirely in test — never split across the
    cross-section within a date.

    embargo: number of unique dates to drop between each train block and its
    test block (passed to TimeSeriesSplit's `gap`). Set to the forward-return
    horizon so a training row's forward-looking label (which peeks `horizon`
    days ahead) can't reach into the test window — the leakage guard for a
    multi-day Target.

    Yields (train_idx, test_idx) row-index arrays usable with panel_df.loc[].
    Requires panel_df to have a clean 0..n-1 index (see model.panel.assemble_panel).
    """
    dates = np.sort(panel_df["Date"].unique())
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=embargo)
    for train_date_pos, test_date_pos in tscv.split(dates):
        train_dates = set(dates[train_date_pos])
        test_dates = set(dates[test_date_pos])
        train_idx = panel_df.index[panel_df["Date"].isin(train_dates)].to_numpy()
        test_idx = panel_df.index[panel_df["Date"].isin(test_dates)].to_numpy()
        yield train_idx, test_idx
