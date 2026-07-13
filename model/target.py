from __future__ import annotations

import numpy as np
import pandas as pd


def add_target(panel_df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional target: does this stock beat today's median stock's
    next-day return? Nets out market-wide moves so the model learns
    stock-specific factor exposure rather than "did the market go up."

    Requires fwd_return already present (data.prices.add_forward_return).
    Target is left NaN wherever fwd_return is NaN (last row per ticker),
    not silently coerced to 0 by the comparison, so callers can dropna
    explicitly rather than the missing data silently voting "underperform."
    """
    panel_df = panel_df.copy()
    cs_median = panel_df.groupby("Date")["fwd_return"].transform("median")
    beats_median = (panel_df["fwd_return"] > cs_median).astype(float)
    panel_df["Target"] = np.where(panel_df["fwd_return"].notna(), beats_median, np.nan)
    return panel_df
