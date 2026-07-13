from __future__ import annotations

import pandas as pd


def add_score(panel_df: pd.DataFrame, feature_columns: list[str], weights: dict | None = None,
              clip: float = 0.0, require_all: bool = False) -> pd.DataFrame:
    """Add a cross-sectional `Score`: each factor is z-scored *within each date*
    (so different-scale factors combine fairly and the ranking is purely
    cross-sectional), then averaged (or weighted). Go long the highest Score.

    clip: optional winsorization to +/- `clip` sigma. Default 0 (off) —
    data-error outliers are guarded at the data layer (e.g. B/M in
    data.fundamentals_edgar). Winsorizing here by default would create a
    plateau of tied values at the cap for sparse/heavy-tailed factors (mostly-
    zero PEAD), making top-decile selection non-deterministic. Opt in when
    combining several unguarded factors.

    require_all=True: Score is NaN unless *every* factor is present (a clean
    multi-factor comparison); default False averages whatever is available
    (broader screener coverage).
    """
    df = panel_df.copy()
    zcols = []
    for c in feature_columns:
        z = df.groupby("Date")[c].transform(
            lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) > 0 else s * 0.0
        )
        if clip:
            z = z.clip(-clip, clip)
        zc = f"{c}_z"
        df[zc] = z
        zcols.append(zc)

    if weights:
        # omitted factors default to 0 weight (reindex) rather than KeyError
        w = pd.Series({f"{k}_z": v for k, v in weights.items()}).reindex(zcols).fillna(0.0)
        df["Score"] = df[zcols].mul(w, axis=1).sum(axis=1, min_count=1)
    else:
        df["Score"] = df[zcols].mean(axis=1)

    if require_all:
        df.loc[df[zcols].isna().any(axis=1), "Score"] = float("nan")
    return df
