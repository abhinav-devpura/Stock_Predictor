from __future__ import annotations

from pathlib import Path

import pandas as pd


def ensure_cache_dir(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)


def read_parquet_cache(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_parquet(path)
    return None


def write_parquet_cache(df: pd.DataFrame, path: Path) -> None:
    ensure_cache_dir(path.parent)
    df.to_parquet(path, index=False)
