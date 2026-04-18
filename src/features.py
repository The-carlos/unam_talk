"""Feature helper functions shared by training experiments if needed.

Production inference should prefer a serialized sklearn Pipeline containing
all preprocessing steps, so these helpers are intentionally small.
"""

from __future__ import annotations

import pandas as pd


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with stripped column names."""

    result = df.copy()
    result.columns = [str(column).strip() for column in result.columns]
    return result

