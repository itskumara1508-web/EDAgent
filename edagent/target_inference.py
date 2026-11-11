"""Heuristic and statistical target column inference for EDAgent."""

from __future__ import annotations

import re
from typing import Optional, Tuple
import pandas as pd


COMMON_TARGET_NAMES = [
    "target", "label", "churn", "status", "price", "salary",
    "is_churn", "purchased", "outcome", "class", "default",
    "converted", "revenue", "sales", "score", "approved", "survived"
]


def infer_target_column(
    df: pd.DataFrame,
    specified_target: Optional[str] = None
) -> Tuple[Optional[str], str, bool]:
    """Infers the target column if not explicitly given by the user.
    
    Returns:
        (target_column_name, explanation, is_inferred)
    """
    if specified_target:
        # Match case-insensitively
        for col in df.columns:
            if col.strip().lower() == specified_target.strip().lower():
                return col, f"Using user-specified target column: '{col}'.", False
        return (
            None,
            f"Specified target '{specified_target}' was not found in the dataset columns: {list(df.columns)}.",
            False
        )

    cols = list(df.columns)
    if not cols:
        return None, "Dataset contains no columns.", False

    # 1. Exact or partial match with common target names
    for pattern in COMMON_TARGET_NAMES:
        for col in cols:
            col_clean = re.sub(r"[_\s\-]+", "", col.lower())
            pat_clean = re.sub(r"[_\s\-]+", "", pattern.lower())
            if col_clean == pat_clean:
                return (
                    col,
                    f"Inferred '{col}' as target because its name directly matches the standard outcome pattern '{pattern}'.",
                    True
                )

    # 2. Look for binary categorical/integer columns (excluding obvious ID/index columns)
    for col in cols:
        if re.search(r"(id|code|index|uuid|key)$", col.lower()):
            continue
        nunique = df[col].dropna().nunique()
        if nunique == 2:
            return (
                col,
                f"Inferred '{col}' as target because it is a binary variable (2 unique classes: {list(df[col].dropna().unique())[:2]}), suitable for classification.",
                True
            )

    # 3. Look for the last numerical or categorical non-ID column
    for col in reversed(cols):
        if re.search(r"(id|code|index|uuid|key)$", col.lower()):
            continue
        nunique = df[col].dropna().nunique()
        if 2 <= nunique <= 20:
            return (
                col,
                f"Inferred '{col}' as target based on categorical outcome structure ({nunique} discrete classes).",
                True
            )

    # 4. Fallback: Last non-ID column
    for col in reversed(cols):
        if not re.search(r"(id|code|index|uuid|key)$", col.lower()):
            return (
                col,
                f"Inferred '{col}' as default target as it is the final descriptive feature in the spreadsheet.",
                True
            )

    # Default to first column if everything else fails
    fallback_col = cols[-1]
    return (
        fallback_col,
        f"Selected '{fallback_col}' as fallback target for baseline modeling.",
        True
    )
