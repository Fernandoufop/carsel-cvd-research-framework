"""Deterministic ingestion, encoding checks, and auditable row rejection."""
from __future__ import annotations

from dataclasses import dataclass
import io

import numpy as np
import pandas as pd

from .schema import ALIASES, FEATURES, NUMERIC


class DatasetError(ValueError):
    pass


@dataclass
class DatasetSummary:
    source_rows: int
    accepted_rows: int
    rejected_rows: int
    age_unit_source: str
    aliases_applied: dict[str, str]
    reason_counts: dict[str, int]


def load_dataset(
    source: str | io.BytesIO,
    *,
    target: str = "cardio",
    age_unit: str = "years",
    drop_invalid: bool = False,
    max_rows: int = 200_000,
) -> tuple[pd.DataFrame, DatasetSummary]:
    """Load a training CSV. No outcome label can be a predictor.

    'age_unit' must be explicitly set; days are converted to years *before*
    splitting since this conversion has no learned parameters.
    """
    if age_unit not in {"years", "days"}:
        raise DatasetError("age_unit must be 'years' or 'days'")
    try:
        df = pd.read_csv(source)
    except Exception as exc:
        raise DatasetError(f"Could not parse CSV: {exc}") from exc
    if not len(df):
        raise DatasetError("The dataset is empty")
    if len(df) > max_rows:
        raise DatasetError(f"Dataset exceeds the limit of {max_rows:,} rows")
    df.columns = [str(x).strip() for x in df.columns]
    original_columns = list(df.columns)
    if len(set(original_columns)) != len(original_columns):
        raise DatasetError("Duplicate CSV column names")
    aliases = {k: v for k, v in ALIASES.items() if k in df.columns}
    for old, new in aliases.items():
        if new in df.columns:
            raise DatasetError(f"Conflicting aliases: both '{old}' and '{new}' are present")
    df = df.rename(columns=aliases)
    needed = FEATURES + [target]
    missing = [column for column in needed if column not in df.columns]
    if missing:
        raise DatasetError(f"Missing required columns: {', '.join(missing)}")
    if target in FEATURES:
        raise DatasetError("The target must not be one of the predictor fields")
    df = df[needed].copy()
    for column in needed:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if age_unit == "days":
        df["age"] = df["age"] / 365.25

    checks: dict[str, pd.Series] = {}
    bounds = {"age": (18, 110), "height": (100, 230), "weight": (25, 350),
              "ap_hi": (60, 300), "ap_lo": (30, 200)}
    for column, (lower, upper) in bounds.items():
        checks[column] = df[column].between(lower, upper) & np.isfinite(df[column])
    checks["ap_hi>ap_lo"] = df["ap_hi"] > df["ap_lo"]
    codes = {"sex": [1, 2], "cholesterol": [1, 2, 3], "glucose": [1, 2, 3],
             "smoking": [0, 1], "alcohol": [0, 1], "physical_activity": [0, 1],
             target: [0, 1]}
    for column, allowed in codes.items():
        checks[column] = df[column].isin(allowed)
    mask = pd.Series(True, index=df.index)
    for result in checks.values():
        mask &= result
    reason_counts = {name: int((~check).sum()) for name, check in checks.items() if (~check).any()}
    rejected = int((~mask).sum())
    if rejected and not drop_invalid:
        idx = df.index[~mask][:5].tolist()
        raise DatasetError(
            f"{rejected} invalid rows (zero-based indices: {idx}); issue counts: {reason_counts}. "
            "Correct the CSV or explicitly enable drop_invalid."
        )
    df = df.loc[mask].reset_index(drop=True)
    if len(df) < 60:
        raise DatasetError("At least 60 valid labeled observations are required (research minimum)")
    counts = df[target].value_counts()
    if len(counts) != 2 or counts.min() < 12:
        raise DatasetError("Both outcome classes must have at least 12 observations")
    for column in ["sex", "cholesterol", "glucose", "smoking", "alcohol", "physical_activity", target]:
        df[column] = df[column].astype(int)
    return df, DatasetSummary(
        source_rows=int(len(mask)), accepted_rows=int(len(df)), rejected_rows=rejected,
        age_unit_source=age_unit, aliases_applied=aliases, reason_counts=reason_counts,
    )
