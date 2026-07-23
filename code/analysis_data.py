#!/usr/bin/env python3
"""Analyze unique values of every column and write one combined CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data_summary" / "all_battery_metadata.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "analysis_results" / "column_value_analysis.csv"

NUMERIC_COLUMNS = [
    "nominal_capacity_in_Ah",
    "depth_of_charge",
    "depth_of_discharge",
    "already_spent_cycles",
    "max_voltage_limit_in_V",
    "min_voltage_limit_in_V",
    "max_current_limit_in_A",
    "min_current_limit_in_A",
]
def load_data(csv_path: Path) -> pd.DataFrame:
    """Load metadata and coerce the expected numeric columns."""
    data = pd.read_csv(csv_path, encoding="utf-8-sig")
    for column in NUMERIC_COLUMNS:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def unique_value_analysis(data: pd.DataFrame) -> pd.DataFrame:
    """Combine per-column summary and value frequencies into one table."""
    value_tables: list[pd.DataFrame] = []
    for column in data.columns:
        counts = (
            data[column]
            .fillna("<missing>")
            .value_counts(dropna=False)
            .rename_axis("value")
            .reset_index(name="count")
        )
        counts.insert(0, "column", column)
        counts.insert(1, "unique_without_missing", data[column].nunique(dropna=True))
        counts.insert(2, "unique_with_missing", data[column].nunique(dropna=False))
        counts.insert(3, "missing_count", data[column].isna().sum())
        counts["ratio_percent"] = np.round(counts["count"] / len(data) * 100.0, 2)
        value_tables.append(counts)
    return pd.concat(value_tables, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"CSV file not found: {args.input}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data = load_data(args.input)
    analysis = unique_value_analysis(data)
    analysis.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"Input: {args.input}")
    print(f"Shape: {data.shape[0]} rows x {data.shape[1]} columns")
    print(f"Result: {args.output}")
    print("\nUnique counts by column:")
    print(
        analysis[
            ["column", "unique_without_missing", "unique_with_missing", "missing_count"]
        ]
        .drop_duplicates("column")
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
