#!/usr/bin/env python3
"""Export battery-level metadata from selected pickle datasets to CSV files."""

from __future__ import annotations

import argparse
import csv
import pickle
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data_summary"
DATASETS = ("CALCE", "HUST", "MATR", "SNL")
FIELDS = (
    "cell_id",
    "form_factor",
    "anode_material",
    "cathode_material",
    "electrolyte_material",
    "nominal_capacity_in_Ah",
    "depth_of_charge",
    "depth_of_discharge",
    "already_spent_cycles",
    "max_voltage_limit_in_V",
    "min_voltage_limit_in_V",
    "max_current_limit_in_A",
    "min_current_limit_in_A",
)


def natural_key(path: Path) -> list[int | str]:
    """Return a key that sorts embedded numbers naturally."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.stem)]


def csv_value(value: Any) -> Any:
    """Convert missing and container values into portable CSV cells."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple, dict, set)):
        return str(value)
    # Convert NumPy scalar-like values without importing NumPy.
    return value.item() if hasattr(value, "item") else value


def export_dataset(dataset: str, data_root: Path, output_dir: Path) -> tuple[Path, int]:
    """Export one dataset and return its output path and number of rows."""
    dataset_dir = data_root / dataset
    pkl_files = sorted(dataset_dir.glob("*.pkl"), key=natural_key)
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files found in {dataset_dir}")

    output_path = output_dir / f"{dataset}_metadata.csv"
    with output_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
        writer.writeheader()

        for pkl_path in pkl_files:
            # Pickle files must come from a trusted source.
            with pkl_path.open("rb") as pkl_file:
                battery: dict[str, Any] = pickle.load(pkl_file)
            row = {field: csv_value(battery.get(field)) for field in FIELDS}
            writer.writerow(row)

    return output_path, len(pkl_files)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    for dataset in DATASETS:
        output_path, row_count = export_dataset(dataset, args.data_root, args.output_dir)
        total_rows += row_count
        print(f"[saved] {dataset}: {row_count} rows -> {output_path}")
    print(f"Total: {total_rows} battery records")


if __name__ == "__main__":
    main()
