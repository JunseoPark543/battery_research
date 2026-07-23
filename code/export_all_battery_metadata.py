#!/usr/bin/env python3
"""Combine metadata from every battery pickle under data/ into one CSV."""

from __future__ import annotations

import argparse
import csv
import pickle
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data_summary" / "all_battery_metadata.csv"

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
    """Sort paths containing numbers in natural order."""
    relative = str(path).replace("\\", "/")
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", relative)
    ]


def csv_value(value: Any) -> Any:
    """Convert missing, NumPy scalar, and container values into CSV-safe values."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple, dict, set)):
        return str(value)
    return value.item() if hasattr(value, "item") else value


def load_metadata(pkl_path: Path) -> dict[str, Any]:
    """Load only the requested battery-level metadata fields."""
    # Pickle can execute code while loading; only use trusted dataset files.
    with pkl_path.open("rb") as pkl_file:
        battery = pickle.load(pkl_file)
    if not isinstance(battery, dict):
        raise TypeError(f"top-level object is {type(battery).__name__}, not dict")
    return {field: csv_value(battery.get(field)) for field in FIELDS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    pkl_files = sorted(args.data_root.rglob("*.pkl"), key=natural_key)
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files found under {args.data_root}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped: list[tuple[Path, str]] = []
    with args.output.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
        writer.writeheader()
        for pkl_path in pkl_files:
            try:
                writer.writerow(load_metadata(pkl_path))
                written += 1
            except (OSError, TypeError, ValueError, pickle.UnpicklingError, EOFError) as error:
                skipped.append((pkl_path, str(error)))

    print(f"[saved] {written} rows, {len(FIELDS)} columns -> {args.output}")
    print(f"[scanned] {len(pkl_files)} pickle files")
    if skipped:
        print(f"[skipped] {len(skipped)} files:")
        for path, reason in skipped:
            print(f"  - {path}: {reason}")


if __name__ == "__main__":
    main()
