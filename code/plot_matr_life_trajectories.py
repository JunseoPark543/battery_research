#!/usr/bin/env python3
"""Draw all MATR battery life trajectories together in one plot."""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "MATR"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "plot" / "MATR" / "life_trajectories"


def natural_key(path: Path) -> list[int | str]:
    """Sort names containing numbers in their natural order."""
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", path.stem)]


def load_trajectory(pkl_path: Path) -> tuple[str, np.ndarray, np.ndarray]:
    """Extract cycle number and maximum discharge capacity from one cell."""
    # Only unpickle files from a trusted source.
    with pkl_path.open("rb") as file:
        battery: dict[str, Any] = pickle.load(file)

    cell_id = str(battery.get("cell_id", pkl_path.stem))
    cycles: list[float] = []
    capacities: list[float] = []

    for index, cycle in enumerate(battery.get("cycle_data", []), start=1):
        values = np.asarray(cycle.get("discharge_capacity_in_Ah", []), dtype=float)
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            continue
        cycles.append(float(cycle.get("cycle_number", index)))
        capacities.append(float(np.max(finite_values)))

    if not capacities:
        raise ValueError("no valid discharge-capacity values")
    if capacities[0] <= 0:
        raise ValueError("initial discharge capacity must be positive")

    return cell_id, np.asarray(cycles), np.asarray(capacities)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    pkl_files = sorted(args.data_dir.glob("*.pkl"), key=natural_key)
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files found in {args.data_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    trajectories: list[tuple[str, np.ndarray, np.ndarray]] = []
    skipped: list[tuple[str, str]] = []
    for pkl_path in pkl_files:
        try:
            trajectory = load_trajectory(pkl_path)
            trajectories.append(trajectory)
            print(f"[loaded] {trajectory[0]}")
        except (KeyError, TypeError, ValueError, pickle.UnpicklingError, EOFError) as error:
            skipped.append((pkl_path.name, str(error)))
            print(f"[skipped] {pkl_path.name}: {error}")

    if not trajectories:
        raise RuntimeError("No valid MATR trajectories were found.")

    fig, axis = plt.subplots(figsize=(12, 7))
    color_map = plt.get_cmap("turbo")
    for index, (cell_id, cycles, capacities) in enumerate(trajectories):
        axis.plot(
            cycles,
            capacities / capacities[0] * 100.0,
            color=color_map(index / max(len(trajectories) - 1, 1)),
            linewidth=0.75,
            alpha=0.7,
            label=cell_id,
        )

    axis.axhline(80.0, color="black", linestyle="--", linewidth=1.2, label="80% SOH")
    axis.set(
        title=f"MATR Battery Life Trajectories ({len(trajectories)} cells)",
        xlabel="Cycle number",
        ylabel="SOH relative to cycle 1 (%)",
    )
    axis.grid(alpha=0.25)
    # MATR has many cells, so omit per-cell legend to keep the data area readable.
    axis.legend(handles=[axis.lines[-1]], loc="lower left")
    fig.tight_layout()

    output_path = args.output_dir / "MATR_all_life_trajectories.png"
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"\nCreated 1 plot with {len(trajectories)} cells: {output_path}")
    if skipped:
        print(f"Skipped {len(skipped)} file(s).")


if __name__ == "__main__":
    main()
