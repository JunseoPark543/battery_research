#!/usr/bin/env python3
"""Validate metadata clusters against observed capacity-degradation trajectories.

Each cell's per-cycle discharge capacity is normalized by its early-life
capacity, smoothed, and interpolated onto a common relative-life grid. The
script measures within-cluster compactness, separation between cluster mean
trajectories, silhouette score, explained trajectory variance, and a
label-permutation significance test.
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSIGNMENTS = (
    PROJECT_ROOT
    / "data_summary"
    / "optimal_clustering"
    / "cluster_assignments.csv"
)
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "data_summary" / "optimal_clustering" / "trajectory_validation"
)


def cycle_capacity(cycle: Mapping[str, Any]) -> float:
    """Use maximum discharge capacity in one cycle, matching existing plots."""
    values = cycle.get("discharge_capacity_in_Ah")
    if values is None:
        return np.nan
    try:
        array = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return np.nan
    finite = array[np.isfinite(array) & (array > 0)]
    return float(np.max(finite)) if finite.size else np.nan


def rolling_median(values: np.ndarray, window: int = 5) -> np.ndarray:
    """Apply a centered rolling median without adding a scipy dependency."""
    return (
        pd.Series(values)
        .rolling(window=window, center=True, min_periods=1)
        .median()
        .to_numpy(dtype=float)
    )


def load_trajectory(
    path: Path, grid_points: int, baseline_cycles: int, min_valid_cycles: int
) -> tuple[np.ndarray, dict[str, float]]:
    """Load, normalize, clean, and interpolate one capacity trajectory."""
    with path.open("rb") as stream:
        battery = pickle.load(stream)
    if not isinstance(battery, Mapping):
        raise TypeError("top-level pickle object is not a mapping")
    raw_cycles = battery.get("cycle_data")
    if not isinstance(raw_cycles, Sequence) or isinstance(raw_cycles, (str, bytes)):
        raise TypeError("cycle_data is not a sequence")

    cycle_numbers: list[float] = []
    capacities: list[float] = []
    for index, cycle in enumerate(raw_cycles, start=1):
        if not isinstance(cycle, Mapping):
            continue
        capacity = cycle_capacity(cycle)
        if not np.isfinite(capacity):
            continue
        try:
            number = float(cycle.get("cycle_number", index))
        except (TypeError, ValueError):
            number = float(index)
        if np.isfinite(number):
            cycle_numbers.append(number)
            capacities.append(capacity)
    if len(capacities) < min_valid_cycles:
        raise ValueError(
            f"only {len(capacities)} valid capacity cycles "
            f"(minimum={min_valid_cycles})"
        )

    cycles = np.asarray(cycle_numbers, dtype=float)
    capacity = np.asarray(capacities, dtype=float)
    order = np.argsort(cycles)
    cycles, capacity = cycles[order], capacity[order]
    cycles, unique_indices = np.unique(cycles, return_index=True)
    capacity = capacity[unique_indices]
    if len(capacity) < min_valid_cycles:
        raise ValueError("too few unique valid cycle numbers")

    baseline = float(np.median(capacity[: min(baseline_cycles, len(capacity))]))
    if not np.isfinite(baseline) or baseline <= 0:
        raise ValueError("invalid early-life capacity baseline")
    soh = rolling_median(capacity / baseline, window=5)

    # Reject clearly malformed unit/trajectory records rather than clipping them.
    if np.nanmedian(soh) < 0.05 or np.nanmedian(soh) > 3.0:
        raise ValueError("implausible normalized-capacity scale")
    progress = (cycles - cycles[0]) / (cycles[-1] - cycles[0])
    grid = np.linspace(0.0, 1.0, grid_points)
    interpolated = np.interp(grid, progress, soh)

    early_end = max(2, int(round(0.2 * (grid_points - 1))) + 1)
    late_start = min(grid_points - 2, int(round(0.8 * (grid_points - 1))))
    full_slope = float(np.polyfit(grid, interpolated, 1)[0])
    early_slope = float(
        np.polyfit(grid[:early_end], interpolated[:early_end], 1)[0]
    )
    late_slope = float(
        np.polyfit(grid[late_start:], interpolated[late_start:], 1)[0]
    )
    line = interpolated[0] + (interpolated[-1] - interpolated[0]) * grid
    knee_index = int(np.argmax(line - interpolated))
    features = {
        "valid_cycle_count": float(len(capacity)),
        "first_cycle_number": float(cycles[0]),
        "last_cycle_number": float(cycles[-1]),
        "observed_cycle_span": float(cycles[-1] - cycles[0]),
        "baseline_capacity_Ah": baseline,
        "final_soh": float(interpolated[-1]),
        "minimum_soh": float(np.min(interpolated)),
        "trajectory_auc": float(np.trapezoid(interpolated, grid)),
        "full_life_slope": full_slope,
        "early_life_slope": early_slope,
        "late_life_slope": late_slope,
        "knee_relative_life": float(grid[knee_index]),
        "knee_deviation": float((line - interpolated)[knee_index]),
    }
    return interpolated, features


def trajectory_statistics(
    trajectories: np.ndarray, labels: np.ndarray
) -> tuple[pd.DataFrame, np.ndarray, float, float]:
    """Calculate cluster compactness and mean trajectories."""
    clusters = np.unique(labels)
    means = np.vstack([trajectories[labels == cluster].mean(axis=0) for cluster in clusters])
    global_mean = trajectories.mean(axis=0)
    total_sse = float(np.sum((trajectories - global_mean) ** 2))
    within_sse = 0.0
    rows = []
    for cluster, mean in zip(clusters, means):
        group = trajectories[labels == cluster]
        residual = group - mean
        cell_rmse = np.sqrt(np.mean(residual**2, axis=1))
        correlations = []
        for curve in group:
            if np.std(curve) > 0 and np.std(mean) > 0:
                correlations.append(float(np.corrcoef(curve, mean)[0, 1]))
        within_sse += float(np.sum(residual**2))
        other_distances = [
            float(np.sqrt(np.mean((mean - other_mean) ** 2)))
            for other_cluster, other_mean in zip(clusters, means)
            if other_cluster != cluster
        ]
        nearest_between = min(other_distances) if other_distances else np.nan
        mean_within = float(np.mean(cell_rmse))
        rows.append(
            {
                "cluster": int(cluster),
                "cell_count": len(group),
                "within_rmse_mean": mean_within,
                "within_rmse_median": float(np.median(cell_rmse)),
                "within_rmse_p90": float(np.quantile(cell_rmse, 0.9)),
                "curve_to_centroid_correlation_mean": (
                    float(np.mean(correlations)) if correlations else np.nan
                ),
                "nearest_centroid_rmse": nearest_between,
                "separation_to_compactness_ratio": (
                    nearest_between / mean_within if mean_within > 0 else np.nan
                ),
            }
        )
    explained_fraction = 1.0 - within_sse / total_sse if total_sse > 0 else np.nan
    return pd.DataFrame(rows), means, within_sse, explained_fraction


def permutation_test(
    trajectories: np.ndarray,
    labels: np.ndarray,
    observed_within_sse: float,
    permutations: int,
    seed: int,
) -> tuple[float, np.ndarray]:
    """Test whether observed labels are more compact than size-matched random labels."""
    rng = np.random.default_rng(seed)
    null_sse = np.empty(permutations, dtype=float)
    clusters = np.unique(labels)
    for permutation in range(permutations):
        shuffled = rng.permutation(labels)
        sse = 0.0
        for cluster in clusters:
            group = trajectories[shuffled == cluster]
            sse += float(np.sum((group - group.mean(axis=0)) ** 2))
        null_sse[permutation] = sse
    p_value = float(
        (1 + np.sum(null_sse <= observed_within_sse)) / (permutations + 1)
    )
    return p_value, null_sse


def save_cluster_plot(
    trajectories: np.ndarray,
    labels: np.ndarray,
    means: np.ndarray,
    output_path: Path,
) -> None:
    """Plot cluster median and interquartile trajectory bands."""
    clusters = np.unique(labels)
    columns = 3
    rows = int(np.ceil(len(clusters) / columns))
    grid = np.linspace(0, 100, trajectories.shape[1])
    fig, axes = plt.subplots(rows, columns, figsize=(15, 4 * rows), sharex=True)
    axes_array = np.asarray(axes).reshape(-1)
    for axis, cluster, mean in zip(axes_array, clusters, means):
        group = trajectories[labels == cluster]
        q25, median, q75 = np.quantile(group, [0.25, 0.5, 0.75], axis=0)
        axis.fill_between(grid, q25 * 100, q75 * 100, alpha=0.25)
        axis.plot(grid, median * 100, linewidth=2, label="median")
        axis.plot(grid, mean * 100, linewidth=1, linestyle="--", label="mean")
        axis.axhline(80, color="black", linewidth=0.8, linestyle=":")
        axis.set_title(f"Cluster {cluster} (n={len(group)})")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    for axis in axes_array[len(clusters) :]:
        axis.set_visible(False)
    fig.supxlabel("Relative observed life (%)")
    fig.supylabel("SOH relative to early-life baseline (%)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_centroid_plot(
    means: np.ndarray, clusters: np.ndarray, output_path: Path
) -> None:
    grid = np.linspace(0, 100, means.shape[1])
    fig, axis = plt.subplots(figsize=(11, 7))
    for cluster, mean in zip(clusters, means):
        axis.plot(grid, mean * 100, linewidth=1.8, label=f"Cluster {cluster}")
    axis.axhline(80, color="black", linewidth=1, linestyle=":")
    axis.set(
        title="Mean degradation trajectory by metadata cluster",
        xlabel="Relative observed life (%)",
        ylabel="Mean normalized capacity (%)",
    )
    axis.grid(alpha=0.2)
    axis.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--grid-points", type=int, default=101)
    parser.add_argument("--baseline-cycles", type=int, default=5)
    parser.add_argument("--min-valid-cycles", type=int, default=20)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assignments = pd.read_csv(args.assignments, encoding="utf-8-sig")
    required = {"dataset", "source_file", "cell_id", "cluster"}
    missing = required.difference(assignments.columns)
    if missing:
        raise ValueError(f"Assignment columns missing: {sorted(missing)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    curves: list[np.ndarray] = []
    feature_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, row in assignments.iterrows():
        pkl_path = args.data_root / Path(str(row["source_file"]))
        try:
            curve, features = load_trajectory(
                pkl_path,
                args.grid_points,
                args.baseline_cycles,
                args.min_valid_cycles,
            )
            curves.append(curve)
            feature_rows.append(
                {
                    "dataset": row["dataset"],
                    "source_file": row["source_file"],
                    "cell_id": row["cell_id"],
                    "cluster": int(row["cluster"]),
                    **features,
                }
            )
        except Exception as error:
            failures.append(
                {
                    "dataset": row["dataset"],
                    "source_file": row["source_file"],
                    "cell_id": row["cell_id"],
                    "cluster": row["cluster"],
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        if (index + 1) % 50 == 0 or index + 1 == len(assignments):
            print(
                f"[progress] {index + 1}/{len(assignments)}; "
                f"valid={len(curves)}, failed={len(failures)}",
                flush=True,
            )
    # Always preserve extraction diagnostics, including runs that cannot proceed.
    pd.DataFrame(failures).to_csv(
        args.output_dir / "extraction_failures.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if len(curves) < 3:
        raise RuntimeError("Fewer than 3 valid trajectories were extracted")

    trajectory_matrix = np.vstack(curves)
    features = pd.DataFrame(feature_rows)
    labels = features["cluster"].to_numpy(dtype=int)
    present_clusters = np.unique(labels)
    if len(present_clusters) < 2:
        raise RuntimeError("Fewer than 2 clusters have valid trajectories")

    cluster_stats, means, within_sse, explained_fraction = trajectory_statistics(
        trajectory_matrix, labels
    )
    silhouette = float(silhouette_score(trajectory_matrix, labels))
    p_value, null_sse = permutation_test(
        trajectory_matrix,
        labels,
        within_sse,
        args.permutations,
        args.seed,
    )
    null_mean = float(np.mean(null_sse))
    compactness_improvement = 1.0 - within_sse / null_mean

    trajectory_columns = [
        f"relative_life_{index:03d}" for index in range(args.grid_points)
    ]
    trajectory_frame = pd.concat(
        [
            features[["dataset", "source_file", "cell_id", "cluster"]],
            pd.DataFrame(trajectory_matrix, columns=trajectory_columns),
        ],
        axis=1,
    )
    feature_summary = (
        features.groupby("cluster")
        .agg(
            cell_count=("cell_id", "size"),
            observed_cycle_span_median=("observed_cycle_span", "median"),
            final_soh_median=("final_soh", "median"),
            final_soh_iqr=(
                "final_soh",
                lambda values: values.quantile(0.75) - values.quantile(0.25),
            ),
            trajectory_auc_median=("trajectory_auc", "median"),
            full_life_slope_median=("full_life_slope", "median"),
            early_life_slope_median=("early_life_slope", "median"),
            late_life_slope_median=("late_life_slope", "median"),
            knee_relative_life_median=("knee_relative_life", "median"),
        )
        .reset_index()
    )

    features.to_csv(
        args.output_dir / "trajectory_features.csv",
        index=False,
        encoding="utf-8-sig",
    )
    trajectory_frame.to_csv(
        args.output_dir / "normalized_trajectories.csv",
        index=False,
        encoding="utf-8-sig",
    )
    cluster_stats.to_csv(
        args.output_dir / "cluster_consistency.csv",
        index=False,
        encoding="utf-8-sig",
    )
    feature_summary.to_csv(
        args.output_dir / "cluster_degradation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    save_cluster_plot(
        trajectory_matrix,
        labels,
        means,
        args.output_dir / "cluster_trajectory_bands.png",
    )
    save_centroid_plot(
        means,
        present_clusters,
        args.output_dir / "cluster_mean_trajectories.png",
    )

    summary = {
        "assigned_cell_count": len(assignments),
        "valid_trajectory_count": len(features),
        "failed_trajectory_count": len(failures),
        "cluster_count_with_valid_data": len(present_clusters),
        "grid_points": args.grid_points,
        "baseline_cycles": args.baseline_cycles,
        "minimum_valid_cycles": args.min_valid_cycles,
        "trajectory_silhouette_score": silhouette,
        "trajectory_variance_explained_by_clusters": explained_fraction,
        "observed_within_cluster_sse": within_sse,
        "permuted_within_cluster_sse_mean": null_mean,
        "compactness_improvement_over_random": compactness_improvement,
        "permutation_count": args.permutations,
        "permutation_p_value": p_value,
        "interpretation_rule": {
            "silhouette": (
                ">0.50 strong, 0.25-0.50 moderate, 0-0.25 weak, <0 conflicting"
            ),
            "separation_to_compactness_ratio": (
                ">1 means nearest cluster centroid is farther than average "
                "within-cluster trajectory deviation"
            ),
            "permutation_p_value": (
                "<0.05 means metadata clusters are more trajectory-compact "
                "than random groups of the same sizes"
            ),
        },
    }
    (args.output_dir / "validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"[done] output={args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
