#!/usr/bin/env python3
"""Summarize battery pickle metadata and cluster cells by metadata similarity.

Run this file from the battery_research project root. Pickle files must be from
a trusted source because unpickling can execute arbitrary code.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data_summary"
DEFAULT_DATASETS = (
    "CALCE",
    "HNEI",
    "HUST",
    "ISU_ILCC",
    "MICH",
    "MICH_EXP",
    "RWTH",
    "SDU",
    "SNL",
    "Stanford_2",
    "Tongji",
    "UL_PUR",
    "XJTU",
)

# Identity/source columns are deliberately excluded from clustering.
NUMERIC_CLUSTER_FEATURES = (
    "nominal_capacity_in_Ah",
    "depth_of_charge",
    "depth_of_discharge",
    "already_spent_cycles",
    "max_voltage_limit_in_V",
    "min_voltage_limit_in_V",
    "max_current_limit_in_A",
    "min_current_limit_in_A",
    "soc_interval_start",
    "soc_interval_end",
    "charge_step_count",
    "charge_rate_min_C",
    "charge_rate_max_C",
    "charge_rate_mean_C",
    "charge_current_max_A",
    "charge_voltage_max_V",
    "discharge_step_count",
    "discharge_rate_min_C",
    "discharge_rate_max_C",
    "discharge_rate_mean_C",
    "discharge_current_max_A",
    "discharge_voltage_min_V",
)
CATEGORICAL_CLUSTER_FEATURES = (
    "form_factor",
    "anode_material",
    "cathode_material",
    "electrolyte_material",
)


def natural_key(path: Path) -> list[int | str]:
    """Sort paths with embedded numbers in human order."""
    return [
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", str(path).replace("\\", "/"))
    ]


def to_builtin(value: Any) -> Any:
    """Convert common NumPy/pandas values into JSON-compatible Python values."""
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    if isinstance(value, Mapping):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def csv_metadata_value(value: Any) -> Any:
    """Keep scalar metadata scalar and serialize containers consistently."""
    value = to_builtin(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def finite_numbers(values: Sequence[Any]) -> list[float]:
    """Return finite values that can safely be converted to float."""
    result: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            result.append(number)
    return result


def protocol_features(protocol: Any, prefix: str) -> dict[str, Any]:
    """Create fixed-size clustering features from a variable-length protocol."""
    steps = (
        list(protocol)
        if isinstance(protocol, Sequence) and not isinstance(protocol, (str, bytes))
        else []
    )
    mappings = [step for step in steps if isinstance(step, Mapping)]
    rates = finite_numbers([step.get("rate_in_C") for step in mappings])
    currents = finite_numbers([step.get("current_in_A") for step in mappings])
    voltages = finite_numbers([step.get("voltage_in_V") for step in mappings])

    result: dict[str, Any] = {
        f"{prefix}_step_count": len(mappings),
        f"{prefix}_rate_min_C": min(rates) if rates else np.nan,
        f"{prefix}_rate_max_C": max(rates) if rates else np.nan,
        f"{prefix}_rate_mean_C": float(np.mean(rates)) if rates else np.nan,
        f"{prefix}_current_max_A": max(map(abs, currents)) if currents else np.nan,
    }
    voltage_key = (
        f"{prefix}_voltage_max_V"
        if prefix == "charge"
        else f"{prefix}_voltage_min_V"
    )
    result[voltage_key] = (
        (max(voltages) if prefix == "charge" else min(voltages))
        if voltages
        else np.nan
    )
    return result


def extract_record(pkl_path: Path, dataset: str, data_root: Path) -> dict[str, Any]:
    """Load one pickle and extract all top-level metadata except cycle_data."""
    with pkl_path.open("rb") as stream:
        battery = pickle.load(stream)
    if not isinstance(battery, Mapping):
        raise TypeError(f"top-level object is {type(battery).__name__}, not a mapping")

    record: dict[str, Any] = {
        "dataset": dataset,
        "source_file": pkl_path.relative_to(data_root).as_posix(),
        "file_size_bytes": pkl_path.stat().st_size,
    }
    for key, value in battery.items():
        if key != "cycle_data":
            record[str(key)] = csv_metadata_value(value)

    cycle_data = battery.get("cycle_data")
    cycles = (
        list(cycle_data)
        if isinstance(cycle_data, Sequence)
        and not isinstance(cycle_data, (str, bytes))
        else []
    )
    cycle_numbers = finite_numbers(
        [
            cycle.get("cycle_number")
            for cycle in cycles
            if isinstance(cycle, Mapping)
        ]
    )
    record["cycle_count"] = len(cycles)
    record["first_cycle_number"] = min(cycle_numbers) if cycle_numbers else np.nan
    record["last_cycle_number"] = max(cycle_numbers) if cycle_numbers else np.nan

    soc_interval = battery.get("SOC_interval")
    if (
        isinstance(soc_interval, Sequence)
        and not isinstance(soc_interval, (str, bytes))
        and len(soc_interval) >= 2
    ):
        interval = finite_numbers(soc_interval[:2])
        record["soc_interval_start"] = interval[0] if len(interval) > 0 else np.nan
        record["soc_interval_end"] = interval[1] if len(interval) > 1 else np.nan
    else:
        record["soc_interval_start"] = np.nan
        record["soc_interval_end"] = np.nan

    record.update(protocol_features(battery.get("charge_protocol"), "charge"))
    record.update(protocol_features(battery.get("discharge_protocol"), "discharge"))
    return record


def scan_metadata(
    data_root: Path, datasets: Sequence[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scan requested folders and return metadata plus a per-folder scan log."""
    records: list[dict[str, Any]] = []
    scan_rows: list[dict[str, Any]] = []

    for dataset in datasets:
        dataset_dir = data_root / dataset
        files = (
            sorted(dataset_dir.rglob("*.pkl"), key=natural_key)
            if dataset_dir.is_dir()
            else []
        )
        succeeded = 0
        errors: list[str] = []
        for pkl_path in files:
            try:
                records.append(extract_record(pkl_path, dataset, data_root))
                succeeded += 1
            except Exception as error:  # Continue so one corrupt file does not lose the run.
                errors.append(f"{pkl_path.name}: {type(error).__name__}: {error}")
        scan_rows.append(
            {
                "dataset": dataset,
                "folder_exists": dataset_dir.is_dir(),
                "pkl_files_found": len(files),
                "files_processed": succeeded,
                "files_failed": len(errors),
                "errors": " | ".join(errors),
            }
        )

    if not records:
        raise FileNotFoundError(
            f"No readable .pkl battery files found for the requested datasets under {data_root}"
        )
    return pd.DataFrame(records), pd.DataFrame(scan_rows)


def build_dataset_summary(metadata: pd.DataFrame) -> pd.DataFrame:
    """Build one compact coverage/statistics row per requested dataset."""
    numeric = [
        column
        for column in NUMERIC_CLUSTER_FEATURES
        if column in metadata.columns
    ]
    grouped = metadata.groupby("dataset", dropna=False)
    summary = grouped.agg(
        cell_count=("source_file", "size"),
        total_cycles=("cycle_count", "sum"),
        median_cycles=("cycle_count", "median"),
        total_file_size_bytes=("file_size_bytes", "sum"),
    )
    for column in numeric:
        summary[f"{column}__median"] = grouped[column].median()
    return summary.reset_index()


def build_missingness(metadata: pd.DataFrame) -> pd.DataFrame:
    """Report availability and unique-value counts for every output column."""
    rows = []
    total = len(metadata)
    for column in metadata.columns:
        missing = int(metadata[column].isna().sum())
        rows.append(
            {
                "column": column,
                "dtype": str(metadata[column].dtype),
                "non_missing_count": total - missing,
                "missing_count": missing,
                "missing_percent": round(100.0 * missing / total, 3),
                "unique_non_missing": int(metadata[column].nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def prepare_cluster_matrix(
    metadata: pd.DataFrame,
) -> tuple[np.ndarray, list[str], list[str], list[str]]:
    """Impute, scale, and one-hot encode usable clustering columns."""
    numeric = [
        column
        for column in NUMERIC_CLUSTER_FEATURES
        if column in metadata.columns
        and pd.to_numeric(metadata[column], errors="coerce").notna().any()
    ]
    categorical = [
        column
        for column in CATEGORICAL_CLUSTER_FEATURES
        if column in metadata.columns and metadata[column].notna().any()
    ]
    if not numeric and not categorical:
        raise ValueError("No usable metadata features are available for clustering")

    frame = metadata[numeric + categorical].copy()
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in categorical:
        frame[column] = frame[column].astype("string")

    transformers = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical,
            )
        )
    preprocessor = ColumnTransformer(transformers, remainder="drop")
    matrix = np.asarray(preprocessor.fit_transform(frame), dtype=float)
    feature_names = list(preprocessor.get_feature_names_out())
    return matrix, feature_names, numeric, categorical


def choose_cluster_count(
    matrix: np.ndarray, requested_clusters: int | None, max_clusters: int
) -> tuple[int, pd.DataFrame]:
    """Use silhouette score to choose k, or evaluate the explicitly supplied k."""
    sample_count = len(matrix)
    if sample_count < 3:
        raise ValueError("At least 3 battery records are required for clustering")
    if requested_clusters is not None:
        if not 2 <= requested_clusters < sample_count:
            raise ValueError(
                f"--n-clusters must be between 2 and {sample_count - 1}"
            )
        candidates = [requested_clusters]
    else:
        candidates = list(range(2, min(max_clusters, sample_count - 1) + 1))

    rows = []
    for clusters in candidates:
        model = KMeans(n_clusters=clusters, random_state=42, n_init=20)
        labels = model.fit_predict(matrix)
        score = (
            silhouette_score(matrix, labels)
            if len(np.unique(labels)) > 1
            else np.nan
        )
        rows.append(
            {
                "n_clusters": clusters,
                "silhouette_score": score,
                "inertia": model.inertia_,
            }
        )
    scores = pd.DataFrame(rows)
    if scores["silhouette_score"].notna().any():
        best = int(scores.loc[scores["silhouette_score"].idxmax(), "n_clusters"])
    else:
        best = int(candidates[0])
    return best, scores


def cluster_profiles(
    assignments: pd.DataFrame, numeric: Sequence[str], categorical: Sequence[str]
) -> pd.DataFrame:
    """Summarize cluster size, numeric medians, and categorical modes."""
    rows: list[dict[str, Any]] = []
    for cluster, group in assignments.groupby("cluster", sort=True):
        row: dict[str, Any] = {"cluster": cluster, "cell_count": len(group)}
        row["datasets"] = json.dumps(
            group["dataset"].value_counts().to_dict(),
            ensure_ascii=False,
            sort_keys=True,
        )
        for column in numeric:
            row[f"{column}__median"] = pd.to_numeric(
                group[column], errors="coerce"
            ).median()
        for column in categorical:
            mode = group[column].dropna().mode()
            row[f"{column}__mode"] = mode.iloc[0] if not mode.empty else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def save_pca_plot(assignments: pd.DataFrame, output_path: Path) -> None:
    """Save a PCA scatter plot colored by cluster."""
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(
        assignments["pca_1"],
        assignments["pca_2"],
        c=assignments["cluster"],
        cmap="tab10",
        s=36,
        alpha=0.8,
        edgecolors="none",
    )
    ax.set_title("Battery metadata clusters (PCA projection)")
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    legend = ax.legend(*scatter.legend_elements(), title="Cluster")
    ax.add_artist(legend)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Dataset folder names under data/ (default: the 13 requested datasets)",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=None,
        help="Fixed k. If omitted, select k by the best silhouette score.",
    )
    parser.add_argument(
        "--max-clusters",
        type=int,
        default=10,
        help="Maximum k considered during automatic selection (default: 10)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[scan] data root: {data_root}")
    metadata, scan_log = scan_metadata(data_root, args.datasets)
    metadata = metadata.sort_values(
        ["dataset", "source_file"], key=lambda col: col.map(str.casefold)
    ).reset_index(drop=True)

    # Stabilize expected numeric columns even when some pickle values are strings.
    for column in set(NUMERIC_CLUSTER_FEATURES) | {
        "cycle_count",
        "first_cycle_number",
        "last_cycle_number",
        "file_size_bytes",
    }:
        if column in metadata.columns:
            metadata[column] = pd.to_numeric(metadata[column], errors="coerce")

    matrix, encoded_names, numeric, categorical = prepare_cluster_matrix(metadata)
    best_k, scores = choose_cluster_count(
        matrix, args.n_clusters, args.max_clusters
    )
    model = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    labels = model.fit_predict(matrix)

    pca_components = min(2, matrix.shape[0], matrix.shape[1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        projected = PCA(n_components=pca_components).fit_transform(matrix)
    assignments = metadata.copy()
    assignments["cluster"] = labels
    assignments["pca_1"] = projected[:, 0]
    assignments["pca_2"] = projected[:, 1] if pca_components > 1 else 0.0

    outputs = {
        "all_metadata.csv": metadata,
        "dataset_scan_log.csv": scan_log,
        "dataset_summary.csv": build_dataset_summary(metadata),
        "metadata_missingness.csv": build_missingness(metadata),
        "cluster_k_evaluation.csv": scores,
        "cluster_assignments.csv": assignments,
        "cluster_profiles.csv": cluster_profiles(
            assignments, numeric, categorical
        ),
        "cluster_encoded_features.csv": pd.DataFrame(
            {"encoded_feature": encoded_names}
        ),
    }
    for filename, frame in outputs.items():
        path = output_dir / filename
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"[saved] {path.name}: {len(frame)} rows")

    plot_path = output_dir / "cluster_pca.png"
    save_pca_plot(assignments, plot_path)
    print(f"[saved] {plot_path.name}")
    print(
        f"[done] {len(metadata)} cells, {matrix.shape[1]} encoded features, "
        f"k={best_k}, output={output_dir}"
    )


if __name__ == "__main__":
    main()
