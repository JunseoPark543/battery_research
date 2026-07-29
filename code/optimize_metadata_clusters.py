#!/usr/bin/env python3
"""Find a robust K-Means cluster count for battery metadata.

This script reads data_summary/all_metadata.csv, cleans metadata-only features,
evaluates a range of k values over repeated random seeds, and saves the best
model and diagnostics to data_summary/optimal_clustering/.
"""

from __future__ import annotations

import argparse
import json
import re
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data_summary" / "all_metadata.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data_summary" / "optimal_clustering"

NUMERIC_CANDIDATES = (
    "nominal_capacity_in_Ah",
    "max_voltage_limit_in_V",
    "min_voltage_limit_in_V",
    "max_current_limit_in_A",
    "soc_interval_start",
    "soc_interval_end",
    "charge_step_count",
    "charge_rate_min_C",
    "charge_rate_max_C",
    "charge_rate_mean_C",
    "charge_voltage_max_V",
    "discharge_step_count",
    "discharge_rate_min_C",
    "discharge_rate_max_C",
    "discharge_rate_mean_C",
)
CATEGORICAL_CANDIDATES = (
    "form_factor",
    "anode_material",
    "cathode_material",
)
PROTOCOL_NUMERIC = (
    "charge_step_count",
    "charge_rate_min_C",
    "charge_rate_max_C",
    "charge_rate_mean_C",
    "charge_voltage_max_V",
    "discharge_step_count",
    "discharge_rate_min_C",
    "discharge_rate_max_C",
    "discharge_rate_mean_C",
)


def normalize_category(value: object) -> object:
    """Normalize harmless case/spacing differences without merging chemistries."""
    if pd.isna(value):
        return np.nan
    text = re.sub(r"\s+", " ", str(value).strip())
    return text.casefold()


def clean_features(metadata: pd.DataFrame) -> pd.DataFrame:
    """Correct missing protocol semantics and normalize categorical spelling."""
    frame = metadata.copy()
    for column in NUMERIC_CANDIDATES:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    # An empty protocol means "not recorded", not a real zero-step experiment.
    for prefix in ("charge", "discharge"):
        protocol_column = f"{prefix}_protocol"
        step_column = f"{prefix}_step_count"
        if protocol_column in frame and step_column in frame:
            empty = frame[protocol_column].fillna("").astype(str).str.strip().isin(
                ("", "[]", "null", "None")
            )
            affected = [
                column
                for column in PROTOCOL_NUMERIC
                if column.startswith(f"{prefix}_") and column in frame
            ]
            frame.loc[empty, affected] = np.nan

    for column in CATEGORICAL_CANDIDATES:
        if column in frame:
            frame[column] = frame[column].map(normalize_category)
    return frame


def select_features(
    frame: pd.DataFrame, max_missing_percent: float
) -> tuple[list[str], list[str], pd.DataFrame]:
    """Keep informative features below the requested missingness threshold."""
    numeric: list[str] = []
    categorical: list[str] = []
    rows: list[dict[str, object]] = []
    for column in (*NUMERIC_CANDIDATES, *CATEGORICAL_CANDIDATES):
        if column not in frame:
            rows.append(
                {
                    "feature": column,
                    "type": "unknown",
                    "missing_percent": 100.0,
                    "unique_non_missing": 0,
                    "selected": False,
                    "reason": "column_not_found",
                }
            )
            continue
        missing = float(frame[column].isna().mean() * 100.0)
        unique = int(frame[column].nunique(dropna=True))
        kind = "numeric" if column in NUMERIC_CANDIDATES else "categorical"
        selected = missing <= max_missing_percent and unique > 1
        reason = (
            "selected"
            if selected
            else ("too_much_missing" if missing > max_missing_percent else "constant")
        )
        rows.append(
            {
                "feature": column,
                "type": kind,
                "missing_percent": missing,
                "unique_non_missing": unique,
                "selected": selected,
                "reason": reason,
            }
        )
        if selected:
            (numeric if kind == "numeric" else categorical).append(column)
    if not numeric and not categorical:
        raise ValueError("No usable features remain after filtering")
    return numeric, categorical, pd.DataFrame(rows)


def preprocess(
    frame: pd.DataFrame, numeric: list[str], categorical: list[str]
) -> tuple[np.ndarray, list[str]]:
    """Median-impute and standardize numeric data; one-hot encode categories."""
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
                                handle_unknown="ignore", sparse_output=False
                            ),
                        ),
                    ]
                ),
                categorical,
            )
        )
    transformer = ColumnTransformer(transformers, remainder="drop")
    matrix = np.asarray(transformer.fit_transform(frame), dtype=float)
    return matrix, list(transformer.get_feature_names_out())


def evaluate_candidates(
    matrix: np.ndarray, k_min: int, k_max: int, repeats: int
) -> tuple[pd.DataFrame, dict[int, list[np.ndarray]]]:
    """Evaluate each k over repeated initializations and quantify stability."""
    if not 2 <= k_min <= k_max < len(matrix):
        raise ValueError(
            f"Require 2 <= k-min <= k-max < sample count ({len(matrix)})"
        )
    all_labels: dict[int, list[np.ndarray]] = {}
    rows = []
    for k in range(k_min, k_max + 1):
        run_rows = []
        label_sets = []
        for seed in range(repeats):
            model = KMeans(n_clusters=k, random_state=seed, n_init=20)
            labels = model.fit_predict(matrix)
            label_sets.append(labels)
            run_rows.append(
                {
                    "silhouette": silhouette_score(matrix, labels),
                    "calinski_harabasz": calinski_harabasz_score(matrix, labels),
                    "davies_bouldin": davies_bouldin_score(matrix, labels),
                    "inertia": model.inertia_,
                }
            )
        all_labels[k] = label_sets
        run = pd.DataFrame(run_rows)
        pairwise_ari = [
            adjusted_rand_score(label_sets[left], label_sets[right])
            for left, right in combinations(range(repeats), 2)
        ]
        rows.append(
            {
                "n_clusters": k,
                "silhouette_mean": run["silhouette"].mean(),
                "silhouette_std": run["silhouette"].std(ddof=0),
                "calinski_harabasz_mean": run["calinski_harabasz"].mean(),
                "davies_bouldin_mean": run["davies_bouldin"].mean(),
                "inertia_mean": run["inertia"].mean(),
                "stability_ari_mean": float(np.mean(pairwise_ari)),
                "stability_ari_min": float(np.min(pairwise_ari)),
            }
        )
    scores = pd.DataFrame(rows)

    # Rank aggregation prevents one metric's numerical scale from dominating.
    scores["rank_silhouette"] = scores["silhouette_mean"].rank(
        ascending=False, method="min"
    )
    scores["rank_calinski_harabasz"] = scores[
        "calinski_harabasz_mean"
    ].rank(ascending=False, method="min")
    scores["rank_davies_bouldin"] = scores["davies_bouldin_mean"].rank(
        ascending=True, method="min"
    )
    scores["rank_stability"] = scores["stability_ari_mean"].rank(
        ascending=False, method="min"
    )
    scores["rank_sum"] = scores[
        [
            "rank_silhouette",
            "rank_calinski_harabasz",
            "rank_davies_bouldin",
            "rank_stability",
        ]
    ].sum(axis=1)
    return scores, all_labels


def choose_best_k(scores: pd.DataFrame) -> int:
    """Choose the lowest aggregate rank, breaking ties by silhouette then k."""
    ordered = scores.sort_values(
        ["rank_sum", "silhouette_mean", "n_clusters"],
        ascending=[True, False, True],
    )
    return int(ordered.iloc[0]["n_clusters"])


def save_metric_plot(scores: pd.DataFrame, best_k: int, path: Path) -> None:
    """Save scale-independent metric curves using one panel per metric."""
    metrics = (
        ("silhouette_mean", "Silhouette (higher is better)"),
        ("calinski_harabasz_mean", "Calinski-Harabasz (higher is better)"),
        ("davies_bouldin_mean", "Davies-Bouldin (lower is better)"),
        ("stability_ari_mean", "Stability ARI (higher is better)"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, (column, title) in zip(axes.flat, metrics):
        ax.plot(scores["n_clusters"], scores[column], marker="o")
        ax.axvline(best_k, color="tab:red", linestyle="--", alpha=0.8)
        ax.set_title(title)
        ax.grid(alpha=0.25)
    for ax in axes[-1]:
        ax.set_xlabel("Number of clusters (k)")
    fig.suptitle(f"Metadata clustering diagnostics (selected k={best_k})")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_pca_plot(
    assignments: pd.DataFrame, explained: np.ndarray, path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(
        assignments["pca_1"],
        assignments["pca_2"],
        c=assignments["cluster"],
        cmap="tab20",
        s=30,
        alpha=0.78,
        edgecolors="none",
    )
    ax.set_title("Optimal battery metadata clusters")
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}%)")
    ax.legend(*scatter.legend_elements(), title="Cluster")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def cluster_profiles(
    assignments: pd.DataFrame, numeric: list[str], categorical: list[str]
) -> pd.DataFrame:
    rows = []
    for cluster, group in assignments.groupby("cluster", sort=True):
        row: dict[str, object] = {
            "cluster": cluster,
            "cell_count": len(group),
            "cell_percent": len(group) / len(assignments) * 100.0,
            "datasets": json.dumps(
                group["dataset"].value_counts().to_dict(),
                ensure_ascii=False,
                sort_keys=True,
            ),
        }
        for column in numeric:
            row[f"{column}__median"] = group[column].median()
        for column in categorical:
            mode = group[column].dropna().mode()
            row[f"{column}__mode"] = mode.iloc[0] if not mode.empty else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=15)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--max-missing-percent",
        type=float,
        default=35.0,
        help="Exclude features above this missingness (default: 35)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Metadata CSV not found: {args.input}")
    if args.repeats < 2:
        raise ValueError("--repeats must be at least 2 for stability analysis")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(args.input, encoding="utf-8-sig")
    clean = clean_features(metadata)
    numeric, categorical, feature_report = select_features(
        clean, args.max_missing_percent
    )
    matrix, encoded_names = preprocess(clean, numeric, categorical)
    scores, _ = evaluate_candidates(
        matrix, args.k_min, args.k_max, args.repeats
    )
    best_k = choose_best_k(scores)

    # Refit the selected solution with a fixed, reproducible high n_init.
    final_model = KMeans(n_clusters=best_k, random_state=42, n_init=100)
    labels = final_model.fit_predict(matrix)
    pca = PCA(n_components=2)
    projected = pca.fit_transform(matrix)

    assignments = metadata.copy()
    assignments["cluster"] = labels
    assignments["pca_1"] = projected[:, 0]
    assignments["pca_2"] = projected[:, 1]
    profiles = cluster_profiles(
        pd.concat(
            [
                assignments[["dataset", "cell_id", "cluster"]],
                clean[numeric + categorical],
            ],
            axis=1,
        ),
        numeric,
        categorical,
    )
    cross_counts = pd.crosstab(assignments["dataset"], assignments["cluster"])
    cross_percent = cross_counts.div(cross_counts.sum(axis=1), axis=0) * 100.0

    scores["selected"] = scores["n_clusters"].eq(best_k)
    outputs = {
        "k_evaluation.csv": scores,
        "feature_selection.csv": feature_report,
        "encoded_features.csv": pd.DataFrame({"feature": encoded_names}),
        "cluster_assignments.csv": assignments,
        "cluster_profiles.csv": profiles,
        "dataset_cluster_counts.csv": cross_counts.reset_index(),
        "dataset_cluster_percent.csv": cross_percent.reset_index(),
    }
    for filename, frame in outputs.items():
        frame.to_csv(
            output_dir / filename, index=False, encoding="utf-8-sig"
        )
        print(f"[saved] {filename}: {len(frame)} rows")

    save_metric_plot(scores, best_k, output_dir / "k_diagnostics.png")
    save_pca_plot(
        assignments, pca.explained_variance_ratio_, output_dir / "cluster_pca.png"
    )
    summary = {
        "sample_count": len(metadata),
        "selected_k": best_k,
        "selection_rule": (
            "minimum rank sum across silhouette, Calinski-Harabasz, "
            "Davies-Bouldin, and repeated-seed ARI stability"
        ),
        "k_range": [args.k_min, args.k_max],
        "repeats_per_k": args.repeats,
        "max_missing_percent": args.max_missing_percent,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "encoded_feature_count": matrix.shape[1],
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "final_inertia": final_model.inertia_,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[saved] k_diagnostics.png, cluster_pca.png, run_summary.json")
    print(
        f"[done] selected k={best_k}; {len(metadata)} cells; "
        f"{matrix.shape[1]} encoded features; output={output_dir}"
    )


if __name__ == "__main__":
    main()
