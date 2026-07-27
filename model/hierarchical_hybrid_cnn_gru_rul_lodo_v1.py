#!/usr/bin/env python
# VERSION: 1.0
# ARCHITECTURE: Parallel intra-cycle CNN/GRU fusion + inter-cycle GRU
# FILE: hierarchical_hybrid_cnn_gru_rul_lodo_v1.py
# PATH POLICY:
#   script -> project/model/
#   battery data -> project/data/{HUST,MATR,RWTH,SNL,Stanford,Tongji}/
#   life labels -> project/data/1. Life labels/
#   outputs -> project/run/<model_name>_target_<dataset>_<Asia-Seoul timestamp>/
"""
BatteryLife multi-source RUL baseline
=====================================

Model C:
    Per-cycle curve
      -> [1D CNN branch || intra-cycle GRU branch]
      -> concatenation and fusion
      -> variable-length inter-cycle GRU
      -> RUL regression

Evaluation:
    Leave-One-Dataset-Out (LODO)
    - The target dataset is completely excluded from supervised training.
    - Target life labels are opened only inside the evaluation stage.
    - Every prediction uses all cycles available up to the selected observation point.

Expected project structure:
    project/
    ├─ data/
    │  ├─ HUST/
    │  │  ├─ HUST_1-1.pkl
    │  │  └─ ...
    │  ├─ HUST/
    │  ├─ MATR/
    │  ├─ RWTH/
    │  ├─ SNL/
    │  ├─ Stanford/
    │  └─ Tongji/
    ├─ 1. Life labels/
    │  ├─ HUST_labels.json
    │  ├─ MATR_labels.json
    │  ├─ RWTH_labels.json
    │  ├─ SNL_labels.json
    │  ├─ Stanford_labels.json
    │  └─ Tongji_labels.json
    ├─ run/
    │  └─ hierarchical_hybrid_cnn_gru_target_HUST_YYYYMMDD_HHMMSS/
    └─ hierarchical_hybrid_cnn_gru_rul_lodo_v1.py

Example:
    python model/hierarchical_hybrid_cnn_gru_rul_lodo_v1.py --target HUST --model-name hierarchical_hybrid_cnn_gru

Useful quick test:
    python model/hierarchical_hybrid_cnn_gru_rul_lodo_v1.py --target HUST \
        --epochs 2 --prefixes-per-battery 3 --max-batteries-per-dataset 5

Notes:
    1. The JSON value is assumed to be the battery life/EOL cycle.
    2. cycle_data is assumed to be a list of dictionaries.
    3. The default input channels are:
       current, voltage, charge capacity, discharge capacity.
    4. temperature and internal resistance are excluded because they can be None.
    5. Each cycle is encoded by parallel CNN and intra-cycle GRU branches.
    6. The fused cycle embeddings are processed by an inter-cycle GRU.
    7. Source prefixes are sampled uniformly over 10%-80% of life.
    8. Target evaluation uses both life-fraction and absolute-cycle benchmarks.
    9. Interpolated cycle tensors are cached as float16 .npy files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset


FEATURE_KEYS: Tuple[str, ...] = (
    "current_in_A",
    "voltage_in_V",
    "charge_capacity_in_Ah",
    "discharge_capacity_in_Ah",
)
TIME_KEY = "time_in_s"

CACHE_FORMAT_VERSION = "v5_ordinal_cycle_age_and_flexible_label_match"

# Only these six BatteryLife datasets participate in the experiment.
ALLOWED_DATASETS: Tuple[str, ...] = (
    "HUST",
    "MATR",
    "RWTH",
    "SNL",
    "Stanford",
    "Tongji",
)
ALLOWED_DATASET_LOOKUP: Dict[str, str] = {
    name.lower(): name for name in ALLOWED_DATASETS
}


@dataclass(frozen=True)
class BatteryRecord:
    dataset: str
    file_path: Path
    file_name: str
    life: int


@dataclass(frozen=True)
class PrefixSample:
    record: BatteryRecord
    prefix_length: int
    current_cycle: int
    rul: float
    evaluation_scheme: str = "source"
    evaluation_point: float = float("nan")
    evaluation_label: str = "source"
    observation_fraction: float = float("nan")


@dataclass
class ChannelScaler:
    mean: np.ndarray
    std: np.ndarray

    def normalize(self, x: np.ndarray) -> np.ndarray:
        # x: [T, C, L]
        return (x - self.mean[None, :, None]) / self.std[None, :, None]

    def to_dict(self) -> Dict[str, List[float]]:
        return {
            "mean": self.mean.astype(float).tolist(),
            "std": self.std.astype(float).tolist(),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("BatteryLife LODO Model C: parallel intra-cycle CNN/GRU fusion ""+ inter-cycle GRU for RUL prediction."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help=(
            "Project root. When omitted, it is inferred as the parent directory "
            "of the folder containing this script (expected: project/model/)."
        ),
    )
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument(
        "--label-dir",
        type=str,
        default="1. Life labels",
        help="Life-label folder name located inside --data-dir.",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default="run",
        help="Root directory for timestamped experiment outputs.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="hierarchical_hybrid_cnn_gru",
        help="Name included in the timestamped run folder.",
    )
    parser.add_argument("--cache-dir", type=str, default=".cache_battery_rul")
    parser.add_argument("--target", type=str, required=True)

    parser.add_argument("--interp-length", type=int, default=128)
    parser.add_argument("--min-prefix-cycles", type=int, default=20)
    parser.add_argument("--prefixes-per-battery", type=int, default=12)
    parser.add_argument(
        "--source-min-fraction",
        type=float,
        default=0.10,
        help="Earliest life fraction used to create supervised source prefixes.",
    )
    parser.add_argument(
        "--source-max-fraction",
        type=float,
        default=0.80,
        help="Latest life fraction used to create supervised source prefixes.",
    )
    parser.add_argument(
        "--target-life-fractions",
        type=float,
        nargs="+",
        default=[0.20, 0.40, 0.60, 0.80],
        help=(
            "Target benchmark points expressed as fractions of the true life. "
            "Target labels are used only after source training to construct evaluation points."
        ),
    )
    parser.add_argument(
        "--target-cycle-counts",
        type=int,
        nargs="+",
        default=[20, 50, 100, 200, 500],
        help="Target benchmark points expressed as absolute observed cycle counts.",
    )
    parser.add_argument(
        "--max-sequence-cycles",
        type=int,
        default=0,
        help="0 keeps every observed cycle. A positive value uniformly subsamples long sequences.",
    )
    parser.add_argument(
        "--max-batteries-per-dataset",
        type=int,
        default=0,
        help="Debug option. 0 uses every battery.",
    )

    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=8)

    parser.add_argument(
        "--cnn-branch-dim",
        type=int,
        default=64,
        help="Output dimension of the intra-cycle CNN branch.",
    )
    parser.add_argument(
        "--intra-gru-hidden-dim",
        type=int,
        default=64,
        help="Hidden/output dimension of the intra-cycle GRU branch.",
    )
    parser.add_argument(
        "--cycle-embedding-dim",
        type=int,
        default=128,
        help="Dimension after concatenating and fusing the two cycle branches.",
    )
    parser.add_argument(
        "--gru-hidden-dim",
        type=int,
        default=128,
        help="Hidden dimension of the inter-cycle GRU.",
    )
    parser.add_argument(
        "--gru-layers",
        type=int,
        default=1,
        help="Number of inter-cycle GRU layers.",
    )
    parser.add_argument(
        "--cycle-encoder-chunk-size",
        type=int,
        default=512,
        help=(
            "Number of valid cycles encoded at once by the parallel cycle "
            "encoder. Lower this value if GPU memory is insufficient. "
            "0 encodes all valid cycles at once."
        ),
    )
    parser.add_argument("--dropout", type=float, default=0.20)

    parser.add_argument(
        "--scaler-cycles-per-battery",
        type=int,
        default=32,
        help="Number of source cycles sampled per battery to fit normalization. 0 uses all cycles.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_json(path: Path) -> Dict[str, int]:
    try:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Failed to read label JSON: {path}\n{exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError(f"Label JSON must be a dictionary: {path}")

    labels: Dict[str, int] = {}
    for key, value in raw.items():
        try:
            labels[str(key)] = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid life label in {path}: {key} -> {value}") from exc
    return labels



def normalize_label_key(value: str) -> str:
    """
    Normalize a label key and a pkl filename to the same comparison form.

    Handles:
      - keys containing folders
      - Windows or Linux separators
      - missing or different file extensions
      - case differences
      - spaces, hyphens, and underscores
    """
    raw = str(value).replace("\\", "/")
    basename = raw.rsplit("/", 1)[-1]
    stem = Path(basename).stem
    return re.sub(r"[^a-z0-9]+", "", stem.lower())


def build_label_indexes(
    labels: Dict[str, int],
) -> Tuple[
    Dict[str, int],
    Dict[str, int],
    Dict[str, List[Tuple[str, int]]],
]:
    basename_index: Dict[str, int] = {}
    stem_index: Dict[str, int] = {}
    normalized_index: Dict[str, List[Tuple[str, int]]] = {}

    for original_key, life in labels.items():
        normalized_path = str(original_key).replace("\\", "/")
        basename = normalized_path.rsplit("/", 1)[-1]
        basename_index.setdefault(basename.lower(), int(life))
        stem_index.setdefault(Path(basename).stem.lower(), int(life))

        normalized = normalize_label_key(original_key)
        normalized_index.setdefault(normalized, []).append(
            (str(original_key), int(life))
        )

    return basename_index, stem_index, normalized_index


def match_life_label(
    pkl_path: Path,
    labels: Dict[str, int],
    basename_index: Dict[str, int],
    stem_index: Dict[str, int],
    normalized_index: Dict[str, List[Tuple[str, int]]],
) -> Tuple[Optional[int], str]:
    # 1. Exact filename match
    if pkl_path.name in labels:
        return int(labels[pkl_path.name]), "exact"

    # 2. Case-insensitive basename match
    basename_match = basename_index.get(pkl_path.name.lower())
    if basename_match is not None:
        return int(basename_match), "basename_casefold"

    # 3. Case-insensitive stem match, allowing a missing/different extension
    stem_match = stem_index.get(pkl_path.stem.lower())
    if stem_match is not None:
        return int(stem_match), "stem_casefold"

    # 4. Punctuation-insensitive normalized stem match
    normalized = normalize_label_key(pkl_path.name)
    candidates = normalized_index.get(normalized, [])
    if len(candidates) == 1:
        return int(candidates[0][1]), "normalized_stem"

    if len(candidates) > 1:
        return None, "ambiguous_normalized_stem"

    return None, "not_found"

def find_label_file(label_root: Path, dataset_name: str) -> Path:
    preferred = label_root / f"{dataset_name}_labels.json"
    if preferred.exists():
        return preferred

    candidates = [
        p
        for p in label_root.glob("*.json")
        if p.stem.lower() == f"{dataset_name}_labels".lower()
    ]
    if len(candidates) == 1:
        return candidates[0]

    fuzzy = [
        p
        for p in label_root.glob("*.json")
        if dataset_name.lower() in p.stem.lower()
    ]
    if len(fuzzy) == 1:
        return fuzzy[0]

    raise FileNotFoundError(
        f"Could not uniquely find a label JSON for dataset '{dataset_name}' "
        f"under: {label_root}"
    )


def discover_records(
    data_root: Path,
    label_root: Path,
    max_batteries_per_dataset: int,
    target_dataset: str,
) -> Dict[str, List[BatteryRecord]]:
    if not data_root.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_root}")
    if not label_root.exists():
        raise FileNotFoundError(f"Label directory does not exist: {label_root}")

    discovered_dirs = {
        p.name.lower(): p
        for p in data_root.iterdir()
        if p.is_dir() and p.name.lower() in ALLOWED_DATASET_LOOKUP
    }

    missing_datasets = [
        canonical_name
        for canonical_name in ALLOWED_DATASETS
        if canonical_name.lower() not in discovered_dirs
    ]
    if missing_datasets:
        raise FileNotFoundError(
            "The following required dataset folders are missing under "
            f"{data_root}: {', '.join(missing_datasets)}"
        )

    dataset_dirs = [
        discovered_dirs[canonical_name.lower()]
        for canonical_name in ALLOWED_DATASETS
    ]
    print(
        "[dataset policy] Using only: "
        + ", ".join(ALLOWED_DATASETS)
    )

    all_records: Dict[str, List[BatteryRecord]] = {}

    for dataset_dir in dataset_dirs:
        dataset_name = ALLOWED_DATASET_LOOKUP[dataset_dir.name.lower()]
        pkl_files = sorted(dataset_dir.glob("*.pkl"))
        is_target = dataset_name.lower() == target_dataset.lower()

        # Strict LODO rule: target labels are not opened during discovery/training.
        if is_target:
            records = [
                BatteryRecord(
                    dataset=dataset_name,
                    file_path=pkl_path.resolve(),
                    file_name=pkl_path.name,
                    life=-1,
                )
                for pkl_path in pkl_files
            ]
            if max_batteries_per_dataset > 0:
                records = records[:max_batteries_per_dataset]
            if records:
                all_records[dataset_name] = records
                print(
                    f"[dataset] {dataset_name}: {len(records)} unlabeled target batteries "
                    "(target JSON not opened)"
                )
            continue

        try:
            label_file = find_label_file(label_root, dataset_name)
        except FileNotFoundError:
            print(f"[skip] No unique label file for source dataset: {dataset_name}")
            continue

        labels = read_json(label_file)
        basename_index, stem_index, normalized_index = build_label_indexes(labels)

        records: List[BatteryRecord] = []
        missing_labels: List[str] = []
        ambiguous_labels: List[str] = []
        match_counts: Dict[str, int] = {}

        for pkl_path in pkl_files:
            life, match_method = match_life_label(
                pkl_path=pkl_path,
                labels=labels,
                basename_index=basename_index,
                stem_index=stem_index,
                normalized_index=normalized_index,
            )
            match_counts[match_method] = match_counts.get(match_method, 0) + 1

            if life is None:
                if match_method == "ambiguous_normalized_stem":
                    ambiguous_labels.append(pkl_path.name)
                else:
                    missing_labels.append(pkl_path.name)
                continue

            records.append(
                BatteryRecord(
                    dataset=dataset_name,
                    file_path=pkl_path.resolve(),
                    file_name=pkl_path.name,
                    life=int(life),
                )
            )

        print(
            f"[label audit] {dataset_name}: pkl={len(pkl_files)}, "
            f"json keys={len(labels)}, matched={len(records)}, "
            f"unmatched={len(missing_labels)}, ambiguous={len(ambiguous_labels)}, "
            f"methods={match_counts}"
        )

        if missing_labels:
            print(
                f"[warning] {dataset_name}: unmatched pkl examples: "
                + ", ".join(missing_labels[:5])
            )
            print(
                f"[warning] {dataset_name}: label key examples: "
                + ", ".join(list(labels.keys())[:5])
            )

        if ambiguous_labels:
            print(
                f"[warning] {dataset_name}: ambiguous normalized matches: "
                + ", ".join(ambiguous_labels[:5])
            )

        if max_batteries_per_dataset > 0:
            records = records[:max_batteries_per_dataset]

        if records:
            all_records[dataset_name] = records
            print(
                f"[dataset] {dataset_name}: {len(records)} matched source batteries "
                f"(label file: {label_file.name})"
            )
        else:
            print(
                f"[dataset error] {dataset_name}: no pkl file could be matched "
                f"to {label_file.name}."
            )

    if not all_records:
        raise RuntimeError("No matched battery-label pairs were found.")

    return all_records


def safe_float_array(value: object) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size == 0:
        return None
    return arr


def unique_sorted_xy(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x, kind="stable")
    x_sorted = x[order]
    y_sorted = y[order]

    unique_x, unique_indices = np.unique(x_sorted, return_index=True)
    unique_y = y_sorted[unique_indices]
    return unique_x, unique_y


def interpolate_feature(
    time_axis: np.ndarray,
    values: np.ndarray,
    interp_length: int,
) -> np.ndarray:
    valid = np.isfinite(time_axis) & np.isfinite(values)
    time_axis = time_axis[valid]
    values = values[valid]

    if values.size == 0:
        return np.zeros(interp_length, dtype=np.float32)
    if values.size == 1:
        return np.full(interp_length, values[0], dtype=np.float32)

    time_axis, values = unique_sorted_xy(time_axis, values)
    if time_axis.size == 1 or math.isclose(float(time_axis[-1]), float(time_axis[0])):
        return np.full(interp_length, values[-1], dtype=np.float32)

    normalized_time = (time_axis - time_axis[0]) / (time_axis[-1] - time_axis[0])
    target_time = np.linspace(0.0, 1.0, interp_length, dtype=np.float64)
    out = np.interp(target_time, normalized_time, values)
    return out.astype(np.float32)


def cycle_to_array(cycle: dict, interp_length: int) -> Optional[np.ndarray]:
    if not isinstance(cycle, dict):
        return None

    feature_arrays = [safe_float_array(cycle.get(key)) for key in FEATURE_KEYS]
    if any(arr is None for arr in feature_arrays):
        return None

    assert all(arr is not None for arr in feature_arrays)
    lengths = [arr.size for arr in feature_arrays if arr is not None]
    time_arr = safe_float_array(cycle.get(TIME_KEY))

    if time_arr is None:
        common_length = min(lengths)
        if common_length < 2:
            return None
        time_arr = np.arange(common_length, dtype=np.float64)
    else:
        common_length = min([time_arr.size] + lengths)
        if common_length < 2:
            return None
        time_arr = time_arr[:common_length]

    output: List[np.ndarray] = []
    for arr in feature_arrays:
        assert arr is not None
        output.append(
            interpolate_feature(
                time_axis=time_arr,
                values=arr[:common_length],
                interp_length=interp_length,
            )
        )

    # [C, L]
    return np.stack(output, axis=0).astype(np.float32)


def infer_cycle_numbers(
    cycle_data: Sequence[dict],
    already_spent_cycles: int,
) -> np.ndarray:
    """
    Returns an estimated absolute cycle number for every retained cycle.

    Rule:
    - If cycle_number values are valid and monotonic, use them.
    - If already_spent_cycles > 0 and the first cycle_number looks relative
      (close to 1), add already_spent_cycles.
    - Otherwise fall back to 1..N plus already_spent_cycles.
    """
    raw_numbers: List[int] = []
    valid = True

    for index, cycle in enumerate(cycle_data):
        try:
            number = int(cycle.get("cycle_number", index + 1))
        except (TypeError, ValueError, AttributeError):
            valid = False
            break
        raw_numbers.append(number)

    if valid and raw_numbers:
        arr = np.asarray(raw_numbers, dtype=np.int64)
        monotonic = bool(np.all(np.diff(arr) >= 0))
        positive = bool(np.all(arr > 0))
        if monotonic and positive:
            if already_spent_cycles > 0 and arr[0] <= 2:
                arr = arr + int(already_spent_cycles)
            return arr

    return (
        np.arange(1, len(cycle_data) + 1, dtype=np.int64)
        + int(max(already_spent_cycles, 0))
    )


def cache_signature(record: BatteryRecord, interp_length: int) -> str:
    stat = record.file_path.stat()
    text = (
        f"{CACHE_FORMAT_VERSION}|{record.file_path}|{stat.st_size}|"
        f"{stat.st_mtime_ns}|{interp_length}|{'|'.join(FEATURE_KEYS)}"
    )
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def cache_paths(
    cache_root: Path,
    record: BatteryRecord,
    interp_length: int,
) -> Tuple[Path, Path, Path]:
    dataset_cache = cache_root / record.dataset
    dataset_cache.mkdir(parents=True, exist_ok=True)
    signature = cache_signature(record, interp_length)
    stem = f"{record.file_path.stem}_{signature}"
    return (
        dataset_cache / f"{stem}_curves.npy",
        dataset_cache / f"{stem}_cycles.npy",
        dataset_cache / f"{stem}_meta.json",
    )


def build_battery_cache(
    record: BatteryRecord,
    cache_root: Path,
    interp_length: int,
    rebuild: bool,
) -> Tuple[Path, Path]:
    curves_path, cycles_path, meta_path = cache_paths(
        cache_root, record, interp_length
    )

    if (
        not rebuild
        and curves_path.exists()
        and cycles_path.exists()
        and meta_path.exists()
    ):
        return curves_path, cycles_path

    try:
        with record.file_path.open("rb") as f:
            battery = pickle.load(f)
    except (OSError, pickle.UnpicklingError, EOFError) as exc:
        raise RuntimeError(f"Failed to read battery file: {record.file_path}") from exc

    if not isinstance(battery, dict):
        raise ValueError(f"Battery file must contain a dict: {record.file_path}")

    cycle_data = battery.get("cycle_data")
    if not isinstance(cycle_data, list) or not cycle_data:
        raise ValueError(
            f"'cycle_data' must be a non-empty list: {record.file_path}"
        )

    try:
        already_spent = int(battery.get("already_spent_cycles") or 0)
    except (TypeError, ValueError):
        already_spent = 0

    # Raw cycle_number is not consistent across all BatteryLife datasets.
    # Keep it only for diagnostics. RUL age uses the ordered position in
    # cycle_data plus already_spent_cycles, which is consistent with life labels
    # represented as total cycle counts.
    raw_cycle_numbers = infer_cycle_numbers(cycle_data, already_spent_cycles=0)
    ordinal_cycle_ages = (
        np.arange(1, len(cycle_data) + 1, dtype=np.int64)
        + int(max(already_spent, 0))
    )

    retained_curves: List[np.ndarray] = []
    retained_numbers: List[int] = []
    retained_raw_numbers: List[int] = []
    skipped = 0

    for cycle, ordinal_age, raw_cycle_number in zip(
        cycle_data,
        ordinal_cycle_ages,
        raw_cycle_numbers,
    ):
        curve = cycle_to_array(cycle, interp_length)
        if curve is None or not np.all(np.isfinite(curve)):
            skipped += 1
            continue
        retained_curves.append(curve)
        retained_numbers.append(int(ordinal_age))
        retained_raw_numbers.append(int(raw_cycle_number))

    if not retained_curves:
        raise ValueError(f"No valid cycles could be extracted: {record.file_path}")

    curves = np.stack(retained_curves, axis=0).astype(np.float16)
    cycle_numbers = np.asarray(retained_numbers, dtype=np.int64)

    # Atomic-ish writes through temporary files.
    curves_tmp = curves_path.with_name(curves_path.name + ".tmp.npy")
    cycles_tmp = cycles_path.with_name(cycles_path.name + ".tmp.npy")
    np.save(curves_tmp, curves)
    np.save(cycles_tmp, cycle_numbers)
    curves_tmp.replace(curves_path)
    cycles_tmp.replace(cycles_path)

    metadata = {
        "dataset": record.dataset,
        "file_name": record.file_name,
        "life": record.life,
        "original_cycle_count": len(cycle_data),
        "retained_cycle_count": int(curves.shape[0]),
        "skipped_cycle_count": skipped,
        "cycle_age_definition": (
            "ordered cycle_data position + already_spent_cycles"
        ),
        "first_cycle_age": int(cycle_numbers[0]),
        "last_cycle_age": int(cycle_numbers[-1]),
        "raw_first_cycle_number": (
            int(retained_raw_numbers[0]) if retained_raw_numbers else None
        ),
        "raw_last_cycle_number": (
            int(retained_raw_numbers[-1]) if retained_raw_numbers else None
        ),
        "already_spent_cycles": already_spent,
        "cache_format_version": CACHE_FORMAT_VERSION,
        "interp_length": interp_length,
        "features": list(FEATURE_KEYS),
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return curves_path, cycles_path


def prepare_all_caches(
    records: Iterable[BatteryRecord],
    cache_root: Path,
    interp_length: int,
    rebuild: bool,
) -> Dict[Path, Tuple[Path, Path]]:
    cache_map: Dict[Path, Tuple[Path, Path]] = {}
    records = list(records)
    total = len(records)

    for i, record in enumerate(records, start=1):
        print(f"[cache {i:>4}/{total}] {record.dataset}/{record.file_name}")
        try:
            cache_map[record.file_path] = build_battery_cache(
                record=record,
                cache_root=cache_root,
                interp_length=interp_length,
                rebuild=rebuild,
            )
        except Exception as exc:
            print(f"[cache skip] {record.file_name}: {exc}")

    if not cache_map:
        raise RuntimeError("No valid battery caches were created.")
    return cache_map


def load_cached_battery(
    cache_pair: Tuple[Path, Path],
) -> Tuple[np.ndarray, np.ndarray]:
    curves_path, cycles_path = cache_pair
    curves = np.load(curves_path, mmap_mode="r")
    cycle_numbers = np.load(cycles_path, mmap_mode="r")
    return curves, cycle_numbers


def filter_valid_records(
    records: Sequence[BatteryRecord],
    cache_map: Dict[Path, Tuple[Path, Path]],
    min_prefix_cycles: int,
) -> List[BatteryRecord]:
    valid_records: List[BatteryRecord] = []
    no_cache = 0
    too_short = 0
    life_too_short = 0

    for record in records:
        cache_pair = cache_map.get(record.file_path)
        if cache_pair is None:
            no_cache += 1
            continue

        curves, cycle_numbers = load_cached_battery(cache_pair)
        if len(curves) < min_prefix_cycles:
            too_short += 1
            continue

        valid_until = np.flatnonzero(cycle_numbers < record.life)
        if valid_until.size < min_prefix_cycles:
            life_too_short += 1
            continue

        valid_records.append(record)

    dataset_name = records[0].dataset if records else "unknown"
    print(
        f"[validity audit] {dataset_name}: input={len(records)}, "
        f"valid={len(valid_records)}, no_cache={no_cache}, "
        f"too_short={too_short}, life_conflict={life_too_short}"
    )
    return valid_records


def filter_unlabeled_target_records(
    records: Sequence[BatteryRecord],
    cache_map: Dict[Path, Tuple[Path, Path]],
    min_prefix_cycles: int,
) -> List[BatteryRecord]:
    """Filter the target using input availability only; no target life is consulted."""
    valid_records: List[BatteryRecord] = []
    for record in records:
        cache_pair = cache_map.get(record.file_path)
        if cache_pair is None:
            continue
        curves, _ = load_cached_battery(cache_pair)
        if len(curves) >= min_prefix_cycles:
            valid_records.append(record)
    return valid_records


def attach_target_labels_for_evaluation(
    records: Sequence[BatteryRecord],
    label_root: Path,
) -> List[BatteryRecord]:
    """Open the held-out target JSON only after supervised source training."""
    if not records:
        return []

    dataset_name = records[0].dataset
    label_file = find_label_file(label_root, dataset_name)
    labels = read_json(label_file)
    basename_index, stem_index, normalized_index = build_label_indexes(labels)

    labeled_records: List[BatteryRecord] = []
    missing: List[str] = []

    for record in records:
        life, _ = match_life_label(
            pkl_path=record.file_path,
            labels=labels,
            basename_index=basename_index,
            stem_index=stem_index,
            normalized_index=normalized_index,
        )
        if life is None:
            missing.append(record.file_name)
            continue
        labeled_records.append(
            BatteryRecord(
                dataset=record.dataset,
                file_path=record.file_path,
                file_name=record.file_name,
                life=int(life),
            )
        )

    if missing:
        print(
            f"[warning] Target evaluation skipped {len(missing)} files "
            "without matching labels."
        )
    print(
        f"[target evaluation] Opened {label_file.name}: "
        f"{len(labeled_records)} matched labels."
    )
    return labeled_records


def split_source_records(
    source_records_by_dataset: Dict[str, List[BatteryRecord]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[BatteryRecord], List[BatteryRecord]]:
    rng = random.Random(seed)
    train_records: List[BatteryRecord] = []
    val_records: List[BatteryRecord] = []

    for dataset, records in sorted(source_records_by_dataset.items()):
        shuffled = list(records)
        rng.shuffle(shuffled)

        if len(shuffled) == 1 or val_ratio <= 0:
            train_part = shuffled
            val_part: List[BatteryRecord] = []
        else:
            n_val = max(1, int(round(len(shuffled) * val_ratio)))
            n_val = min(n_val, len(shuffled) - 1)
            val_part = shuffled[:n_val]
            train_part = shuffled[n_val:]

        train_records.extend(train_part)
        val_records.extend(val_part)
        print(
            f"[split] {dataset}: train batteries={len(train_part)}, "
            f"validation batteries={len(val_part)}"
        )

    return train_records, val_records


def validate_benchmark_arguments(args: argparse.Namespace) -> None:
    if not (0.0 < args.source_min_fraction < args.source_max_fraction < 1.0):
        raise ValueError(
            "--source-min-fraction and --source-max-fraction must satisfy "
            "0 < min < max < 1."
        )

    if args.prefixes_per_battery < 1:
        raise ValueError("--prefixes-per-battery must be at least 1.")

    if not args.target_life_fractions:
        raise ValueError("--target-life-fractions must not be empty.")
    if any(not (0.0 < value < 1.0) for value in args.target_life_fractions):
        raise ValueError(
            "Every --target-life-fractions value must be between 0 and 1."
        )

    if not args.target_cycle_counts:
        raise ValueError("--target-cycle-counts must not be empty.")
    if any(value < 1 for value in args.target_cycle_counts):
        raise ValueError(
            "Every --target-cycle-counts value must be a positive integer."
        )

    positive_architecture_values = {
        "--cnn-branch-dim": args.cnn_branch_dim,
        "--intra-gru-hidden-dim": args.intra_gru_hidden_dim,
        "--cycle-embedding-dim": args.cycle_embedding_dim,
        "--gru-hidden-dim": args.gru_hidden_dim,
        "--gru-layers": args.gru_layers,
    }
    invalid = [
        name
        for name, value in positive_architecture_values.items()
        if value < 1
    ]
    if invalid:
        raise ValueError(
            "The following architecture arguments must be positive: "
            + ", ".join(invalid)
        )

    if args.cycle_encoder_chunk_size < 0:
        raise ValueError("--cycle-encoder-chunk-size must be >= 0.")


def prefix_length_for_cycle_age(
    cycle_numbers: np.ndarray,
    desired_cycle_age: int,
    min_prefix_cycles: int,
    max_valid_length: int,
) -> Optional[int]:
    """
    Return the number of available cycles at or before desired_cycle_age.

    cycle_numbers contains the absolute observed age assigned during caching:
    ordered cycle position + already_spent_cycles.
    """
    usable_cycle_numbers = np.asarray(
        cycle_numbers[:max_valid_length],
        dtype=np.int64,
    )
    prefix_length = int(
        np.searchsorted(
            usable_cycle_numbers,
            int(desired_cycle_age),
            side="right",
        )
    )
    if prefix_length < min_prefix_cycles:
        return None
    return min(prefix_length, max_valid_length)


def build_source_samples(
    records: Sequence[BatteryRecord],
    cache_map: Dict[Path, Tuple[Path, Path]],
    min_prefix_cycles: int,
    prefixes_per_battery: int,
    source_min_fraction: float,
    source_max_fraction: float,
) -> List[PrefixSample]:
    """
    Build balanced supervised prefixes from the early-to-late life range.

    Fractions are evenly spaced between source_min_fraction and
    source_max_fraction. Prefixes after 80% life are excluded by default,
    preventing late-life/small-RUL samples from dominating training.
    """
    samples: List[PrefixSample] = []
    requested_fractions = np.linspace(
        source_min_fraction,
        source_max_fraction,
        num=prefixes_per_battery,
        dtype=np.float64,
    )

    for record in records:
        _, cycle_numbers = load_cached_battery(cache_map[record.file_path])
        valid_indices = np.flatnonzero(cycle_numbers < record.life)
        if valid_indices.size < min_prefix_cycles:
            continue

        max_valid_length = int(valid_indices[-1] + 1)
        used_prefix_lengths: set[int] = set()

        for requested_fraction in requested_fractions:
            desired_cycle_age = int(round(record.life * requested_fraction))
            prefix_length = prefix_length_for_cycle_age(
                cycle_numbers=cycle_numbers,
                desired_cycle_age=desired_cycle_age,
                min_prefix_cycles=min_prefix_cycles,
                max_valid_length=max_valid_length,
            )
            if prefix_length is None or prefix_length in used_prefix_lengths:
                continue

            current_cycle = int(cycle_numbers[prefix_length - 1])
            rul = float(record.life - current_cycle)
            if rul <= 0:
                continue

            used_prefix_lengths.add(prefix_length)
            actual_fraction = float(current_cycle / record.life)
            samples.append(
                PrefixSample(
                    record=record,
                    prefix_length=prefix_length,
                    current_cycle=current_cycle,
                    rul=rul,
                    evaluation_scheme="source_life_fraction",
                    evaluation_point=float(requested_fraction),
                    evaluation_label=f"{requested_fraction * 100:.1f}%",
                    observation_fraction=actual_fraction,
                )
            )

    return samples


def build_target_benchmark_samples(
    records: Sequence[BatteryRecord],
    cache_map: Dict[Path, Tuple[Path, Path]],
    min_prefix_cycles: int,
    life_fractions: Sequence[float],
    absolute_cycle_counts: Sequence[int],
) -> Tuple[List[PrefixSample], List[dict]]:
    """
    Create two complementary target benchmarks.

    1. life_fraction:
       Controls degradation progress across batteries with different lives.
       The target life label is used only to choose the held-out evaluation
       time and compute the RUL label, after supervised source training.

    2. absolute_cycle:
       Represents deployment-like questions such as:
       "How well can RUL be predicted with 100 observed cycles?"
    """
    samples: List[PrefixSample] = []
    coverage: Dict[Tuple[str, str], dict] = {}

    unique_life_fractions = sorted(set(float(v) for v in life_fractions))
    unique_cycle_counts = sorted(set(int(v) for v in absolute_cycle_counts))

    for fraction in unique_life_fractions:
        label = f"{fraction * 100:.0f}%"
        coverage[("life_fraction", label)] = {
            "evaluation_scheme": "life_fraction",
            "evaluation_label": label,
            "evaluation_point": fraction,
            "eligible_batteries": 0,
            "total_batteries": len(records),
        }

    for cycle_count in unique_cycle_counts:
        label = f"{cycle_count}_cycles"
        coverage[("absolute_cycle", label)] = {
            "evaluation_scheme": "absolute_cycle",
            "evaluation_label": label,
            "evaluation_point": cycle_count,
            "eligible_batteries": 0,
            "total_batteries": len(records),
        }

    for record in records:
        _, cycle_numbers = load_cached_battery(cache_map[record.file_path])
        valid_indices = np.flatnonzero(cycle_numbers < record.life)
        if valid_indices.size < min_prefix_cycles:
            continue

        max_valid_length = int(valid_indices[-1] + 1)

        for fraction in unique_life_fractions:
            desired_cycle_age = int(round(record.life * fraction))
            prefix_length = prefix_length_for_cycle_age(
                cycle_numbers=cycle_numbers,
                desired_cycle_age=desired_cycle_age,
                min_prefix_cycles=min_prefix_cycles,
                max_valid_length=max_valid_length,
            )
            if prefix_length is None:
                continue

            current_cycle = int(cycle_numbers[prefix_length - 1])
            # Do not count a point as available when the stored trajectory
            # ends materially before the requested benchmark point.
            if current_cycle < desired_cycle_age:
                continue

            rul = float(record.life - current_cycle)
            if rul <= 0:
                continue

            label = f"{fraction * 100:.0f}%"
            samples.append(
                PrefixSample(
                    record=record,
                    prefix_length=prefix_length,
                    current_cycle=current_cycle,
                    rul=rul,
                    evaluation_scheme="life_fraction",
                    evaluation_point=float(fraction),
                    evaluation_label=label,
                    observation_fraction=float(current_cycle / record.life),
                )
            )
            coverage[("life_fraction", label)]["eligible_batteries"] += 1

        for cycle_count in unique_cycle_counts:
            prefix_length = prefix_length_for_cycle_age(
                cycle_numbers=cycle_numbers,
                desired_cycle_age=cycle_count,
                min_prefix_cycles=min_prefix_cycles,
                max_valid_length=max_valid_length,
            )
            if prefix_length is None:
                continue

            current_cycle = int(cycle_numbers[prefix_length - 1])
            if current_cycle < cycle_count:
                continue

            rul = float(record.life - current_cycle)
            if rul <= 0:
                continue

            label = f"{cycle_count}_cycles"
            samples.append(
                PrefixSample(
                    record=record,
                    prefix_length=prefix_length,
                    current_cycle=current_cycle,
                    rul=rul,
                    evaluation_scheme="absolute_cycle",
                    evaluation_point=float(cycle_count),
                    evaluation_label=label,
                    observation_fraction=float(current_cycle / record.life),
                )
            )
            coverage[("absolute_cycle", label)]["eligible_batteries"] += 1

    coverage_rows = list(coverage.values())
    for row in coverage_rows:
        total = max(int(row["total_batteries"]), 1)
        row["coverage_percent"] = (
            100.0 * int(row["eligible_batteries"]) / total
        )

    return samples, coverage_rows


def fit_channel_scaler(
    records: Sequence[BatteryRecord],
    cache_map: Dict[Path, Tuple[Path, Path]],
    cycles_per_battery: int,
    seed: int,
) -> ChannelScaler:
    rng = np.random.default_rng(seed)

    channel_sum = np.zeros(len(FEATURE_KEYS), dtype=np.float64)
    channel_sq_sum = np.zeros(len(FEATURE_KEYS), dtype=np.float64)
    channel_count = 0

    for record in records:
        curves, cycle_numbers = load_cached_battery(cache_map[record.file_path])
        valid_indices = np.flatnonzero(cycle_numbers < record.life)
        if valid_indices.size == 0:
            continue

        if cycles_per_battery > 0 and valid_indices.size > cycles_per_battery:
            selected = rng.choice(
                valid_indices, size=cycles_per_battery, replace=False
            )
        else:
            selected = valid_indices

        x = np.asarray(curves[selected], dtype=np.float32)  # [T, C, L]
        channel_sum += x.sum(axis=(0, 2), dtype=np.float64)
        channel_sq_sum += np.square(x, dtype=np.float64).sum(
            axis=(0, 2), dtype=np.float64
        )
        channel_count += x.shape[0] * x.shape[2]

    if channel_count == 0:
        raise RuntimeError("Could not fit scaler: no source cycles available.")

    mean = channel_sum / channel_count
    variance = channel_sq_sum / channel_count - np.square(mean)
    std = np.sqrt(np.maximum(variance, 1e-8))
    std = np.maximum(std, 1e-4)

    print("[scaler]")
    for key, mu, sigma in zip(FEATURE_KEYS, mean, std):
        print(f"  {key:<30} mean={mu: .6f}, std={sigma: .6f}")

    return ChannelScaler(
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
    )


def uniform_subsample_indices(length: int, max_length: int) -> np.ndarray:
    if max_length <= 0 or length <= max_length:
        return np.arange(length, dtype=np.int64)
    return np.linspace(0, length - 1, num=max_length).round().astype(np.int64)


class BatteryPrefixDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[PrefixSample],
        cache_map: Dict[Path, Tuple[Path, Path]],
        scaler: ChannelScaler,
        max_sequence_cycles: int,
    ) -> None:
        self.samples = list(samples)
        self.cache_map = cache_map
        self.scaler = scaler
        self.max_sequence_cycles = max_sequence_cycles

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        curves, cycle_numbers = load_cached_battery(
            self.cache_map[sample.record.file_path]
        )

        x = np.asarray(
            curves[: sample.prefix_length],
            dtype=np.float32,
        )
        cycle_ids = np.asarray(
            cycle_numbers[: sample.prefix_length],
            dtype=np.int64,
        )

        keep = uniform_subsample_indices(
            length=x.shape[0],
            max_length=self.max_sequence_cycles,
        )
        x = x[keep]
        cycle_ids = cycle_ids[keep]

        x = self.scaler.normalize(x).astype(np.float32)

        return {
            "x": torch.from_numpy(x),  # [T, C, L]
            "length": int(x.shape[0]),
            "rul": torch.tensor(sample.rul, dtype=torch.float32),
            "log_rul": torch.tensor(
                math.log1p(sample.rul), dtype=torch.float32
            ),
            "life": torch.tensor(sample.record.life, dtype=torch.float32),
            "current_cycle": torch.tensor(
                sample.current_cycle, dtype=torch.float32
            ),
            "cycle_ids": torch.from_numpy(cycle_ids),
            "dataset": sample.record.dataset,
            "file_name": sample.record.file_name,
            "evaluation_scheme": sample.evaluation_scheme,
            "evaluation_point": float(sample.evaluation_point),
            "evaluation_label": sample.evaluation_label,
            "observation_fraction": torch.tensor(
                sample.observation_fraction,
                dtype=torch.float32,
            ),
        }


def collate_prefix_batch(batch: Sequence[dict]) -> dict:
    lengths = torch.tensor(
        [item["length"] for item in batch],
        dtype=torch.long,
    )
    max_t = int(lengths.max().item())
    channels = int(batch[0]["x"].shape[1])
    curve_length = int(batch[0]["x"].shape[2])

    x = torch.zeros(
        len(batch),
        max_t,
        channels,
        curve_length,
        dtype=torch.float32,
    )
    cycle_ids = torch.zeros(len(batch), max_t, dtype=torch.long)
    mask = torch.zeros(len(batch), max_t, dtype=torch.bool)

    for i, item in enumerate(batch):
        t = item["length"]
        x[i, :t] = item["x"]
        cycle_ids[i, :t] = item["cycle_ids"]
        mask[i, :t] = True

    return {
        "x": x,
        "lengths": lengths,
        "mask": mask,
        "cycle_ids": cycle_ids,
        "rul": torch.stack([item["rul"] for item in batch]),
        "log_rul": torch.stack([item["log_rul"] for item in batch]),
        "life": torch.stack([item["life"] for item in batch]),
        "current_cycle": torch.stack(
            [item["current_cycle"] for item in batch]
        ),
        "dataset": [item["dataset"] for item in batch],
        "file_name": [item["file_name"] for item in batch],
        "evaluation_scheme": [
            item["evaluation_scheme"] for item in batch
        ],
        "evaluation_point": torch.tensor(
            [item["evaluation_point"] for item in batch],
            dtype=torch.float32,
        ),
        "evaluation_label": [
            item["evaluation_label"] for item in batch
        ],
        "observation_fraction": torch.stack(
            [item["observation_fraction"] for item in batch]
        ),
    }


class ParallelCycleCNNGRU(nn.Module):
    """
    Hybrid intra-cycle feature extractor.

    Input:
        x: [N_cycles, C_features, L_measurement_points]

    Branch 1:
        1D CNN extracts local curve morphology.

    Branch 2:
        GRU scans the L measurement points and extracts sequential context.

    Output:
        fused cycle embedding: [N_cycles, cycle_embedding_dim]
    """

    def __init__(
        self,
        in_channels: int,
        cnn_branch_dim: int,
        intra_gru_hidden_dim: int,
        cycle_embedding_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.cnn_encoder = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.cnn_projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, cnn_branch_dim),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(dropout),
        )

        # The paper uses one GRU layer in this branch due to computational cost.
        self.intra_cycle_gru = nn.GRU(
            input_size=in_channels,
            hidden_size=intra_gru_hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.gru_projection = nn.Sequential(
            nn.Linear(intra_gru_hidden_dim, intra_gru_hidden_dim),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(dropout),
        )

        fused_dim = cnn_branch_dim + intra_gru_hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, cycle_embedding_dim),
            nn.LayerNorm(cycle_embedding_dim),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                "ParallelCycleCNNGRU expects [N, C, L], "
                f"but received shape {tuple(x.shape)}."
            )

        # CNN branch: [N, C, L] -> [N, cnn_branch_dim]
        cnn_feature = self.cnn_projection(self.cnn_encoder(x))

        # GRU branch: [N, C, L] -> [N, L, C] -> final hidden state
        sequence = x.transpose(1, 2).contiguous()
        _, hidden = self.intra_cycle_gru(sequence)
        gru_feature = self.gru_projection(hidden[-1])

        fused = torch.cat([cnn_feature, gru_feature], dim=-1)
        return self.fusion(fused)


class BatteryRULModel(nn.Module):
    """
    Hierarchical Model C.

    Level 1, intra-cycle:
        CNN || GRU -> concat/fusion -> one embedding per cycle.

    Level 2, inter-cycle:
        variable-length GRU over all observed cycle embeddings.

    Padding cycles are not passed through the cycle encoder. This avoids
    contaminating CNN BatchNorm statistics and reduces unnecessary computation.
    """

    def __init__(
        self,
        in_channels: int,
        cnn_branch_dim: int,
        intra_gru_hidden_dim: int,
        cycle_embedding_dim: int,
        gru_hidden_dim: int,
        gru_layers: int,
        cycle_encoder_chunk_size: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.cycle_embedding_dim = cycle_embedding_dim
        self.cycle_encoder_chunk_size = cycle_encoder_chunk_size

        self.cycle_encoder = ParallelCycleCNNGRU(
            in_channels=in_channels,
            cnn_branch_dim=cnn_branch_dim,
            intra_gru_hidden_dim=intra_gru_hidden_dim,
            cycle_embedding_dim=cycle_embedding_dim,
            dropout=dropout,
        )

        inter_gru_dropout = dropout if gru_layers > 1 else 0.0
        self.sequence_encoder = nn.GRU(
            input_size=cycle_embedding_dim,
            hidden_size=gru_hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=inter_gru_dropout,
            bidirectional=False,
        )

        self.rul_head = nn.Sequential(
            nn.Linear(gru_hidden_dim, gru_hidden_dim),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden_dim, 1),
        )

    def encode_cycles_in_chunks(
        self,
        valid_cycles: torch.Tensor,
    ) -> torch.Tensor:
        num_cycles = int(valid_cycles.shape[0])
        if num_cycles == 0:
            raise ValueError("A batch must contain at least one valid cycle.")

        chunk_size = self.cycle_encoder_chunk_size
        if chunk_size <= 0 or num_cycles <= chunk_size:
            return self.cycle_encoder(valid_cycles)

        outputs: List[torch.Tensor] = []
        for start in range(0, num_cycles, chunk_size):
            end = min(start + chunk_size, num_cycles)
            outputs.append(self.cycle_encoder(valid_cycles[start:end]))
        return torch.cat(outputs, dim=0)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        # x: [B, T, C, L]
        if x.ndim != 4:
            raise ValueError(
                "BatteryRULModel expects [B, T, C, L], "
                f"but received shape {tuple(x.shape)}."
            )

        batch_size, max_t, _, _ = x.shape
        if lengths.ndim != 1 or lengths.shape[0] != batch_size:
            raise ValueError(
                "lengths must be [B] and match the input batch size."
            )

        time_index = torch.arange(max_t, device=x.device).unsqueeze(0)
        valid_mask = time_index < lengths.to(x.device).unsqueeze(1)

        # [total valid cycles, C, L]
        valid_cycles = x[valid_mask]
        valid_embeddings = self.encode_cycles_in_chunks(valid_cycles)

        # Scatter valid embeddings back to the padded battery tensor.
        cycle_embeddings = x.new_zeros(
            batch_size,
            max_t,
            self.cycle_embedding_dim,
        )
        cycle_embeddings[valid_mask] = valid_embeddings

        packed = pack_padded_sequence(
            cycle_embeddings,
            lengths=lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.sequence_encoder(packed)
        battery_embedding = hidden[-1]

        predicted_log_rul = self.rul_head(battery_embedding).squeeze(-1)
        return predicted_log_rul


def inverse_log_rul(predicted_log_rul: torch.Tensor) -> torch.Tensor:
    return torch.expm1(predicted_log_rul).clamp(min=0.0)


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    if y_true.size == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "mape": float("nan")}

    error = y_pred - y_true
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(np.square(error))))
    nonzero = np.abs(y_true) > 1e-8
    mape = (
        float(np.mean(np.abs(error[nonzero] / y_true[nonzero])) * 100.0)
        if np.any(nonzero)
        else float("nan")
    )
    return {"mae": mae, "rmse": rmse, "mape": mape}


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True)
        if torch.is_tensor(value)
        else value
        for key, value in batch.items()
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        pred_log_rul = model(batch["x"], batch["lengths"])
        loss = F.smooth_l1_loss(pred_log_rul, batch["log_rul"])

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss: {loss.item()}")

        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        batch_size = int(batch["x"].shape[0])
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    collect_rows: bool = False,
) -> Tuple[float, Dict[str, float], List[dict]]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    true_values: List[np.ndarray] = []
    pred_values: List[np.ndarray] = []
    rows: List[dict] = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        pred_log_rul = model(batch["x"], batch["lengths"])
        loss = F.smooth_l1_loss(pred_log_rul, batch["log_rul"])

        pred_rul = inverse_log_rul(pred_log_rul)
        true_rul = batch["rul"]

        batch_size = int(batch["x"].shape[0])
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size

        true_np = true_rul.detach().cpu().numpy()
        pred_np = pred_rul.detach().cpu().numpy()
        true_values.append(true_np)
        pred_values.append(pred_np)

        if collect_rows:
            lives = batch["life"].detach().cpu().numpy()
            current_cycles = batch["current_cycle"].detach().cpu().numpy()
            evaluation_points = (
                batch["evaluation_point"].detach().cpu().numpy()
            )
            observation_fractions = (
                batch["observation_fraction"].detach().cpu().numpy()
            )
            for i in range(batch_size):
                absolute_error = float(abs(pred_np[i] - true_np[i]))
                rows.append(
                    {
                        "dataset": batch["dataset"][i],
                        "file_name": batch["file_name"][i],
                        "evaluation_scheme": batch["evaluation_scheme"][i],
                        "evaluation_label": batch["evaluation_label"][i],
                        "evaluation_point": float(evaluation_points[i]),
                        "life": int(lives[i]),
                        "current_cycle": int(current_cycles[i]),
                        "observation_fraction": float(
                            observation_fractions[i]
                        ),
                        "true_rul": float(true_np[i]),
                        "predicted_rul": float(pred_np[i]),
                        "absolute_error": absolute_error,
                        "life_normalized_absolute_error_percent": float(
                            100.0 * absolute_error / max(float(lives[i]), 1.0)
                        ),
                    }
                )

    y_true = np.concatenate(true_values) if true_values else np.array([])
    y_pred = np.concatenate(pred_values) if pred_values else np.array([])
    metrics = regression_metrics(y_true, y_pred)
    avg_loss = total_loss / max(total_count, 1)
    return avg_loss, metrics, rows


def save_predictions(rows: Sequence[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "dataset",
        "file_name",
        "evaluation_scheme",
        "evaluation_label",
        "evaluation_point",
        "life",
        "current_cycle",
        "observation_fraction",
        "true_rul",
        "predicted_rul",
        "absolute_error",
        "life_normalized_absolute_error_percent",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(columns) + "\n")
        for row in rows:
            values = []
            for column in columns:
                value = row[column]
                if isinstance(value, str):
                    value = '"' + value.replace('"', '""') + '"'
                values.append(str(value))
            f.write(",".join(values) + "\n")


def metrics_from_prediction_rows(rows: Sequence[dict]) -> Dict[str, float]:
    if not rows:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "mape": float("nan"),
            "life_normalized_mae_percent": float("nan"),
        }

    y_true = np.asarray([row["true_rul"] for row in rows], dtype=np.float64)
    y_pred = np.asarray(
        [row["predicted_rul"] for row in rows],
        dtype=np.float64,
    )
    metrics = regression_metrics(y_true, y_pred)
    metrics["life_normalized_mae_percent"] = float(
        np.mean(
            [
                row["life_normalized_absolute_error_percent"]
                for row in rows
            ]
        )
    )
    return metrics


def summarize_target_benchmark(
    rows: Sequence[dict],
    coverage_rows: Sequence[dict],
) -> dict:
    grouped: Dict[Tuple[str, str], List[dict]] = {}
    battery_groups: Dict[str, List[dict]] = {}

    for row in rows:
        grouped.setdefault(
            (row["evaluation_scheme"], row["evaluation_label"]),
            [],
        ).append(row)
        battery_groups.setdefault(row["file_name"], []).append(row)

    group_summaries: List[dict] = []
    coverage_lookup = {
        (row["evaluation_scheme"], row["evaluation_label"]): row
        for row in coverage_rows
    }

    scheme_order = {"life_fraction": 0, "absolute_cycle": 1}
    sorted_groups = sorted(
        grouped.items(),
        key=lambda item: (
            scheme_order.get(item[0][0], 99),
            float(item[1][0]["evaluation_point"]),
        ),
    )

    for (scheme, label), group_rows in sorted_groups:
        metrics = metrics_from_prediction_rows(group_rows)
        observation_fractions = np.asarray(
            [row["observation_fraction"] for row in group_rows],
            dtype=np.float64,
        )
        coverage = coverage_lookup.get((scheme, label), {})
        group_summaries.append(
            {
                "evaluation_scheme": scheme,
                "evaluation_label": label,
                "evaluation_point": float(
                    group_rows[0]["evaluation_point"]
                ),
                "samples": len(group_rows),
                "batteries": len(
                    {row["file_name"] for row in group_rows}
                ),
                "coverage_percent": float(
                    coverage.get("coverage_percent", float("nan"))
                ),
                "mean_observation_fraction": float(
                    observation_fractions.mean()
                ),
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "mape": metrics["mape"],
                "life_normalized_mae_percent": metrics[
                    "life_normalized_mae_percent"
                ],
            }
        )

    per_battery_metrics: List[dict] = []
    for file_name, battery_rows in battery_groups.items():
        battery_metrics = metrics_from_prediction_rows(battery_rows)
        per_battery_metrics.append(
            {
                "file_name": file_name,
                **battery_metrics,
            }
        )

    if per_battery_metrics:
        macro_battery = {
            metric_name: float(
                np.mean(
                    [
                        row[metric_name]
                        for row in per_battery_metrics
                        if np.isfinite(row[metric_name])
                    ]
                )
            )
            for metric_name in (
                "mae",
                "rmse",
                "mape",
                "life_normalized_mae_percent",
            )
        }
    else:
        macro_battery = {
            "mae": float("nan"),
            "rmse": float("nan"),
            "mape": float("nan"),
            "life_normalized_mae_percent": float("nan"),
        }

    return {
        "overall_micro": metrics_from_prediction_rows(rows),
        "overall_macro_battery": macro_battery,
        "evaluated_batteries": len(battery_groups),
        "evaluation_samples": len(rows),
        "groups": group_summaries,
        "coverage": list(coverage_rows),
    }


def save_target_metric_files(
    summary: dict,
    json_path: Path,
    csv_path: Path,
) -> None:
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    columns = [
        "evaluation_scheme",
        "evaluation_label",
        "evaluation_point",
        "samples",
        "batteries",
        "coverage_percent",
        "mean_observation_fraction",
        "mae",
        "rmse",
        "mape",
        "life_normalized_mae_percent",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(columns) + "\n")
        for row in summary["groups"]:
            f.write(",".join(str(row[column]) for column in columns) + "\n")


def print_target_benchmark_summary(summary: dict) -> None:
    print("-" * 104)
    print(
        f"{'Benchmark':<18} {'Point':<12} {'N':>5} "
        f"{'Coverage':>10} {'MAE':>10} {'RMSE':>10} "
        f"{'Life-NMAE':>11}"
    )
    print("-" * 104)

    for row in summary["groups"]:
        print(
            f"{row['evaluation_scheme']:<18} "
            f"{row['evaluation_label']:<12} "
            f"{row['samples']:>5d} "
            f"{row['coverage_percent']:>9.1f}% "
            f"{row['mae']:>10.2f} "
            f"{row['rmse']:>10.2f} "
            f"{row['life_normalized_mae_percent']:>10.2f}%"
        )

    macro = summary["overall_macro_battery"]
    print("-" * 104)
    print(
        "Overall macro-by-battery | "
        f"MAE={macro['mae']:.2f}, "
        f"RMSE={macro['rmse']:.2f}, "
        f"MAPE={macro['mape']:.2f}%, "
        f"Life-NMAE={macro['life_normalized_mae_percent']:.2f}%"
    )
    print("-" * 104)


def create_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_prefix_batch,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )


def save_checkpoint(
    output_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    scaler: ChannelScaler,
    args: argparse.Namespace,
    dataset_names: Sequence[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
        "scaler": scaler.to_dict(),
        "feature_keys": list(FEATURE_KEYS),
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "source_datasets": list(dataset_names),
        "target_dataset": args.target,
    }
    torch.save(checkpoint, output_path)


def main() -> None:
    args = parse_args()
    validate_benchmark_arguments(args)
    set_seed(args.seed)
    device = resolve_device(args.device)

    target_key = args.target.lower()
    if target_key not in ALLOWED_DATASET_LOOKUP:
        raise ValueError(
            f"--target must be one of: {', '.join(ALLOWED_DATASETS)}"
        )
    args.target = ALLOWED_DATASET_LOOKUP[target_key]
    run_started_at = datetime.now(ZoneInfo("Asia/Seoul"))

    if args.project_root is None:
        # Expected script location: project/model/hierarchical_hybrid_cnn_gru_rul_lodo_v1.py
        project_root = Path(__file__).resolve().parent.parent
    else:
        project_root = args.project_root.resolve()

    data_root = project_root / args.data_dir
    label_root = data_root / args.label_dir
    run_root = project_root / args.run_dir
    cache_root = project_root / args.cache_dir

    print("=" * 80)
    print("BatteryLife multi-source RUL baseline")
    print(f"script path  : {Path(__file__).resolve()}")
    print(f"project root : {project_root}")
    print(f"data root    : {data_root}")
    print(f"label root   : {label_root}")
    print(f"target       : {args.target}")
    print(f"datasets     : {', '.join(ALLOWED_DATASETS)}")
    print(f"run root     : {run_root}")
    print(f"device       : {device}")
    print(
        "architecture : intra-cycle [CNN || GRU] -> concat/fusion "
        "-> inter-cycle GRU"
    )
    print(
        "dimensions   : "
        f"CNN={args.cnn_branch_dim}, "
        f"intra-GRU={args.intra_gru_hidden_dim}, "
        f"cycle-embedding={args.cycle_embedding_dim}, "
        f"inter-GRU={args.gru_hidden_dim}"
    )
    print("=" * 80)

    records_by_dataset = discover_records(
        data_root=data_root,
        label_root=label_root,
        max_batteries_per_dataset=args.max_batteries_per_dataset,
        target_dataset=args.target,
    )

    target_name = args.target
    if target_name not in records_by_dataset:
        raise ValueError(
            f"Target dataset '{target_name}' could not be loaded."
        )

    target_records = records_by_dataset[target_name]
    source_records_by_dataset = {
        name: records
        for name, records in records_by_dataset.items()
        if name != target_name
    }
    if not source_records_by_dataset:
        raise RuntimeError("LODO training requires at least one source dataset.")

    all_records = [
        record
        for records in records_by_dataset.values()
        for record in records
    ]

    cache_map = prepare_all_caches(
        records=all_records,
        cache_root=cache_root,
        interp_length=args.interp_length,
        rebuild=args.rebuild_cache,
    )

    source_records_by_dataset = {
        name: filter_valid_records(
            records,
            cache_map,
            args.min_prefix_cycles,
        )
        for name, records in source_records_by_dataset.items()
    }
    source_records_by_dataset = {
        name: records
        for name, records in source_records_by_dataset.items()
        if records
    }
    target_records = filter_unlabeled_target_records(
        target_records,
        cache_map,
        args.min_prefix_cycles,
    )

    expected_sources = {
        name for name in ALLOWED_DATASETS if name != target_name
    }
    missing_sources = sorted(
        expected_sources - set(source_records_by_dataset)
    )
    if missing_sources:
        raise RuntimeError(
            "No valid source batteries remain for required datasets: "
            + ", ".join(missing_sources)
            + ". Check the preceding [label audit], [cache skip], and "
              "[validity audit] lines for the exact cause."
        )
    if not target_records:
        raise RuntimeError("No valid target batteries remain after preprocessing.")

    train_records, val_records = split_source_records(
        source_records_by_dataset=source_records_by_dataset,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    if not train_records:
        raise RuntimeError("No source training batteries were created.")

    scaler = fit_channel_scaler(
        records=train_records,
        cache_map=cache_map,
        cycles_per_battery=args.scaler_cycles_per_battery,
        seed=args.seed,
    )

    train_samples = build_source_samples(
        records=train_records,
        cache_map=cache_map,
        min_prefix_cycles=args.min_prefix_cycles,
        prefixes_per_battery=args.prefixes_per_battery,
        source_min_fraction=args.source_min_fraction,
        source_max_fraction=args.source_max_fraction,
    )
    val_samples = build_source_samples(
        records=val_records,
        cache_map=cache_map,
        min_prefix_cycles=args.min_prefix_cycles,
        prefixes_per_battery=max(2, args.prefixes_per_battery // 2),
        source_min_fraction=args.source_min_fraction,
        source_max_fraction=args.source_max_fraction,
    )
    if not train_samples:
        raise RuntimeError("No source training prefixes were created.")

    # If the source pool is very small, use training prefixes for monitoring only.
    if not val_samples:
        print("[warning] No independent validation samples. Reusing train samples for monitoring.")
        val_samples = train_samples

    print(
        f"[samples] train={len(train_samples)}, validation={len(val_samples)}"
    )
    print(
        "[source-prefix policy] Uniform life-fraction sampling from "
        f"{args.source_min_fraction:.0%} to {args.source_max_fraction:.0%}; "
        "late-life prefixes beyond the maximum fraction are excluded."
    )
    print(
        "[target-label policy] The target JSON has not been opened. "
        "Target curves are unlabeled throughout supervised source training."
    )

    train_dataset = BatteryPrefixDataset(
        samples=train_samples,
        cache_map=cache_map,
        scaler=scaler,
        max_sequence_cycles=args.max_sequence_cycles,
    )
    val_dataset = BatteryPrefixDataset(
        samples=val_samples,
        cache_map=cache_map,
        scaler=scaler,
        max_sequence_cycles=args.max_sequence_cycles,
    )
    train_loader = create_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        device=device,
    )
    val_loader = create_loader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        device=device,
    )
    model = BatteryRULModel(
        in_channels=len(FEATURE_KEYS),
        cnn_branch_dim=args.cnn_branch_dim,
        intra_gru_hidden_dim=args.intra_gru_hidden_dim,
        cycle_embedding_dim=args.cycle_embedding_dim,
        gru_hidden_dim=args.gru_hidden_dim,
        gru_layers=args.gru_layers,
        cycle_encoder_chunk_size=args.cycle_encoder_chunk_size,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    source_names = sorted(source_records_by_dataset)

    run_timestamp = run_started_at.strftime("%Y%m%d_%H%M%S")
    safe_model_name = "".join(
        char if char.isalnum() or char in ("-", "_") else "_"
        for char in args.model_name.strip()
    ).strip("_")
    if not safe_model_name:
        raise ValueError("--model-name must contain at least one valid character.")

    run_name = (
        f"{safe_model_name}_target_{target_name}_{run_timestamp}"
    )
    run_path = run_root / run_name
    run_path.mkdir(parents=True, exist_ok=False)

    checkpoint_path = run_path / "best_model.pt"
    prediction_path = run_path / "target_predictions.csv"
    target_metrics_json_path = run_path / "target_metrics.json"
    target_metrics_csv_path = run_path / "target_metrics_by_point.csv"
    config_path = run_path / "config.json"
    with config_path.open("w", encoding="utf-8") as f:
        serializable_args = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        json.dump(
            {
                "args": serializable_args,
                "feature_keys": list(FEATURE_KEYS),
                "scaler": scaler.to_dict(),
                "model_name": safe_model_name,
                "architecture": {
                    "name": "hierarchical_parallel_cnn_gru_then_inter_gru",
                    "intra_cycle_cnn_branch_dim": args.cnn_branch_dim,
                    "intra_cycle_gru_hidden_dim": args.intra_gru_hidden_dim,
                    "fused_cycle_embedding_dim": args.cycle_embedding_dim,
                    "inter_cycle_gru_hidden_dim": args.gru_hidden_dim,
                    "inter_cycle_gru_layers": args.gru_layers,
                    "cycle_encoder_chunk_size": args.cycle_encoder_chunk_size,
                },
                "run_name": run_name,
                "run_timestamp_asia_seoul": run_timestamp,
                "run_path": str(run_path),
                "allowed_datasets": list(ALLOWED_DATASETS),
                "source_datasets": source_names,
                "target_dataset": target_name,
                "train_batteries": len(train_records),
                "validation_batteries": len(val_records),
                "target_batteries_unlabeled": len(target_records),
                "train_samples": len(train_samples),
                "validation_samples": len(val_samples),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip=args.grad_clip,
        )
        val_loss, val_metrics, _ = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            collect_rows=False,
        )

        print(
            f"[epoch {epoch:03d}] "
            f"train loss={train_loss:.5f} | "
            f"val loss={val_loss:.5f} | "
            f"val MAE={val_metrics['mae']:.2f} | "
            f"val RMSE={val_metrics['rmse']:.2f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            save_checkpoint(
                output_path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_loss=val_loss,
                scaler=scaler,
                args=args,
                dataset_names=source_names,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"[early stop] No validation improvement for {args.patience} epochs.")
                break

    if not checkpoint_path.exists():
        raise RuntimeError("Training ended without creating a checkpoint.")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Evaluation begins here. This is the first point at which target labels are opened.
    target_records_labeled = attach_target_labels_for_evaluation(
        records=target_records,
        label_root=label_root,
    )
    target_samples, target_coverage = build_target_benchmark_samples(
        records=target_records_labeled,
        cache_map=cache_map,
        min_prefix_cycles=args.min_prefix_cycles,
        life_fractions=args.target_life_fractions,
        absolute_cycle_counts=args.target_cycle_counts,
    )
    if not target_samples:
        raise RuntimeError("No target benchmark samples were created.")

    print(
        f"[target benchmark] samples={len(target_samples)}, "
        f"life fractions={args.target_life_fractions}, "
        f"absolute cycles={args.target_cycle_counts}"
    )
    for coverage_row in target_coverage:
        print(
            "[target coverage] "
            f"{coverage_row['evaluation_scheme']}/"
            f"{coverage_row['evaluation_label']}: "
            f"{coverage_row['eligible_batteries']}/"
            f"{coverage_row['total_batteries']} "
            f"({coverage_row['coverage_percent']:.1f}%)"
        )

    target_dataset = BatteryPrefixDataset(
        samples=target_samples,
        cache_map=cache_map,
        scaler=scaler,
        max_sequence_cycles=args.max_sequence_cycles,
    )
    target_loader = create_loader(
        target_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        device=device,
    )

    target_loss, target_metrics, prediction_rows = evaluate(
        model=model,
        loader=target_loader,
        device=device,
        collect_rows=True,
    )
    save_predictions(prediction_rows, prediction_path)

    target_summary = summarize_target_benchmark(
        rows=prediction_rows,
        coverage_rows=target_coverage,
    )
    save_target_metric_files(
        summary=target_summary,
        json_path=target_metrics_json_path,
        csv_path=target_metrics_csv_path,
    )

    elapsed = time.time() - start_time
    print("=" * 80)
    print(f"Best validation epoch : {checkpoint['epoch']}")
    print(f"Target dataset        : {target_name}")
    print(f"Target log-space loss : {target_loss:.5f}")
    print_target_benchmark_summary(target_summary)
    print(f"Run directory         : {run_path}")
    print(f"Checkpoint            : {checkpoint_path}")
    print(f"Predictions           : {prediction_path}")
    print(f"Metrics JSON          : {target_metrics_json_path}")
    print(f"Metrics by point CSV  : {target_metrics_csv_path}")
    print(f"Configuration         : {config_path}")
    print(f"Elapsed               : {elapsed / 60.0:.1f} minutes")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"\n[error] {exc}", file=sys.stderr)
        raise
