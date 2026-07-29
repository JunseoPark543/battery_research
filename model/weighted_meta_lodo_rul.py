#!/usr/bin/env python3
"""Target-aware weighted meta-learning for battery RUL under dataset LODO.

Each dataset is treated as one domain. For an outer LODO fold, the real target
dataset is excluded from all supervised training. During meta-training, one of
the remaining source datasets is sampled as a pseudo-target. Its *unlabelled*
context embedding is compared with the other source-domain prototypes, and
the resulting soft weights control a differentiable inner gradient update.
The pseudo-target query RUL loss trains both the RUL initialization and gate.

At evaluation, metadata and observed cycles from the unseen target are allowed
to compute source weights, but target life/RUL labels are opened only for final
metrics. This is target-aware unsupervised LODO, not strict source-only domain
generalization.

The preprocessing, cache format, labels, reference attention, and metrics are
reused from model/msda_rul.py.

Examples
--------
Prepare dQ(V) caches once:
    python model/msda_rul.py prepare --data-root data --cache-root cache \
        --labels-path "data/1. Life lables"

Quick smoke test:
    python model/weighted_meta_lodo_rul.py --target HUST --meta-steps 5 \
        --batch-per-domain 2 --max-domains 4 --device cpu

One target fold:
    python model/weighted_meta_lodo_rul.py --target HUST --device cuda

All available dataset folds:
    python model/weighted_meta_lodo_rul.py --all-targets --device cuda
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

import msda_rul as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class MetaConfig:
    meta_steps: int = 800
    inner_steps: int = 1
    inner_lr: float = 2e-3
    outer_lr: float = 5e-4
    weight_decay: float = 1e-4
    batch_per_domain: int = 6
    target_context_size: int = 8
    target_query_size: int = 8
    embedding_dim: int = 32
    metadata_hidden: int = 24
    gate_temperature: float = 0.25
    learn_temperature: bool = True
    min_temperature: float = 0.05
    max_temperature: float = 2.0
    entropy_weight: float = 1e-3
    weight_mode: str = "hybrid"
    grad_clip: float = 2.0
    log_every: int = 25
    seed: int = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalized_text(value: object) -> str:
    return " ".join(str(value).strip().casefold().split())


@dataclass
class MetadataSchema:
    """Fold-specific schema fitted only on source cells."""

    numeric_mean: np.ndarray
    numeric_std: np.ndarray
    cathode_to_index: Dict[str, int]

    @property
    def dimension(self) -> int:
        # nominal Ah, extracted V-low/high + branch(2) + cathode + unknown.
        return 3 + 2 + len(self.cathode_to_index) + 1

    @classmethod
    def fit(cls, cells: Sequence[base.Cell]) -> "MetadataSchema":
        numeric = np.asarray(
            [
                [
                    cell.cache.nominal_ah,
                    cell.cache.v_lo,
                    cell.cache.v_hi,
                ]
                for cell in cells
            ],
            dtype=np.float64,
        )
        mean = np.nanmean(numeric, axis=0)
        std = np.nanstd(numeric, axis=0)
        std[~np.isfinite(std) | (std < 1e-8)] = 1.0
        cathodes = sorted(
            {
                normalized_text(cell.cache.cathode)
                for cell in cells
                if normalized_text(cell.cache.cathode)
                not in {"", "none", "nan"}
            }
        )
        return cls(
            numeric_mean=mean.astype(np.float32),
            numeric_std=std.astype(np.float32),
            cathode_to_index={name: index for index, name in enumerate(cathodes)},
        )

    def encode(self, cell: base.Cell) -> np.ndarray:
        numeric = np.asarray(
            [cell.cache.nominal_ah, cell.cache.v_lo, cell.cache.v_hi],
            dtype=np.float32,
        )
        numeric = np.nan_to_num(
            (numeric - self.numeric_mean) / self.numeric_std,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        branch = np.zeros(2, dtype=np.float32)
        branch[0 if cell.cache.branch == "charge" else 1] = 1.0
        chemistry = np.zeros(
            len(self.cathode_to_index) + 1, dtype=np.float32
        )
        name = normalized_text(cell.cache.cathode)
        chemistry[self.cathode_to_index.get(name, len(self.cathode_to_index))] = 1.0
        return np.concatenate([numeric, branch, chemistry])

    def to_dict(self) -> dict:
        return {
            "numeric_mean": self.numeric_mean.tolist(),
            "numeric_std": self.numeric_std.tolist(),
            "cathode_to_index": self.cathode_to_index,
            "dimension": self.dimension,
        }


def collate_with_metadata(
    cells: Sequence[base.Cell],
    grids: Dict[str, np.ndarray],
    cfg: base.Config,
    schema: MetadataSchema,
) -> Tuple[torch.Tensor, ...]:
    dq, cyc, mask, y, observed = base.collate(list(cells), grids, cfg)
    metadata = torch.as_tensor(
        np.stack([schema.encode(cell) for cell in cells]),
        dtype=torch.float32,
        device=cfg.device,
    )
    return dq, cyc, mask, metadata, y, observed


class HybridWeightedMetaRUL(nn.Module):
    """RUL backbone plus metadata/cycle embedding used by the source gate."""

    def __init__(
        self, cfg: base.Config, meta_cfg: MetaConfig, metadata_dim: int
    ) -> None:
        super().__init__()
        self.backbone = base.Net(cfg)
        trajectory_dim = len(cfg.ref_grid) * cfg.m_bottleneck
        self.trajectory_projector = nn.Sequential(
            nn.Linear(trajectory_dim, meta_cfg.embedding_dim),
            nn.LayerNorm(meta_cfg.embedding_dim),
            nn.GELU(),
        )
        self.metadata_projector = nn.Sequential(
            nn.Linear(metadata_dim, meta_cfg.metadata_hidden),
            nn.GELU(),
            nn.Linear(meta_cfg.metadata_hidden, meta_cfg.embedding_dim),
        )
        self.fusion = nn.Sequential(
            nn.Linear(2 * meta_cfg.embedding_dim, meta_cfg.embedding_dim),
            nn.LayerNorm(meta_cfg.embedding_dim),
            nn.GELU(),
        )
        initial = math.log(math.exp(meta_cfg.gate_temperature) - 1.0)
        self.raw_temperature = nn.Parameter(
            torch.tensor(initial, dtype=torch.float32),
            requires_grad=meta_cfg.learn_temperature,
        )
        self.meta_cfg = meta_cfg

    def temperature(self) -> torch.Tensor:
        value = F.softplus(self.raw_temperature)
        return value.clamp(
            min=self.meta_cfg.min_temperature,
            max=self.meta_cfg.max_temperature,
        )

    def forward(
        self,
        dq: torch.Tensor,
        cyc: torch.Tensor,
        mask: torch.Tensor,
        metadata: torch.Tensor,
    ) -> dict:
        output = self.backbone(dq, cyc, mask)
        trajectory = self.trajectory_projector(output["u"].flatten(1))
        static = self.metadata_projector(metadata)
        embedding = F.normalize(
            self.fusion(torch.cat([trajectory, static], dim=-1)), dim=-1
        )
        return {
            **output,
            "embedding": embedding,
            "cycle_embedding": F.normalize(trajectory, dim=-1),
            "metadata_embedding": F.normalize(static, dim=-1),
        }


def sample_cells(
    cells: Sequence[base.Cell], count: int, rng: np.random.Generator
) -> List[base.Cell]:
    if not cells:
        raise ValueError("Cannot sample an empty cell collection")
    replace = len(cells) < count
    indices = rng.choice(len(cells), size=count, replace=replace)
    return [cells[int(index)] for index in indices]


def split_context_query(
    cells: Sequence[base.Cell],
    context_size: int,
    query_size: int,
    rng: np.random.Generator,
) -> tuple[List[base.Cell], List[base.Cell]]:
    total = context_size + query_size
    replace = len(cells) < total
    indices = rng.choice(len(cells), size=total, replace=replace)
    context = [cells[int(index)] for index in indices[:context_size]]
    query = [cells[int(index)] for index in indices[context_size:]]
    return context, query


def domain_weights(
    target_embedding: torch.Tensor,
    source_embeddings: Sequence[torch.Tensor],
    temperature: torch.Tensor,
) -> torch.Tensor:
    """Cosine-softmax weights from a target centroid to source centroids."""
    target_prototype = F.normalize(target_embedding.mean(dim=0), dim=0)
    source_prototypes = torch.stack(
        [
            F.normalize(embedding.mean(dim=0), dim=0)
            for embedding in source_embeddings
        ]
    )
    similarity = source_prototypes @ target_prototype
    return torch.softmax(similarity / temperature, dim=0)


def output_embedding(output: dict, mode: str) -> torch.Tensor:
    if mode == "hybrid":
        return output["embedding"]
    if mode == "metadata":
        return output["metadata_embedding"]
    if mode == "cycle":
        return output["cycle_embedding"]
    raise ValueError(f"Embedding is undefined for weight mode: {mode}")


def entropy(weights: torch.Tensor) -> torch.Tensor:
    return -(weights.clamp_min(1e-9) * weights.clamp_min(1e-9).log()).sum()


def named_parameters_dict(model: nn.Module) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(model.named_parameters())


def run_model(
    model: HybridWeightedMetaRUL,
    params: Mapping[str, torch.Tensor] | None,
    batch: Tuple[torch.Tensor, ...],
) -> dict:
    inputs = batch[:4]
    if params is None:
        return model(*inputs)
    return functional_call(model, params, inputs)


def weighted_inner_update(
    model: HybridWeightedMetaRUL,
    params: OrderedDict[str, torch.Tensor],
    source_batches: Sequence[Tuple[torch.Tensor, ...]],
    target_context_batch: Tuple[torch.Tensor, ...],
    label_mean: float,
    label_std: float,
    inner_lr: float,
    create_graph: bool,
) -> tuple[OrderedDict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """One differentiable update from target-weighted source-domain losses."""
    target_output = run_model(model, params, target_context_batch)
    source_outputs = [run_model(model, params, batch) for batch in source_batches]
    if model.meta_cfg.weight_mode == "uniform":
        weights = torch.full(
            (len(source_outputs),),
            1.0 / len(source_outputs),
            device=target_output["y"].device,
        )
    else:
        weights = domain_weights(
            output_embedding(target_output, model.meta_cfg.weight_mode),
            [
                output_embedding(output, model.meta_cfg.weight_mode)
                for output in source_outputs
            ],
            model.temperature(),
        )
    losses = torch.stack(
        [
            F.mse_loss(
                output["y"],
                (batch[4] - label_mean) / label_std,
            )
            for output, batch in zip(source_outputs, source_batches)
        ]
    )
    weighted_loss = torch.sum(weights * losses)
    gradients = torch.autograd.grad(
        weighted_loss,
        tuple(params.values()),
        create_graph=create_graph,
        allow_unused=True,
    )
    updated = OrderedDict()
    for (name, parameter), gradient in zip(params.items(), gradients):
        updated[name] = (
            parameter
            if gradient is None
            else parameter - inner_lr * gradient
        )
    return updated, weights, losses


def group_by_domain(
    cells: Iterable[base.Cell],
) -> Dict[str, List[base.Cell]]:
    groups: Dict[str, List[base.Cell]] = {}
    for cell in cells:
        groups.setdefault(cell.dataset, []).append(cell)
    return groups


def enforce_official_labels(
    cells: Sequence[base.Cell], labels_path: Path, minimum_life: int
) -> List[base.Cell]:
    """Drop unmatched cells and overwrite labels with official cycle lives."""
    scoped = base.load_life_labels(labels_path)

    def canonical(value: object) -> str:
        text = str(value).strip().upper()
        text = re.sub(r"\.(PKL|PICKLE|JSON|CSV)$", "", text)
        return re.sub(r"[^A-Z0-9]+", "_", text).strip("_")

    normalized_tables = {
        canonical(hint): {canonical(key): value for key, value in table.items()}
        for hint, table in scoped.items()
    }

    def lookup(dataset: str, cell_id: str) -> float | None:
        ds, cell = canonical(dataset), canonical(cell_id)
        matching_tables = [
            table
            for hint, table in normalized_tables.items()
            if hint == ds or hint.startswith(ds) or ds.startswith(hint)
        ]
        hits = [table[cell] for table in matching_tables if cell in table]
        if len(hits) == 1:
            return float(hits[0])
        global_hits = [
            table[cell] for table in normalized_tables.values() if cell in table
        ]
        return float(global_hits[0]) if len(global_hits) == 1 else None

    kept: List[base.Cell] = []
    dropped = 0
    for cell in cells:
        life = lookup(cell.dataset, cell.cache.cell_id)
        if life is None or not np.isfinite(life) or life < minimum_life:
            dropped += 1
            continue
        cell.y_raw = float(np.log(life))
        kept.append(cell)
    print(
        f"[label policy] official-only: kept={len(kept)}, dropped={dropped}",
        flush=True,
    )
    if not kept:
        raise RuntimeError("No cells matched official life labels")
    return kept


def enforce_consistent_branches(
    cells: Sequence[base.Cell],
    grids: Dict[str, np.ndarray],
    cfg: base.Config,
) -> tuple[List[base.Cell], Dict[str, np.ndarray]]:
    """Keep the majority charge/discharge branch and rebuild valid grids."""
    groups = group_by_domain(cells)
    kept: List[base.Cell] = []
    rebuilt: Dict[str, np.ndarray] = {}
    for dataset, domain_cells in sorted(groups.items()):
        counts: Dict[str, int] = {}
        for cell in domain_cells:
            counts[cell.cache.branch] = counts.get(cell.cache.branch, 0) + 1
        branch = max(sorted(counts), key=lambda name: counts[name])
        selected = [
            cell for cell in domain_cells if cell.cache.branch == branch
        ]
        removed = len(domain_cells) - len(selected)
        lo = max(cell.cache.v_lo for cell in selected)
        hi = min(cell.cache.v_hi for cell in selected)
        absolute_grid = np.isfinite(lo) and np.isfinite(hi) and lo < hi
        if absolute_grid:
            grid = np.linspace(lo, hi, cfg.grid_points).astype(np.float32)
            for cell in selected:
                cell._cyc = None
                cell._dq = None
                cell.regrid(grid)
            grid_text = f"absolute V=[{lo:.3f},{hi:.3f}]"
        else:
            # Heterogeneous chemistry/protocol datasets (notably SNL) may have
            # no voltage interval shared by every cell even within one branch.
            # Preserve curve shape on each cell's stable interval by mapping
            # its local voltage coordinate to [0,1].
            grid = np.linspace(0.0, 1.0, cfg.grid_points).astype(np.float32)
            for cell in selected:
                keep = cell.cache.cycles <= cell.L
                cell._cyc = cell.cache.cycles[keep].astype(np.int64)
                local = cell.cache.dq[keep]
                old_position = np.linspace(
                    0.0, 1.0, local.shape[1], dtype=np.float32
                )
                cell._dq = np.stack(
                    [np.interp(grid, old_position, row) for row in local]
                ).astype(np.float32)
            grid_text = "relative voltage coordinate [0,1]"
        kept.extend(selected)
        rebuilt[dataset] = grid
        print(
            f"[branch policy] {dataset}: {branch}, kept={len(selected)}, "
            f"removed={removed}, {grid_text}",
            flush=True,
        )
    return kept, rebuilt


def source_normalization(cells: Sequence[base.Cell]) -> tuple[float, float]:
    labels = np.asarray([cell.y_raw for cell in cells], dtype=np.float64)
    return float(labels.mean()), float(labels.std() + 1e-8)


def meta_train(
    model: HybridWeightedMetaRUL,
    source_groups: Dict[str, List[base.Cell]],
    grids: Dict[str, np.ndarray],
    cfg: base.Config,
    meta_cfg: MetaConfig,
    schema: MetadataSchema,
) -> list[dict]:
    """Nested pseudo-LODO meta-training using source domains only."""
    domains = sorted(source_groups)
    if len(domains) < 3:
        raise ValueError("At least 3 source domains are required for pseudo-LODO")
    all_source = [cell for cells in source_groups.values() for cell in cells]
    label_mean, label_std = source_normalization(all_source)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=meta_cfg.outer_lr,
        weight_decay=meta_cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=meta_cfg.meta_steps
    )
    rng = np.random.default_rng(meta_cfg.seed)
    history: list[dict] = []

    for step in range(1, meta_cfg.meta_steps + 1):
        pseudo_target = domains[(step - 1) % len(domains)]
        pseudo_sources = [domain for domain in domains if domain != pseudo_target]
        context_cells, query_cells = split_context_query(
            source_groups[pseudo_target],
            meta_cfg.target_context_size,
            meta_cfg.target_query_size,
            rng,
        )
        context_batch = collate_with_metadata(
            context_cells, grids, cfg, schema
        )
        query_batch = collate_with_metadata(query_cells, grids, cfg, schema)
        source_batches = [
            collate_with_metadata(
                sample_cells(
                    source_groups[domain], meta_cfg.batch_per_domain, rng
                ),
                grids,
                cfg,
                schema,
            )
            for domain in pseudo_sources
        ]

        params = named_parameters_dict(model)
        last_weights = None
        last_source_losses = None
        for _ in range(meta_cfg.inner_steps):
            params, last_weights, last_source_losses = weighted_inner_update(
                model,
                params,
                source_batches,
                context_batch,
                label_mean,
                label_std,
                meta_cfg.inner_lr,
                create_graph=True,
            )
        query_output = run_model(model, params, query_batch)
        query_loss = F.mse_loss(
            query_output["y"],
            (query_batch[4] - label_mean) / label_std,
        )
        # A tiny negative-entropy term discourages premature one-source collapse.
        regularizer = -meta_cfg.entropy_weight * entropy(last_weights)
        objective = query_loss + regularizer

        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), meta_cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        row = {
            "step": step,
            "pseudo_target": pseudo_target,
            "query_loss": float(query_loss.detach()),
            "objective": float(objective.detach()),
            "temperature": float(model.temperature().detach()),
            "weight_entropy": float(entropy(last_weights).detach()),
            "max_weight": float(last_weights.max().detach()),
        }
        history.append(row)
        if step == 1 or step % meta_cfg.log_every == 0:
            weight_text = ", ".join(
                f"{domain}={float(weight):.3f}"
                for domain, weight in zip(
                    pseudo_sources, last_weights.detach().cpu()
                )
            )
            print(
                f"[meta {step:4d}/{meta_cfg.meta_steps}] "
                f"pseudo-target={pseudo_target} query={row['query_loss']:.4f} "
                f"T={row['temperature']:.3f} | {weight_text}",
                flush=True,
            )
    return history


def adapt_to_real_target(
    model: HybridWeightedMetaRUL,
    source_groups: Dict[str, List[base.Cell]],
    target_cells: Sequence[base.Cell],
    grids: Dict[str, np.ndarray],
    cfg: base.Config,
    meta_cfg: MetaConfig,
    schema: MetadataSchema,
) -> tuple[OrderedDict[str, torch.Tensor], np.ndarray]:
    """Unsupervised target-aware weighting followed by weighted source update."""
    rng = np.random.default_rng(meta_cfg.seed + 10_000)
    all_source = [cell for cells in source_groups.values() for cell in cells]
    label_mean, label_std = source_normalization(all_source)
    domains = sorted(source_groups)
    source_batches = [
        collate_with_metadata(
            sample_cells(
                source_groups[domain], meta_cfg.batch_per_domain, rng
            ),
            grids,
            cfg,
            schema,
        )
        for domain in domains
    ]
    context_cells = sample_cells(
        target_cells,
        min(max(meta_cfg.target_context_size, 1), len(target_cells)),
        rng,
    )
    target_context = collate_with_metadata(
        context_cells, grids, cfg, schema
    )
    params = named_parameters_dict(model)
    accumulated = []
    for _ in range(meta_cfg.inner_steps):
        params, weights, _ = weighted_inner_update(
            model,
            params,
            source_batches,
            target_context,
            label_mean,
            label_std,
            meta_cfg.inner_lr,
            create_graph=False,
        )
        accumulated.append(weights.detach().cpu().numpy())
    return params, np.mean(accumulated, axis=0)


@torch.no_grad()
def evaluate_target(
    model: HybridWeightedMetaRUL,
    params: Mapping[str, torch.Tensor],
    target_cells: Sequence[base.Cell],
    grids: Dict[str, np.ndarray],
    cfg: base.Config,
    schema: MetadataSchema,
    source_cells: Sequence[base.Cell],
) -> tuple[dict, list[dict]]:
    label_mean, label_std = source_normalization(source_cells)
    predictions = []
    rows = []
    batch_size = max(1, cfg.batch_target)
    for start in range(0, len(target_cells), batch_size):
        cells = list(target_cells[start : start + batch_size])
        batch = collate_with_metadata(cells, grids, cfg, schema)
        output = run_model(model, params, batch)
        prediction_log = (
            output["y"] * label_std + label_mean
        ).detach().cpu().numpy()
        predictions.extend(prediction_log.tolist())
        for cell, predicted in zip(cells, prediction_log):
            life_true = float(np.exp(cell.y_raw))
            life_pred = float(np.exp(predicted))
            rows.append(
                {
                    "dataset": cell.dataset,
                    "cell_id": cell.cache.cell_id,
                    "observed_cycle": cell.L,
                    "true_life": life_true,
                    "predicted_life": life_pred,
                    "true_rul": life_true - cell.L,
                    "predicted_rul": max(life_pred - cell.L, 0.0),
                }
            )
    metrics = base._metrics(
        np.asarray(predictions),
        np.asarray([cell.y_raw for cell in target_cells]),
        list(target_cells),
        cfg,
    )
    return metrics, rows


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def target_name_lookup(cells: Sequence[base.Cell], requested: str) -> str:
    lookup = {cell.dataset.casefold(): cell.dataset for cell in cells}
    key = requested.casefold()
    if key not in lookup:
        raise ValueError(
            f"Target '{requested}' not found. Available: {sorted(set(lookup.values()))}"
        )
    return lookup[key]


def run_fold(
    target: str,
    cells: Sequence[base.Cell],
    grids: Dict[str, np.ndarray],
    cfg: base.Config,
    meta_cfg: MetaConfig,
    output_root: Path,
) -> dict:
    target = target_name_lookup(cells, target)
    target_cells = [cell for cell in cells if cell.dataset == target]
    source_cells = [cell for cell in cells if cell.dataset != target]
    source_groups = group_by_domain(source_cells)
    if len(source_groups) < 3:
        raise ValueError(
            f"Fold {target} has only {len(source_groups)} source domains"
        )
    fold_dir = output_root / f"target_{target}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    schema = MetadataSchema.fit(source_cells)
    model = HybridWeightedMetaRUL(cfg, meta_cfg, schema.dimension).to(cfg.device)
    print(
        f"\n[fold] target={target}; target cells={len(target_cells)}; "
        f"source domains={len(source_groups)}; source cells={len(source_cells)}",
        flush=True,
    )
    history = meta_train(
        model, source_groups, grids, cfg, meta_cfg, schema
    )
    adapted_params, weights = adapt_to_real_target(
        model,
        source_groups,
        target_cells,
        grids,
        cfg,
        meta_cfg,
        schema,
    )
    metrics, prediction_rows = evaluate_target(
        model,
        adapted_params,
        target_cells,
        grids,
        cfg,
        schema,
        source_cells,
    )
    domains = sorted(source_groups)
    weight_rows = [
        {
            "target_dataset": target,
            "source_dataset": domain,
            "weight": float(weight),
        }
        for domain, weight in zip(domains, weights)
    ]
    checkpoint = {
        "model_state": model.state_dict(),
        "adapted_parameters": {
            name: value.detach().cpu() for name, value in adapted_params.items()
        },
        "base_config": asdict(cfg),
        "meta_config": asdict(meta_cfg),
        "metadata_schema": schema.to_dict(),
        "target_dataset": target,
        "source_domains": domains,
        "target_weights": dict(zip(domains, map(float, weights))),
        "metrics": metrics,
    }
    torch.save(checkpoint, fold_dir / "checkpoint.pt")
    write_csv(fold_dir / "training_history.csv", history)
    write_csv(fold_dir / "source_weights.csv", weight_rows)
    write_csv(fold_dir / "target_predictions.csv", prediction_rows)
    (fold_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[result] target={target} "
        f"MAE_RUL={metrics['MAE_rul_cyc']:.2f} "
        f"RMSE_RUL={metrics['RMSE_rul_cyc']:.2f}",
        flush=True,
    )
    return {
        "target": target,
        **{key: value for key, value in metrics.items() if key != "stratified"},
        "fold_dir": str(fold_dir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--all-targets", action="store_true")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--cache-root", type=Path, default=Path("cache"))
    parser.add_argument(
        "--labels-path", type=Path, default=Path("data/1. Life lables")
    )
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--max-domains", type=int, default=0)
    parser.add_argument("--meta-steps", type=int, default=800)
    parser.add_argument("--inner-steps", type=int, default=1)
    parser.add_argument("--inner-lr", type=float, default=2e-3)
    parser.add_argument("--outer-lr", type=float, default=5e-4)
    parser.add_argument("--batch-per-domain", type=int, default=6)
    parser.add_argument("--target-context-size", type=int, default=8)
    parser.add_argument("--target-query-size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--fixed-temperature", action="store_true")
    parser.add_argument("--entropy-weight", type=float, default=1e-3)
    parser.add_argument(
        "--weight-mode",
        choices=("uniform", "metadata", "cycle", "hybrid"),
        default="hybrid",
        help="Information used to compute target-source weights",
    )
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--L-min", type=int, default=20)
    parser.add_argument("--L-max", type=int, default=100)
    parser.add_argument("--batch-target", type=int, default=32)
    parser.add_argument(
        "--label-source",
        choices=("auto", "official", "derived"),
        default="official",
    )
    args = parser.parse_args()
    if not args.all_targets and not args.target:
        parser.error("Specify --target DATASET or --all-targets")
    if args.all_targets and args.target:
        parser.error("Use only one of --target and --all-targets")
    if args.inner_steps < 1:
        parser.error("--inner-steps must be >= 1")
    if args.meta_steps < 1:
        parser.error("--meta-steps must be >= 1")
    return args


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    cache_root = (
        args.cache_root
        if args.cache_root.is_absolute()
        else project_root / args.cache_root
    )
    labels_path = (
        args.labels_path
        if args.labels_path.is_absolute()
        else project_root / args.labels_path
    )
    device = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    set_seed(args.seed)
    datasets = tuple(args.datasets or ())
    cfg = base.Config(
        cache_root=str(cache_root),
        labels_path=str(labels_path),
        label_source=args.label_source,
        datasets=datasets,
        device=device,
        seed=args.seed,
        L_min=args.L_min,
        L_max=args.L_max,
        batch_target=args.batch_target,
    )
    meta_cfg = MetaConfig(
        meta_steps=args.meta_steps,
        inner_steps=args.inner_steps,
        inner_lr=args.inner_lr,
        outer_lr=args.outer_lr,
        batch_per_domain=args.batch_per_domain,
        target_context_size=args.target_context_size,
        target_query_size=args.target_query_size,
        gate_temperature=args.temperature,
        learn_temperature=not args.fixed_temperature,
        entropy_weight=args.entropy_weight,
        weight_mode=args.weight_mode,
        log_every=args.log_every,
        seed=args.seed,
    )
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_root = (
        args.output_root
        if args.output_root.is_absolute()
        else project_root / args.output_root
    )
    run_dir = output_root / f"weighted_meta_lodo_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    base.setup_logging(run_dir / "data_loading.log")
    print(f"[device] {device}")
    print(f"[run] {run_dir}")
    cells, grids = base.load_cells(cfg)
    if args.label_source == "official":
        cells = enforce_official_labels(cells, labels_path, cfg.min_life)
    cells, grids = enforce_consistent_branches(cells, grids, cfg)
    available = sorted({cell.dataset for cell in cells})
    if args.max_domains > 0:
        available = available[: args.max_domains]
        cells = [cell for cell in cells if cell.dataset in available]
        grids = {key: value for key, value in grids.items() if key in available}
    targets = available if args.all_targets else [args.target]
    summaries = [
        run_fold(target, cells, grids, cfg, meta_cfg, run_dir)
        for target in targets
    ]
    write_csv(run_dir / "lodo_summary.csv", summaries)
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "base_config": asdict(cfg),
                "meta_config": asdict(meta_cfg),
                "targets": targets,
                "available_datasets": available,
                "protocol": (
                    "Target labels excluded from training; target metadata and "
                    "observed cycles used for unsupervised source weighting."
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[done] results={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
