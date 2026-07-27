#!/usr/bin/env python3
"""
Target-aware weighted meta-learning pilot experiment in one file.

What this script does
---------------------
1. Creates several synthetic 2D binary-classification source tasks.
2. Creates a target task with separate splits for:
   - transferability estimation
   - target adaptation
   - final testing
3. Trains a separate probe network on pooled source data.
4. Fits one linear head per source and one target head on the frozen probe backbone.
5. Computes paper-inspired transferability coefficients alpha* by solving

       min_alpha ||g_target - sum_i alpha_i g_i||^2
                 + variance_lambda * sum_i alpha_i^2 / n_i
       s.t. alpha_i >= 0, sum_i alpha_i = 1

   The candidate set includes the target head (alpha_0) and all source heads.
   Source-only meta-weights are obtained by renormalizing alpha_1,...,alpha_K.
6. Runs:
   - pooled supervised baseline
   - best-source supervised baseline
   - uniform/estimated-weight FOMAML with multiple rho values
   - oracle-weight FOMAML using the known synthetic angle distance

       w_i(rho) = (1-rho)/K + rho * normalized_source_alpha_i

   rho=0 is uniform FOMAML and rho=1 is fully transferability-weighted FOMAML.
7. Evaluates target accuracy after 0, 1, 5, and 10 adaptation steps.
8. Saves CSV/JSON summaries and plots.

Requirements
------------
- Python 3.9+
- PyTorch
- NumPy
- Matplotlib (optional; CSV files are still saved if unavailable)

Example
-------
python weighted_meta_transferability_experiment.py --device cuda

Quick smoke test
----------------
python weighted_meta_transferability_experiment.py \
    --device cuda --seeds 0 --meta-iters 20 --probe-steps 50 \
    --head-fit-steps 50 --supervised-steps 50
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import time
from collections import OrderedDict, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional
    plt = None


# -----------------------------------------------------------------------------
# Configuration and reproducibility
# -----------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    source_angles: List[float]
    target_angle: float
    source_size: int
    target_weight_size: int
    target_adapt_size: int
    target_test_size: int
    label_noise: float

    hidden_dim: int
    feature_dim: int

    probe_steps: int
    probe_batch_size: int
    probe_lr: float

    head_fit_steps: int
    head_fit_lr: float
    head_weight_decay: float

    variance_lambda: float
    qp_max_iter: int
    qp_tol: float

    meta_iters: int
    support_size: int
    query_size: int
    inner_steps: int
    inner_lr: float
    outer_lr: float
    grad_clip: float

    supervised_steps: int
    supervised_batch_size: int
    supervised_lr: float

    adaptation_steps: List[int]
    adaptation_lr: float

    rhos: List[float]
    oracle_temperature: float
    seeds: List[int]
    device: str
    output_dir: str


@dataclass
class TaskData:
    x: torch.Tensor
    y: torch.Tensor
    angle_deg: float
    name: str

    @property
    def n(self) -> int:
        return int(self.x.shape[0])


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("[warning] CUDA was requested but is unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


# -----------------------------------------------------------------------------
# Synthetic task generation
# -----------------------------------------------------------------------------


def generate_linear_task(
    angle_deg: float,
    n: int,
    label_noise: float,
    seed: int,
    name: str,
    device: torch.device,
) -> TaskData:
    """Generate x ~ N(0, I), y = 1[x dot w(angle) > 0] with optional flips."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    x = torch.randn(n, 2, generator=generator)
    angle_rad = math.radians(angle_deg)
    w = torch.tensor([math.cos(angle_rad), math.sin(angle_rad)], dtype=torch.float32)
    logits = x @ w
    y = (logits > 0).float()

    if label_noise > 0:
        flips = torch.rand(n, generator=generator) < label_noise
        y = torch.where(flips, 1.0 - y, y)

    return TaskData(
        x=x.to(device),
        y=y.to(device),
        angle_deg=float(angle_deg),
        name=name,
    )


def build_tasks(cfg: ExperimentConfig, seed: int, device: torch.device) -> Tuple[List[TaskData], TaskData, TaskData, TaskData]:
    sources: List[TaskData] = []
    for idx, angle in enumerate(cfg.source_angles):
        sources.append(
            generate_linear_task(
                angle_deg=angle,
                n=cfg.source_size,
                label_noise=cfg.label_noise,
                seed=seed * 10_000 + 100 + idx,
                name=f"source_{idx + 1}",
                device=device,
            )
        )

    target_weight = generate_linear_task(
        angle_deg=cfg.target_angle,
        n=cfg.target_weight_size,
        label_noise=cfg.label_noise,
        seed=seed * 10_000 + 1_001,
        name="target_weight",
        device=device,
    )
    target_adapt = generate_linear_task(
        angle_deg=cfg.target_angle,
        n=cfg.target_adapt_size,
        label_noise=cfg.label_noise,
        seed=seed * 10_000 + 1_002,
        name="target_adapt",
        device=device,
    )
    target_test = generate_linear_task(
        angle_deg=cfg.target_angle,
        n=cfg.target_test_size,
        label_noise=cfg.label_noise,
        seed=seed * 10_000 + 1_003,
        name="target_test",
        device=device,
    )
    return sources, target_weight, target_adapt, target_test


def sample_batch(task: TaskData, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if batch_size <= task.n:
        idx = torch.randperm(task.n, device=task.x.device)[:batch_size]
    else:
        idx = torch.randint(0, task.n, (batch_size,), device=task.x.device)
    return task.x[idx], task.y[idx]


def sample_support_query(
    task: TaskData,
    support_size: int,
    query_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    total = support_size + query_size
    if total <= task.n:
        idx = torch.randperm(task.n, device=task.x.device)[:total]
    else:
        idx = torch.randint(0, task.n, (total,), device=task.x.device)

    support_idx = idx[:support_size]
    query_idx = idx[support_size:]
    return task.x[support_idx], task.y[support_idx], task.x[query_idx], task.y[query_idx]


# -----------------------------------------------------------------------------
# Model and functional forward for FOMAML
# -----------------------------------------------------------------------------


class BinaryMLP(nn.Module):
    def __init__(self, hidden_dim: int = 32, feature_dim: int = 16) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, feature_dim)
        self.head = nn.Linear(feature_dim, 1)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.extract_features(x)
        return self.head(z).squeeze(-1)


def functional_forward(x: torch.Tensor, params: Mapping[str, torch.Tensor]) -> torch.Tensor:
    x = F.linear(x, params["fc1.weight"], params["fc1.bias"])
    x = F.relu(x)
    x = F.linear(x, params["fc2.weight"], params["fc2.bias"])
    x = F.relu(x)
    x = F.linear(x, params["head.weight"], params["head.bias"])
    return x.squeeze(-1)


def bce_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, y)


@torch.no_grad()
def accuracy(model: nn.Module, task: TaskData) -> float:
    logits = model(task.x)
    preds = (logits >= 0).float()
    return float((preds == task.y).float().mean().item())


# -----------------------------------------------------------------------------
# Probe network and task-head fitting
# -----------------------------------------------------------------------------


def concatenate_tasks(tasks: Sequence[TaskData]) -> TaskData:
    return TaskData(
        x=torch.cat([task.x for task in tasks], dim=0),
        y=torch.cat([task.y for task in tasks], dim=0),
        angle_deg=float("nan"),
        name="pooled_sources",
    )


def train_probe_network(
    cfg: ExperimentConfig,
    sources: Sequence[TaskData],
    device: torch.device,
) -> BinaryMLP:
    model = BinaryMLP(cfg.hidden_dim, cfg.feature_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.probe_lr)
    pooled = concatenate_tasks(sources)

    model.train()
    for _ in range(cfg.probe_steps):
        x, y = sample_batch(pooled, cfg.probe_batch_size)
        loss = bce_loss(model(x), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model


def fit_frozen_linear_head(
    probe_model: BinaryMLP,
    task: TaskData,
    cfg: ExperimentConfig,
) -> torch.Tensor:
    """Fit a binary linear head on frozen probe features and return [weight, bias]."""
    probe_model.eval()
    with torch.no_grad():
        features = probe_model.extract_features(task.x).detach()

    head = nn.Linear(cfg.feature_dim, 1).to(task.x.device)
    with torch.no_grad():
        head.weight.copy_(probe_model.head.weight)
        head.bias.copy_(probe_model.head.bias)

    optimizer = torch.optim.Adam(
        head.parameters(),
        lr=cfg.head_fit_lr,
        weight_decay=cfg.head_weight_decay,
    )

    for _ in range(cfg.head_fit_steps):
        logits = head(features).squeeze(-1)
        loss = bce_loss(logits, task.y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    vector = torch.cat([head.weight.detach().flatten(), head.bias.detach().flatten()])
    return vector


# -----------------------------------------------------------------------------
# Paper-inspired alpha* optimization
# -----------------------------------------------------------------------------


def project_to_simplex(v: torch.Tensor) -> torch.Tensor:
    """Euclidean projection onto {x >= 0, sum(x)=1}."""
    if v.ndim != 1:
        raise ValueError("Simplex projection expects a 1D tensor.")

    sorted_v, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(sorted_v, dim=0) - 1.0
    index = torch.arange(1, v.numel() + 1, device=v.device, dtype=v.dtype)
    condition = sorted_v - cssv / index > 0
    if not torch.any(condition):
        return torch.full_like(v, 1.0 / v.numel())

    rho = torch.nonzero(condition, as_tuple=False)[-1, 0]
    theta = cssv[rho] / (rho.to(v.dtype) + 1.0)
    return torch.clamp(v - theta, min=0.0)


def solve_transferability_qp(
    target_head: torch.Tensor,
    source_heads: Sequence[torch.Tensor],
    target_n: int,
    source_ns: Sequence[int],
    variance_lambda: float,
    max_iter: int,
    tol: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Solve a convex simplex-constrained quadratic problem.

    Candidate 0 is the target head itself. Candidates 1..K are source heads.

      min_a ||g_t - sum_i a_i g_i||^2 + lambda * sum_i a_i^2 / n_i
      s.t. a_i >= 0, sum_i a_i = 1

    This is a simplified, paper-inspired projected-head-distance objective.
    """
    candidates = torch.stack([target_head, *source_heads], dim=0).double()
    target = target_head.double()
    sample_sizes = torch.tensor([target_n, *source_ns], device=candidates.device, dtype=torch.double)

    # Scale all head vectors jointly for numerical stability. This does not change
    # the relative geometry; it only changes the balance with variance_lambda.
    scale = torch.sqrt(torch.mean(candidates.square())).clamp_min(1e-8)
    candidates = candidates / scale
    target = target / scale

    regularizer = variance_lambda * torch.diag(1.0 / sample_sizes)
    hessian_half = candidates @ candidates.T + regularizer
    linear_term = candidates @ target

    eig_max = torch.linalg.eigvalsh(hessian_half).max().clamp_min(1e-8)
    step_size = float(0.9 / (2.0 * eig_max).item())

    alpha = torch.full(
        (candidates.shape[0],),
        1.0 / candidates.shape[0],
        device=candidates.device,
        dtype=torch.double,
    )

    iterations = 0
    for iterations in range(1, max_iter + 1):
        grad = 2.0 * (hessian_half @ alpha - linear_term)
        updated = project_to_simplex(alpha - step_size * grad)
        if torch.max(torch.abs(updated - alpha)).item() < tol:
            alpha = updated
            break
        alpha = updated

    reconstruction = alpha @ candidates
    head_error = torch.sum((target - reconstruction) ** 2)
    variance_penalty = variance_lambda * torch.sum(alpha.square() / sample_sizes)
    objective = head_error + variance_penalty

    diagnostics = {
        "iterations": int(iterations),
        "objective": float(objective.item()),
        "head_error": float(head_error.item()),
        "variance_penalty": float(variance_penalty.item()),
        "step_size": float(step_size),
        "scale": float(scale.item()),
    }
    return alpha.float(), diagnostics


def normalized_source_weights(alpha_all: torch.Tensor) -> torch.Tensor:
    source_alpha = alpha_all[1:].clamp_min(0.0)
    total = source_alpha.sum()
    if float(total.item()) <= 1e-12:
        return torch.full_like(source_alpha, 1.0 / source_alpha.numel())
    return source_alpha / total


# -----------------------------------------------------------------------------
# FOMAML and supervised baselines
# -----------------------------------------------------------------------------


def first_order_adapted_parameters(
    model: BinaryMLP,
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    inner_steps: int,
    inner_lr: float,
) -> OrderedDict[str, torch.Tensor]:
    params: OrderedDict[str, torch.Tensor] = OrderedDict(model.named_parameters())

    for _ in range(inner_steps):
        support_logits = functional_forward(support_x, params)
        support_loss = bce_loss(support_logits, support_y)
        grads = torch.autograd.grad(
            support_loss,
            tuple(params.values()),
            create_graph=False,
            retain_graph=False,
        )
        # Detaching the inner gradients yields the first-order MAML approximation.
        params = OrderedDict(
            (name, param - inner_lr * grad.detach())
            for (name, param), grad in zip(params.items(), grads)
        )
    return params


def train_fomaml(
    cfg: ExperimentConfig,
    init_state: Mapping[str, torch.Tensor],
    sources: Sequence[TaskData],
    task_weights: torch.Tensor,
    device: torch.device,
) -> BinaryMLP:
    model = BinaryMLP(cfg.hidden_dim, cfg.feature_dim).to(device)
    model.load_state_dict(init_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.outer_lr)

    weights = task_weights.to(device=device, dtype=torch.float32)
    weights = weights / weights.sum().clamp_min(1e-12)

    model.train()
    for iteration in range(1, cfg.meta_iters + 1):
        meta_loss = torch.zeros((), device=device)

        for task_idx, task in enumerate(sources):
            sx, sy, qx, qy = sample_support_query(
                task,
                support_size=cfg.support_size,
                query_size=cfg.query_size,
            )
            adapted = first_order_adapted_parameters(
                model,
                sx,
                sy,
                inner_steps=cfg.inner_steps,
                inner_lr=cfg.inner_lr,
            )
            query_logits = functional_forward(qx, adapted)
            query_loss = bce_loss(query_logits, qy)
            meta_loss = meta_loss + weights[task_idx] * query_loss

        optimizer.zero_grad(set_to_none=True)
        meta_loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

    return model


def train_supervised_baseline(
    cfg: ExperimentConfig,
    init_state: Mapping[str, torch.Tensor],
    train_task: TaskData,
    device: torch.device,
) -> BinaryMLP:
    model = BinaryMLP(cfg.hidden_dim, cfg.feature_dim).to(device)
    model.load_state_dict(init_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.supervised_lr)

    model.train()
    for _ in range(cfg.supervised_steps):
        x, y = sample_batch(train_task, cfg.supervised_batch_size)
        loss = bce_loss(model(x), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
    return model


# -----------------------------------------------------------------------------
# Target adaptation and evaluation
# -----------------------------------------------------------------------------


def evaluate_adaptation_curve(
    cfg: ExperimentConfig,
    trained_state: Mapping[str, torch.Tensor],
    target_adapt: TaskData,
    target_test: TaskData,
    device: torch.device,
) -> Dict[int, float]:
    model = BinaryMLP(cfg.hidden_dim, cfg.feature_dim).to(device)
    model.load_state_dict(trained_state)

    requested_steps = sorted(set(cfg.adaptation_steps))
    if not requested_steps or requested_steps[0] < 0:
        raise ValueError("adaptation_steps must contain non-negative integers.")

    max_steps = max(requested_steps)
    results: Dict[int, float] = {}
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.adaptation_lr)

    for step in range(max_steps + 1):
        if step in requested_steps:
            model.eval()
            results[step] = accuracy(model, target_test)

        if step == max_steps:
            break

        model.train()
        logits = model(target_adapt.x)
        loss = bce_loss(logits, target_adapt.y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return results


# -----------------------------------------------------------------------------
# Saving and plotting
# -----------------------------------------------------------------------------


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_results(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    grouped: MutableMapping[Tuple[str, int], List[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), int(row["adaptation_step"]))].append(float(row["accuracy"]))

    summary: List[Dict[str, object]] = []
    for (method, step), values in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        arr = np.asarray(values, dtype=np.float64)
        summary.append(
            {
                "method": method,
                "adaptation_step": step,
                "mean_accuracy": float(arr.mean()),
                "std_accuracy": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                "num_seeds": int(arr.size),
            }
        )
    return summary


def plot_alpha(alpha_rows: Sequence[Mapping[str, object]], output_dir: Path) -> None:
    if plt is None or not alpha_rows:
        return

    by_source: MutableMapping[int, List[float]] = defaultdict(list)
    source_angles: Dict[int, float] = {}
    distances: Dict[int, float] = {}
    for row in alpha_rows:
        source_idx = int(row["source_index"])
        by_source[source_idx].append(float(row["source_meta_weight"]))
        source_angles[source_idx] = float(row["source_angle"])
        distances[source_idx] = float(row["angle_distance"])

    indices = sorted(by_source)
    means = [float(np.mean(by_source[i])) for i in indices]
    stds = [float(np.std(by_source[i], ddof=1)) if len(by_source[i]) > 1 else 0.0 for i in indices]
    labels = [f"S{i}\n{source_angles[i]:g}°\nΔ={distances[i]:g}°" for i in indices]

    plt.figure(figsize=(9, 5))
    plt.bar(labels, means, yerr=stds, capsize=4)
    plt.ylabel("Normalized source meta-weight")
    plt.title("Transferability-derived source weights")
    plt.tight_layout()
    plt.savefig(output_dir / "alpha_source_weights.png", dpi=180)
    plt.close()


def plot_accuracy_summary(summary_rows: Sequence[Mapping[str, object]], output_dir: Path) -> None:
    if plt is None or not summary_rows:
        return

    by_method: MutableMapping[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in summary_rows:
        by_method[str(row["method"])].append(row)

    plt.figure(figsize=(10, 6))
    for method, rows in sorted(by_method.items()):
        ordered = sorted(rows, key=lambda row: int(row["adaptation_step"]))
        steps = [int(row["adaptation_step"]) for row in ordered]
        means = [float(row["mean_accuracy"]) for row in ordered]
        stds = [float(row["std_accuracy"]) for row in ordered]
        plt.errorbar(steps, means, yerr=stds, marker="o", capsize=3, label=method)

    plt.xlabel("Target adaptation steps")
    plt.ylabel("Target test accuracy")
    plt.ylim(0.45, 1.02)
    plt.xticks(sorted({int(row["adaptation_step"]) for row in summary_rows}))
    plt.grid(alpha=0.25)
    plt.legend()
    plt.title("Target adaptation performance")
    plt.tight_layout()
    plt.savefig(output_dir / "target_adaptation_accuracy.png", dpi=180)
    plt.close()


def print_seed_alpha_table(
    cfg: ExperimentConfig,
    seed: int,
    alpha_all: torch.Tensor,
    source_weights: torch.Tensor,
    diagnostics: Mapping[str, float],
) -> None:
    print(f"\n[seed {seed}] alpha_0(target) = {float(alpha_all[0]):.4f}")
    print("source | angle | |angle-target| | raw alpha | meta weight")
    print("-------+-------+----------------+-----------+------------")
    for idx, (angle, raw_alpha, meta_weight) in enumerate(
        zip(cfg.source_angles, alpha_all[1:].tolist(), source_weights.tolist()),
        start=1,
    ):
        distance = abs(float(angle) - float(cfg.target_angle))
        print(
            f"S{idx:<5} | {angle:>5.1f} | {distance:>14.1f} | "
            f"{raw_alpha:>9.4f} | {meta_weight:>10.4f}"
        )
    print(
        "QP diagnostics: "
        f"objective={diagnostics['objective']:.6f}, "
        f"head_error={diagnostics['head_error']:.6f}, "
        f"variance={diagnostics['variance_penalty']:.6f}, "
        f"iterations={int(diagnostics['iterations'])}"
    )


# -----------------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------------


def run_experiment(cfg: ExperimentConfig) -> None:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(cfg.device)
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(device)}")

    with (output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(asdict(cfg), file, indent=2, ensure_ascii=False)

    all_result_rows: List[Dict[str, object]] = []
    all_alpha_rows: List[Dict[str, object]] = []
    qp_rows: List[Dict[str, object]] = []

    experiment_start = time.time()

    for seed in cfg.seeds:
        seed_start = time.time()
        set_global_seed(seed)
        sources, target_weight, target_adapt, target_test = build_tasks(cfg, seed, device)

        # A separate probe model estimates head-level task transferability.
        probe_model = train_probe_network(cfg, sources, device)
        target_head = fit_frozen_linear_head(probe_model, target_weight, cfg)
        source_heads = [fit_frozen_linear_head(probe_model, task, cfg) for task in sources]

        alpha_all, diagnostics = solve_transferability_qp(
            target_head=target_head,
            source_heads=source_heads,
            target_n=target_weight.n,
            source_ns=[task.n for task in sources],
            variance_lambda=cfg.variance_lambda,
            max_iter=cfg.qp_max_iter,
            tol=cfg.qp_tol,
        )
        source_weights = normalized_source_weights(alpha_all)
        print_seed_alpha_table(cfg, seed, alpha_all, source_weights, diagnostics)

        qp_rows.append({"seed": seed, **diagnostics, "alpha_target": float(alpha_all[0].item())})
        for idx, angle in enumerate(cfg.source_angles):
            all_alpha_rows.append(
                {
                    "seed": seed,
                    "source_index": idx + 1,
                    "source_angle": float(angle),
                    "target_angle": float(cfg.target_angle),
                    "angle_distance": abs(float(angle) - float(cfg.target_angle)),
                    "raw_alpha": float(alpha_all[idx + 1].item()),
                    "source_meta_weight": float(source_weights[idx].item()),
                }
            )

        # Every compared model starts from exactly the same fresh initialization.
        set_global_seed(seed + 50_000)
        base_model = BinaryMLP(cfg.hidden_dim, cfg.feature_dim).to(device)
        init_state = copy.deepcopy(base_model.state_dict())

        trained_models: Dict[str, BinaryMLP] = {}

        pooled_task = concatenate_tasks(sources)
        trained_models["pooled_supervised"] = train_supervised_baseline(
            cfg, init_state, pooled_task, device
        )

        best_source_idx = int(torch.argmax(source_weights).item())
        trained_models[f"best_source_S{best_source_idx + 1}"] = train_supervised_baseline(
            cfg, init_state, sources[best_source_idx], device
        )

        uniform = torch.full_like(source_weights, 1.0 / len(sources))

        # Oracle weighting is available only because this is a synthetic experiment.
        # It separates "is task weighting useful?" from "is alpha estimation accurate?".
        angle_distances = torch.tensor(
            [abs(float(angle) - float(cfg.target_angle)) for angle in cfg.source_angles],
            device=device,
            dtype=torch.float32,
        )
        oracle_weights = torch.softmax(
            -angle_distances / max(cfg.oracle_temperature, 1e-8), dim=0
        )
        trained_models["fomaml_oracle"] = train_fomaml(
            cfg=cfg,
            init_state=init_state,
            sources=sources,
            task_weights=oracle_weights,
            device=device,
        )

        for rho in cfg.rhos:
            if not 0.0 <= rho <= 1.0:
                raise ValueError(f"rho must lie in [0,1], got {rho}")
            mixed_weights = (1.0 - rho) * uniform + rho * source_weights
            method_name = f"fomaml_rho_{rho:.2f}"
            trained_models[method_name] = train_fomaml(
                cfg=cfg,
                init_state=init_state,
                sources=sources,
                task_weights=mixed_weights,
                device=device,
            )

        for method_name, model in trained_models.items():
            curve = evaluate_adaptation_curve(
                cfg=cfg,
                trained_state=model.state_dict(),
                target_adapt=target_adapt,
                target_test=target_test,
                device=device,
            )
            print(
                f"[seed {seed}] {method_name}: "
                + ", ".join(f"step {step}={acc:.4f}" for step, acc in sorted(curve.items()))
            )
            for step, acc in sorted(curve.items()):
                all_result_rows.append(
                    {
                        "seed": seed,
                        "method": method_name,
                        "adaptation_step": int(step),
                        "accuracy": float(acc),
                    }
                )

        print(f"[seed {seed}] elapsed: {time.time() - seed_start:.1f}s")

    summary_rows = summarize_results(all_result_rows)

    write_csv(output_dir / "results_long.csv", all_result_rows)
    write_csv(output_dir / "results_summary.csv", summary_rows)
    write_csv(output_dir / "alpha_weights.csv", all_alpha_rows)
    write_csv(output_dir / "qp_diagnostics.csv", qp_rows)

    plot_alpha(all_alpha_rows, output_dir)
    plot_accuracy_summary(summary_rows, output_dir)

    print("\n=== Mean target accuracies ===")
    for row in summary_rows:
        print(
            f"{row['method']:<24} | step={int(row['adaptation_step']):>2} | "
            f"mean={float(row['mean_accuracy']):.4f} | "
            f"std={float(row['std_accuracy']):.4f}"
        )

    print(f"\nSaved results to: {output_dir.resolve()}")
    print(f"Total elapsed: {time.time() - experiment_start:.1f}s")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(
        description="One-file pilot experiment for transferability-weighted meta-learning."
    )

    parser.add_argument("--source-angles", nargs="+", type=float, default=[0, 15, 40, 70, 100])
    parser.add_argument("--target-angle", type=float, default=20.0)
    parser.add_argument("--source-size", type=int, default=1000)
    parser.add_argument("--target-weight-size", type=int, default=20)
    parser.add_argument("--target-adapt-size", type=int, default=20)
    parser.add_argument("--target-test-size", type=int, default=2000)
    parser.add_argument("--label-noise", type=float, default=0.03)

    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--feature-dim", type=int, default=16)

    parser.add_argument("--probe-steps", type=int, default=600)
    parser.add_argument("--probe-batch-size", type=int, default=128)
    parser.add_argument("--probe-lr", type=float, default=1e-3)

    parser.add_argument("--head-fit-steps", type=int, default=300)
    parser.add_argument("--head-fit-lr", type=float, default=3e-2)
    parser.add_argument("--head-weight-decay", type=float, default=1e-3)

    parser.add_argument("--variance-lambda", type=float, default=1.0)
    parser.add_argument("--qp-max-iter", type=int, default=20_000)
    parser.add_argument("--qp-tol", type=float, default=1e-10)

    parser.add_argument("--meta-iters", type=int, default=600)
    parser.add_argument("--support-size", type=int, default=10)
    parser.add_argument("--query-size", type=int, default=20)
    parser.add_argument("--inner-steps", type=int, default=1)
    parser.add_argument("--inner-lr", type=float, default=0.05)
    parser.add_argument("--outer-lr", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=5.0)

    parser.add_argument("--supervised-steps", type=int, default=1500)
    parser.add_argument("--supervised-batch-size", type=int, default=128)
    parser.add_argument("--supervised-lr", type=float, default=1e-3)

    parser.add_argument("--adaptation-steps", nargs="+", type=int, default=[0, 1, 5, 10])
    parser.add_argument("--adaptation-lr", type=float, default=0.05)

    parser.add_argument("--rhos", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument(
        "--oracle-temperature",
        type=float,
        default=20.0,
        help="Softmax temperature in degrees for the synthetic oracle task weights.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="weighted_meta_results")

    args = parser.parse_args()

    if args.target_weight_size < 2 or args.target_adapt_size < 2:
        raise ValueError("Target weight/adaptation sets should each contain at least 2 samples.")
    if args.support_size < 1 or args.query_size < 1:
        raise ValueError("support_size and query_size must be positive.")
    if not args.source_angles:
        raise ValueError("At least one source task is required.")

    return ExperimentConfig(
        source_angles=list(args.source_angles),
        target_angle=args.target_angle,
        source_size=args.source_size,
        target_weight_size=args.target_weight_size,
        target_adapt_size=args.target_adapt_size,
        target_test_size=args.target_test_size,
        label_noise=args.label_noise,
        hidden_dim=args.hidden_dim,
        feature_dim=args.feature_dim,
        probe_steps=args.probe_steps,
        probe_batch_size=args.probe_batch_size,
        probe_lr=args.probe_lr,
        head_fit_steps=args.head_fit_steps,
        head_fit_lr=args.head_fit_lr,
        head_weight_decay=args.head_weight_decay,
        variance_lambda=args.variance_lambda,
        qp_max_iter=args.qp_max_iter,
        qp_tol=args.qp_tol,
        meta_iters=args.meta_iters,
        support_size=args.support_size,
        query_size=args.query_size,
        inner_steps=args.inner_steps,
        inner_lr=args.inner_lr,
        outer_lr=args.outer_lr,
        grad_clip=args.grad_clip,
        supervised_steps=args.supervised_steps,
        supervised_batch_size=args.supervised_batch_size,
        supervised_lr=args.supervised_lr,
        adaptation_steps=list(args.adaptation_steps),
        adaptation_lr=args.adaptation_lr,
        rhos=list(args.rhos),
        oracle_temperature=args.oracle_temperature,
        seeds=list(args.seeds),
        device=args.device,
        output_dir=args.output_dir,
    )


def main() -> None:
    cfg = parse_args()
    run_experiment(cfg)


if __name__ == "__main__":
    main()
