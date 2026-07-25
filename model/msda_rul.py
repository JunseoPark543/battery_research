#!/usr/bin/env python3
# =============================================================================
# msda_rul.py
#
# Multi-source domain adaptation for battery cycle-life prediction from
# variable-length early-life windows.
#
#   data/<DATASET>/<cell>.pkl   ->  prepare (cache)  ->  train  ->  evaluate
#
# Run everything:      python model/msda_rul.py all
# Individual stages:   prepare | diagnose | baseline | train | evaluate | selftest
#
# Sections
#   01 IMPORTS
#   02 CONFIG
#   03 LOGGING
#   04 CACHE IO
#   05 FEATURES
#   06 SPLIT
#   07 BATCHING
#   08 CYCLE ENCODER
#   09 REFERENCE ATTENTION
#   10 HEAD
#   11 MODEL
#   12 LOSSES
#   13 BASELINES
#   14 TRAIN
#   15 EVAL
#   16 DIAGNOSTICS
#   17 TESTS
#   18 CLI
# =============================================================================

# ===== 01. IMPORTS ===========================================================
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import pickle
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None

LOG = logging.getLogger("msda")


# ===== 02. CONFIG ============================================================
@dataclass(frozen=True)
class Config:
    # -- paths ---------------------------------------------------------------
    data_root: str = "data"
    cache_root: str = "cache"
    run_root: str = "runs"
    labels_path: str = "data/1. Life labels"   # BatteryLife "Life labels"
    label_source: str = "auto"      # auto | official | derived

    # -- feature extraction --------------------------------------------------
    grid_points: int = 128          # |V|
    ref_cycle: int = 10             # baseline cycle for the difference curve
    cache_max_cycle: int = 300      # cycles cached per cell (>= L_max + margin)
    seg_min_points: int = 30        # min points for a stage to count as CC
    crate_bin: float = 0.25         # C-rate quantisation when finding CC stages
    seg_min_vspan: float = 0.05     # V; rejects the CV taper, which has no span
    datasets: Tuple[str, ...] = ()  # empty = every folder found under data/
    charge_segment: bool = True     # fallback when a dataset has no profile
    relative_q: bool = True         # subtract Q at the segment start
    grid_inset: float = 0.02        # shrink common voltage window by this frac

    # -- labels / protocol ---------------------------------------------------
    eol_frac: float = 0.80          # of the reference capacity below
    eol_reference: str = "nominal"  # nominal | initial -- the official
    #   BatteryLife labels use NOMINAL capacity. Verified on HUST_1-1: the
    #   official life is 1542 while the stored record stops at cycle 1504 with
    #   0.8942 Ah, and extrapolating the late fade rate to 0.88 Ah (= 80% of the
    #   1.1 Ah rating) lands exactly on 1542. Deriving a life from the stored
    #   trace therefore CANNOT reproduce the official label for such cells.
    init_cycles: int = 5            # cycles averaged for initial capacity
    min_life: int = 150             # drop cells with shorter life
    L_min: int = 20
    L_max: int = 100
    truncate_source: bool = True    # source cells get the same L_i distribution
    seed: int = 0
    n_seeds: int = 1                # repeats per fold

    # -- reference grid ------------------------------------------------------
    ref_grid: Tuple[int, ...] = (20, 35, 50, 65, 80)

    # -- model ---------------------------------------------------------------
    d_cycle: int = 16               # cycle feature width (rank of dQ is tiny)
    enc_hidden: int = 32
    d_attn: int = 16
    n_fourier: int = 8
    m_bottleneck: int = 8
    head_hidden: int = 64
    dropout: float = 0.2

    # -- losses --------------------------------------------------------------
    lambda_align: float = 0.0       # start at 0; turn on only after a baseline
    align_warmup: int = 300
    gamma_ssl: float = 0.0

    # -- optimisation --------------------------------------------------------
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    batch_source: int = 32
    batch_target: int = 16
    max_steps: int = 1500
    log_every: int = 50
    eval_every: int = 100
    patience: int = 400
    val_frac: float = 0.15

    # -- misc ----------------------------------------------------------------
    device: str = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
    strat_edges: Tuple[int, ...] = (20, 40, 60, 80, 101)

    def ref_grid_np(self) -> np.ndarray:
        return np.asarray(self.ref_grid, dtype=np.float32)


# Which branch of the cycle is comparable ACROSS CELLS differs per dataset and
# is the single most consequential preprocessing choice here. MATR holds the
# discharge fixed while varying the fast-charge protocol; HUST does the exact
# opposite. Using the wrong branch makes the model encode the protocol instead
# of the degradation.
DATASET_PROFILES: Dict[str, Dict] = {
    "MATR":     {"segment": "discharge"},   # 81 charge protocols, 1 discharge
    "MATR1":    {"segment": "discharge"},
    "MATR2":    {"segment": "discharge"},
    "CLO":      {"segment": "discharge"},
    "HUST":     {"segment": "charge"},      # 77 discharge protocols, 1 charge
    "RWTH":     {"segment": "charge"},      # single protocol for all cells
    "UL_PUR":   {"segment": "charge"},
    "HNEI":     {"segment": "charge"},
    "CALCE":    {"segment": "charge"},
    "STANFORD": {"segment": "charge"},
    "SNL":      {"segment": "auto"},        # temperature / DOD / rate vary
    "TONGJI":   {"segment": "auto"},
    "MICH":     {"segment": "auto"},
    "MICH_EXP": {"segment": "auto"},
    "XJTU":     {"segment": "auto"},
    "ISU_ILCC": {"segment": "auto"},
}
DEFAULT_PROFILE = {"segment": "auto"}


def profile_for(dataset: str) -> Dict:
    return DATASET_PROFILES.get(dataset.upper().replace("-", "_"), DEFAULT_PROFILE)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ===== 03. LOGGING ===========================================================
class _Fmt(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        record.elapsed = f"{time.time() - _T0:7.1f}s"
        return super().format(record)


_T0 = time.time()


def setup_logging(logfile: Optional[Path] = None, level: int = logging.INFO) -> None:
    LOG.setLevel(level)
    LOG.handlers.clear()
    fmt = _Fmt("[%(elapsed)s] %(levelname)-5s %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    LOG.addHandler(sh)
    if logfile is not None:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile, encoding="utf-8")
        fh.setFormatter(fmt)
        LOG.addHandler(fh)
    LOG.propagate = False


class Stage:
    """Context manager that brackets a pipeline stage in the log."""

    def __init__(self, title: str, index: str = ""):
        self.title = title
        self.index = index
        self.t0 = 0.0

    def __enter__(self) -> "Stage":
        self.t0 = time.time()
        tag = f"{self.index} " if self.index else ""
        LOG.info("=" * 72)
        LOG.info(f"{tag}{self.title}")
        LOG.info("=" * 72)
        return self

    def __exit__(self, *exc) -> None:
        LOG.info(f"-- done ({time.time() - self.t0:.1f}s): {self.title}")


def log_table(rows: List[Sequence], header: Sequence[str]) -> None:
    cols = [len(str(h)) for h in header]
    for r in rows:
        for i, v in enumerate(r):
            cols[i] = max(cols[i], len(str(v)))
    line = "  ".join(str(h).ljust(cols[i]) for i, h in enumerate(header))
    LOG.info("  " + line)
    LOG.info("  " + "-" * len(line))
    for r in rows:
        LOG.info("  " + "  ".join(str(v).ljust(cols[i]) for i, v in enumerate(r)))


# ===== 04. CACHE IO ==========================================================
@dataclass
class CellCache:
    """Per-cell cached features. Small: ~O(cache_max_cycle x grid_points)."""
    cell_id: str
    dataset: str
    cathode: str
    nominal_ah: float
    v_lo: float
    v_hi: float
    grid: np.ndarray          # (G,) per-cell voltage grid
    cycles: np.ndarray        # (n,) absolute cycle numbers with valid curves
    dq: np.ndarray            # (n, G) difference curves vs ref cycle
    capacity: np.ndarray      # (N,) discharge capacity for every recorded cycle
    cap_cycles: np.ndarray    # (N,) cycle numbers for capacity
    c_eol: Optional[int]
    initial_capacity: float
    branch: str = "charge"

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            meta=json.dumps({
                "cell_id": self.cell_id, "dataset": self.dataset,
                "cathode": self.cathode, "nominal_ah": self.nominal_ah,
                "v_lo": self.v_lo, "v_hi": self.v_hi,
                "c_eol": self.c_eol, "initial_capacity": self.initial_capacity,
                "branch": self.branch,
            }),
            grid=self.grid, cycles=self.cycles, dq=self.dq,
            capacity=self.capacity, cap_cycles=self.cap_cycles,
        )

    @staticmethod
    def load(path: Path) -> "CellCache":
        z = np.load(path, allow_pickle=False)
        m = json.loads(str(z["meta"]))
        return CellCache(
            cell_id=m["cell_id"], dataset=m["dataset"], cathode=m["cathode"],
            nominal_ah=m["nominal_ah"], v_lo=m["v_lo"], v_hi=m["v_hi"],
            grid=z["grid"], cycles=z["cycles"], dq=z["dq"],
            capacity=z["capacity"], cap_cycles=z["cap_cycles"],
            c_eol=m["c_eol"], initial_capacity=m["initial_capacity"],
            branch=m.get("branch", "charge"),
        )


_LIFE_KEYS = ("life", "cycle_life", "label", "eol", "rul", "life_label",
              "cycle_life_label", "y")


def _norm_id(x: str) -> str:
    """Label files key cells by FILENAME ('HUST_1-1.pkl'), the caches key them
    by cell_id ('HUST_1-1'), so the extension has to go."""
    t = str(x).strip()
    for ext in (".pkl", ".pickle", ".json", ".csv"):
        if t.lower().endswith(ext):
            t = t[: -len(ext)]
            break
    return t.upper().replace(" ", "").replace("-", "_")


def _flatten_labels(obj, out: Dict[str, float]) -> None:
    """Accept {cell: life}, {dataset: {cell: life}}, {cell: {life: ...}}, or a
    list of records. The release layout has changed between versions, so this
    stays deliberately permissive."""
    if isinstance(obj, (list, tuple)):
        for item in obj:
            if isinstance(item, dict):
                cid = next((item[k] for k in
                            ("cell_id", "cell", "battery", "id", "name")
                            if k in item), None)
                val = next((item[k] for k in _LIFE_KEYS if k in item), None)
                if cid is not None and isinstance(val, (int, float)):
                    out[_norm_id(cid)] = float(val)
        return
    if not isinstance(obj, dict):
        return
    for k, v in obj.items():
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            out[_norm_id(k)] = float(v)
        elif isinstance(v, dict):
            hit = next((kk for kk in v if str(kk).lower() in _LIFE_KEYS), None)
            if hit is not None and isinstance(v[hit], (int, float)):
                out[_norm_id(k)] = float(v[hit])
            else:
                _flatten_labels(v, out)
        elif isinstance(v, (list, tuple, np.ndarray)):
            arr = np.asarray(v).ravel()
            if arr.size == 1 and np.issubdtype(arr.dtype, np.number):
                out[_norm_id(k)] = float(arr[0])
            else:
                _flatten_labels(v, out)


def _dataset_hint(fname: str) -> str:
    """'ISU-ILCC_labels.json' -> 'ISU_ILCC'."""
    stem = Path(fname).stem
    for suf in ("_labels", "_label", "-labels"):
        if stem.lower().endswith(suf):
            stem = stem[: -len(suf)]
            break
    return _norm_id(stem)


def load_life_labels(path: Path) -> Dict[str, Dict[str, float]]:
    """Returns {DATASET_HINT: {CELL_ID: life}}.

    Scoping by dataset matters: cell identifiers are only unique within a
    dataset, so a flat lookup can silently attach the wrong life to a cell.
    """
    if not path.exists():
        LOG.warning(f"no label folder at '{path}' -- lives will be derived "
                    f"from the capacity trace instead")
        return {}
    files = sorted(f for f in (path.rglob("*") if path.is_dir() else [path])
                   if f.is_file() and f.suffix.lower() in
                   (".json", ".pkl", ".pickle", ".csv"))
    scoped: Dict[str, Dict[str, float]] = {}
    for f in files:
        one: Dict[str, float] = {}
        try:
            if f.suffix.lower() == ".json":
                _flatten_labels(json.loads(f.read_text(encoding="utf-8")), one)
            elif f.suffix.lower() in (".pkl", ".pickle"):
                with open(f, "rb") as fh:
                    _flatten_labels(pickle.load(fh), one)
            else:
                import csv as _csv
                with open(f, newline="", encoding="utf-8") as fh:
                    _flatten_labels(list(_csv.DictReader(fh)), one)
        except Exception as e:                                   # noqa: BLE001
            LOG.warning(f"  could not read {f.name}: {e}")
            continue
        if not one:
            LOG.warning(f"  {f.name}: parsed 0 labels -- unexpected structure")
            continue
        hint = _dataset_hint(f.name)
        scoped.setdefault(hint, {}).update(one)
        ex = list(one.items())[:2]
        LOG.info(f"  {f.name}: {len(one)} labels (hint={hint}) e.g. {ex}")
    LOG.info(f"official life labels: {sum(len(v) for v in scoped.values())} "
             f"entries across {len(scoped)} files")
    return scoped


def lookup_life(scoped: Dict[str, Dict[str, float]], dataset: str,
                cell_id: str) -> Optional[float]:
    """Dataset-scoped first, then any file. Returns None if unmatched."""
    cid = _norm_id(cell_id)
    ds = _norm_id(dataset)
    for hint, table in scoped.items():
        if hint == ds and cid in table:
            return table[cid]
    for hint, table in scoped.items():
        if (hint.startswith(ds) or ds.startswith(hint)) and cid in table:
            return table[cid]
    hits = [t[cid] for t in scoped.values() if cid in t]
    if len(hits) == 1:
        return hits[0]
    return None


def list_datasets(data_root: Path, wanted: Sequence[str] = ()) -> List[str]:
    if not data_root.exists():
        return []
    found = sorted(p.name for p in data_root.iterdir()
                   if p.is_dir() and any(p.glob("*.pkl")))
    if not wanted:
        return found
    lut = {f.upper(): f for f in found}
    out, missing = [], []
    for w in wanted:
        key = w.upper()
        (out.append(lut[key]) if key in lut else missing.append(w))
    if missing:
        LOG.warning(f"requested datasets not present under {data_root}: {missing}")
    return out


# ===== 05. FEATURES ==========================================================
def _cc_segment(cycle: dict, nominal_ah: float, cfg: Config, charge: bool
                ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Lowest-rate constant-current stage of the charge (or discharge) step.

    Lowest rate is chosen deliberately: polarisation is smallest there, so the
    Q(V) shape is best resolved. For HUST this selects the 1C charge stage,
    which is identical across cells (the discharge protocol is not).
    """
    I = np.asarray(cycle.get("current_in_A") or [], dtype=np.float64)
    V = np.asarray(cycle.get("voltage_in_V") or [], dtype=np.float64)
    key = "charge_capacity_in_Ah" if charge else "discharge_capacity_in_Ah"
    Q = np.asarray(cycle.get(key) or [], dtype=np.float64)
    if I.size == 0 or I.size != V.size or I.size != Q.size:
        return None

    sign = 1.0 if charge else -1.0
    idx = np.nonzero(sign * I > 0.02 * max(nominal_ah, 1e-6))[0]
    if idx.size < cfg.seg_min_points:
        return None

    crate = np.abs(I[idx]) / max(nominal_ah, 1e-6)
    binned = np.round(crate / cfg.crate_bin) * cfg.crate_bin

    # A candidate stage needs enough points AND a real voltage span. The span
    # test is what rejects the constant-voltage taper: it sits at a single
    # voltage while its current decays through every low C-rate bin, so without
    # it the "lowest rate" stage is the CV hold and Q(V) is undefined.
    best = None
    for lvl in np.unique(binned):
        sel = idx[binned == lvl]
        if sel.size < cfg.seg_min_points:
            continue
        if float(V[sel].max() - V[sel].min()) < cfg.seg_min_vspan:
            continue
        if best is None or lvl < best[0]:
            best = (float(lvl), sel)
    if best is None:
        return None
    return V[best[1]], Q[best[1]]


def _qv(V: np.ndarray, Q: np.ndarray, grid: np.ndarray,
        relative: bool) -> Optional[np.ndarray]:
    o = np.argsort(V)
    V, Q = V[o], Q[o]
    V, ix = np.unique(V, return_index=True)
    Q = Q[ix]
    if V.size < 10 or V[0] > grid[0] or V[-1] < grid[-1]:
        return None
    q = np.interp(grid, V, Q)
    return q - q[0] if relative else q


def _extract_branch(cd: list, nominal: float, cfg: Config, charge: bool):
    """Grid + difference curves for one branch (charge or discharge).

    Returns (grid, cycle_numbers, dq, lo, hi) or None.
    """
    horizon = cfg.cache_max_cycle
    lo, hi = -np.inf, np.inf
    segs: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for c in cd:
        n = int(c["cycle_number"])
        if n > horizon:
            continue
        seg = _cc_segment(c, nominal, cfg, charge)
        if seg is None:
            continue
        segs[n] = seg
        lo = max(lo, float(seg[0].min()))
        hi = min(hi, float(seg[0].max()))
    if not segs or not np.isfinite(lo) or not np.isfinite(hi):
        return None
    span = hi - lo
    if span < cfg.seg_min_vspan:
        return None
    lo += cfg.grid_inset * span
    hi -= cfg.grid_inset * span
    grid = np.linspace(lo, hi, cfg.grid_points).astype(np.float32)

    qv = {n: _qv(v, q, grid, cfg.relative_q) for n, (v, q) in segs.items()}
    qv = {n: a for n, a in qv.items() if a is not None}
    if cfg.ref_cycle not in qv:
        return None
    ref = qv[cfg.ref_cycle]
    nums = np.array(sorted(n for n in qv if n > cfg.ref_cycle), dtype=np.int64)
    if nums.size < 5:
        return None
    dq = np.stack([qv[n] - ref for n in nums]).astype(np.float32)
    return grid, nums, dq, float(lo), float(hi)


def build_cell_cache(pkl_path: Path, dataset: str, cfg: Config) -> Optional[CellCache]:
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    cd = d.get("cycle_data") or []
    if len(cd) < cfg.min_life // 2:
        LOG.warning(f"    skip {pkl_path.name}: only {len(cd)} cycles")
        return None

    nominal = float(d.get("nominal_capacity_in_Ah") or 1.0)

    # --- capacity trajectory over the whole record (cheap) ------------------
    cap, cap_cyc = [], []
    for c in cd:
        q = c.get("discharge_capacity_in_Ah")
        if q is None or len(q) == 0:
            continue
        cap.append(float(np.nanmax(np.asarray(q, dtype=np.float64))))
        cap_cyc.append(int(c["cycle_number"]))
    if len(cap) < cfg.init_cycles + 1:
        LOG.warning(f"    skip {pkl_path.name}: no capacity trace")
        return None
    cap = np.asarray(cap, dtype=np.float32)
    cap_cyc = np.asarray(cap_cyc, dtype=np.int64)

    initial = float(np.mean(cap[: cfg.init_cycles]))
    ref_cap = nominal if cfg.eol_reference == "nominal" else initial
    below = np.nonzero(cap < cfg.eol_frac * ref_cap)[0]
    c_eol = int(cap_cyc[below[0]]) if below.size else None

    # --- pick the branch that is comparable across cells of this dataset ----
    mode = profile_for(dataset)["segment"]
    if mode == "charge":
        cands = [(True, _extract_branch(cd, nominal, cfg, True))]
    elif mode == "discharge":
        cands = [(False, _extract_branch(cd, nominal, cfg, False))]
    else:
        cands = [(True, _extract_branch(cd, nominal, cfg, True)),
                 (False, _extract_branch(cd, nominal, cfg, False))]
    cands = [(ch, r) for ch, r in cands if r is not None]
    if not cands:
        LOG.warning(f"    skip {pkl_path.name}: no usable CC segment "
                    f"(mode={mode})")
        return None
    # auto: prefer the branch with the wider stable voltage window, which is the
    # one whose operating point is shared across cycles
    charge, (grid, nums, dq, lo, hi) = max(cands, key=lambda t: t[1][4] - t[1][3])

    return CellCache(
        cell_id=str(d.get("cell_id") or pkl_path.stem), dataset=dataset,
        cathode=str(d.get("cathode_material")), nominal_ah=nominal,
        v_lo=lo, v_hi=hi, grid=grid, cycles=nums, dq=dq,
        capacity=cap, cap_cycles=cap_cyc, c_eol=c_eol,
        initial_capacity=initial, branch="charge" if charge else "discharge",
    )


def stage_prepare(cfg: Config, force: bool = False) -> Dict[str, List[Path]]:
    data_root, cache_root = Path(cfg.data_root), Path(cfg.cache_root)
    datasets = list_datasets(data_root, cfg.datasets)
    if not datasets:
        raise FileNotFoundError(f"no dataset folders with *.pkl under {data_root.resolve()}")
    LOG.info(f"datasets found: {datasets}")

    out: Dict[str, List[Path]] = {}
    for ds in datasets:
        pkls = sorted((data_root / ds).glob("*.pkl"))
        LOG.info(f"[{ds}] {len(pkls)} cell files, "
                 f"segment profile = {profile_for(ds)['segment']}")
        paths: List[Path] = []
        for i, p in enumerate(pkls, 1):
            cpath = cache_root / ds / (p.stem + ".npz")
            if cpath.exists() and not force:
                paths.append(cpath)
                LOG.info(f"  ({i}/{len(pkls)}) {p.name}: cached")
                continue
            t0 = time.time()
            cc = build_cell_cache(p, ds, cfg)
            if cc is None:
                continue
            cc.save(cpath)
            paths.append(cpath)
            LOG.info(f"  ({i}/{len(pkls)}) {p.name}: {cc.dq.shape[0]} curves, "
                     f"C0={cc.initial_capacity:.4f}Ah, c_eol={cc.c_eol}, "
                     f"{cc.branch} V=[{cc.v_lo:.3f},{cc.v_hi:.3f}] "
                     f"({time.time()-t0:.1f}s)")
        out[ds] = paths
        LOG.info(f"[{ds}] cached {len(paths)} cells")
    return out


# ===== 06. SPLIT =============================================================
@dataclass
class Cell:
    cache: CellCache
    L: int                 # observation window length (cycles)
    y_raw: float           # log cycle life
    _cyc: Optional[np.ndarray] = None      # regridded once, at load time
    _dq: Optional[np.ndarray] = None

    @property
    def dataset(self) -> str:
        return self.cache.dataset

    def regrid(self, grid: np.ndarray) -> None:
        """Project onto the dataset-common voltage grid and clip to L. Done once
        per run; doing it inside collate would dominate the step time."""
        m = self.cache.cycles <= self.L
        cyc = self.cache.cycles[m].astype(np.int64)
        dq = self.cache.dq[m]
        if not np.array_equal(grid, self.cache.grid):
            dq = np.stack([np.interp(grid, self.cache.grid, r) for r in dq])
        self._cyc, self._dq = cyc, dq.astype(np.float32)

    def window(self, grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self._cyc is None:
            self.regrid(grid)
        return self._cyc, self._dq


def load_cells(cfg: Config) -> Tuple[List[Cell], Dict[str, np.ndarray]]:
    """Load caches, drop short-lived cells, assign L_i independently of life."""
    cache_root = Path(cfg.cache_root)
    wanted = {d.upper() for d in cfg.datasets} if cfg.datasets else None
    caches: List[CellCache] = []
    for ds in sorted(q.name for q in cache_root.iterdir() if q.is_dir()):
        if wanted is not None and ds.upper() not in wanted:
            continue
        for q in sorted((cache_root / ds).glob("*.npz")):
            caches.append(CellCache.load(q))
    LOG.info(f"loaded {len(caches)} cached cells")

    # ---- labels ---------------------------------------------------------
    # The pkl files carry no label: BatteryLife ships cycle life separately.
    # Prefer the official value so results line up with the published
    # benchmark; fall back to deriving it from the capacity trace.
    official = ({} if cfg.label_source == "derived"
                else load_life_labels(Path(cfg.labels_path)))
    if cfg.label_source == "official" and not official:
        raise FileNotFoundError(
            f"label_source='official' but no labels under {cfg.labels_path}. "
            f"Download the BatteryLife 'Life labels' folder, or use "
            f"--label-source derived.")

    kept, dropped, lives, n_official, diffs = [], [], [], 0, []
    for c in caches:
        off = lookup_life(official, c.dataset, c.cell_id)
        if off is not None:
            life = float(off)
            n_official += 1
            if c.c_eol is not None:
                diffs.append(abs(life - c.c_eol) / max(life, 1.0))
        else:
            life = float(c.c_eol) if c.c_eol is not None else float("nan")
        if not np.isfinite(life) or life < cfg.min_life:
            dropped.append((c.cell_id, None if not np.isfinite(life) else int(life)))
        else:
            kept.append(c)
            lives.append(life)

    src = ("official" if n_official == len(caches) else
           "derived" if n_official == 0 else "mixed")
    LOG.info(f"label source: {src} ({n_official}/{len(caches)} cells matched "
             f"an official label)")
    if n_official < len(caches):
        LOG.warning(
            f"{len(caches) - n_official} cells fell back to a DERIVED life. "
            f"Stored records can stop before the EOL threshold is crossed, in "
            f"which case the derived value is missing or wrong (HUST_1-1: "
            f"official 1542, record ends at cycle 1504). Check that the cell "
            f"ids in the label json match the pkl file names.")
    if diffs:
        d = np.asarray(diffs)
        LOG.info(f"  official vs derived: median |rel diff| = {np.median(d):.3%}, "
                 f"max = {d.max():.3%}")
        if np.median(d) > 0.05:
            LOG.warning("  labels disagree by more than 5% -- check eol_frac / "
                        "capacity_reference before trusting any result")
    if dropped:
        LOG.warning(f"dropped {len(dropped)} cells (no life or life < "
                    f"{cfg.min_life}): {dropped[:8]}"
                    f"{'...' if len(dropped) > 8 else ''}")

    # L_i is drawn from a fixed RNG keyed only by cell_id order, never by life.
    rng = np.random.default_rng(cfg.seed)
    order = sorted(range(len(kept)), key=lambda i: (kept[i].dataset, kept[i].cell_id))
    kept = [kept[i] for i in order]
    lives = [lives[i] for i in order]
    Ls = rng.integers(cfg.L_min, cfg.L_max + 1, size=len(kept))
    cells = [Cell(cache=c, L=int(L), y_raw=float(np.log(v)))
             for c, v, L in zip(kept, lives, Ls)]

    # dataset-common voltage grid
    grids: Dict[str, np.ndarray] = {}
    for ds in sorted({c.dataset for c in cells}):
        sub = [c.cache for c in cells if c.dataset == ds]
        lo = max(s.v_lo for s in sub)
        hi = min(s.v_hi for s in sub)
        grids[ds] = np.linspace(lo, hi, cfg.grid_points).astype(np.float32)
        br = sorted({s2.branch for s2 in sub})
        LOG.info(f"[{ds}] {len(sub)} cells, branch={'/'.join(br)}, "
                 f"common V=[{lo:.3f},{hi:.3f}]")
        if len(br) > 1:
            LOG.warning(f"[{ds}] cells disagree on the branch ({br}); pin it in "
                        f"DATASET_PROFILES or the curves are not comparable")

    t0 = time.time()
    for c in cells:
        c.regrid(grids[c.dataset])
    LOG.info(f"regridded {len(cells)} cells onto the common grids "
             f"({time.time() - t0:.1f}s)")

    if len(cells) > 2:
        L = np.array([c.L for c in cells], dtype=float)
        y = np.array([c.y_raw for c in cells], dtype=float)
        r = float(np.corrcoef(L, y)[0, 1])
        LOG.info(f"LEAK CHECK  corr(L_i, log c_eol) = {r:+.4f} "
                 f"(must be ~0; |r|>0.2 invalidates the protocol)")
    return cells, grids


def folds(cells: List[Cell]) -> List[str]:
    return sorted({c.dataset for c in cells})


# ===== 07. BATCHING ==========================================================
def collate(cells: List[Cell], grids: Dict[str, np.ndarray], cfg: Config):
    """Ragged windows -> padded tensors. mask is True at VALID positions."""
    wins = [c.window(grids[c.dataset]) for c in cells]
    Lmax = max(w[0].size for w in wins)
    B, G = len(cells), cfg.grid_points
    dq = np.zeros((B, Lmax, G), dtype=np.float32)
    cyc = np.zeros((B, Lmax), dtype=np.float32)
    msk = np.zeros((B, Lmax), dtype=bool)
    for i, (cc, dd) in enumerate(wins):
        n = cc.size
        dq[i, :n] = dd
        cyc[i, :n] = cc
        msk[i, :n] = True
    y = np.array([c.y_raw for c in cells], dtype=np.float32)
    Ls = np.array([c.L for c in cells], dtype=np.float32)
    dev = cfg.device
    t = lambda a, d=torch.float32: torch.as_tensor(a, dtype=d, device=dev)
    return (t(dq), t(cyc), torch.as_tensor(msk, device=dev), t(y), t(Ls))


# ===== 08. CYCLE ENCODER =====================================================
class CycleEncoder(nn.Module):
    """dQ(V) -> cycle feature.

    Deliberately small. The SVD of dQ within a cell is close to rank one, so a
    wide convolutional stack would only rediscover a one-dimensional subspace
    while adding overfitting capacity on a few hundred cell-level labels.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.grid_points, cfg.enc_hidden),
            nn.GELU(),
            nn.Linear(cfg.enc_hidden, cfg.d_cycle),
        )

    def forward(self, dq: torch.Tensor) -> torch.Tensor:      # (B,L,G)->(B,L,d)
        return self.net(dq)


def fourier(x: torch.Tensor, n: int, scale: float) -> torch.Tensor:
    """Fourier features of a continuous cycle index. x: (...,) -> (..., 2n)."""
    freqs = torch.arange(n, device=x.device, dtype=x.dtype)
    w = (2.0 ** freqs) * math.pi / scale
    a = x.unsqueeze(-1) * w
    return torch.cat([torch.sin(a), torch.cos(a)], dim=-1)


# ===== 09. REFERENCE ATTENTION ===============================================
class ReferenceGridAttention(nn.Module):
    """Cross-attention from a fixed grid of reference cycles to observed cycles.

    Output size is K x d regardless of how many cycles the cell has, and slot k
    always means 'state at cycle r_k'. Length therefore cannot leak into the
    representation the way it does with masked mean pooling, where a 20-cycle
    and a 100-cycle cell average over different ageing windows.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("ref", torch.as_tensor(cfg.ref_grid_np()))
        ff = 2 * cfg.n_fourier
        self.q = nn.Linear(ff, cfg.d_attn)
        self.k = nn.Linear(ff, cfg.d_attn)
        self.v = nn.Linear(cfg.d_cycle, cfg.d_cycle)
        self.scale = cfg.L_max

    def forward(self, h: torch.Tensor, cyc: torch.Tensor, mask: torch.Tensor):
        B, L, d = h.shape
        K = self.ref.numel()
        q = self.q(fourier(self.ref, self.cfg.n_fourier, self.scale))       # (K,da)
        k = self.k(fourier(cyc, self.cfg.n_fourier, self.scale))            # (B,L,da)
        logits = torch.einsum("kd,bld->bkl", q, k) / math.sqrt(k.shape[-1])
        logits = logits.masked_fill(~mask.unsqueeze(1), float("-inf"))
        w = torch.softmax(logits, dim=-1)                                   # (B,K,L)
        z = torch.einsum("bkl,bld->bkd", w, self.v(h))                      # (B,K,d)

        # coverage: how well each reference slot is actually supported
        cyc_e = cyc.unsqueeze(1)                                            # (B,1,L)
        ref_e = self.ref.view(1, K, 1)
        big = torch.full_like(cyc_e, 1e4)
        dist = torch.where(mask.unsqueeze(1), (cyc_e - ref_e).abs(), big)
        nearest = dist.min(dim=-1).values / self.scale                      # (B,K)
        ent = -(w.clamp_min(1e-9).log() * w).sum(-1) / math.log(L + 1)      # (B,K)
        last = torch.where(mask, cyc, torch.zeros_like(cyc)).max(dim=1).values
        extrap = (ref_e.squeeze(-1) > last.unsqueeze(1)).float()            # (B,K)
        frac = (mask.unsqueeze(1) & (cyc_e <= ref_e)).float().sum(-1) / \
               mask.float().sum(-1, keepdim=True).clamp_min(1.0)
        cov = torch.stack([nearest, ent, extrap, frac], dim=-1)             # (B,K,4)
        return z, cov, w


# ===== 10. HEAD ==============================================================
class Head(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        K = len(cfg.ref_grid)
        self.proj = nn.Sequential(
            nn.Linear(cfg.d_cycle + 4, cfg.m_bottleneck), nn.GELU(),
            nn.Linear(cfg.m_bottleneck, cfg.m_bottleneck),
        )
        n_feat = K * cfg.m_bottleneck + (K - 1) * cfg.m_bottleneck
        self.mlp = nn.Sequential(
            nn.Linear(n_feat, cfg.head_hidden), nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden, 1),
        )

    def forward(self, z: torch.Tensor, cov: torch.Tensor):
        u = self.proj(torch.cat([z, cov], dim=-1))          # (B,K,m)
        du = u[:, 1:] - u[:, :-1]                           # learned slopes
        feat = torch.cat([u.flatten(1), du.flatten(1)], dim=-1)
        return self.mlp(feat).squeeze(-1), u


# ===== 11. MODEL =============================================================
class Net(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.encoder = CycleEncoder(cfg)
        self.attn = ReferenceGridAttention(cfg)
        self.head = Head(cfg)
        self.ssl = nn.Linear(cfg.d_cycle, cfg.grid_points)

    def forward(self, dq, cyc, mask):
        h = self.encoder(dq)
        z, cov, w = self.attn(h, cyc, mask)
        yhat, u = self.head(z, cov)
        return {"y": yhat, "u": u, "h": h, "attn": w, "cov": cov}

    def n_params(self) -> Dict[str, int]:
        f = lambda m: sum(p.numel() for p in m.parameters())
        return {"encoder": f(self.encoder), "attn": f(self.attn),
                "head": f(self.head), "ssl": f(self.ssl), "total": f(self)}


# ===== 12. LOSSES ============================================================
def loss_regression(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, y)


def loss_align(u_src: torch.Tensor, u_tgt: torch.Tensor) -> torch.Tensor:
    """Per-reference-slot first/second moment matching.

    Mean + diagonal covariance, not a kernel MMD: a domain here contributes only
    a few dozen cells, which is far too few for a reliable kernel estimate.
    """
    if u_src.shape[0] < 2 or u_tgt.shape[0] < 2:
        return u_src.sum() * 0.0
    ms, mt = u_src.mean(0), u_tgt.mean(0)
    ss, st = u_src.std(0, unbiased=False), u_tgt.std(0, unbiased=False)
    return ((ms - mt) ** 2).mean() + ((ss - st) ** 2).mean()


def loss_ssl(model: Net, out: dict, dq: torch.Tensor,
             mask: torch.Tensor) -> torch.Tensor:
    """Predict the next cycle's difference curve. Needs no labels, so it also
    runs on the unlabelled target cells."""
    pred = model.ssl(out["h"][:, :-1])
    tgt = dq[:, 1:]
    m = (mask[:, :-1] & mask[:, 1:]).unsqueeze(-1)
    if m.sum() == 0:
        return pred.sum() * 0.0
    return (((pred - tgt) ** 2) * m).sum() / m.sum().clamp_min(1) / dq.shape[-1]


# ===== 13. BASELINES =========================================================
def summary_features(cell: Cell, grid: np.ndarray) -> np.ndarray:
    """Severson-style summary of the difference curve, fitted linearly in cycle.

    Returns (slope, intercept, residual sd) for each summary statistic plus a
    precision proxy, which is the classical counterpart of the coverage signal
    the network gets.
    """
    cyc, dq = cell.window(grid)
    if cyc.size < 4:
        return np.zeros(10, dtype=np.float32)
    stats = np.stack([
        np.log(dq.var(axis=1) + 1e-12),
        np.log(np.abs(dq.min(axis=1)) + 1e-12),
        dq.mean(axis=1),
    ], axis=1)                                            # (n,3)
    x = cyc.astype(np.float64)
    feats = []
    for j in range(stats.shape[1]):
        b, a = np.polyfit(x, stats[:, j], 1)
        resid = stats[:, j] - (a + b * x)
        feats += [b, a, resid.std()]
    feats.append(1.0 / math.sqrt(max(cyc.size, 1)))
    return np.asarray(feats, dtype=np.float32)


def run_baselines(train: List[Cell], test: List[Cell], grids, cfg: Config) -> Dict:
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    ytr = np.array([c.y_raw for c in train])
    yte = np.array([c.y_raw for c in test])
    res = {"dummy": _metrics(np.full_like(yte, ytr.mean()), yte, test, cfg)}

    Xtr = np.stack([summary_features(c, grids[c.dataset]) for c in train])
    Xte = np.stack([summary_features(c, grids[c.dataset]) for c in test])
    Xtr = np.nan_to_num(Xtr, nan=0.0, posinf=0.0, neginf=0.0)
    Xte = np.nan_to_num(Xte, nan=0.0, posinf=0.0, neginf=0.0)
    sc = StandardScaler().fit(Xtr)
    mdl = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(sc.transform(Xtr), ytr)
    res["L0_ridge"] = _metrics(mdl.predict(sc.transform(Xte)), yte, test, cfg)
    return res


# ===== 14. TRAIN =============================================================
def _sample(cells: List[Cell], n: int, rng: np.random.Generator) -> List[Cell]:
    idx = rng.choice(len(cells), size=min(n, len(cells)), replace=False)
    return [cells[i] for i in idx]


def train_fold(target_ds: str, cells: List[Cell], grids, cfg: Config,
               seed: int, out_dir: Path) -> Dict:
    set_seed(seed)
    rng = np.random.default_rng(seed)

    src_all = [c for c in cells if c.dataset != target_ds]
    tgt = [c for c in cells if c.dataset == target_ds]
    if len(src_all) < 4 or len(tgt) < 2:
        LOG.warning(f"fold {target_ds}: too few cells (src={len(src_all)}, "
                    f"tgt={len(tgt)}) -- skipped")
        return {}

    perm = rng.permutation(len(src_all))
    n_val = max(1, int(cfg.val_frac * len(src_all)))
    val = [src_all[i] for i in perm[:n_val]]
    src = [src_all[i] for i in perm[n_val:]]

    mu = float(np.mean([c.y_raw for c in src]))
    sd = float(np.std([c.y_raw for c in src]) + 1e-8)
    LOG.info(f"fold={target_ds} seed={seed} | source {len(src)} train / "
             f"{len(val)} val, target {len(tgt)} | label mu={mu:.3f} sd={sd:.3f}")
    LOG.info(f"  source datasets: {sorted({c.dataset for c in src_all})}")

    model = Net(cfg).to(cfg.device)
    LOG.info(f"  params: {model.n_params()}")
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                           weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.max_steps)

    best = {"val": float("inf"), "step": -1, "state": None}
    hist = []
    for step in range(1, cfg.max_steps + 1):
        model.train()
        bs = _sample(src, cfg.batch_source, rng)
        dq, cyc, msk, y, _ = collate(bs, grids, cfg)
        out = model(dq, cyc, msk)
        l_reg = loss_regression(out["y"], (y - mu) / sd)
        loss = l_reg
        parts = {"reg": l_reg.item()}

        if cfg.lambda_align > 0:
            bt = _sample(tgt, cfg.batch_target, rng)
            dq_t, cyc_t, msk_t, _, _ = collate(bt, grids, cfg)
            out_t = model(dq_t, cyc_t, msk_t)
            l_al = loss_align(out["u"], out_t["u"])
            w = cfg.lambda_align * min(1.0, step / max(cfg.align_warmup, 1))
            loss = loss + w * l_al
            parts["align"] = l_al.item()
        if cfg.gamma_ssl > 0:
            l_ssl = loss_ssl(model, out, dq, msk)
            loss = loss + cfg.gamma_ssl * l_ssl
            parts["ssl"] = l_ssl.item()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()

        if step % cfg.log_every == 0 or step == 1:
            LOG.info(f"  step {step:5d}/{cfg.max_steps}  loss={loss.item():.4f}  "
                     + "  ".join(f"{k}={v:.4f}" for k, v in parts.items())
                     + f"  lr={sched.get_last_lr()[0]:.2e}")

        if step % cfg.eval_every == 0:
            model.eval()
            with torch.no_grad():
                dq_v, cyc_v, msk_v, y_v, _ = collate(val, grids, cfg)
                pv = model(dq_v, cyc_v, msk_v)["y"] * sd + mu
                vmse = float(F.mse_loss(pv, y_v))
            hist.append({"step": step, "val_mse": vmse})
            flag = ""
            if vmse < best["val"]:
                best = {"val": vmse, "step": step,
                        "state": {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}}
                flag = "  <- best"
            LOG.info(f"  step {step:5d}  val_mse(log-life)={vmse:.4f}{flag}")
            if step - best["step"] >= cfg.patience:
                LOG.info(f"  early stop at step {step} "
                         f"(no improvement for {cfg.patience})")
                break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    LOG.info(f"  restored best checkpoint from step {best['step']} "
             f"(val_mse={best['val']:.4f})")

    # ---- REQUIREMENT 1: persist parameters --------------------------------
    ckpt_dir = out_dir / f"fold_{target_ds}" / f"seed_{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "checkpoint.pt"
    torch.save({
        "model_state": model.state_dict(),
        "config": asdict(cfg),
        "label_mu": mu, "label_sd": sd,
        "target_dataset": target_ds, "seed": seed,
        "best_step": best["step"], "best_val_mse": best["val"],
        "grids": {k: v for k, v in grids.items()},
        "train_cells": [c.cache.cell_id for c in src],
        "val_cells": [c.cache.cell_id for c in val],
        "target_cells": [c.cache.cell_id for c in tgt],
        "history": hist,
    }, ckpt_path)
    LOG.info(f"  saved checkpoint -> {ckpt_path}")

    model.eval()
    with torch.no_grad():
        dq_t, cyc_t, msk_t, y_t, _ = collate(tgt, grids, cfg)
        pred = (model(dq_t, cyc_t, msk_t)["y"] * sd + mu).cpu().numpy()
    metrics = _metrics(pred, np.array([c.y_raw for c in tgt]), tgt, cfg)

    base = run_baselines(src_all, tgt, grids, cfg)
    return {"model": metrics, **base, "ckpt": str(ckpt_path),
            "best_step": best["step"], "best_val_mse": best["val"]}


# ===== 15. EVAL ==============================================================
def _metrics(pred_log: np.ndarray, true_log: np.ndarray,
             cells: List[Cell], cfg: Config) -> Dict:
    life_p = np.exp(pred_log)
    life_t = np.exp(true_log)
    L = np.array([c.L for c in cells], dtype=float)
    rul_p, rul_t = np.maximum(life_p - L, 0.0), life_t - L
    out = {
        "n": int(len(cells)),
        "MAPE_life_%": float(np.mean(np.abs(life_p - life_t) / life_t) * 100),
        "MAE_rul_cyc": float(np.mean(np.abs(rul_p - rul_t))),
        "RMSE_rul_cyc": float(np.sqrt(np.mean((rul_p - rul_t) ** 2))),
        "RMSE_log": float(np.sqrt(np.mean((pred_log - true_log) ** 2))),
    }
    strat = {}
    e = cfg.strat_edges
    for lo, hi in zip(e[:-1], e[1:]):
        m = (L >= lo) & (L < hi)
        if m.sum() == 0:
            continue
        strat[f"L[{lo},{hi})"] = {
            "n": int(m.sum()),
            "MAPE_life_%": float(np.mean(np.abs(life_p[m] - life_t[m]) / life_t[m]) * 100),
        }
    out["stratified"] = strat
    return out


def report(results: Dict[str, Dict], cfg: Config) -> None:
    methods = ["dummy", "L0_ridge", "model"]
    rows = []
    for ds, r in results.items():
        if not r:
            continue
        for m in methods:
            if m in r:
                rows.append([ds, m, r[m]["n"],
                             f"{r[m]['MAPE_life_%']:.2f}",
                             f"{r[m]['MAE_rul_cyc']:.1f}",
                             f"{r[m]['RMSE_log']:.4f}"])
    LOG.info("")
    LOG.info("PER-FOLD RESULTS")
    log_table(rows, ["target", "method", "n", "MAPE%", "MAE_RUL", "RMSE_log"])

    LOG.info("")
    LOG.info("MEAN OVER FOLDS")
    agg = []
    for m in methods:
        vals = [r[m]["MAPE_life_%"] for r in results.values() if r and m in r]
        if vals:
            agg.append([m, len(vals), f"{np.mean(vals):.2f}", f"{np.std(vals):.2f}"])
    log_table(agg, ["method", "folds", "MAPE% mean", "sd"])

    LOG.info("")
    LOG.info("MODEL ERROR STRATIFIED BY OBSERVATION LENGTH")
    rows = []
    for ds, r in results.items():
        if r and "model" in r:
            for k, v in r["model"]["stratified"].items():
                rows.append([ds, k, v["n"], f"{v['MAPE_life_%']:.2f}"])
    if rows:
        log_table(rows, ["target", "bucket", "n", "MAPE%"])


# ===== 16. DIAGNOSTICS =======================================================
def stage_diagnose(cfg: Config) -> Dict:
    cells, grids = load_cells(cfg)
    if not cells:
        LOG.warning("no cells to diagnose")
        return {}

    LOG.info("")
    LOG.info("[A] effective rank of the difference curves (per cell)")
    rows = []
    for c in cells[:12]:
        X = c.cache.dq[c.cache.cycles <= cfg.L_max]
        if X.shape[0] < 5:
            continue
        s = np.linalg.svd(X - X.mean(0), compute_uv=False)
        ev = s ** 2 / max((s ** 2).sum(), 1e-30)
        rows.append([c.cache.cell_id, X.shape[0],
                     f"{ev[0]:.4f}", f"{ev[:2].sum():.4f}", f"{ev[:3].sum():.4f}"])
    log_table(rows, ["cell", "cycles", "PC1", "cum2", "cum3"])

    LOG.info("")
    LOG.info("[B] slope consistency across window lengths "
             "(divergence => length-dependent bias, which a linear summary "
             "cannot absorb)")
    rows = []
    for c in cells[:12]:
        cyc_all, dq_all = c.cache.cycles, c.cache.dq
        a = np.log(dq_all.var(axis=1) + 1e-12)
        vals = []
        for L in (20, 50, 100, 200):
            m = cyc_all <= L
            if m.sum() >= 5:
                vals.append(f"{np.polyfit(cyc_all[m], a[m], 1)[0]:.2e}")
            else:
                vals.append("-")
        rows.append([c.cache.cell_id] + vals)
    log_table(rows, ["cell", "L=20", "L=50", "L=100", "L=200"])

    LOG.info("")
    LOG.info("[C] label and protocol summary")
    rows = []
    for ds in sorted({c.dataset for c in cells}):
        sub = [c for c in cells if c.dataset == ds]
        life = np.array([np.exp(c.y_raw) for c in sub])
        rows.append([ds, len(sub), f"{life.min():.0f}", f"{np.median(life):.0f}",
                     f"{life.max():.0f}", f"{np.mean([c.L for c in sub]):.1f}"])
    log_table(rows, ["dataset", "cells", "life min", "med", "max", "mean L"])
    return {"n_cells": len(cells)}


# ===== 17. TESTS =============================================================
def _toy_cell(cell_id: str, ds: str, life: int, L: int, cfg: Config,
              rng: np.random.Generator) -> Cell:
    G = cfg.grid_points
    n = max(L - cfg.ref_cycle, 5)
    cyc = np.arange(cfg.ref_cycle + 1, cfg.ref_cycle + 1 + n, dtype=np.int64)
    profile = np.sin(np.linspace(0, np.pi, G)).astype(np.float32)
    amp = (cyc / life).astype(np.float32) ** 1.3
    dq = (amp[:, None] * profile[None, :]).astype(np.float32)
    dq += rng.normal(0, 1e-4, dq.shape).astype(np.float32)
    grid = np.linspace(3.38, 3.60, G).astype(np.float32)
    cc = CellCache(cell_id, ds, "LFP", 1.1, 3.38, 3.60, grid, cyc, dq,
                   np.ones(life, np.float32), np.arange(1, life + 1), life, 1.1)
    return Cell(cache=cc, L=L, y_raw=float(np.log(life)))


def _toy_population(cfg: Config, n_per: int = 24) -> Tuple[List[Cell], Dict]:
    rng = np.random.default_rng(0)
    cells = []
    for di, ds in enumerate(["DS_A", "DS_B", "DS_C"]):
        for i in range(n_per):
            life = int(np.exp(rng.normal(6.6 + 0.15 * di, 0.35)))
            L = int(rng.integers(cfg.L_min, cfg.L_max + 1))
            cells.append(_toy_cell(f"{ds}_{i}", ds, max(life, 200), L, cfg, rng))
    grids = {ds: np.linspace(3.38, 3.60, cfg.grid_points).astype(np.float32)
             for ds in ["DS_A", "DS_B", "DS_C"]}
    return cells, grids


def test_mask_polarity(cfg: Config) -> None:
    cells, grids = _toy_population(cfg, 6)
    dq, cyc, msk, _, _ = collate(cells[:4], grids, cfg)
    assert msk.dtype == torch.bool
    model = Net(cfg).to(cfg.device).eval()
    with torch.no_grad():
        a = model(dq, cyc, msk)["y"].clone()
        dq2 = dq.clone()
        dq2[~msk] = 123.0                       # garbage at padded slots
        b = model(dq2, cyc, msk)["y"]
    assert torch.allclose(a, b, atol=1e-5), "padding leaks into the output"


def test_output_shape_invariant_to_length(cfg: Config) -> None:
    cells, grids = _toy_population(cfg, 4)
    model = Net(cfg).to(cfg.device).eval()
    outs = []
    for c in cells[:3]:
        dq, cyc, msk, _, _ = collate([c], grids, cfg)
        with torch.no_grad():
            outs.append(model(dq, cyc, msk)["u"].shape)
    assert len(set(outs)) == 1, f"representation shape varies with L: {outs}"


def test_no_label_leak(cfg: Config) -> None:
    cells, _ = _toy_population(cfg, 40)
    L = np.array([c.L for c in cells], float)
    y = np.array([c.y_raw for c in cells], float)
    r = abs(float(np.corrcoef(L, y)[0, 1]))
    assert r < 0.25, f"L_i correlates with life (r={r:.3f})"


def test_metrics_sane(cfg: Config) -> None:
    cells, _ = _toy_population(cfg, 8)
    y = np.array([c.y_raw for c in cells])
    m = _metrics(y.copy(), y, cells, cfg)
    assert m["MAPE_life_%"] < 1e-6 and m["RMSE_log"] < 1e-6


def test_train_smoke(cfg: Config) -> None:
    cells, grids = _toy_population(cfg, 20)
    cfg2 = Config(**{**asdict(cfg), "max_steps": 60, "eval_every": 30,
                     "log_every": 30, "lambda_align": 0.01, "gamma_ssl": 0.01})
    out = Path("/tmp/msda_test")
    r = train_fold("DS_C", cells, grids, cfg2, seed=0, out_dir=out)
    assert "model" in r and Path(r["ckpt"]).exists()


def stage_selftest(cfg: Config) -> bool:
    tests = [test_mask_polarity, test_output_shape_invariant_to_length,
             test_no_label_leak, test_metrics_sane, test_train_smoke]
    rows, ok = [], True
    for t in tests:
        try:
            t(cfg)
            rows.append([t.__name__, "PASS", ""])
        except Exception as e:                       # noqa: BLE001
            ok = False
            rows.append([t.__name__, "FAIL", str(e)[:60]])
    log_table(rows, ["test", "result", "detail"])
    return ok


# ===== 18. CLI ===============================================================
def stage_train(cfg: Config, run_dir: Path) -> Dict:
    cells, grids = load_cells(cfg)
    ds_list = folds(cells)
    LOG.info(f"leave-one-dataset-out over {len(ds_list)} folds: {ds_list}")
    if len(ds_list) < 2:
        LOG.error("need at least 2 datasets for leave-one-dataset-out; "
                  "put more dataset folders under data/")
        return {}
    results: Dict[str, Dict] = {}
    for i, ds in enumerate(ds_list, 1):
        with Stage(f"FOLD {i}/{len(ds_list)} — target = {ds}", "[train]"):
            per_seed = []
            for s in range(cfg.n_seeds):
                r = train_fold(ds, cells, grids, cfg, cfg.seed + s, run_dir)
                if r:
                    per_seed.append(r)
            if per_seed:
                results[ds] = per_seed[0] if len(per_seed) == 1 else _avg(per_seed)
    return results


def _avg(rs: List[Dict]) -> Dict:
    out = dict(rs[0])
    for m in ("model", "dummy", "L0_ridge"):
        if m not in rs[0]:
            continue
        keys = [k for k, v in rs[0][m].items() if isinstance(v, float)]
        out[m] = dict(rs[0][m])
        for k in keys:
            out[m][k] = float(np.mean([r[m][k] for r in rs]))
    return out


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", nargs="?", default="all",
                   choices=["all", "prepare", "diagnose", "baseline",
                            "train", "selftest"])
    p.add_argument("--data-root", default="data")
    p.add_argument("--cache-root", default="cache")
    p.add_argument("--run-root", default="runs")
    p.add_argument("--force", action="store_true", help="rebuild the cache")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--n-seeds", type=int, default=None)
    p.add_argument("--lambda-align", type=float, default=None)
    p.add_argument("--gamma-ssl", type=float, default=None)
    p.add_argument("--labels-path", default=None,
                   help="BatteryLife 'Life labels' file or folder")
    p.add_argument("--label-source", default=None,
                   choices=["auto", "official", "derived"])
    p.add_argument("--datasets", nargs="*", default=None,
                   help="dataset folder names to use; default = all found")
    p.add_argument("--device", default=None)
    a = p.parse_args(argv)

    over = {"data_root": a.data_root, "cache_root": a.cache_root,
            "run_root": a.run_root}
    if a.datasets:
        over["datasets"] = tuple(a.datasets)
    for k, v in [("labels_path", a.labels_path),
                 ("label_source", a.label_source),
                 ("max_steps", a.max_steps), ("n_seeds", a.n_seeds),
                 ("lambda_align", a.lambda_align), ("gamma_ssl", a.gamma_ssl),
                 ("device", a.device)]:
        if v is not None:
            over[k] = v
    cfg = Config(**{**asdict(Config()), **over})

    run_dir = Path(cfg.run_root) / time.strftime("%Y%m%d_%H%M%S")
    setup_logging(run_dir / "run.log")
    LOG.info(f"run directory: {run_dir.resolve()}")
    if torch is not None and cfg.device.startswith("cuda"):
        if torch.cuda.is_available():
            i = torch.cuda.current_device()
            LOG.info(f"device: {cfg.device}  ({torch.cuda.get_device_name(i)}, "
                     f"{torch.cuda.get_device_properties(i).total_memory/2**30:.1f} GiB, "
                     f"torch {torch.__version__})")
        else:
            LOG.warning("cuda requested but not available -- falling back to cpu")
            cfg = Config(**{**asdict(cfg), "device": "cpu"})
    else:
        LOG.info(f"device: {cfg.device}  (torch {torch.__version__ if torch else 'n/a'})")
    LOG.info(f"config: {json.dumps(asdict(cfg), default=str)}")
    set_seed(cfg.seed)

    if a.stage == "selftest":
        with Stage("SELF TEST", "[1/1]"):
            return 0 if stage_selftest(cfg) else 1

    if a.stage in ("all", "prepare"):
        with Stage("PREPARE — parse pkl files and cache features", "[1/4]"):
            stage_prepare(cfg, force=a.force)
        if a.stage == "prepare":
            return 0

    if a.stage in ("all", "diagnose"):
        with Stage("DIAGNOSE — rank, slope consistency, label summary", "[2/4]"):
            stage_diagnose(cfg)
        if a.stage == "diagnose":
            return 0

    results: Dict[str, Dict] = {}
    if a.stage in ("all", "train", "baseline"):
        with Stage("TRAIN — leave-one-dataset-out", "[3/4]"):
            results = stage_train(cfg, run_dir)

    if results:
        with Stage("EVALUATE", "[4/4]"):
            report(results, cfg)
            (run_dir / "results.json").write_text(
                json.dumps(results, indent=2, default=str), encoding="utf-8")
            LOG.info(f"results written -> {run_dir / 'results.json'}")
    LOG.info(f"total elapsed {time.time() - _T0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
