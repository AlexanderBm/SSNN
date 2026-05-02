"""
Step 4: Validation pipeline using PUMAS on real GWAS summary statistics.

Orchestrates the full comparison of PRS methods on summary-stat data:
    1. C+T (clumping + thresholding)
    2. LDpred2-inf (infinitesimal ridge)
    3. PRS-CS (continuous shrinkage)
    4. Gaussian NN (summary-stat neural network, Part 1)
    5. Edgeworth NN (summary-stat NN + cumulant corrections, Part 2)

Uses PUMAS pseudo-subset splits for train/validation evaluation.

For real GWAS data, the user provides:
    - Sigma_beta: marginal association vector from GWAS
    - Sigma: LD matrix from a reference panel
    - maf: minor allele frequencies
    - E_y2: estimated phenotypic variance (or 1.0 for standardized traits)
    - N: GWAS sample size

The pipeline can also run on synthetic data (using the simulation module)
to validate the pipeline itself before applying to real data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .baselines import clump_and_threshold, ldpred2_inf, prs_cs
from .optimizer import train, TrainResult
from .edgeworth_optimizer import train_edgeworth
from .pumas import (
    PUMASSplit,
    generate_pumas_splits,
    pumas_summary_r2,
    pumas_nn_summary_r2,
)
from .utils import linear_prs_weights


# ===================================================================
# Configuration
# ===================================================================

@dataclass
class TraitConfig:
    """Configuration for a real-data trait analysis."""
    name: str
    Sigma_beta: np.ndarray
    Sigma: np.ndarray
    maf: np.ndarray
    E_y2: float
    N: int
    # Optional: hyperparameter search grids
    h2_grid: list[float] = field(default_factory=lambda: [0.1, 0.3, 0.5, 0.7])
    p_causal_grid: list[float] = field(default_factory=lambda: [0.01, 0.1, 0.5, 1.0])
    ct_p_thresholds: list[float] = field(default_factory=lambda: [5e-8, 1e-5, 1e-3, 0.01, 0.05])
    ct_r2_thresholds: list[float] = field(default_factory=lambda: [0.1, 0.2, 0.5])
    nn_hidden_units: list[int] = field(default_factory=lambda: [3, 5, 10])
    nn_lr: float = 0.01
    nn_max_iters: int = 3000


@dataclass
class MethodResult:
    """Result from a single method on a single PUMAS split."""
    method: str
    split_idx: int
    summary_r2_val: float
    params: dict = field(default_factory=dict)


@dataclass
class TraitResult:
    """Aggregated results for one trait."""
    trait_name: str
    method_results: list[MethodResult]
    best_per_method: dict = field(default_factory=dict)


# ===================================================================
# Individual method runners
# ===================================================================

def _run_ct_on_split(
    split: PUMASSplit,
    Sigma: np.ndarray,
    N: int,
    p_threshold: float,
    r2_threshold: float,
) -> tuple[np.ndarray, float]:
    """Run C+T on a PUMAS training split, evaluate on validation."""
    weights = clump_and_threshold(
        Sigma, split.Sigma_beta_train,
        p_threshold=p_threshold,
        r2_threshold=r2_threshold,
        n=split.n_train,
    )
    r2 = pumas_summary_r2(split.Sigma_beta_val, weights, Sigma, split.E_y2_val)
    return weights, r2


def _run_ldpred2_on_split(
    split: PUMASSplit,
    Sigma: np.ndarray,
    h2: float,
    p_causal: float,
) -> tuple[np.ndarray, float]:
    """Run LDpred2-inf on a PUMAS training split, evaluate on validation."""
    weights = ldpred2_inf(
        Sigma, split.Sigma_beta_train,
        h2=h2, p_causal=p_causal, n=split.n_train,
    )
    r2 = pumas_summary_r2(split.Sigma_beta_val, weights, Sigma, split.E_y2_val)
    return weights, r2


def _run_prs_cs_on_split(
    split: PUMASSplit,
    Sigma: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Run PRS-CS on a PUMAS training split, evaluate on validation."""
    weights = prs_cs(Sigma, split.Sigma_beta_train, n=split.n_train)
    r2 = pumas_summary_r2(split.Sigma_beta_val, weights, Sigma, split.E_y2_val)
    return weights, r2


def _run_gaussian_nn_on_split(
    split: PUMASSplit,
    Sigma: np.ndarray,
    m: int,
    activation: str,
    lr: float,
    max_iters: int,
    rng: np.random.Generator,
) -> tuple[TrainResult, float]:
    """Train Gaussian NN on PUMAS training split, evaluate on validation."""
    result = train(
        Sigma, split.Sigma_beta_train, split.E_y2_train,
        m=m, activation=activation, lr=lr, max_iters=max_iters,
        rng=rng,
    )
    r2 = pumas_nn_summary_r2(
        split.Sigma_beta_val, split.E_y2_val,
        result.a, result.W, Sigma, activation=activation,
    )
    return result, r2


def _run_edgeworth_nn_on_split(
    split: PUMASSplit,
    Sigma: np.ndarray,
    maf: np.ndarray,
    m: int,
    activation: str,
    lr: float,
    max_iters: int,
    rng: np.random.Generator,
    gauss_result: TrainResult | None = None,
) -> tuple[TrainResult, float]:
    """Train Edgeworth NN on PUMAS training split, evaluate on validation."""
    a_init = gauss_result.a if gauss_result is not None else None
    W_init = gauss_result.W if gauss_result is not None else None

    result = train_edgeworth(
        Sigma, split.Sigma_beta_train, split.E_y2_train, maf,
        m=m, activation=activation,
        lr=lr * 0.1 if gauss_result is not None else lr,
        max_iters=max_iters // 2 if gauss_result is not None else max_iters,
        rng=rng,
        loss_floor=0.0, grad_clip=0.5, max_backtracks=10,
        a_init=a_init, W_init=W_init,
    )
    r2 = pumas_nn_summary_r2(
        split.Sigma_beta_val, split.E_y2_val,
        result.a, result.W, Sigma, maf=maf,
        activation=activation, use_edgeworth=True,
    )
    return result, r2


# ===================================================================
# Full pipeline
# ===================================================================

def run_validation(
    config: TraitConfig,
    n_splits: int = 5,
    train_fraction: float = 0.8,
    activation: str = "relu",
    seed: int = 42,
    verbose: bool = False,
) -> TraitResult:
    """Run the full Step 4 validation pipeline for one trait.

    For each method:
        1. Generate PUMAS train/val splits.
        2. Train on each training split (with hyperparameter grid where applicable).
        3. Evaluate summary-stat R^2 on each validation split.
        4. Select best hyperparameters by mean validation R^2 across splits.

    Args:
        config: TraitConfig with summary stats and hyperparameter grids.
        n_splits: Number of PUMAS splits.
        train_fraction: Training fraction for PUMAS.
        activation: NN activation function.
        seed: Random seed.
        verbose: Print progress.

    Returns:
        TraitResult with all method results and best configs.
    """
    splits = generate_pumas_splits(
        config.Sigma_beta, config.E_y2, config.Sigma,
        config.N, n_splits=n_splits,
        train_fraction=train_fraction, seed=seed,
    )

    all_results: list[MethodResult] = []

    # --- 1. C+T with grid search ---
    if verbose:
        print(f"[{config.name}] Running C+T ...")
    best_ct_r2 = -np.inf
    best_ct_params = {}
    for p_thresh in config.ct_p_thresholds:
        for r2_thresh in config.ct_r2_thresholds:
            split_r2s = []
            for si, split in enumerate(splits):
                _, r2 = _run_ct_on_split(split, config.Sigma, config.N, p_thresh, r2_thresh)
                split_r2s.append(r2)
                all_results.append(MethodResult(
                    "C+T", si, r2,
                    {"p_threshold": p_thresh, "r2_threshold": r2_thresh},
                ))
            mean_r2 = np.mean(split_r2s)
            if mean_r2 > best_ct_r2:
                best_ct_r2 = mean_r2
                best_ct_params = {"p_threshold": p_thresh, "r2_threshold": r2_thresh}

    # --- 2. LDpred2-inf with grid search ---
    if verbose:
        print(f"[{config.name}] Running LDpred2-inf ...")
    best_ldpred_r2 = -np.inf
    best_ldpred_params = {}
    for h2 in config.h2_grid:
        for pc in config.p_causal_grid:
            split_r2s = []
            for si, split in enumerate(splits):
                _, r2 = _run_ldpred2_on_split(split, config.Sigma, h2, pc)
                split_r2s.append(r2)
                all_results.append(MethodResult(
                    "LDpred2-inf", si, r2,
                    {"h2": h2, "p_causal": pc},
                ))
            mean_r2 = np.mean(split_r2s)
            if mean_r2 > best_ldpred_r2:
                best_ldpred_r2 = mean_r2
                best_ldpred_params = {"h2": h2, "p_causal": pc}

    # --- 3. PRS-CS ---
    if verbose:
        print(f"[{config.name}] Running PRS-CS ...")
    prs_cs_r2s = []
    for si, split in enumerate(splits):
        _, r2 = _run_prs_cs_on_split(split, config.Sigma)
        prs_cs_r2s.append(r2)
        all_results.append(MethodResult("PRS-CS", si, r2))

    # --- 4. Gaussian NN with architecture search ---
    if verbose:
        print(f"[{config.name}] Running Gaussian NN ...")
    best_gnn_r2 = -np.inf
    best_gnn_params = {}
    best_gnn_results = {}
    for m in config.nn_hidden_units:
        split_r2s = []
        split_results = []
        for si, split in enumerate(splits):
            rng = np.random.default_rng(seed + 1000 + si)
            result, r2 = _run_gaussian_nn_on_split(
                split, config.Sigma, m, activation,
                config.nn_lr, config.nn_max_iters, rng,
            )
            split_r2s.append(r2)
            split_results.append(result)
            all_results.append(MethodResult(
                "Gaussian NN", si, r2, {"m": m},
            ))
        mean_r2 = np.mean(split_r2s)
        if mean_r2 > best_gnn_r2:
            best_gnn_r2 = mean_r2
            best_gnn_params = {"m": m}
            best_gnn_results = {si: r for si, r in enumerate(split_results)}

    # --- 5. Edgeworth NN (warm-started from best Gaussian NN) ---
    if verbose:
        print(f"[{config.name}] Running Edgeworth NN ...")
    best_m = best_gnn_params.get("m", config.nn_hidden_units[0])
    for si, split in enumerate(splits):
        rng = np.random.default_rng(seed + 2000 + si)
        gauss_init = best_gnn_results.get(si, None)
        result, r2 = _run_edgeworth_nn_on_split(
            split, config.Sigma, config.maf, best_m, activation,
            config.nn_lr, config.nn_max_iters, rng,
            gauss_result=gauss_init,
        )
        all_results.append(MethodResult(
            "Edgeworth NN", si, r2, {"m": best_m},
        ))

    # --- Aggregate best results ---
    best_per_method = _aggregate_best(all_results, n_splits)

    return TraitResult(
        trait_name=config.name,
        method_results=all_results,
        best_per_method=best_per_method,
    )


def _aggregate_best(
    results: list[MethodResult],
    n_splits: int,
) -> dict[str, dict]:
    """For each method, find the best hyperparameter config and its mean R^2."""
    from collections import defaultdict

    method_configs: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for r in results:
        key = _params_key(r.params)
        method_configs[r.method][key].append(r.summary_r2_val)

    best = {}
    for method, configs in method_configs.items():
        best_key = max(configs, key=lambda k: np.mean(configs[k]))
        r2s = configs[best_key]
        best[method] = {
            "mean_r2": float(np.mean(r2s)),
            "std_r2": float(np.std(r2s)),
            "params": best_key,
            "n_splits": len(r2s),
        }

    return best


def _params_key(params: dict) -> str:
    """Hashable string key for a parameter dict."""
    if not params:
        return "default"
    return ";".join(f"{k}={v}" for k, v in sorted(params.items()))


# ===================================================================
# Synthetic data convenience function
# ===================================================================

def run_synthetic_validation(
    p: int = 50,
    N: int = 50000,
    maf_spectrum: str = "mixed",
    ld_decay: float = 0.5,
    heritability: float = 0.5,
    sparsity: float = 0.2,
    n_splits: int = 5,
    seed: int = 42,
    verbose: bool = False,
) -> TraitResult:
    """Run the full validation pipeline on synthetic GWAS data.

    Generates genotypes from Binomial(2, p) with realistic LD and MAF,
    simulates phenotypes, computes summary statistics, and runs
    all methods through the PUMAS validation pipeline.

    This validates the pipeline before applying to real data.

    Args:
        p: Number of SNPs.
        N: Simulated GWAS sample size.
        maf_spectrum: MAF spectrum for generate_maf_spectrum.
        ld_decay: LD decay parameter.
        heritability: Target h^2.
        sparsity: Fraction of causal SNPs.
        n_splits: Number of PUMAS validation splits.
        seed: Random seed.
        verbose: Print progress.

    Returns:
        TraitResult with all method comparisons.
    """
    from .simulation import (
        generate_binomial_genotypes,
        generate_effect_sizes,
        generate_maf_spectrum as gen_maf,
        compute_summary_stats_from_genotypes,
    )
    from .utils import generate_ld_matrix

    rng = np.random.default_rng(seed)

    maf = gen_maf(p, maf_spectrum, rng)
    Sigma = generate_ld_matrix(p, decay=ld_decay)
    beta_star, sigma_eps = generate_effect_sizes(p, Sigma, heritability, sparsity, rng)

    X = generate_binomial_genotypes(N, maf, Sigma, rng)
    col_means = X.mean(axis=0)
    X_centered = X - col_means
    y = X_centered @ beta_star + rng.normal(0, sigma_eps, N)

    stats = compute_summary_stats_from_genotypes(X, y, Sigma)

    config = TraitConfig(
        name=f"Synthetic(p={p}, h2={heritability}, MAF={maf_spectrum})",
        Sigma_beta=stats["Sigma_beta_hat"],
        Sigma=Sigma,
        maf=stats["maf"],
        E_y2=stats["E_y2_hat"],
        N=N,
    )

    return run_validation(
        config, n_splits=n_splits,
        seed=seed, verbose=verbose,
    )
