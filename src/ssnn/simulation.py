"""
Simulation study for Step 3 of the research plan.

Generates Binomial(2, p) genotype data with realistic allele frequency
spectra and LD structure, simulates phenotypes with known beta*, and
compares five methods:

    1. Linear PRS: Sigma^{-1} Sigma_beta (optimal linear from summary stats)
    2. Gaussian NN: summary-stat NN under Gaussian genotype assumption
    3. Edgeworth NN: summary-stat NN with cumulant corrections
    4. Interaction NN: summary-stat NN using second-order interaction
       statistics Gamma_ij = E[x_i x_j y] to break the linearity barrier
    5. Oracle NN: individual-level NN trained on raw genotype data

Metrics: prediction R^2 on held-out test data, weight recovery accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.stats import norm

from .cumulants import snp_cumulants
from .optimizer import train
from .edgeworth_optimizer import train_edgeworth
from .interaction_optimizer import train_interaction
from .utils import (
    generate_ld_matrix,
    linear_prs_weights,
    nn_predict,
    nn_prediction_r2,
    prediction_r2,
)


# ===================================================================
# Data generation
# ===================================================================

def generate_maf_spectrum(
    p: int,
    spectrum: str | np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample minor allele frequencies from a named spectrum.

    Args:
        p: Number of SNPs.
        spectrum: One of "common", "rare", "mixed", or an explicit (p,) array.
        rng: Random generator.

    Returns:
        (p,) array of MAFs in (0, 1).
    """
    if isinstance(spectrum, np.ndarray):
        if spectrum.shape != (p,):
            raise ValueError(f"Explicit MAF array must have shape ({p},)")
        return spectrum

    if spectrum == "common":
        return rng.uniform(0.10, 0.50, size=p)
    elif spectrum == "rare":
        return rng.uniform(0.01, 0.05, size=p)
    elif spectrum == "mixed":
        # Realistic: ~60% common, ~25% low-frequency, ~15% rare
        n_common = int(0.60 * p)
        n_lowfreq = int(0.25 * p)
        n_rare = p - n_common - n_lowfreq
        mafs = np.concatenate([
            rng.uniform(0.10, 0.50, size=n_common),
            rng.uniform(0.05, 0.10, size=n_lowfreq),
            rng.uniform(0.01, 0.05, size=n_rare),
        ])
        rng.shuffle(mafs)
        return mafs
    else:
        raise ValueError(f"Unknown MAF spectrum: {spectrum!r}")


def generate_binomial_genotypes(
    n: int,
    maf: np.ndarray,
    Sigma: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate correlated Binomial(2, p) genotypes via liability thresholding.

    Each genotype X_j in {0, 1, 2} has marginal distribution Binomial(2, p_j).
    Correlation structure is induced by generating correlated Gaussian latents
    from N(0, Sigma) and thresholding into the three genotype categories using
    quantiles that match the Binomial(2, p_j) probabilities:
        P(X_j = 0) = (1 - p_j)^2
        P(X_j = 1) = 2 p_j (1 - p_j)
        P(X_j = 2) = p_j^2

    Args:
        n: Number of individuals.
        maf: (p,) minor allele frequencies in (0, 1).
        Sigma: (p, p) LD covariance matrix (used for the latent Gaussian).
        rng: Random generator.

    Returns:
        (n, p) integer array of genotypes in {0, 1, 2}.
    """
    p = len(maf)
    Z = rng.multivariate_normal(np.zeros(p), Sigma, size=n)

    q = 1.0 - maf
    thresh_0 = norm.ppf(q ** 2)          # P(X=0) = (1-p)^2
    thresh_01 = norm.ppf(q ** 2 + 2 * maf * q)  # P(X<=1) = (1-p)^2 + 2p(1-p)

    X = np.zeros((n, p), dtype=np.float64)
    X[Z >= thresh_0] = 1.0
    X[Z >= thresh_01] = 2.0

    return X


def generate_effect_sizes(
    p: int,
    Sigma: np.ndarray,
    heritability: float,
    sparsity: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Generate sparse effect sizes scaled to a target heritability.

    Args:
        p: Number of SNPs.
        Sigma: (p, p) LD covariance matrix.
        heritability: Target h^2 = Var(beta*'x) / Var(y).
        sparsity: Fraction of SNPs with nonzero effects.
        rng: Random generator.

    Returns:
        (beta_star, sigma_eps) where beta_star is (p,) and sigma_eps is
        the noise std such that h^2 = beta*'Sigma beta* / (beta*'Sigma beta* + sigma_eps^2).
    """
    n_causal = max(1, int(sparsity * p))
    causal_idx = rng.choice(p, size=n_causal, replace=False)

    beta_star = np.zeros(p)
    beta_star[causal_idx] = rng.standard_normal(n_causal)

    genetic_var = float(beta_star @ Sigma @ beta_star)
    if genetic_var < 1e-15:
        beta_star[causal_idx] = 1.0
        genetic_var = float(beta_star @ Sigma @ beta_star)

    # Scale to target heritability: h^2 = gvar / (gvar + sigma_eps^2)
    # => sigma_eps^2 = gvar * (1 - h^2) / h^2
    sigma_eps = np.sqrt(genetic_var * (1.0 - heritability) / heritability)

    return beta_star, float(sigma_eps)


def compute_summary_stats_from_genotypes(
    X_train: np.ndarray,
    y_train: np.ndarray,
    Sigma_ref: np.ndarray,
) -> dict:
    """Compute GWAS-style summary statistics from discrete genotype data.

    Genotypes are centered (mean-subtracted) before computing associations,
    consistent with the zero-mean assumption in the SSNN framework.

    Args:
        X_train: (n, p) genotype matrix (values in {0, 1, 2}).
        y_train: (n,) phenotype vector.
        Sigma_ref: (p, p) reference-panel LD matrix (population-level).

    Returns:
        Dictionary with Sigma_beta_hat, E_y2_hat, Sigma (reference), maf.
    """
    n, p = X_train.shape

    col_means = X_train.mean(axis=0)
    X_centered = X_train - col_means

    Sigma_beta_hat = X_centered.T @ y_train / n
    E_y2_hat = float(np.mean(y_train ** 2))
    maf_hat = col_means / 2.0
    maf_hat = np.clip(maf_hat, 0.01, 0.99)

    Gamma_hat = X_centered.T @ (X_centered * y_train[:, None]) / n

    Cov_ref = X_centered.T @ X_centered / n

    return {
        "Sigma_beta_hat": Sigma_beta_hat,
        "E_y2_hat": E_y2_hat,
        "Sigma": Sigma_ref,
        "maf": maf_hat,
        "Gamma_hat": Gamma_hat,
        "Cov_ref": Cov_ref,
    }


# ===================================================================
# Oracle individual-level NN
# ===================================================================

def _apply_activation(Z: np.ndarray, activation: str) -> np.ndarray:
    if activation == "relu":
        return np.maximum(0.0, Z)
    elif activation == "sigmoid":
        return 1.0 / (1.0 + np.exp(-np.clip(Z, -500, 500)))
    elif activation == "identity":
        return Z
    else:
        raise ValueError(f"Unknown activation: {activation!r}")


def _apply_activation_derivative(Z: np.ndarray, activation: str) -> np.ndarray:
    if activation == "relu":
        return (Z > 0).astype(float)
    elif activation == "sigmoid":
        s = 1.0 / (1.0 + np.exp(-np.clip(Z, -500, 500)))
        return s * (1.0 - s)
    elif activation == "identity":
        return np.ones_like(Z)
    else:
        raise ValueError(f"Unknown activation: {activation!r}")


def train_oracle_nn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    m: int = 5,
    activation: str = "relu",
    lr: float = 0.001,
    max_iters: int = 2000,
    batch_size: int = 256,
    rng: np.random.Generator | None = None,
    init_scale: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Train a 1-hidden-layer NN on individual-level data via mini-batch SGD.

    f(x) = sum_k a_k sigma(w_k^T x),  trained to minimize (1/n) sum (y_i - f(x_i))^2.

    Uses plain SGD with a linearly decaying learning rate.

    Args:
        X_train: (n, p) centered genotype matrix.
        y_train: (n,) phenotype vector.
        m: Number of hidden units.
        activation: Activation function.
        lr: Initial learning rate.
        max_iters: Number of SGD steps.
        batch_size: Mini-batch size.
        rng: Random generator.
        init_scale: Scale for weight initialization.

    Returns:
        (a, W, loss_history) -- trained parameters and training loss curve.
    """
    if rng is None:
        rng = np.random.default_rng()

    n, p = X_train.shape
    W = rng.standard_normal((m, p)) * init_scale
    a = rng.standard_normal(m) * init_scale

    loss_history = []

    for t in range(max_iters):
        idx = rng.choice(n, size=min(batch_size, n), replace=False)
        X_b = X_train[idx]
        y_b = y_train[idx]
        nb = len(idx)

        current_lr = lr * (1.0 - t / max_iters)

        # Forward
        pre_act = X_b @ W.T               # (nb, m)
        hidden = _apply_activation(pre_act, activation)  # (nb, m)
        y_pred = hidden @ a                # (nb,)

        residual = y_pred - y_b            # (nb,)
        loss = float(np.mean(residual ** 2))
        loss_history.append(loss)

        # Backward: dL/da_k = (2/nb) sum_i residual_i * hidden_ik
        grad_a = (2.0 / nb) * (hidden.T @ residual)

        # dL/dW_{k,j} = (2/nb) sum_i residual_i * a_k * sigma'(pre_act_{i,k}) * X_{i,j}
        act_deriv = _apply_activation_derivative(pre_act, activation)  # (nb, m)
        delta = (2.0 / nb) * (residual[:, None] * act_deriv) * a[None, :]  # (nb, m)
        grad_W = delta.T @ X_b             # (m, p)

        a -= current_lr * grad_a
        W -= current_lr * grad_W

    return a, W, loss_history


# ===================================================================
# Scenario configuration and runner
# ===================================================================

@dataclass
class SimulationScenario:
    """Configuration for one simulation scenario."""
    p: int = 50
    m: int = 5
    n_train: int = 5000
    n_test: int = 2000
    maf_spectrum: str | np.ndarray = "common"
    ld_decay: float = 0.5
    heritability: float = 0.5
    sparsity: float = 0.2
    activation: str = "relu"
    # DGP type: "linear" or "nonlinear"
    dgp_type: str = "linear"
    # Fraction of genetic variance from nonlinear component (when dgp_type="nonlinear")
    nonlinear_frac: float = 0.25
    # Method-specific overrides
    sumstat_lr: float = 0.01
    sumstat_max_iters: int = 3000
    interaction_lr: float = 0.005
    interaction_max_iters: int = 3000
    oracle_lr: float = 0.01
    oracle_max_iters: int = 5000
    oracle_batch_size: int = 256


@dataclass
class ScenarioResult:
    """Results from running one replicate of a scenario."""
    method: str
    r2: float
    weight_cosine: float
    mean_abs_kappa3: float


def _weight_cosine_similarity(
    a: np.ndarray,
    W: np.ndarray,
    beta_star: np.ndarray,
) -> float:
    """Cosine similarity between Wa (effective linear weights) and beta*."""
    effective = W.T @ a
    norm_eff = np.linalg.norm(effective)
    norm_beta = np.linalg.norm(beta_star)
    if norm_eff < 1e-15 or norm_beta < 1e-15:
        return 0.0
    return float(np.dot(effective, beta_star) / (norm_eff * norm_beta))


def _calibrate_gamma(
    X: np.ndarray,
    beta_star: np.ndarray,
    w_star: np.ndarray,
    target_nonlinear_frac: float,
) -> float:
    """Scale gamma so Var(gamma * relu(w*^T x)) / Var(beta*^T x) == target."""
    var_linear = np.var(X @ beta_star)
    relu_vals = np.maximum(0.0, X @ w_star)
    var_relu = np.var(relu_vals)
    if var_relu < 1e-15:
        return 0.0
    return float(np.sqrt(target_nonlinear_frac * var_linear / var_relu))


def run_single_rep(
    scenario: SimulationScenario,
    rng: np.random.Generator,
) -> list[ScenarioResult]:
    """Run one replicate of a simulation scenario.

    Returns a list of ScenarioResult, one per method.
    """
    p = scenario.p

    # 1. Generate MAFs
    maf = generate_maf_spectrum(p, scenario.maf_spectrum, rng)

    # 2. Generate LD matrix
    Sigma = generate_ld_matrix(p, decay=scenario.ld_decay)

    # 3. Generate effect sizes (linear component)
    beta_star, sigma_eps = generate_effect_sizes(
        p, Sigma, scenario.heritability, scenario.sparsity, rng,
    )

    # 4. Generate genotype data
    X_train_raw = generate_binomial_genotypes(scenario.n_train, maf, Sigma, rng)
    X_test_raw = generate_binomial_genotypes(scenario.n_test, maf, Sigma, rng)

    # Center genotypes
    train_means = X_train_raw.mean(axis=0)
    X_train = X_train_raw - train_means
    X_test = X_test_raw - train_means

    # 5. Generate phenotypes
    if scenario.dgp_type == "nonlinear":
        w_star = rng.standard_normal(p) * 0.3
        gamma = _calibrate_gamma(
            X_train, beta_star, w_star, scenario.nonlinear_frac,
        )
        var_linear = np.var(X_train @ beta_star)
        var_nonlinear = np.var(gamma * np.maximum(0.0, X_train @ w_star))
        total_genetic_var = var_linear + var_nonlinear
        sigma_eps = float(np.sqrt(
            total_genetic_var * (1 - scenario.heritability) / scenario.heritability
        ))
        y_train = (
            X_train @ beta_star
            + gamma * np.maximum(0.0, X_train @ w_star)
            + rng.normal(0, sigma_eps, scenario.n_train)
        )
        y_test = (
            X_test @ beta_star
            + gamma * np.maximum(0.0, X_test @ w_star)
            + rng.normal(0, sigma_eps, scenario.n_test)
        )
    else:
        y_train = X_train @ beta_star + rng.normal(0, sigma_eps, scenario.n_train)
        y_test = X_test @ beta_star + rng.normal(0, sigma_eps, scenario.n_test)

    # 6. Summary statistics (including interaction tensor)
    stats = compute_summary_stats_from_genotypes(X_train_raw, y_train, Sigma)
    Sigma_beta_hat = stats["Sigma_beta_hat"]
    E_y2_hat = stats["E_y2_hat"]
    maf_hat = stats["maf"]
    Gamma_hat = stats["Gamma_hat"]
    Cov_ref = stats["Cov_ref"]

    # Compute mean |kappa_3| for this scenario
    cum = snp_cumulants(maf)
    std_kappa3 = cum["kappa3"] / (cum["kappa2"] ** 1.5 + 1e-30)
    mean_abs_k3 = float(np.mean(np.abs(std_kappa3)))

    results = []

    # --- Method 1: Linear PRS ---
    try:
        beta_linear = linear_prs_weights(Sigma, Sigma_beta_hat)
        r2_linear = prediction_r2(X_test, y_test, beta_linear)
        wc_linear = float(np.dot(beta_linear, beta_star) / (
            np.linalg.norm(beta_linear) * np.linalg.norm(beta_star) + 1e-30
        ))
    except np.linalg.LinAlgError:
        r2_linear, wc_linear = 0.0, 0.0

    results.append(ScenarioResult("Linear PRS", r2_linear, wc_linear, mean_abs_k3))

    # --- Method 2: Gaussian NN ---
    gauss_result = train(
        Sigma, Sigma_beta_hat, E_y2_hat,
        m=scenario.m,
        activation=scenario.activation,
        lr=scenario.sumstat_lr,
        max_iters=scenario.sumstat_max_iters,
        init_scale=0.01,
        rng=np.random.default_rng(rng.integers(2**32)),
    )
    r2_gauss = nn_prediction_r2(
        X_test, y_test, gauss_result.a, gauss_result.W, scenario.activation,
    )
    wc_gauss = _weight_cosine_similarity(gauss_result.a, gauss_result.W, beta_star)
    results.append(ScenarioResult("Gaussian NN", r2_gauss, wc_gauss, mean_abs_k3))

    # --- Method 3: Edgeworth NN (warm-started from Gaussian solution) ---
    # Use a smaller LR and fewer iterations since we're fine-tuning
    # from the Gaussian optimum, not training from scratch.
    ew_result = train_edgeworth(
        Sigma, Sigma_beta_hat, E_y2_hat, maf_hat,
        m=scenario.m,
        activation=scenario.activation,
        lr=scenario.sumstat_lr * 0.1,
        max_iters=scenario.sumstat_max_iters // 2,
        rng=np.random.default_rng(rng.integers(2**32)),
        loss_floor=0.0,
        grad_clip=0.5,
        max_backtracks=10,
        a_init=gauss_result.a,
        W_init=gauss_result.W,
    )
    r2_ew = nn_prediction_r2(
        X_test, y_test, ew_result.a, ew_result.W, scenario.activation,
    )
    wc_ew = _weight_cosine_similarity(ew_result.a, ew_result.W, beta_star)
    results.append(ScenarioResult("Edgeworth NN", r2_ew, wc_ew, mean_abs_k3))

    # --- Method 4: Interaction NN (warm-started from Gaussian solution) ---
    int_result = train_interaction(
        Sigma, Sigma_beta_hat, E_y2_hat, Gamma_hat,
        m=scenario.m,
        activation=scenario.activation,
        lr=scenario.interaction_lr,
        max_iters=scenario.interaction_max_iters,
        rng=np.random.default_rng(rng.integers(2**32)),
        grad_clip=0.5,
        max_backtracks=10,
        a_init=gauss_result.a,
        W_init=gauss_result.W,
        Cov_ref=Cov_ref,
    )
    r2_int = nn_prediction_r2(
        X_test, y_test, int_result.a, int_result.W, scenario.activation,
    )
    wc_int = _weight_cosine_similarity(int_result.a, int_result.W, beta_star)
    results.append(ScenarioResult("Interaction NN", r2_int, wc_int, mean_abs_k3))

    # --- Method 5: Oracle NN ---
    oracle_a, oracle_W, _ = train_oracle_nn(
        X_train, y_train,
        m=scenario.m,
        activation=scenario.activation,
        lr=scenario.oracle_lr,
        max_iters=scenario.oracle_max_iters,
        batch_size=scenario.oracle_batch_size,
        rng=np.random.default_rng(rng.integers(2**32)),
    )
    r2_oracle = nn_prediction_r2(
        X_test, y_test, oracle_a, oracle_W, scenario.activation,
    )
    wc_oracle = _weight_cosine_similarity(oracle_a, oracle_W, beta_star)
    results.append(ScenarioResult("Oracle NN", r2_oracle, wc_oracle, mean_abs_k3))

    return results


def run_scenario(
    scenario: SimulationScenario,
    n_reps: int = 10,
    seed: int = 42,
    verbose: bool = False,
) -> list[dict]:
    """Run multiple replicates of a simulation scenario.

    Args:
        scenario: The simulation configuration.
        n_reps: Number of independent replicates.
        seed: Base random seed.
        verbose: Print progress.

    Returns:
        List of dicts with keys: method, rep, r2, weight_cosine, mean_abs_kappa3.
    """
    rows = []
    for rep in range(n_reps):
        if verbose:
            print(f"  Replicate {rep + 1}/{n_reps} ...", flush=True)
        rng = np.random.default_rng(seed + rep)
        rep_results = run_single_rep(scenario, rng)
        for sr in rep_results:
            rows.append({
                "method": sr.method,
                "rep": rep,
                "r2": sr.r2,
                "weight_cosine": sr.weight_cosine,
                "mean_abs_kappa3": sr.mean_abs_kappa3,
            })
    return rows
