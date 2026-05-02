"""
Utilities: synthetic data generation, LD matrix construction, linear PRS.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def generate_ld_matrix(
    p: int,
    n_blocks: int | None = None,
    decay: float = 0.5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a realistic block-diagonal LD (covariance) matrix.

    Each block has exponentially decaying correlations:
        Sigma_{ij} = decay^|i-j| within the block, 0 across blocks.

    Args:
        p: Total number of SNPs.
        n_blocks: Number of LD blocks. Defaults to p // 5 (blocks of ~5 SNPs).
        decay: Correlation decay parameter. 0.5 mimics moderate LD.
        rng: Random generator (unused, kept for API consistency).

    Returns:
        (p, p) positive definite covariance matrix.
    """
    if n_blocks is None:
        n_blocks = max(1, p // 5)

    block_sizes = [p // n_blocks] * n_blocks
    for i in range(p % n_blocks):
        block_sizes[i] += 1

    blocks = []
    for size in block_sizes:
        block = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                block[i, j] = decay ** abs(i - j)
        blocks.append(block)

    return np.block([
        [blocks[i] if i == j else np.zeros((block_sizes[i], block_sizes[j]))
         for j in range(n_blocks)]
        for i in range(n_blocks)
    ])


def generate_gwas_summary_stats(
    Sigma: np.ndarray,
    beta_star: np.ndarray,
    n: int,
    sigma_eps: float = 1.0,
    rng: np.random.Generator | None = None,
    return_individual_data: bool = False,
) -> dict:
    """Simulate a GWAS and return summary statistics.

    Generates individual-level data x ~ N(0, Sigma), y = beta*^T x + eps,
    then computes summary statistics.

    Args:
        Sigma: (p, p) LD covariance matrix.
        beta_star: (p,) true effect sizes.
        n: Sample size.
        sigma_eps: Noise standard deviation.
        rng: Random generator.
        return_individual_data: If True, also return (X, y).

    Returns:
        Dictionary with keys:
            Sigma_beta: (p,) = Sigma @ beta* (population-level, for infinite n)
            Sigma_beta_hat: (p,) = X^T y / n (finite-sample estimate)
            E_y2: scalar = beta*^T Sigma beta* + sigma_eps^2 (population)
            E_y2_hat: scalar = mean(y^2) (finite-sample estimate)
            Sigma, beta_star, n, sigma_eps: inputs echoed back
            X, y: (optional) individual-level data
    """
    if rng is None:
        rng = np.random.default_rng()

    p = len(beta_star)
    X = rng.multivariate_normal(np.zeros(p), Sigma, size=n)
    eps = rng.normal(0, sigma_eps, size=n)
    y = X @ beta_star + eps

    result = {
        "Sigma_beta": Sigma @ beta_star,
        "Sigma_beta_hat": X.T @ y / n,
        "E_y2": float(beta_star @ Sigma @ beta_star + sigma_eps**2),
        "E_y2_hat": float(np.mean(y**2)),
        "Sigma": Sigma,
        "beta_star": beta_star,
        "n": n,
        "sigma_eps": sigma_eps,
    }

    if return_individual_data:
        result["X"] = X
        result["y"] = y

    return result


def linear_prs_weights(Sigma: np.ndarray, Sigma_beta: np.ndarray) -> np.ndarray:
    """Compute optimal linear PRS weights: beta* = Sigma^{-1} Sigma_beta.

    Under the model y = beta*^T x + eps with x ~ N(0, Sigma), the optimal
    linear predictor uses weights beta* = Sigma^{-1} (Sigma beta*).

    Uses a stable solve rather than explicit inversion.
    """
    return np.linalg.solve(Sigma, Sigma_beta)


def prediction_r2(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Compute prediction R^2 = 1 - MSE / Var(y) on held-out data."""
    y_pred = X @ weights
    ss_res = np.mean((y - y_pred) ** 2)
    ss_tot = np.var(y)
    if ss_tot == 0.0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def nn_predict(
    X: np.ndarray,
    a: np.ndarray,
    W: np.ndarray,
    activation: str = "relu",
) -> np.ndarray:
    """Predict y from individual-level data using the trained NN.

    f(x) = sum_k a_k sigma(w_k^T x)
    """
    hidden = X @ W.T  # (n, m)

    if activation == "relu":
        hidden = np.maximum(0, hidden)
    elif activation == "sigmoid":
        hidden = 1.0 / (1.0 + np.exp(-hidden))
    elif activation == "identity":
        pass
    else:
        raise ValueError(f"Unknown activation: {activation!r}")

    return hidden @ a


def nn_prediction_r2(
    X: np.ndarray,
    y: np.ndarray,
    a: np.ndarray,
    W: np.ndarray,
    activation: str = "relu",
) -> float:
    """Compute R^2 for the neural network predictor."""
    y_pred = nn_predict(X, a, W, activation)
    ss_res = np.mean((y - y_pred) ** 2)
    ss_tot = np.var(y)
    if ss_tot == 0.0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)
