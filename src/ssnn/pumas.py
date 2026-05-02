"""
PUMAS: Pseudo-subset summary statistics for train/validation splitting.

Implements the conditional distribution from Zhao et al.:

    beta_hat_tr | beta_hat ~ N(
        (n_tr / N) * beta_hat,
        (n_tr * (N - n_tr) / N^2) * (1/n_tr) * Sigma^{-1}
    )

This allows splitting a single set of GWAS summary statistics into
pseudo-training and pseudo-validation subsets *without* individual-level data.

Two uses:
    1. Train/validation splitting for hyperparameter tuning.
    2. Summary-stat R^2 evaluation (PUMAS Eq. 20).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve


@dataclass
class PUMASSplit:
    """One train/validation split of pseudo-subset summary statistics."""
    Sigma_beta_train: np.ndarray
    Sigma_beta_val: np.ndarray
    E_y2_train: float
    E_y2_val: float
    n_train: int
    n_val: int


def _stable_inverse_sqrt(Sigma: np.ndarray) -> np.ndarray:
    """Sigma^{-1/2} via eigendecomposition, clipping small eigenvalues."""
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    eigvals = np.maximum(eigvals, 1e-10)
    return eigvecs @ np.diag(eigvals ** (-0.5)) @ eigvecs.T


def generate_pumas_split(
    Sigma_beta: np.ndarray,
    E_y2: float,
    Sigma: np.ndarray,
    N: int,
    n_train: int,
    rng: np.random.Generator,
) -> PUMASSplit:
    """Generate one PUMAS pseudo train/validation split.

    Given full-sample summary statistics (Sigma_beta = X^T y / N, E_y2),
    draw pseudo-training summary stats from the PUMAS conditional
    distribution and derive the validation residual.

    Args:
        Sigma_beta: (p,) full-sample marginal association vector (= Sigma @ beta_hat).
        E_y2: Full-sample E[y^2].
        Sigma: (p, p) LD covariance matrix.
        N: Full GWAS sample size.
        n_train: Desired training subset size.
        rng: Random generator.

    Returns:
        PUMASSplit with pseudo-training and pseudo-validation summary stats.
    """
    if n_train >= N:
        raise ValueError(f"n_train ({n_train}) must be < N ({N})")

    p = len(Sigma_beta)
    n_val = N - n_train
    frac = n_train / N

    # Mean of the conditional: (n_tr / N) * beta_hat
    # where beta_hat = Sigma^{-1} Sigma_beta (per-SNP marginal associations)
    # But we work with Sigma_beta directly: mean of Sigma_beta_train = frac * Sigma_beta
    mean_train = frac * Sigma_beta

    # Covariance of the conditional on Sigma_beta_train:
    # Cov = (n_tr * (N - n_tr) / N^2) * (Sigma / n_tr)
    #      = ((N - n_tr) / N^2) * Sigma
    # This is because Sigma_beta_train = Sigma @ beta_hat_train,
    # and Cov(beta_hat_train) = (n_tr(N-n_tr)/N^2) * (1/n_tr) * Sigma^{-1}
    # so Cov(Sigma @ beta_hat_train) = Sigma * Cov(beta_hat_train) * Sigma
    #    = (n_tr(N-n_tr)/N^2) * (1/n_tr) * Sigma
    #    = ((N-n_tr)/(N^2)) * Sigma
    cov_scale = float(n_val) / (float(N) ** 2)

    # Draw from multivariate normal with covariance cov_scale * Sigma
    L = np.linalg.cholesky(Sigma)
    z = rng.standard_normal(p)
    noise = np.sqrt(cov_scale) * (L @ z)

    Sigma_beta_train = mean_train + noise

    # Validation summary stats from the residual:
    # Sigma_beta_val = (N * Sigma_beta - n_train * Sigma_beta_train) / n_val
    Sigma_beta_val = (N * Sigma_beta - n_train * Sigma_beta_train) / n_val

    # E[y^2] partitioning:
    # The per-sample E[y^2] is approximately the same across subsets,
    # with noise proportional to 1/sqrt(n). We add a small perturbation
    # to make the splits not identical.
    E_y2_noise_scale = np.sqrt(2.0 * E_y2**2 / N)
    E_y2_train = E_y2 + rng.normal(0, E_y2_noise_scale * np.sqrt(N / n_train - 1))
    E_y2_val = E_y2 + rng.normal(0, E_y2_noise_scale * np.sqrt(N / n_val - 1))
    E_y2_train = max(E_y2_train, 0.01)
    E_y2_val = max(E_y2_val, 0.01)

    return PUMASSplit(
        Sigma_beta_train=Sigma_beta_train,
        Sigma_beta_val=Sigma_beta_val,
        E_y2_train=E_y2_train,
        E_y2_val=E_y2_val,
        n_train=n_train,
        n_val=n_val,
    )


def generate_pumas_splits(
    Sigma_beta: np.ndarray,
    E_y2: float,
    Sigma: np.ndarray,
    N: int,
    n_splits: int = 5,
    train_fraction: float = 0.8,
    seed: int = 42,
) -> list[PUMASSplit]:
    """Generate multiple PUMAS pseudo train/validation splits.

    Args:
        Sigma_beta: (p,) full-sample marginal association vector.
        E_y2: Full-sample E[y^2].
        Sigma: (p, p) LD covariance matrix.
        N: Full GWAS sample size.
        n_splits: Number of random splits.
        train_fraction: Fraction of N used for training in each split.
        seed: Base random seed.

    Returns:
        List of PUMASSplit objects.
    """
    n_train = int(train_fraction * N)
    n_train = max(1, min(n_train, N - 1))

    splits = []
    for i in range(n_splits):
        rng = np.random.default_rng(seed + i)
        splits.append(
            generate_pumas_split(Sigma_beta, E_y2, Sigma, N, n_train, rng)
        )
    return splits


def pumas_summary_r2(
    Sigma_beta_val: np.ndarray,
    weights: np.ndarray,
    Sigma: np.ndarray,
    E_y2_val: float,
) -> float:
    """Compute the summary-statistic R^2 on a PUMAS validation set.

    This is the PUMAS Eq. 20 analog:
        R^2 = 1 - (E[y^2] - 2 w^T Sigma_beta_val + w^T Sigma w) / E[y^2]
            = (2 w^T Sigma_beta_val - w^T Sigma w) / E[y^2]

    This measures how well the PRS weights explain phenotypic variance
    in the pseudo-validation sample.

    Args:
        Sigma_beta_val: (p,) validation-set marginal associations.
        weights: (p,) PRS weight vector.
        Sigma: (p, p) LD covariance matrix.
        E_y2_val: Validation-set E[y^2].

    Returns:
        Summary-stat R^2 (can be negative if weights are poor).
    """
    pred_var = float(2.0 * weights @ Sigma_beta_val - weights @ Sigma @ weights)
    if E_y2_val <= 0:
        return 0.0
    return pred_var / E_y2_val


def pumas_nn_summary_r2(
    Sigma_beta_val: np.ndarray,
    E_y2_val: float,
    a: np.ndarray,
    W: np.ndarray,
    Sigma: np.ndarray,
    maf: np.ndarray | None = None,
    activation: str = "relu",
    use_edgeworth: bool = False,
) -> float:
    """Compute summary-stat R^2 for a neural network predictor.

    For the NN f(x) = sum_k a_k sigma(w_k^T x), the population risk is:
        L = E[y^2] - 2 E[y f(x)] + E[f(x)^2]

    And R^2 = 1 - L / E[y^2].

    Uses the (Edgeworth-corrected) loss function from the SSNN library
    with the validation-set summary statistics.

    Args:
        Sigma_beta_val: (p,) validation-set marginal associations.
        E_y2_val: Validation-set E[y^2].
        a: (m,) second-layer weights.
        W: (m, p) first-layer weights.
        Sigma: (p, p) LD covariance matrix.
        maf: (p,) MAFs (required if use_edgeworth=True).
        activation: Activation function name.
        use_edgeworth: Use Edgeworth-corrected loss.

    Returns:
        Summary-stat R^2.
    """
    if use_edgeworth:
        if maf is None:
            raise ValueError("maf required for Edgeworth R^2")
        from .edgeworth_risk import compute_edgeworth_loss
        loss = compute_edgeworth_loss(
            a, W, Sigma, Sigma_beta_val, E_y2_val, maf,
            activation, loss_floor=None,
        )
    else:
        from .population_risk import compute_loss
        loss = compute_loss(a, W, Sigma, Sigma_beta_val, E_y2_val, activation)

    if E_y2_val <= 0:
        return 0.0
    return 1.0 - loss / E_y2_val
