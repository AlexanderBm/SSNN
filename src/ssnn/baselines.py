"""
Baseline PRS weight estimation methods for benchmarking.

All methods operate on summary statistics only (Sigma, Sigma_beta)
and return a weight vector of the same dimension as the genotype vector.

Methods:
    1. Clumping + Thresholding (C+T):
       Select top SNPs by marginal association, prune by LD, use
       marginal effect sizes as weights.

    2. LDpred2-style ridge shrinkage:
       beta_hat = (Sigma + lambda * I)^{-1} Sigma_beta
       Equivalent to the infinitesimal LDpred model (all SNPs causal).

    3. PRS-CS-style continuous shrinkage:
       Applies a per-SNP adaptive shrinkage via a global-local prior,
       approximated here by iterative coordinate-wise soft-thresholding
       on the penalized summary-stat regression.
"""

from __future__ import annotations

import numpy as np


def clump_and_threshold(
    Sigma: np.ndarray,
    Sigma_beta: np.ndarray,
    p_threshold: float = 0.05,
    r2_threshold: float = 0.1,
    n: int = 10000,
) -> np.ndarray:
    """Clumping + Thresholding PRS from summary statistics.

    Steps:
        1. Compute approximate z-scores: z_j = sqrt(n) * beta_hat_j / se_j,
           where beta_hat = Sigma^{-1} Sigma_beta and se_j ~ sqrt(Sigma_jj / n).
        2. Convert z-scores to p-values.
        3. Greedily select the most significant SNP, then prune all SNPs
           in LD (|r| > r2_threshold^0.5) with it. Repeat.
        4. Return the pruned marginal effect sizes.

    Args:
        Sigma: (p, p) LD covariance matrix.
        Sigma_beta: (p,) marginal association vector.
        p_threshold: P-value threshold for inclusion.
        r2_threshold: LD r^2 threshold for clumping.
        n: GWAS sample size (used for z-score approximation).

    Returns:
        (p,) PRS weight vector.
    """
    p = len(Sigma_beta)

    marginal_beta = Sigma_beta / np.diag(Sigma)
    se = np.sqrt(np.diag(Sigma) / n)
    se = np.maximum(se, 1e-15)
    z_scores = marginal_beta / se
    p_values = 2.0 * _normal_sf(np.abs(z_scores))

    # LD correlation matrix (standardized)
    diag_inv_sqrt = 1.0 / np.sqrt(np.maximum(np.diag(Sigma), 1e-15))
    R = Sigma * np.outer(diag_inv_sqrt, diag_inv_sqrt)

    # Greedy clumping
    order = np.argsort(p_values)
    selected = []
    pruned = set()

    for idx in order:
        if idx in pruned:
            continue
        if p_values[idx] > p_threshold:
            break
        selected.append(idx)
        for j in range(p):
            if j != idx and R[idx, j] ** 2 > r2_threshold:
                pruned.add(j)

    weights = np.zeros(p)
    for idx in selected:
        weights[idx] = marginal_beta[idx]

    return weights


def ldpred2_inf(
    Sigma: np.ndarray,
    Sigma_beta: np.ndarray,
    h2: float = 0.5,
    p_causal: float = 1.0,
    n: int = 10000,
) -> np.ndarray:
    """LDpred2 infinitesimal model (ridge-style shrinkage).

    Solves:  beta = (Sigma + (M / (n * h2)) * I)^{-1} * Sigma_beta

    where M = p (total number of SNPs), n = GWAS sample size, h2 = heritability.
    When p_causal < 1, scales the penalty by 1/p_causal (sparser prior).

    This is equivalent to the posterior mean under a Gaussian prior
    beta ~ N(0, (h2 / M) * I) with LD matrix Sigma.

    Args:
        Sigma: (p, p) LD covariance matrix.
        Sigma_beta: (p,) marginal association vector.
        h2: Assumed heritability.
        p_causal: Fraction of causal SNPs (1.0 = infinitesimal).
        n: GWAS sample size.

    Returns:
        (p,) shrunk PRS weight vector.
    """
    p = len(Sigma_beta)
    M_eff = p / max(p_causal, 1e-6)
    lam = M_eff / (n * max(h2, 1e-6))

    regularized = Sigma + lam * np.eye(p)
    return np.linalg.solve(regularized, Sigma_beta)


def prs_cs(
    Sigma: np.ndarray,
    Sigma_beta: np.ndarray,
    n: int = 10000,
    a: float = 1.0,
    b: float = 0.5,
    phi: float | None = None,
    max_iters: int = 1000,
    tol: float = 1e-6,
) -> np.ndarray:
    """PRS-CS-style continuous shrinkage from summary statistics.

    Implements a coordinate-descent approximation to the PRS-CS posterior
    mean. PRS-CS places a global-local continuous shrinkage prior:

        beta_j | psi_j ~ N(0, psi_j * phi / n)
        psi_j ~ Gamma(a, b)   (strawderman-berger)

    and estimates the posterior mean via iterative coordinate updates.

    When phi is None, it is estimated from the data (auto mode) using
    the marginal associations.

    Args:
        Sigma: (p, p) LD covariance matrix.
        Sigma_beta: (p,) marginal association vector.
        n: GWAS sample size.
        a: Shape parameter for the local shrinkage prior (default 1).
        b: Rate parameter for the local shrinkage prior (default 0.5).
        phi: Global shrinkage parameter. None = auto-estimate.
        max_iters: Maximum coordinate descent iterations.
        tol: Convergence tolerance.

    Returns:
        (p,) posterior mean PRS weight vector.
    """
    p = len(Sigma_beta)

    if phi is None:
        marginal_beta = Sigma_beta / np.diag(Sigma)
        phi = float(np.mean(marginal_beta ** 2) * n)
        phi = max(phi, 1e-4)

    beta = np.zeros(p)
    psi = np.ones(p)

    Sigma_diag = np.diag(Sigma)

    for iteration in range(max_iters):
        beta_old = beta.copy()

        for j in range(p):
            residual_j = Sigma_beta[j] - Sigma[j] @ beta + Sigma_diag[j] * beta[j]

            # Posterior precision for beta_j
            precision_j = Sigma_diag[j] + n / (psi[j] * phi + 1e-30)
            beta[j] = residual_j / precision_j

        # Update local shrinkage psi_j via MAP of the Gamma posterior:
        # psi_j | beta_j ~ InvGamma(a + 0.5, b + n * beta_j^2 / (2 * phi))
        # MAP of InvGamma(alpha, scale) = scale / (alpha + 1)
        for j in range(p):
            scale_j = b + n * beta[j] ** 2 / (2.0 * phi)
            psi[j] = scale_j / (a + 0.5 + 1.0)
            psi[j] = max(psi[j], 1e-10)

        if np.max(np.abs(beta - beta_old)) < tol:
            break

    return beta


def _normal_sf(x: np.ndarray) -> np.ndarray:
    """Standard normal survival function (1 - CDF), avoiding scipy import."""
    from math import erfc, sqrt
    result = np.empty_like(x, dtype=float)
    for i in range(x.size):
        result.flat[i] = 0.5 * erfc(x.flat[i] / sqrt(2.0))
    return result
