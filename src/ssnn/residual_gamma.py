"""
Residualized summary statistics for the Interaction-SSNN.

In genome-wide training, each LD block's Gamma is computed against the
full phenotype y, which contains signal from all other blocks.  This
inflates estimation noise by a factor proportional to Var(y)/Var(y_block).

The fix: subtract the fitted linear PRS prediction before computing Gamma,
giving residuals r = y - y_hat_linear whose variance is
Var(r) = (1 - R^2_linear) * Var(y).

The correction uses only the reference panel (already required for Sigma)
and the fitted linear weights (derived from summary statistics):

    CORRECTION_b = (1/n_ref) X_ref_b^T Diag(X_ref_b beta_hat_b) X_ref_b
    Gamma_b^resid = Gamma_b^raw - CORRECTION_b

This is the Option A (reference panel) correction from the derivation.
The diagonal approximation (Option B) is not implemented because it
assumes within-block linkage equilibrium, which LD blocks violate by
construction (~70% error at rho=0.8).

All functions operate on a single block.  Multi-block orchestration
(accumulating global E_r2 across blocks) lives in the calling code.
"""

from __future__ import annotations

import numpy as np


def compute_gamma_correction(
    X_ref: np.ndarray,
    beta_hat: np.ndarray,
) -> np.ndarray:
    """Compute the within-block third-moment correction for Gamma.

    CORRECTION = (1/n_ref) X_ref^T Diag(X_ref @ beta_hat) X_ref

    This estimates E[x_i x_j (beta_hat^T x)], the expected contamination
    of Gamma^raw from the fitted linear signal.  Using reference panel
    genotypes captures all third moments including LD-induced off-diagonal
    terms, which the diagonal (MAF-only) approximation misses.

    Args:
        X_ref: (n_ref, p) reference panel genotype matrix (centered).
        beta_hat: (p,) fitted linear PRS weights.

    Returns:
        (p, p) symmetric correction matrix.
    """
    v = X_ref @ beta_hat  # (n_ref,) linear prediction on reference panel
    return (X_ref * v[:, None]).T @ X_ref / len(X_ref)


def compute_residual_gamma(
    Gamma_raw: np.ndarray,
    X_ref: np.ndarray,
    beta_hat: np.ndarray,
) -> np.ndarray:
    """Residualize Gamma by removing the fitted linear PRS contamination.

    Gamma^resid = Gamma^raw - CORRECTION

    where CORRECTION = (1/n_ref) X_ref^T Diag(X_ref @ beta_hat) X_ref.

    Cross-block contamination (from other blocks' linear signals) has zero
    expectation under block independence and is left as-is; at GWAS sample
    sizes (n ~ 10^5) this is negligible relative to the within-block term.

    Args:
        Gamma_raw: (p, p) raw interaction tensor E[x_i x_j y].
        X_ref: (n_ref, p) reference panel genotype matrix (centered).
        beta_hat: (p,) fitted linear PRS weights.

    Returns:
        (p, p) residualized interaction tensor.
    """
    return Gamma_raw - compute_gamma_correction(X_ref, beta_hat)


def compute_residual_sigma_beta(
    Sigma_beta: np.ndarray,
    Sigma: np.ndarray,
    beta_hat: np.ndarray,
) -> np.ndarray:
    """Residualize Sigma_beta by removing the fitted linear contribution.

    Sigma_beta^resid = Sigma_beta - Sigma @ beta_hat

    For OLS weights (beta_hat = Sigma^{-1} Sigma_beta) this is exactly 0
    at the population level: the residual is orthogonal to the linear span.

    For ridge weights (beta_hat = (Sigma + lambda*I)^{-1} Sigma_beta) this
    equals lambda * (Sigma + lambda*I)^{-1} Sigma_beta, which is non-zero
    and maintains the Stein gradient term for optimizer conditioning.

    Use ridge weights (lambda > 0) in practice for numerical stability.

    Args:
        Sigma_beta: (p,) marginal associations E[x y].
        Sigma: (p, p) LD covariance matrix.
        beta_hat: (p,) fitted linear weights (OLS or ridge).

    Returns:
        (p,) residualized marginal associations.
    """
    return Sigma_beta - Sigma @ beta_hat


def compute_genome_wide_residual_gamma(
    X_blocks: list[np.ndarray],
    y: np.ndarray,
    beta_hat_blocks: list[np.ndarray],
    n: int,
) -> list[np.ndarray]:
    """Two-pass genome-wide Γ residualization using leave-one-out linear residuals.

    For each block b:
      r_b = y - Σ_{b'≠b} X_{b'} @ beta_hat_{b'}  (leave-one-out residual)
      Γ_b = (1/n) X_b^T diag(r_b) X_b

    This removes cross-block linear contamination from Γ.  The noise in Γ_b
    now comes from Var(r_b) instead of Var(y), reducing the MP threshold by
    approximately (1 - R²_other_blocks).

    Block b's own linear signal remains in r_b, so the within-block correction
    from compute_residual_gamma() is still needed; this function is orthogonal
    to that correction (it handles cross-block contamination only).

    Args:
        X_blocks: list of (n, p_b) centered genotype matrices (training data).
        y: (n,) phenotype vector (training data, centered).
        beta_hat_blocks: list of (p_b,) fitted ridge weights per block.
        n: sample size (must equal len(y)).

    Returns:
        list of (p_b, p_b) Γ matrices computed against leave-one-out residuals.
    """
    B = len(beta_hat_blocks)

    # Compute genome-wide linear PRS once (sum over all blocks)
    prs_total = sum(X_blocks[b] @ beta_hat_blocks[b] for b in range(B))

    Gammas = []
    for b in range(B):
        # Leave-one-out: remove all blocks except b from the phenotype
        prs_other = prs_total - X_blocks[b] @ beta_hat_blocks[b]
        r_b = y - prs_other  # preserves block b's full signal (linear + epistatic)
        Gamma_b = X_blocks[b].T @ (X_blocks[b] * r_b[:, None]) / n
        Gammas.append(Gamma_b)

    return Gammas


def compute_residual_sigma_other2(
    E_y2: float,
    Sigma_beta_blocks: list[np.ndarray],
    beta_hat_blocks: list[np.ndarray],
    block_idx: int,
) -> float:
    """σ²_other for the leave-one-out residual r_b = y - Σ_{b'≠b} β̂_{b'}^T x_{b'}.

    Var(r_b) ≈ E[y²] - Σ_{b'≠b} Σ_β_{b'}^T β̂_{b'}

    This is the correct noise variance to pass to triage_block() and denoise_gamma()
    when Γ was computed using compute_genome_wide_residual_gamma().

    Args:
        E_y2: global E[y²].
        Sigma_beta_blocks: list of (p_b,) marginal association vectors.
        beta_hat_blocks: list of (p_b,) fitted linear weight vectors.
        block_idx: index of the block being processed (excluded from PVE sum).

    Returns:
        Scalar noise variance (clamped to ≥ 1e-6).
    """
    pve_other = sum(
        max(0.0, float(np.dot(Sigma_beta_blocks[b], beta_hat_blocks[b])))
        for b in range(len(beta_hat_blocks))
        if b != block_idx
    )
    return max(1e-6, E_y2 - pve_other)


def compute_residual_e_y2(
    E_y2: float,
    Sigma_beta_blocks: list[np.ndarray],
    beta_hat_blocks: list[np.ndarray],
) -> float:
    """Compute residual variance E[r^2] from summary statistics.

    E[r^2] = E[y^2] - sum_b Sigma_beta_b^T beta_hat_b

    The sum estimates the total proportion of variance explained (PVE) by
    the linear PRS across all blocks.  Under block independence and OLS
    weights this equals E[y^2] * R^2_linear.

    A negative return value indicates overfitting (p/n too large) or
    population mismatch between the LD reference and GWAS cohort.

    Args:
        E_y2: scalar E[y^2] from GWAS summary statistics.
        Sigma_beta_blocks: list of (p_b,) marginal association vectors.
        beta_hat_blocks: list of (p_b,) fitted linear weight vectors.

    Returns:
        Scalar E[r^2].
    """
    pve = sum(
        float(np.dot(sb, bh))
        for sb, bh in zip(Sigma_beta_blocks, beta_hat_blocks)
    )
    return E_y2 - pve
