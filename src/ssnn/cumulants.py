"""
Genotype cumulant computations for the Edgeworth expansion.

A single SNP genotype X_j ~ Binomial(2, p_j) has known cumulants:
    kappa_1 = 2 p_j                             (mean)
    kappa_2 = 2 p_j (1 - p_j)                   (variance)
    kappa_3 = 2 p_j (1 - p_j) (1 - 2 p_j)      (third cumulant)
    kappa_4 = 2 p_j (1 - p_j) (1 - 6 p_j (1 - p_j))  (fourth cumulant)

These are entirely determined by allele frequency, which is routinely
available from GWAS summary statistics or reference panels.

For projections z = w^T x, the standardized cumulants kappa_tilde_3(w)
and kappa_tilde_4(w) drive the Edgeworth corrections to activation
expectations.  When SNPs are correlated (LD), we use a decorrelation
approximation via Sigma^{-1/2}.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import sqrtm


def snp_cumulants(maf: np.ndarray) -> dict[str, np.ndarray]:
    """Compute per-SNP cumulants from minor allele frequencies.

    Args:
        maf: (p,) array of minor allele frequencies in (0, 1).

    Returns:
        Dictionary with keys 'kappa2', 'kappa3', 'kappa4', each shape (p,).
        (kappa1 = 2*maf is not needed since we work with centered genotypes.)
    """
    p = np.asarray(maf, dtype=float)
    q = 1.0 - p
    pq = p * q

    return {
        "kappa2": 2.0 * pq,
        "kappa3": 2.0 * pq * (1.0 - 2.0 * p),
        "kappa4": 2.0 * pq * (1.0 - 6.0 * pq),
    }


def projection_cumulants_independent(
    w: np.ndarray,
    kappa2: np.ndarray,
    kappa3: np.ndarray,
    kappa4: np.ndarray,
) -> tuple[float, float]:
    """Standardized cumulants of z = w^T x for independent SNPs.

    kappa_tilde_3(w) = sum_j w_j^3 kappa_3,j / (sum_j w_j^2 kappa_2,j)^{3/2}
    kappa_tilde_4(w) = sum_j w_j^4 kappa_4,j / (sum_j w_j^2 kappa_2,j)^2

    Args:
        w: (p,) weight vector.
        kappa2, kappa3, kappa4: (p,) per-SNP cumulants.

    Returns:
        (kappa_tilde_3, kappa_tilde_4) standardized projection cumulants.
    """
    var_z = np.sum(w**2 * kappa2)
    if var_z <= 0:
        return 0.0, 0.0

    kt3 = np.sum(w**3 * kappa3) / var_z**1.5
    kt4 = np.sum(w**4 * kappa4) / var_z**2.0

    return float(kt3), float(kt4)


def projection_cumulant_gradients_independent(
    w: np.ndarray,
    kappa2: np.ndarray,
    kappa3: np.ndarray,
    kappa4: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Gradients of the standardized projection cumulants w.r.t. w.

    Let V = sum_j w_j^2 kappa_{2,j},  N3 = sum_j w_j^3 kappa_{3,j},
        N4 = sum_j w_j^4 kappa_{4,j}.

    Then kt3 = N3 / V^{3/2},  kt4 = N4 / V^2.

        d kt3 / d w_j = (3 w_j^2 kappa_{3,j} V^{3/2} - N3 * 3/2 V^{1/2} * 2 w_j kappa_{2,j})
                         / V^3
                       = (3 w_j^2 kappa_{3,j}) / V^{3/2}
                         - 3 kt3 w_j kappa_{2,j} / V

        d kt4 / d w_j = (4 w_j^3 kappa_{4,j}) / V^2
                         - 4 kt4 w_j kappa_{2,j} / V

    Returns:
        (grad_kt3, grad_kt4) each shape (p,).
    """
    V = np.sum(w**2 * kappa2)
    if V <= 0:
        return np.zeros_like(w), np.zeros_like(w)

    kt3, kt4 = projection_cumulants_independent(w, kappa2, kappa3, kappa4)

    grad_kt3 = (3.0 * w**2 * kappa3) / V**1.5 - 3.0 * kt3 * w * kappa2 / V
    grad_kt4 = (4.0 * w**3 * kappa4) / V**2.0 - 4.0 * kt4 * w * kappa2 / V

    return grad_kt3, grad_kt4


def _compute_Sigma_inv_sqrt(Sigma: np.ndarray) -> np.ndarray:
    """Compute Sigma^{-1/2} via eigendecomposition (more stable than sqrtm)."""
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    eigvals = np.maximum(eigvals, 1e-12)
    return eigvecs @ np.diag(eigvals ** (-0.5)) @ eigvecs.T


def decorrelation_matrix(Sigma: np.ndarray) -> np.ndarray:
    """Compute the decorrelation matrix Sigma^{-1/2}.

    Used to transform correlated SNPs into approximately independent
    components: x_tilde = Sigma^{-1/2} x has Cov(x_tilde) = I.

    Args:
        Sigma: (p, p) positive definite LD covariance matrix.

    Returns:
        (p, p) decorrelation matrix Sigma^{-1/2}.
    """
    return _compute_Sigma_inv_sqrt(Sigma)


def projection_cumulants_ld(
    w: np.ndarray,
    maf: np.ndarray,
    Sigma: np.ndarray,
    Sigma_inv_sqrt: np.ndarray | None = None,
) -> tuple[float, float]:
    """Standardized cumulants of z = w^T x for LD-correlated SNPs.

    Uses the decorrelation approximation:
        w_tilde = Sigma^{-1/2} w
        x_tilde = Sigma^{-1/2} x  (uncorrelated but not independent)

    Then applies the independent-SNP formulas to w_tilde and the cumulants
    of the decorrelated components.  This is exact for second-order and
    approximate for higher-order cumulants (neglects cross-cumulant terms).

    Args:
        w: (p,) weight vector.
        maf: (p,) minor allele frequencies.
        Sigma: (p, p) LD covariance matrix.
        Sigma_inv_sqrt: (p, p) precomputed Sigma^{-1/2} (optional).

    Returns:
        (kappa_tilde_3, kappa_tilde_4) standardized projection cumulants.
    """
    if Sigma_inv_sqrt is None:
        Sigma_inv_sqrt = _compute_Sigma_inv_sqrt(Sigma)

    w_tilde = Sigma_inv_sqrt @ w

    cum = snp_cumulants(maf)

    return projection_cumulants_independent(
        w_tilde,
        cum["kappa2"],
        cum["kappa3"],
        cum["kappa4"],
    )


def projection_cumulant_gradients_ld(
    w: np.ndarray,
    maf: np.ndarray,
    Sigma: np.ndarray,
    Sigma_inv_sqrt: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Gradients of LD-adjusted projection cumulants w.r.t. w.

    Since kt3(w) and kt4(w) are computed via w_tilde = Sigma^{-1/2} w,
    the chain rule gives:

        d kt_r / d w_j = sum_i (d kt_r / d w_tilde_i) * (Sigma^{-1/2})_{ij}

    which is Sigma^{-1/2}^T @ grad_w_tilde(kt_r).

    Returns:
        (grad_kt3, grad_kt4) each shape (p,), gradients w.r.t. w.
    """
    if Sigma_inv_sqrt is None:
        Sigma_inv_sqrt = _compute_Sigma_inv_sqrt(Sigma)

    w_tilde = Sigma_inv_sqrt @ w
    cum = snp_cumulants(maf)

    g3_tilde, g4_tilde = projection_cumulant_gradients_independent(
        w_tilde, cum["kappa2"], cum["kappa3"], cum["kappa4"],
    )

    # Sigma^{-1/2} is symmetric, so Sigma_inv_sqrt^T = Sigma_inv_sqrt
    grad_kt3 = Sigma_inv_sqrt @ g3_tilde
    grad_kt4 = Sigma_inv_sqrt @ g4_tilde

    return grad_kt3, grad_kt4
