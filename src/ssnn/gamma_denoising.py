"""
Spectral denoising of the interaction tensor Gamma using Marchenko-Pastur thresholding.

The empirical Gamma = (1/n) X^T diag(y) X is a noisy estimate of the true
interaction tensor. Under the null (y independent of x_i x_j), its eigenvalues
follow the Marchenko-Pastur distribution with upper edge lambda_+ = sigma²(1+sqrt(p/n))².

We threshold eigenvalues: keep those with |lambda| > lambda_+ and reconstruct.
Negative eigenvalues above the threshold correspond to suppressive epistasis and
should be kept with their sign.
"""

from __future__ import annotations
import numpy as np


def marchenko_pastur_edge(p: int, n: int, sigma2: float) -> float:
    """Noise floor threshold for Gamma = (1/n) X^T diag(y) X.

    The classical Wishart MP formula sigma²(1+sqrt(p/n))² applies to
    sample covariance matrices (1/n)X^T X, NOT to the weighted matrix
    (1/n) X^T diag(y) X. For the latter, the max noise eigenvalue scales
    empirically as ~C·sqrt(sigma2·p/n). We use C=3.5 for a ~2.5-sigma
    safety margin above the empirical p99 of noise eigenvalues.

    Args:
        p: Matrix dimension.
        n: Number of samples used to estimate the matrix.
        sigma2: Noise variance (typically sigma_other2 for Gamma estimation).

    Returns:
        Noise floor threshold (scalar).
    """
    gamma = p / max(n, 1)
    return 3.5 * np.sqrt(sigma2 * gamma)


def denoise_gamma(
    Gamma_resid: np.ndarray,
    sigma_other2: float,
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spectrally denoise the residualized Gamma tensor via MP thresholding.

    Steps:
    1. Eigendecompose Gamma_resid using np.linalg.eigh (symmetric matrix)
    2. Compute MP edge from (p, n, sigma_other2)
    3. Zero eigenvalues with |eigenvalue| < MP edge
    4. Reconstruct denoised matrix

    Args:
        Gamma_resid: (p, p) symmetric residualized interaction tensor.
        sigma_other2: Noise variance for Gamma estimation (cross-block signal variance).
        n: Sample size used to estimate Gamma.

    Returns:
        Tuple of (Gamma_denoised, surviving_eigenvectors, surviving_eigenvalues):
        - Gamma_denoised: (p, p) denoised matrix (all zeros if no eigenvalues survive)
        - surviving_eigenvectors: (k, p) array — rows are eigenvectors (may be empty)
        - surviving_eigenvalues: (k,) array — corresponding eigenvalues (may be empty)
    """
    p = Gamma_resid.shape[0]

    # Use eigh for symmetric matrices (more stable, guaranteed real eigenvalues)
    eigenvalues, eigenvectors = np.linalg.eigh(Gamma_resid)  # eigenvalues ascending
    # eigenvectors[:, i] is the i-th eigenvector (columns)

    threshold = marchenko_pastur_edge(p, n, sigma_other2)

    # Keep eigenvalues by absolute value (both positive and negative spikes matter)
    survive_mask = np.abs(eigenvalues) > threshold

    surv_vals = eigenvalues[survive_mask]
    surv_vecs = eigenvectors[:, survive_mask].T  # shape (k, p) — rows are eigenvectors

    if len(surv_vals) == 0:
        return np.zeros_like(Gamma_resid), np.empty((0, p)), np.empty(0)

    # Reconstruct: sum_k lambda_k v_k v_k^T
    Gamma_denoised = sum(
        surv_vals[i] * np.outer(surv_vecs[i], surv_vecs[i])
        for i in range(len(surv_vals))
    )

    return Gamma_denoised, surv_vecs, surv_vals


def triage_block(
    Gamma_resid: np.ndarray,
    sigma_other2: float,
    n: int,
) -> bool:
    """Return True if any eigenvalue of Gamma_resid exceeds the MP edge.

    Quick check to decide whether interaction training is worthwhile for this block.
    """
    p = Gamma_resid.shape[0]
    threshold = marchenko_pastur_edge(p, n, sigma_other2)
    eigenvalues = np.linalg.eigvalsh(Gamma_resid)
    return bool(np.any(np.abs(eigenvalues) > threshold))
