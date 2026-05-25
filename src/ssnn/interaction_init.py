"""
Eigenvector-based initialization for the interaction NN.

The Gaussian NN warm-start locks neurons in the additive-effect subspace
(span of beta). The surviving eigenvectors of Gamma_denoised are the
maximum-SNR epistatic directions. Seeding some neurons there lets gradient
descent find solutions it could never reach from the warm-start alone.
"""

from __future__ import annotations
import numpy as np


def compute_eigenvector_init(
    surviving_eigvecs: np.ndarray,   # (k, p) — rows are eigenvectors
    surviving_eigvals: np.ndarray,   # (k,) eigenvalues
    gauss_a: np.ndarray,             # (m,) from Gaussian NN warm-start
    gauss_W: np.ndarray,             # (m, p) from Gaussian NN warm-start
    m: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create initial (a, W) combining eigenvector directions + Gaussian warm-start.

    Strategy:
    1. If no eigenvectors survive, return (gauss_a, gauss_W) unchanged.
    2. Allocate n_eig = min(k_surv, m // 2) neurons to eigenvectors (round up if m=1).
    3. n_eig neurons get W[i] = surviving_eigvec[i], a[i] = small_scale * sign(eigval_i).
       Small a prevents these neurons from dominating the initial prediction.
    4. Remaining m - n_eig neurons copy the largest-|a| neurons from the warm-start.

    Args:
        surviving_eigvecs: (k, p) eigenvectors with |eigenvalue| > MP edge.
        surviving_eigvals: (k,) corresponding eigenvalues.
        gauss_a: (m,) warm-start output weights.
        gauss_W: (m, p) warm-start hidden weights.
        m: total number of neurons.

    Returns:
        (a_init, W_init) with shapes (m,) and (m, p).
    """
    k_surv = len(surviving_eigvals)

    if k_surv == 0:
        return gauss_a.copy(), gauss_W.copy()

    p = gauss_W.shape[1]
    a_init = np.zeros(m)
    W_init = np.zeros((m, p))

    # Number of neurons to assign to eigenvectors
    n_eig = min(k_surv, max(1, m // 2))

    # Sort surviving eigenvectors by |eigenvalue| descending (strongest signal first)
    order = np.argsort(-np.abs(surviving_eigvals))

    small_scale = 0.01  # small a so eigenvector neurons don't dominate initial prediction

    for i in range(n_eig):
        idx = order[i]
        W_init[i] = surviving_eigvecs[idx]  # already unit norm from eigh
        a_init[i] = small_scale * np.sign(surviving_eigvals[idx])

    # Fill remaining neurons with the largest-|a| warm-start neurons
    n_warm = m - n_eig
    if n_warm > 0:
        # Sort warm-start neurons by |a| descending
        warm_order = np.argsort(-np.abs(gauss_a))
        for j in range(n_warm):
            if j < m:
                w_idx = warm_order[j % len(warm_order)]
                W_init[n_eig + j] = gauss_W[w_idx]
                a_init[n_eig + j] = gauss_a[w_idx]

    return a_init, W_init
