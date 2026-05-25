"""
Cross-block interaction statistics for Phase 4 (cross-block epistasis).

Given a block-additive PRS model F(x) = sum_b f_b(x_b), individual blocks
cannot represent interaction terms of the form gamma * x_i^A * x_j^B that
couple variables across LD blocks.  This module computes the cross-block
interaction signal projected onto effective per-block directions, together
with a triage rule for selecting significant block pairs.
"""

from __future__ import annotations

import numpy as np


def compute_cross_block_scalars(
    X_blocks: list[np.ndarray],
    y: np.ndarray,
    beta_hats: list[np.ndarray],
) -> np.ndarray:
    """Compute the cross-block interaction scalar q_ab for every block pair.

    q_ab = (1/n) sum_i (beta_hat_a^T x_{a,i}) * (beta_hat_b^T x_{b,i}) * y_i

    Args:
        X_blocks: list of B arrays, each (n, p_b) centered genotypes.
        y: (n,) centered phenotype.
        beta_hats: list of B arrays, each (p_b,) per-block ridge weights.

    Returns:
        (B, B) symmetric matrix of cross-block scalars.
    """
    B = len(X_blocks)
    n = len(y)
    projections = np.stack([X_blocks[b] @ beta_hats[b] for b in range(B)], axis=1)
    weighted = projections * y[:, None]
    q = weighted.T @ projections / n
    q = 0.5 * (q + q.T)
    return q


def compute_cross_block_projected(
    X_blocks: list[np.ndarray],
    y: np.ndarray,
    eigvec_blocks: list[np.ndarray | None],
) -> dict[tuple[int, int], np.ndarray]:
    """Compute cross-block scalars in surviving eigenvector bases.

    For each pair (a, b) with both blocks having K_a, K_b > 0 surviving
    eigenvectors:

        Q_ab[r, s] = (1/n) sum_i (u_{a,r}^T x_{a,i}) * (u_{b,s}^T x_{b,i}) * y_i

    Args:
        X_blocks: list of B arrays, each (n, p_b) centered genotypes.
        y: (n,) centered phenotype.
        eigvec_blocks: list of B arrays (p_b, K_b) or None / empty.

    Returns:
        Dict mapping (a, b) with a < b to (K_a, K_b) matrices.
    """
    B = len(X_blocks)
    n = len(y)

    projections = []
    for b in range(B):
        U = eigvec_blocks[b]
        if U is None or (hasattr(U, "size") and U.size == 0):
            projections.append(None)
        else:
            projections.append(X_blocks[b] @ U)

    out: dict[tuple[int, int], np.ndarray] = {}
    for a in range(B):
        Pa = projections[a]
        if Pa is None:
            continue
        for b in range(a + 1, B):
            Pb = projections[b]
            if Pb is None:
                continue
            Q = (Pa * y[:, None]).T @ Pb / n
            out[(a, b)] = Q
    return out


def triage_cross_pairs(
    q_matrix: np.ndarray,
    per_block_Ef2: list[float],
    E_y2: float,
    n: int,
    threshold: float = 3.0,
) -> list[tuple[int, int]]:
    """Select significant cross-block pairs by null-noise z-score.

    Under the null hypothesis of no cross-block interaction, q_ab has
    approximate variance

        Var(q_ab) ~ (1/n) * E[f_a^2] * E[f_b^2] * E[y^2]

    by independence of the three factors when blocks a and b carry no
    coupled signal.  Pairs with |q_ab| above `threshold` standard
    deviations are returned (upper triangle only).
    """
    B = len(per_block_Ef2)
    pairs: list[tuple[int, int]] = []
    for a in range(B):
        for b in range(a + 1, B):
            var_null = per_block_Ef2[a] * per_block_Ef2[b] * E_y2 / max(n, 1)
            if var_null <= 0:
                continue
            z = abs(q_matrix[a, b]) / np.sqrt(var_null)
            if z > threshold:
                pairs.append((a, b))
    return pairs
