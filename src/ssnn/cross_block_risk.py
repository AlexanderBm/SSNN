"""
Stage-2 cross-block coefficient risk and closed-form solver.

Given pre-trained per-block predictors {f_b}, the extended model is

    F(x) = sum_b f_b(x_b) + sum_{(a,b) in S} c_ab * f_a(x_a) * f_b(x_b)

Expanding L_2(c) = E[(y - F(x))^2] and exploiting block independence
together with E[f_b] ~ 0 (centered predictions) collapses the cross
terms.  All E[f_a f_b f_c f_d] mixed moments vanish when at least one
block index is unpaired, so the loss restricted to {c_ab} reduces to a
diagonal quadratic:

    L_2(c) = const - 2 sum c_ab * q_ab + sum c_ab^2 * E[f_a^2] * E[f_b^2]

where q_ab = E[y f_a f_b] is the cross-block scalar from
cross_block_stats.compute_cross_block_scalars.  Each c_ab therefore has
the closed-form optimum q_ab / (E[f_a^2] * E[f_b^2]).
"""

from __future__ import annotations

import numpy as np


def compute_cross_block_loss(
    c_dict: dict[tuple[int, int], float],
    per_block_Ef2: list[float],
    cross_scalars_q: np.ndarray,
    E_y2: float,
    per_block_Eyf: list[float],
    significant_pairs: list[tuple[int, int]],
) -> float:
    """Stage-2 population loss for cross-block coefficients.

    L_2 = E[y^2] - 2 sum_b E[y f_b] + sum_b E[f_b^2]
          - 2 sum_{(a,b)} c_ab q_ab + sum_{(a,b)} c_ab^2 E[f_a^2] E[f_b^2]

    The Stage-1 constant terms (E[y^2], -2 E[y f_b], E[f_b^2]) are
    included so that the absolute loss is meaningful; the (a,b) sums
    iterate only over `significant_pairs`.
    """
    B = len(per_block_Ef2)
    loss = E_y2
    for b in range(B):
        loss -= 2.0 * per_block_Eyf[b]
        loss += per_block_Ef2[b]

    for (a, b) in significant_pairs:
        c_ab = c_dict.get((a, b), 0.0)
        loss -= 2.0 * c_ab * cross_scalars_q[a, b]
        loss += c_ab ** 2 * per_block_Ef2[a] * per_block_Ef2[b]

    return float(loss)


def solve_cross_block_coefficients(
    per_block_Ef2: list[float],
    cross_scalars_q: np.ndarray,
    significant_pairs: list[tuple[int, int]],
    min_Ef2: float = 1e-6,
    E_y2: float = 0.0,
    n: int = 1,
) -> dict[tuple[int, int], float]:
    """Closed-form minimizer of the decoupled quadratic with ridge regularization.

    c_ab* = q_ab / (E[f_a^2] * E[f_b^2] + ridge), where ridge = E[y^2] / n.
    The ridge term prevents division by near-zero denominators when E[f^2] is
    small and also shrinks spurious large coefficients at low sample sizes.
    When E_y2=0 (default), ridge=0 and the formula reduces to the original
    unregularized closed form.

    When either block predictor is essentially trivial (E[f_b^2] below
    `min_Ef2`), the pair is pinned to c_ab = 0.
    """
    ridge = E_y2 / max(n, 1)
    c_dict: dict[tuple[int, int], float] = {}
    for (a, b) in significant_pairs:
        if per_block_Ef2[a] < min_Ef2 or per_block_Ef2[b] < min_Ef2:
            c_dict[(a, b)] = 0.0
            continue
        denom = per_block_Ef2[a] * per_block_Ef2[b] + ridge
        c_dict[(a, b)] = float(cross_scalars_q[a, b] / denom)
    return c_dict
