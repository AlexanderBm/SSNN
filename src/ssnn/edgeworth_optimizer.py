"""
Gradient descent optimizer for the Edgeworth-corrected summary-statistics
neural network.

Trains a 1-hidden-layer network f(x) = sum_k a_k sigma(w_k^T x) by
minimizing the Edgeworth-corrected population risk L_EW(a, W) computed
entirely from summary statistics (Sigma, Sigma_beta) and allele
frequencies.

The Edgeworth surrogate loss is not bounded below for ReLU (the expansion
can produce negative "densities").  This optimizer includes safeguards:
    - loss_floor: clamps the loss at a non-negative floor (default 0.0).
    - grad_clip: limits the gradient norm per step to prevent large jumps
      into the non-physical region.
    - Backtracking: if a step increases the loss, the step size is halved
      and the step is retried (up to max_backtracks times).
"""

from __future__ import annotations

import numpy as np

from .cumulants import decorrelation_matrix
from .edgeworth_risk import (
    _raw_edgeworth_loss,
    compute_edgeworth_loss,
    compute_edgeworth_grad_a,
    compute_edgeworth_grad_W,
)
from .optimizer import TrainResult


def train_edgeworth(
    Sigma: np.ndarray,
    Sigma_beta: np.ndarray,
    E_y2: float,
    maf: np.ndarray,
    m: int = 5,
    activation: str = "relu",
    lr: float = 0.01,
    max_iters: int = 5000,
    tol: float = 1e-8,
    init_scale: float = 0.01,
    rng: np.random.Generator | None = None,
    verbose: bool = False,
    loss_floor: float | None = 0.0,
    grad_clip: float | None = 1.0,
    max_backtracks: int = 5,
    a_init: np.ndarray | None = None,
    W_init: np.ndarray | None = None,
) -> TrainResult:
    """Train a 1-hidden-layer NN on summary statistics with Edgeworth corrections.

    This extends the Gaussian-only optimizer by incorporating genotype
    non-Gaussianity through Edgeworth corrections parameterized by
    allele-frequency-derived cumulants.

    Args:
        Sigma: (p, p) LD covariance matrix.
        Sigma_beta: (p,) = Sigma @ beta*.
        E_y2: scalar E[y^2].
        maf: (p,) minor allele frequencies.
        m: Number of hidden units.
        activation: Activation function name.
        lr: Learning rate.
        max_iters: Maximum gradient descent iterations.
        tol: Convergence tolerance on relative loss change.
        init_scale: Scale for random weight initialization.
        rng: Random generator for initialization.
        verbose: Print progress every 500 iterations.
        loss_floor: lower bound for the loss surface; None disables.
            The true MSE is always >= 0, so 0.0 is a safe default.
        grad_clip: if set, rescale the combined gradient so its norm
            does not exceed this value. Prevents large jumps into the
            region where the Edgeworth expansion breaks down.
        max_backtracks: number of step-halving attempts when a GD step
            increases the (raw) loss. 0 disables backtracking.
        a_init: Optional initial second-layer weights (warm start).
        W_init: Optional initial first-layer weights (warm start).

    Returns:
        TrainResult with optimized (a, W) and loss history.
    """
    if rng is None:
        rng = np.random.default_rng()

    p = Sigma.shape[0]
    if W_init is not None:
        W = W_init.copy()
        m = W.shape[0]
    else:
        W = rng.standard_normal((m, p)) * init_scale
    if a_init is not None:
        a = a_init.copy()
    else:
        a = rng.standard_normal(m) * init_scale

    Sigma_inv_sqrt = decorrelation_matrix(Sigma)

    loss_history = []
    converged = False

    def _loss(a_val, W_val):
        return compute_edgeworth_loss(
            a_val, W_val, Sigma, Sigma_beta, E_y2, maf,
            activation, Sigma_inv_sqrt, loss_floor,
        )

    def _raw_loss(a_val, W_val):
        return _raw_edgeworth_loss(
            a_val, W_val, Sigma, Sigma_beta, E_y2, maf,
            activation, Sigma_inv_sqrt,
        )

    for i in range(max_iters):
        loss = _loss(a, W)
        loss_history.append(loss)

        if verbose and i % 500 == 0:
            print(f"  iter {i:5d}  loss = {loss:.8f}")

        if i > 0:
            rel_change = abs(loss_history[-1] - loss_history[-2]) / (
                abs(loss_history[-2]) + 1e-30
            )
            if rel_change < tol:
                converged = True
                break

        grad_a = compute_edgeworth_grad_a(
            a, W, Sigma, Sigma_beta, maf, activation, Sigma_inv_sqrt
        )
        grad_W = compute_edgeworth_grad_W(
            a, W, Sigma, Sigma_beta, maf, activation, Sigma_inv_sqrt
        )

        if loss_floor is not None and _raw_loss(a, W) <= loss_floor:
            grad_a = np.zeros_like(grad_a)
            grad_W = np.zeros_like(grad_W)

        if grad_clip is not None:
            grad_norm = np.sqrt(np.sum(grad_a**2) + np.sum(grad_W**2))
            if grad_norm > grad_clip:
                scale = grad_clip / grad_norm
                grad_a = grad_a * scale
                grad_W = grad_W * scale

        step_lr = lr
        a_new = a - step_lr * grad_a
        W_new = W - step_lr * grad_W

        for _bt in range(max_backtracks):
            new_loss = _loss(a_new, W_new)
            if new_loss <= loss:
                break
            step_lr *= 0.5
            a_new = a - step_lr * grad_a
            W_new = W - step_lr * grad_W

        a = a_new
        W = W_new

    if not converged:
        loss = _loss(a, W)
        loss_history.append(loss)

    return TrainResult(
        a=a,
        W=W,
        loss_history=loss_history,
        converged=converged,
        n_iters=len(loss_history),
    )
