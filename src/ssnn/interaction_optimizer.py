"""
Gradient descent optimizer for the interaction-SSNN.

Trains a 1-hidden-layer network f(x) = sum_k a_k sigma(w_k^T x) by
minimizing the interaction-extended population risk L_int(a, W), which
uses both first-order (Sigma_beta) and second-order (Gamma) summary
statistics.

Supports optional warm-starting from a pre-trained Gaussian solution,
gradient clipping, and backtracking line search.
"""

from __future__ import annotations

import numpy as np

from .interaction_risk import compute_interaction_loss, compute_interaction_gradients
from .optimizer import TrainResult


def train_interaction(
    Sigma: np.ndarray,
    Sigma_beta: np.ndarray,
    E_y2: float,
    Gamma: np.ndarray,
    m: int = 5,
    activation: str = "relu",
    lr: float = 0.01,
    max_iters: int = 5000,
    tol: float = 1e-8,
    init_scale: float = 0.01,
    rng: np.random.Generator | None = None,
    verbose: bool = False,
    grad_clip: float = 1.0,
    max_backtracks: int = 5,
    a_init: np.ndarray | None = None,
    W_init: np.ndarray | None = None,
    Cov_ref: np.ndarray | None = None,
    reg_a: float = 0.0,
    reg_W: float = 0.0,
    sigma_other2: float | None = None,
    n_train: int | None = None,
    validation_data: dict | None = None,
) -> TrainResult:
    """Train a 1-hidden-layer NN on interaction-extended summary statistics.

    Args:
        Sigma: (p, p) LD covariance matrix.
        Sigma_beta: (p,) = E[x y], marginal associations.
        E_y2: scalar E[y^2].
        Gamma: (p, p) interaction tensor E[x_i x_j y].
        m: Number of hidden units.
        activation: Activation function name.
        lr: Learning rate.
        max_iters: Maximum iterations.
        tol: Convergence tolerance on relative loss change.
        init_scale: Scale for random weight initialization.
        rng: Random generator for initialization.
        verbose: Print progress every 500 iterations.
        grad_clip: Maximum gradient norm.
        max_backtracks: Maximum backtracking steps per iteration.
        a_init: Optional initial second-layer weights (warm start).
        W_init: Optional initial first-layer weights (warm start).
        Cov_ref: (p, p) empirical covariance from a reference panel.
            When provided, corrects the E[f^2] term by using the true
            projection covariances instead of the Gaussian-latent Sigma.
        reg_a: L2 regularization strength for second-layer weights.
        reg_W: L2 regularization strength for first-layer weights.
        sigma_other2: Cross-block phenotype noise variance
            = max(0, E[y^2] - block_b_variance_explained).
            When provided together with n_train, enables rank-1 collapsed
            gradient with James-Stein denoising of q_k.  When SNR < 1 the
            interaction gradient vanishes, recovering the Gaussian NN.
        n_train: Training sample size used to estimate Gamma.  Required to
            activate denoising alongside sigma_other2.
        validation_data: Optional dict with keys "Sigma_beta_val", "Gamma_val",
            "E_y2_val".  When provided, enables validation-based early stopping:
            validation loss is checked every 50 iterations; if it has not improved
            for 200 consecutive iterations, training stops and the weights from the
            best validation iteration are returned.

    Returns:
        TrainResult with optimized (a, W) and loss history.
    """
    if rng is None:
        rng = np.random.default_rng()

    p = Sigma.shape[0]

    if a_init is not None and W_init is not None:
        a = a_init.copy()
        W = W_init.copy()
    else:
        W = rng.standard_normal((m, p)) * init_scale
        a = rng.standard_normal(m) * init_scale

    loss_history = []
    converged = False

    # Validation early-stopping state
    use_val = validation_data is not None
    best_val_loss = float("inf")
    best_a = a.copy()
    best_W = W.copy()
    val_no_improve_iters = 0
    val_check_interval = 50
    val_patience = 200  # stop if no improvement for this many iterations

    for i in range(max_iters):
        loss = compute_interaction_loss(
            a, W, Sigma, Sigma_beta, E_y2, Gamma, activation, Cov_ref,
            reg_a, reg_W, sigma_other2=sigma_other2, n_train=n_train,
        )
        loss_history.append(loss)

        if verbose and i % 500 == 0:
            print(f"  iter {i:5d}  loss = {loss:.8f}")

        if i > 0:
            rel_change = abs(loss_history[-1] - loss_history[-2]) / (abs(loss_history[-2]) + 1e-30)
            if rel_change < tol:
                converged = True
                break

        # Validation-based early stopping
        if use_val and i % val_check_interval == 0:
            val_loss = compute_interaction_loss(
                a, W, Sigma,
                validation_data["Sigma_beta_val"], validation_data["E_y2_val"],
                validation_data["Gamma_val"], activation, Cov_ref,
                reg_a, reg_W, sigma_other2=sigma_other2, n_train=n_train,
            )
            if val_loss < best_val_loss - 1e-10:
                best_val_loss = val_loss
                best_a = a.copy()
                best_W = W.copy()
                val_no_improve_iters = 0
            else:
                val_no_improve_iters += val_check_interval
                if val_no_improve_iters >= val_patience:
                    a = best_a
                    W = best_W
                    converged = True
                    break

        grad_a, grad_W = compute_interaction_gradients(
            a, W, Sigma, Sigma_beta, E_y2, Gamma, activation, Cov_ref,
            reg_a, reg_W,
            sigma_other2=sigma_other2, n_train=n_train,
        )

        combined_norm = np.sqrt(np.sum(grad_a**2) + np.sum(grad_W**2))
        if combined_norm > grad_clip:
            scale = grad_clip / combined_norm
            grad_a = grad_a * scale
            grad_W = grad_W * scale

        step_lr = lr
        for _ in range(max_backtracks):
            a_new = a - step_lr * grad_a
            W_new = W - step_lr * grad_W
            new_loss = compute_interaction_loss(
                a_new, W_new, Sigma, Sigma_beta, E_y2, Gamma, activation, Cov_ref,
                reg_a, reg_W, sigma_other2=sigma_other2, n_train=n_train,
            )
            if new_loss <= loss + 1e-10:
                break
            step_lr *= 0.5
        else:
            a_new = a - step_lr * grad_a
            W_new = W - step_lr * grad_W

        a = a_new
        W = W_new

    if not converged:
        loss = compute_interaction_loss(
            a, W, Sigma, Sigma_beta, E_y2, Gamma, activation, Cov_ref,
            reg_a, reg_W, sigma_other2=sigma_other2, n_train=n_train,
        )
        loss_history.append(loss)

    # "Do no harm": if a warm start was provided and the interaction optimizer
    # ended up at a worse training loss, return the warm start unchanged.
    if a_init is not None and W_init is not None:
        warmstart_loss = compute_interaction_loss(
            a_init, W_init, Sigma, Sigma_beta, E_y2, Gamma, activation, Cov_ref,
            reg_a, reg_W, sigma_other2=sigma_other2, n_train=n_train,
        )
        if loss_history[-1] > warmstart_loss - 1e-10:
            return TrainResult(
                a=a_init.copy(),
                W=W_init.copy(),
                loss_history=loss_history,
                converged=converged,
                n_iters=len(loss_history),
            )

    return TrainResult(
        a=a,
        W=W,
        loss_history=loss_history,
        converged=converged,
        n_iters=len(loss_history),
    )
