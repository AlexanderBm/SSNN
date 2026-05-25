"""
Robust multi-start interaction optimizer with spectral denoising and full gradient.

Pipeline:
1. Residualize Gamma to remove linear contamination
2. Spectrally denoise Gamma_resid via MP thresholding
3. If no eigenvalues survive (pure noise), return Gaussian NN result
4. Run interaction optimizer from multiple starts:
   - Start 1: Gaussian NN warm-start
   - Start 2: Eigenvector initialization
   - Starts 3+: Random small-scale
5. All starts use Gamma_denoised with the FULL gradient (sigma_other2=None)
6. "Do no harm": return Gaussian NN if no start beats it
"""

from __future__ import annotations
import numpy as np
from .optimizer import TrainResult
from .interaction_optimizer import train_interaction
from .residual_gamma import compute_residual_gamma
from .gamma_denoising import denoise_gamma, triage_block
from .interaction_init import compute_eigenvector_init


def train_interaction_robust(
    Sigma: np.ndarray,
    Sigma_beta: np.ndarray,
    E_y2: float,
    Gamma_raw: np.ndarray,
    X_ref: np.ndarray,
    beta_hat: np.ndarray,
    sigma_other2: float,
    n_train: int,
    m: int = 3,
    activation: str = "relu",
    Cov_ref: np.ndarray | None = None,
    gauss_result: TrainResult | None = None,
    n_restarts: int = 3,
    lr: float = 0.005,
    max_iters: int = 300,
    grad_clip: float = 0.5,
    rng: np.random.Generator | None = None,
    validation_data: dict | None = None,
) -> TrainResult:
    """Robust interaction optimizer with spectral denoising and multi-start.

    CRITICAL: We pass sigma_other2=None and n_train=None to train_interaction()
    so it uses the FULL gradient (interaction_cross_moment_grad), NOT the
    rank-1 collapsed gradient (interaction_cross_moment_grad_denoised).
    The denoising is done at the matrix level by denoise_gamma(), not per-neuron.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Step 1: Residualize Gamma
    Gamma_resid = compute_residual_gamma(Gamma_raw, X_ref, beta_hat)

    # Step 2: Spectral denoising
    if not triage_block(Gamma_resid, sigma_other2, n_train):
        # No signal above MP noise floor — return Gaussian result
        if gauss_result is not None:
            return gauss_result
        return TrainResult(
            a=np.zeros(m), W=np.zeros((m, Sigma.shape[0]))
        )

    Gamma_denoised, surv_vecs, surv_vals = denoise_gamma(Gamma_resid, sigma_other2, n_train)

    # Step 3: Set up multiple starts
    p = Sigma.shape[0]
    starts = []

    # Start 1: Gaussian warm-start
    if gauss_result is not None:
        starts.append((gauss_result.a.copy(), gauss_result.W.copy()))
    else:
        starts.append((rng.standard_normal(m) * 0.01, rng.standard_normal((m, p)) * 0.01))

    # Start 2: Eigenvector initialization
    if gauss_result is not None:
        a_eig, W_eig = compute_eigenvector_init(surv_vecs, surv_vals, gauss_result.a, gauss_result.W, m)
    else:
        a_eig, W_eig = compute_eigenvector_init(surv_vecs, surv_vals,
                                                 np.zeros(m), np.zeros((m, p)), m)
    starts.append((a_eig, W_eig))

    # Starts 3+: Random small-scale
    for _ in range(max(0, n_restarts - 2)):
        starts.append((rng.standard_normal(m) * 0.01, rng.standard_normal((m, p)) * 0.01))

    # Step 4: Run all starts with FULL gradient (sigma_other2=None, n_train=None)
    best_result = gauss_result  # fallback
    best_loss = float('inf')

    for i, (a_init, W_init) in enumerate(starts):
        start_rng = np.random.default_rng(rng.integers(2**32))
        try:
            result = train_interaction(
                Sigma=Sigma,
                Sigma_beta=Sigma_beta,
                E_y2=E_y2,
                Gamma=Gamma_denoised,      # pre-denoised
                m=m,
                activation=activation,
                lr=lr,
                max_iters=max_iters,
                a_init=a_init,
                W_init=W_init,
                Cov_ref=Cov_ref,
                rng=start_rng,
                grad_clip=grad_clip,
                sigma_other2=None,          # FULL gradient, not rank-1 collapsed
                n_train=None,               # FULL gradient, not rank-1 collapsed
                validation_data=validation_data,
            )
            final_loss = result.loss_history[-1] if result.loss_history else float('inf')
            if final_loss < best_loss:
                best_loss = final_loss
                best_result = result
        except Exception:
            continue

    # Step 5: "Do no harm" — if no start beat the Gaussian NN, return Gaussian
    if gauss_result is not None and best_result is not None:
        gauss_loss = gauss_result.loss_history[-1] if gauss_result.loss_history else float('inf')
        if best_loss > gauss_loss - 1e-10:
            return gauss_result

    return best_result if best_result is not None else gauss_result
