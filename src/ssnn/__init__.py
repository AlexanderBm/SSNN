"""Summary-Statistics Neural Network for PRS weight estimation."""

from .activations import get_activation, get_activation_derivs
from .population_risk import compute_loss, compute_grad_a, compute_grad_W, compute_gradients
from .optimizer import train, TrainResult
from .utils import (
    generate_ld_matrix,
    generate_gwas_summary_stats,
    linear_prs_weights,
    prediction_r2,
    nn_predict,
    nn_prediction_r2,
)

# Edgeworth-corrected framework (Part 2)
from .cumulants import (
    snp_cumulants,
    projection_cumulants_independent,
    projection_cumulants_ld,
    projection_cumulant_gradients_independent,
    projection_cumulant_gradients_ld,
    decorrelation_matrix,
)
from .edgeworth_integrals import (
    edgeworth_E_sigma,
    edgeworth_E_sigma_prime,
    edgeworth_E_sigma_sigma,
)
from .edgeworth_risk import (
    compute_edgeworth_loss,
    compute_edgeworth_gradients,
    compute_correction_delta,
)
from .edgeworth_optimizer import train_edgeworth

# Interaction-SSNN framework (barrier-breaking)
from .activations import get_activation_double_prime
from .interaction_integrals import (
    interaction_cross_moment,
    interaction_cross_moment_grad,
)
from .interaction_risk import (
    compute_interaction_loss,
    compute_interaction_gradients,
)
from .interaction_optimizer import train_interaction

# Baseline PRS methods
from .baselines import (
    clump_and_threshold,
    ldpred2_inf,
    prs_cs,
)

# PUMAS pseudo-subset splitting
from .pumas import (
    PUMASSplit,
    generate_pumas_split,
    generate_pumas_splits,
    pumas_summary_r2,
    pumas_nn_summary_r2,
)

# PUMAS validation pipeline (Step 4)
from .pumas_validation import (
    TraitConfig,
    TraitResult,
    MethodResult,
    run_validation,
    run_synthetic_validation,
)

# Error analysis (Step 5)
from .error_analysis import (
    ErrorDecomposition,
    edgeworth_truncation_bound,
    decorrelation_bound,
    ld_estimation_bound,
    pumas_variance_bound,
    optimization_bound,
    estimate_smoothness,
    compute_error_decomposition,
)

# Simulation study (Step 3)
from .simulation import (
    SimulationScenario,
    ScenarioResult,
    generate_binomial_genotypes,
    generate_maf_spectrum,
    generate_effect_sizes,
    compute_summary_stats_from_genotypes,
    train_oracle_nn,
    run_single_rep,
    run_scenario,
)
