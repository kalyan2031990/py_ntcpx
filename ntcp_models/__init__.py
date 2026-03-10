"""
NTCP Models Package - Four-Tier Framework (v1.1.0)
==================================================
Tier 1: Legacy-A (QUANTEC LKB / RS, fixed)
Tier 2: Legacy-B (MLE-refitted LKB / RS)
Tier 3: Modern Classical (de Vette multivariable NTCP)
Tier 4: AI (ANN, XGBoost, SHAP, uNTCP, CCS) - in main codebase

Shared utilities such as EPV (Events Per Variable) checking live here so they
can be reused consistently across Tier 3 (logistic) and Tier 4 (ML) models.
"""


class EPVError(ValueError):
    """Raised when Events Per Variable is below the required minimum."""


def check_epv(
    n_events: int,
    n_features: int,
    min_epv: float = 10.0,
    model_name: str = "Model",
) -> float:
    """
    Check events-per-variable (EPV) constraint for a given model.

    Args:
        n_events: Number of outcome events (positives)
        n_features: Number of free parameters/features in the model
        min_epv: Minimum required EPV (default 10 for logistic, 20 for trees)
        model_name: Name for error message

    Returns:
        epv: Computed EPV value

    Raises:
        EPVError: If EPV < min_epv (hard block)
    """
    if n_features <= 0:
        return float("inf")

    epv = n_events / float(n_features)
    if epv < min_epv:
        max_features = int(n_events / min_epv) if min_epv > 0 else 0
        raise EPVError(
            f"{model_name}: EPV = {epv:.1f} ({n_events} events / {n_features} features) "
            f"is below minimum {min_epv}. Reduce features or collect more data. "
            f"Maximum allowed features at this sample size: {max_features}."
        )
    return epv


from .legacy_fixed import LegacyFixedNTCP  # noqa: E402
from .legacy_mle import LegacyMLENTCP  # noqa: E402
from .modern_logistic import ModernLogisticNTCP  # noqa: E402


__all__ = [
    'LegacyFixedNTCP',
    'LegacyMLENTCP',
    'ModernLogisticNTCP',
    'EPVError',
    'check_epv',
]

