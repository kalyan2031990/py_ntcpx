"""
NTCP Models Package - Four-Tier Framework
==========================================
Tier 1: Legacy-A (QUANTEC LKB / RS, fixed)
Tier 2: Legacy-B (MLE-refitted LKB / RS)
Tier 3: Modern Classical (de Vette multivariable NTCP)
Tier 4: AI (ANN, XGBoost, SHAP, uNTCP, CCS) - in main codebase
"""

from .legacy_fixed import LegacyFixedNTCP
from .legacy_mle import LegacyMLENTCP
from .modern_logistic import ModernLogisticNTCP

__all__ = ['LegacyFixedNTCP', 'LegacyMLENTCP', 'ModernLogisticNTCP']

