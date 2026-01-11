#!/usr/bin/env python3
"""
Quality Assurance Modules for NTCP Pipeline
============================================
Uncertainty-Aware NTCP (uNTCP) and Cohort Consistency Score (CCS)
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
import sys

# Windows-safe encoding configuration
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except (AttributeError, ValueError):
    pass


class UncertaintyAwareNTCP:
    """Uncertainty-Aware NTCP with explicit confidence intervals"""
    
    def calculate_untcp(self, ntcp_func, params, dvh):
        """Calculate NTCP with uncertainty propagation using first-order Taylor expansion"""
        from scipy import stats
        
        # Calculate nominal NTCP
        ntcp_nominal = ntcp_func(params, dvh)
        
        # Calculate partial derivatives (numerical)
        epsilon = 1e-6
        gradients = {}
        
        for param_name, param_value in params.items():
            params_perturbed = params.copy()
            params_perturbed[param_name] = param_value + epsilon
            ntcp_perturbed = ntcp_func(params_perturbed, dvh)
            gradients[param_name] = (ntcp_perturbed - ntcp_nominal) / epsilon
        
        # Propagate uncertainty (first-order Taylor expansion - Delta method)
        # sigma^2_NTCP = SUM_j (dNTCP/dtheta_j)^2 * sigma^2_theta_j
        param_uncertainties = {
            'n': params['n'] * 0.1,
            'TD50': params['TD50'] * 0.1,
            'm': params['m'] * 0.1
        }
        
        variance = sum(
            (gradients[p] ** 2) * (param_uncertainties[p] ** 2)
            for p in params.keys() if p in param_uncertainties
        )
        
        std = np.sqrt(variance) if variance > 0 else 0.0
        
        return {
            'ntcp': ntcp_nominal,
            'std': std,
            'ci_lower': max(0, ntcp_nominal - 1.96 * std),
            'ci_upper': min(1, ntcp_nominal + 1.96 * std),
            'uncertainty_contributions': {
                p: (gradients[p] ** 2) * (param_uncertainties[p] ** 2) / variance if variance > 0 else 0
                for p in params.keys() if p in param_uncertainties
            },
            'clinical_interpretation': self._interpret_uncertainty(ntcp_nominal, std)
        }
    
    def _interpret_uncertainty(self, ntcp, std):
        """Provide clinical interpretation of uncertainty"""
        if std < 0.05:
            return f"NTCP = {ntcp:.2f} +/- {std:.2f} (Low uncertainty - reliable prediction)"
        elif std < 0.15:
            return f"NTCP = {ntcp:.2f} +/- {std:.2f} (Moderate uncertainty - use with caution)"
        else:
            return f"NTCP = {ntcp:.2f} +/- {std:.2f} (High uncertainty - prediction unreliable)"


class CohortConsistencyScore:
    """Cohort Consistency Score for QA validation"""
    
    def __init__(self):
        self.training_stats = None
    
    def fit(self, X_train):
        """Fit on training cohort"""
        X_train = np.asarray(X_train)
        self.training_stats = {
            'mean': np.mean(X_train, axis=0),
            'cov': np.cov(X_train.T) + np.eye(X_train.shape[1]) * 1e-6  # Regularization
        }
    
    def calculate_ccs(self, X_new):
        """Calculate Cohort Consistency Score using Mahalanobis distance"""
        if self.training_stats is None:
            raise ValueError("Must fit on training data first")
        
        X_new = np.asarray(X_new).flatten()
        mean = self.training_stats['mean']
        cov = self.training_stats['cov']
        cov_inv = np.linalg.pinv(cov)
        
        # Calculate Mahalanobis distance
        d_squared = mahalanobis(X_new, mean, cov_inv) ** 2
        
        # Convert to CCS (0-1 scale)
        # CCS = exp(-½ D²_Mahalanobis)
        ccs = np.exp(-0.5 * d_squared)
        
        # Interpretation thresholds
        if ccs > 0.95:
            warning_level = "none"
        elif ccs > 0.80:
            warning_level = "low"
        elif ccs > 0.50:
            warning_level = "medium"
        elif ccs > 0.20:
            warning_level = "high"
        else:
            warning_level = "critical"
        
        # Calculate warning boolean: True if warning_level is moderate, high, or critical
        warning = warning_level in ["medium", "high", "critical"]
        
        # Safety: True means model output is safe for clinical interpretation
        # safety = not warning means: safe when no warning (warning=False), unsafe when warning (warning=True)
        safety = not warning
        
        return {
            'ccs': float(ccs),
            'mahalanobis_distance': float(np.sqrt(d_squared)),
            'warning': bool(warning),
            'warning_level': str(warning_level),
            'safety': bool(safety),
            'recommendation': str(self._get_recommendation(warning_level)),
            'per_feature_deviation': np.array(self._calculate_feature_deviations(X_new, mean, np.sqrt(np.diag(cov))))
        }
    
    def _get_recommendation(self, warning):
        recommendations = {
            'none': "[OK] Patient within training cohort distribution - predictions reliable",
            'low': "[WARN] Minor deviation - use predictions with caution",
            'medium': "[WARN] Moderate deviation - verify clinical parameters carefully",
            'high': "[ERROR] Significant deviation - predictions may be unreliable",
            'critical': "[ERROR] Out-of-distribution - DO NOT use predictions for clinical decisions"
        }
        return recommendations[warning]
    
    def _calculate_feature_deviations(self, X_new, mean, std):
        """Calculate per-feature standardized deviations"""
        return (X_new - mean) / (std + 1e-6)

