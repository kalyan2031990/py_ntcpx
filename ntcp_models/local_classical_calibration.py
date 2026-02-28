#!/usr/bin/env python3
"""
Local Classical NTCP Calibration
=================================
Fits a local sigmoid curve to cohort data and extracts D50 and slope parameters
for fair comparison with ML models.

This module provides:
- Local sigmoid fitting (logistic NTCP curve)
- Conversion to LKB and RS parameters
- Fair classical vs ML comparison
"""

import numpy as np
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')


def logistic_ntcp(x, D50, k):
    """
    Logistic NTCP function: NTCP = 1 / (1 + exp(-(x - D50) / k))
    
    Args:
        x: Dose metric (e.g., mean dose, gEUD)
        D50: Dose at 50% NTCP
        k: Slope parameter (smaller k = steeper curve)
    
    Returns:
        NTCP value (0-1)
    """
    return 1.0 / (1.0 + np.exp(-(x - D50) / k))


def fit_local_sigmoid(dose_metric, toxicity):
    """
    Fit local sigmoid curve to cohort data
    
    Args:
        dose_metric: Array of dose metrics (e.g., mean dose, gEUD)
        toxicity: Array of binary toxicity outcomes (0/1)
    
    Returns:
        tuple: (D50_local, k_local) - fitted parameters
    """
    # Remove NaN values
    valid_mask = ~(np.isnan(dose_metric) | np.isnan(toxicity))
    dose_metric_clean = dose_metric[valid_mask]
    toxicity_clean = toxicity[valid_mask].astype(float)
    
    if len(dose_metric_clean) < 5:
        raise ValueError(f"Insufficient data for sigmoid fitting: {len(dose_metric_clean)} samples")
    
    # Initial parameter estimates
    dose_range = np.max(dose_metric_clean) - np.min(dose_metric_clean)
    D50_initial = np.median(dose_metric_clean)
    k_initial = dose_range / 4.0  # Rough estimate
    
    try:
        popt, _ = curve_fit(
            logistic_ntcp,
            dose_metric_clean,
            toxicity_clean,
            p0=[D50_initial, k_initial],
            bounds=([0.0, 0.01], [200.0, 50.0]),
            maxfev=10000,
            method='trf'
        )
        D50_local, k_local = popt
        return float(D50_local), float(k_local)
    except Exception as e:
        # Fallback: use median-based estimates
        print(f"Warning: Sigmoid fitting failed ({e}), using median-based estimates")
        D50_local = np.median(dose_metric_clean)
        k_local = dose_range / 4.0
        return float(D50_local), float(k_local)


def local_sigmoid_to_lkb(D50, k):
    """
    Convert local sigmoid parameters to LKB model parameters
    
    Args:
        D50: Dose at 50% NTCP from sigmoid fit
        k: Slope parameter from sigmoid fit
    
    Returns:
        tuple: (TD50, m) - LKB parameters
    """
    # LKB model: NTCP = Φ((D - TD50) / (m * TD50))
    # For sigmoid: NTCP = 1 / (1 + exp(-(D - D50) / k))
    # Approximate conversion:
    # TD50 ≈ D50
    # m ≈ k / D50 (slope parameter)
    
    TD50 = D50
    m = k / D50 if D50 > 0 else 0.2  # Default m if D50 is invalid
    
    # Ensure reasonable bounds
    m = np.clip(m, 0.01, 1.0)
    
    return float(TD50), float(m)


def local_sigmoid_to_rs(D50, k):
    """
    Convert local sigmoid parameters to RS (Relative Seriality) model parameters
    
    Args:
        D50: Dose at 50% NTCP from sigmoid fit
        k: Slope parameter from sigmoid fit
    
    Returns:
        tuple: (D50_rs, gamma) - RS parameters
    """
    # RS model uses D50 and gamma
    # gamma relates to slope: larger gamma = steeper curve
    # Approximate: gamma ≈ 1 / k (inverse relationship)
    
    D50_rs = D50
    gamma = 1.0 / k if k > 0 else 1.0  # Default gamma if k is invalid
    
    # Ensure reasonable bounds
    gamma = np.clip(gamma, 0.1, 10.0)
    
    return float(D50_rs), float(gamma)

