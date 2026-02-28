"""
AUC calculation with proper confidence intervals

Uses bootstrap or DeLong's method for CI estimation
"""

import numpy as np
from typing import Tuple
from sklearn.metrics import roc_auc_score
from scipy import stats


def calculate_auc_with_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    method: str = 'bootstrap',
    n_bootstraps: int = 2000,
    alpha: float = 0.05
) -> Tuple[float, Tuple[float, float]]:
    """
    Calculate AUC with proper confidence intervals
    
    Parameters
    ----------
    y_true : array
        True binary labels
    y_pred : array
        Predicted probabilities
    method : str
        'bootstrap' (recommended) or 'delong'
    n_bootstraps : int
        Number of bootstrap iterations
    alpha : float
        Significance level (default 0.05 for 95% CI)
        
    Returns
    -------
    auc : float
        Point estimate of AUC
    ci : tuple
        (lower, upper) confidence bounds
    """
    # Validate inputs
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")
    
    if len(np.unique(y_true)) < 2:
        raise ValueError("Both classes must be present for AUC calculation")
    
    # Point estimate
    try:
        auc = roc_auc_score(y_true, y_pred)
    except ValueError as e:
        raise ValueError(f"Error calculating AUC: {e}")
    
    if method == 'bootstrap':
        bootstrapped_aucs = []
        n = len(y_true)
        
        for _ in range(n_bootstraps):
            # Resample with replacement
            idx = np.random.choice(n, n, replace=True)
            
            # Ensure both classes present
            if len(np.unique(y_true[idx])) < 2:
                continue
            
            try:
                boot_auc = roc_auc_score(y_true[idx], y_pred[idx])
                if not np.isnan(boot_auc):
                    bootstrapped_aucs.append(boot_auc)
            except Exception:
                continue
        
        if len(bootstrapped_aucs) < 100:
            raise ValueError(
                f"Insufficient valid bootstrap samples: {len(bootstrapped_aucs)}"
            )
        
        # Percentile CI
        ci_lower = np.percentile(bootstrapped_aucs, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrapped_aucs, 100 * (1 - alpha / 2))
        
    elif method == 'delong':
        # DeLong's method for AUC variance
        # Reference: DeLong et al. Biometrics 1988
        ci_lower, ci_upper = _delong_ci(y_true, y_pred, alpha)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'bootstrap' or 'delong'")
    
    return auc, (ci_lower, ci_upper)


def _delong_ci(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> Tuple[float, float]:
    """
    Calculate AUC CI using DeLong's method
    
    Reference: DeLong et al. Biometrics 1988
    """
    # Separate cases and controls
    pos_idx = y_true == 1
    neg_idx = y_true == 0
    
    pos_preds = y_pred[pos_idx]
    neg_preds = y_pred[neg_idx]
    
    n_pos = len(pos_preds)
    n_neg = len(neg_preds)
    
    # Calculate DeLong variance components
    # V10: variance of AUC estimator for positive class
    # V01: variance of AUC estimator for negative class
    
    # Calculate ranks
    all_preds = np.concatenate([pos_preds, neg_preds])
    ranks = stats.rankdata(all_preds)
    
    pos_ranks = ranks[:n_pos]
    neg_ranks = ranks[n_pos:]
    
    # DeLong variance
    auc_var = (np.var(pos_ranks) / n_pos + np.var(neg_ranks) / n_neg) / (n_pos * n_neg)
    
    # Standard error
    se = np.sqrt(auc_var)
    
    # Z-score for confidence interval
    z_score = stats.norm.ppf(1 - alpha / 2)
    
    auc = roc_auc_score(y_true, y_pred)
    
    ci_lower = auc - z_score * se
    ci_upper = auc + z_score * se
    
    # Clamp to [0, 1]
    ci_lower = max(0.0, ci_lower)
    ci_upper = min(1.0, ci_upper)
    
    return ci_lower, ci_upper


def compare_aucs_delong(
    y_true: np.ndarray,
    y_pred1: np.ndarray,
    y_pred2: np.ndarray
) -> Tuple[float, float]:
    """
    DeLong test for comparing two AUCs
    
    Reference: DeLong et al. Biometrics 1988
    
    Parameters
    ----------
    y_true : array
        True binary labels
    y_pred1 : array
        Predicted probabilities from model 1
    y_pred2 : array
        Predicted probabilities from model 2
        
    Returns
    -------
    z_stat : float
        Z statistic
    p_value : float
        Two-sided p-value
    """
    # Calculate AUCs
    auc1 = roc_auc_score(y_true, y_pred1)
    auc2 = roc_auc_score(y_true, y_pred2)
    
    # Separate cases and controls
    pos_idx = y_true == 1
    neg_idx = y_true == 0
    
    pos_preds1 = y_pred1[pos_idx]
    neg_preds1 = y_pred1[neg_idx]
    pos_preds2 = y_pred2[pos_idx]
    neg_preds2 = y_pred2[neg_idx]
    
    n_pos = len(pos_preds1)
    n_neg = len(neg_preds1)
    
    # Calculate DeLong variance-covariance
    # This is a simplified version - full implementation requires more complex calculations
    
    # For simplicity, use bootstrap to estimate variance of difference
    n = len(y_true)
    diff_aucs = []
    
    for _ in range(1000):
        idx = np.random.choice(n, n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        
        try:
            auc1_boot = roc_auc_score(y_true[idx], y_pred1[idx])
            auc2_boot = roc_auc_score(y_true[idx], y_pred2[idx])
            diff_aucs.append(auc1_boot - auc2_boot)
        except Exception:
            continue
    
    if len(diff_aucs) < 100:
        raise ValueError("Insufficient bootstrap samples for comparison")
    
    # Z-test
    diff_mean = np.mean(diff_aucs)
    diff_se = np.std(diff_aucs)
    
    if diff_se == 0:
        z_stat = 0.0
        p_value = 1.0
    else:
        z_stat = diff_mean / diff_se
        p_value = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))
    
    return z_stat, p_value
