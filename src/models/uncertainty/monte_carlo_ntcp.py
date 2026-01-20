"""
Correct Monte Carlo NTCP with proper uncertainty propagation

References:
- Taylor JR. An Introduction to Error Analysis. 2nd Ed.
- Burman et al. IJROBP 1991 (LKB model uncertainty)
"""

import numpy as np
from typing import Dict, Optional, Callable
import warnings


class MonteCarloNTCPCorrect:
    """
    Correct Monte Carlo NTCP with proper uncertainty propagation
    
    References:
    - Taylor JR. An Introduction to Error Analysis. 2nd Ed.
    - Burman et al. IJROBP 1991 (LKB model uncertainty)
    """
    
    def __init__(self, n_samples: int = 10000, random_seed: int = 42):
        """
        Initialize Monte Carlo NTCP calculator
        
        Parameters
        ----------
        n_samples : int
            Number of Monte Carlo samples
        random_seed : int
            Random seed for reproducibility
        """
        self.n_samples = n_samples
        np.random.seed(random_seed)
    
    def predict_with_parameter_uncertainty(
        self,
        model: Callable,
        geud_values: np.ndarray,
        param_mean: Dict[str, float],
        param_cov: np.ndarray
    ) -> Dict:
        """
        Monte Carlo NTCP prediction with parameter uncertainty
        
        Parameters
        ----------
        model : callable
            NTCP model function that takes gEUD values and parameters
            Example: model(geud_values, TD50=30, m=0.1)
        geud_values : np.ndarray
            gEUD values for each patient
        param_mean : dict
            Mean parameter values {'TD50': float, 'm': float}
        param_cov : np.ndarray
            Parameter covariance matrix (2x2 for LKB)
            
        Returns
        -------
        dict with keys:
            - 'mean': Mean NTCP prediction
            - 'std': Standard deviation
            - 'ci_lower': 2.5th percentile
            - 'ci_upper': 97.5th percentile
            - 'samples': All MC samples (for diagnostics)
        """
        # Sample parameters from multivariate normal
        param_names = list(param_mean.keys())
        param_values = np.array([param_mean[k] for k in param_names])
        
        # Ensure covariance matrix is valid
        if param_cov.shape != (len(param_values), len(param_values)):
            raise ValueError(
                f"Covariance matrix shape {param_cov.shape} doesn't match "
                f"parameter vector length {len(param_values)}"
            )
        
        try:
            param_samples = np.random.multivariate_normal(
                param_values, param_cov, self.n_samples
            )
        except np.linalg.LinAlgError:
            # If covariance matrix is not positive definite, use diagonal only
            warnings.warn("Covariance matrix not positive definite, using diagonal only")
            param_cov_diag = np.diag(np.diag(param_cov))
            param_samples = np.random.multivariate_normal(
                param_values, param_cov_diag, self.n_samples
            )
        
        # Ensure parameters stay in valid range
        # TD50 > 0, m > 0 (for LKB)
        param_samples = np.clip(param_samples, 0.01, None)
        
        # Calculate NTCP for each parameter sample
        n_patients = len(geud_values)
        ntcp_samples = np.zeros((self.n_samples, n_patients))
        
        for i, params in enumerate(param_samples):
            param_dict = dict(zip(param_names, params))
            
            # Call model with parameters
            try:
                ntcp_samples[i] = model(geud_values, **param_dict)
            except Exception as e:
                warnings.warn(f"Error in MC sample {i}: {e}")
                # Use previous valid sample or mean
                if i > 0:
                    ntcp_samples[i] = ntcp_samples[i-1]
                else:
                    ntcp_samples[i] = np.full(n_patients, 0.5)
        
        # Ensure NTCP values are in [0, 1]
        ntcp_samples = np.clip(ntcp_samples, 0.0, 1.0)
        
        # Calculate statistics
        mean_pred = np.mean(ntcp_samples, axis=0)
        std_pred = np.std(ntcp_samples, axis=0)
        ci_lower = np.percentile(ntcp_samples, 2.5, axis=0)
        ci_upper = np.percentile(ntcp_samples, 97.5, axis=0)
        
        return {
            'mean': mean_pred,
            'std': std_pred,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'samples': ntcp_samples
        }
    
    def predict_with_data_uncertainty(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        n_bootstrap: int = 2000
    ) -> Dict:
        """
        Bootstrap NTCP uncertainty (data-driven)
        
        More robust than parameter uncertainty when:
        - Model may be misspecified
        - Parameter covariance is unknown
        
        Parameters
        ----------
        model : sklearn-like model
            Model with fit() and predict_proba() methods
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Binary outcomes
        n_bootstrap : int
            Number of bootstrap iterations
            
        Returns
        -------
        dict with keys:
            - 'mean': Mean NTCP prediction
            - 'std': Standard deviation
            - 'ci_lower': 2.5th percentile
            - 'ci_upper': 97.5th percentile
        """
        from sklearn.base import clone
        
        n_samples = len(y)
        bootstrap_preds = np.zeros((n_bootstrap, n_samples))
        
        # Check if both classes present
        if len(np.unique(y)) < 2:
            raise ValueError("Both classes must be present for bootstrap")
        
        valid_bootstraps = 0
        for b in range(n_bootstrap):
            # Bootstrap resample
            idx = np.random.choice(n_samples, n_samples, replace=True)
            X_boot, y_boot = X[idx], y[idx]
            
            # Check if both classes still present
            if len(np.unique(y_boot)) < 2:
                continue
            
            try:
                # Clone model and refit
                model_boot = clone(model)
                model_boot.fit(X_boot, y_boot)
                
                # Predict on original data
                y_pred = model_boot.predict_proba(X)[:, 1]
                bootstrap_preds[valid_bootstraps] = y_pred
                valid_bootstraps += 1
            except Exception as e:
                warnings.warn(f"Bootstrap iteration {b} failed: {e}")
                continue
        
        if valid_bootstraps == 0:
            raise ValueError("No valid bootstrap iterations completed")
        
        # Trim to valid iterations
        bootstrap_preds = bootstrap_preds[:valid_bootstraps]
        
        # Calculate statistics
        mean_pred = np.mean(bootstrap_preds, axis=0)
        std_pred = np.std(bootstrap_preds, axis=0)
        ci_lower = np.percentile(bootstrap_preds, 2.5, axis=0)
        ci_upper = np.percentile(bootstrap_preds, 97.5, axis=0)
        
        return {
            'mean': mean_pred,
            'std': std_pred,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }
