"""
Calibration Correction (Phase 5.2)

Platt scaling and isotonic regression for post-hoc recalibration
"""

import numpy as np
from typing import Tuple, Optional
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import warnings


class CalibrationCorrector:
    """
    Post-hoc calibration correction for ML models
    
    Methods:
    - Platt scaling (parametric)
    - Isotonic regression (non-parametric)
    """
    
    def __init__(self, method: str = 'platt'):
        """
        Initialize calibration corrector
        
        Parameters
        ----------
        method : str
            'platt' (parametric) or 'isotonic' (non-parametric)
        """
        if method not in ['platt', 'isotonic']:
            raise ValueError(f"Unknown method: {method}. Use 'platt' or 'isotonic'")
        
        self.method = method
        self.calibrator = None
        self._fitted = False
    
    def fit(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Fit calibration corrector
        
        Parameters
        ----------
        y_true : np.ndarray
            True binary labels
        y_pred : np.ndarray
            Predicted probabilities (uncalibrated)
        """
        if len(np.unique(y_true)) < 2:
            raise ValueError("Both classes must be present for calibration")
        
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        # Remove NaN values
        valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_valid = y_true[valid_mask]
        y_pred_valid = y_pred[valid_mask]
        
        if len(y_true_valid) < 10:
            warnings.warn("Insufficient data for calibration correction")
            self._fitted = False
            return
        
        if self.method == 'platt':
            # Platt scaling: logistic regression on log-odds
            # logit(p) = a * logit(p_raw) + b
            y_pred_clipped = np.clip(y_pred_valid, 1e-10, 1 - 1e-10)
            log_odds = np.log(y_pred_clipped / (1 - y_pred_clipped))
            
            # Fit logistic regression
            self.calibrator = LogisticRegression()
            self.calibrator.fit(log_odds.reshape(-1, 1), y_true_valid)
            
        elif self.method == 'isotonic':
            # Isotonic regression (non-parametric)
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(y_pred_valid, y_true_valid)
        
        self._fitted = True
    
    def transform(self, y_pred: np.ndarray) -> np.ndarray:
        """
        Apply calibration correction to predictions
        
        Parameters
        ----------
        y_pred : np.ndarray
            Uncalibrated predictions
            
        Returns
        -------
        np.ndarray
            Calibrated predictions
        """
        if not self._fitted:
            raise ValueError("Calibrator must be fitted before transform")
        
        y_pred = np.asarray(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-10, 1 - 1e-10)
        
        if self.method == 'platt':
            # Convert to log-odds
            log_odds = np.log(y_pred_clipped / (1 - y_pred_clipped))
            # Apply calibration
            calibrated = self.calibrator.predict_proba(log_odds.reshape(-1, 1))[:, 1]
        elif self.method == 'isotonic':
            calibrated = self.calibrator.predict(y_pred_clipped)
        
        return np.clip(calibrated, 0.0, 1.0)
    
    def fit_transform(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step
        
        Parameters
        ----------
        y_true : np.ndarray
            True binary labels
        y_pred : np.ndarray
            Uncalibrated predictions
            
        Returns
        -------
        np.ndarray
            Calibrated predictions
        """
        self.fit(y_true, y_pred)
        return self.transform(y_pred)


def compute_calibration_slope(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10
) -> Tuple[float, float]:
    """
    Compute calibration slope and intercept
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_pred : np.ndarray
        Predicted probabilities
    n_bins : int
        Number of bins for calibration curve
        
    Returns
    -------
    slope : float
        Calibration slope (1.0 = perfect)
    intercept : float
        Calibration intercept (0.0 = perfect)
    """
    from sklearn.calibration import calibration_curve
    
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Remove NaN
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    
    if len(y_true_valid) < n_bins:
        return np.nan, np.nan
    
    try:
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true_valid, y_pred_valid, n_bins=n_bins, strategy='uniform'
        )
        
        # Linear regression: observed = slope * predicted + intercept
        if len(mean_predicted_value) < 2:
            return np.nan, np.nan
        
        x_mean = np.mean(mean_predicted_value)
        y_mean = np.mean(fraction_of_positives)
        
        numerator = np.sum((mean_predicted_value - x_mean) * (fraction_of_positives - y_mean))
        denominator = np.sum((mean_predicted_value - x_mean) ** 2)
        
        if denominator == 0:
            return np.nan, np.nan
        
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        
        return float(slope), float(intercept)
    except Exception:
        return np.nan, np.nan
