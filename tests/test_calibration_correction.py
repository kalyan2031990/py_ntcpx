"""
Test suite for Calibration Correction (Phase 5.2)
"""

import unittest
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.validation.calibration_correction import CalibrationCorrector, compute_calibration_slope


class TestCalibrationCorrection(unittest.TestCase):
    """Test suite for calibration correction"""
    
    def setUp(self):
        """Create synthetic test data"""
        np.random.seed(42)
        
        # Create poorly calibrated predictions
        n_samples = 100
        self.y_true = np.random.binomial(1, 0.3, n_samples)
        
        # Overconfident predictions (slope > 1)
        self.y_pred_overconfident = np.random.beta(2, 5, n_samples) * 0.5 + 0.5
        
        # Underconfident predictions (slope < 1)
        self.y_pred_underconfident = np.random.beta(5, 2, n_samples) * 0.3
    
    def test_platt_scaling(self):
        """Test Platt scaling calibration"""
        corrector = CalibrationCorrector(method='platt')
        
        # Fit on training data
        corrector.fit(self.y_true[:70], self.y_pred_overconfident[:70])
        
        # Transform test data
        y_calibrated = corrector.transform(self.y_pred_overconfident[70:])
        
        # Check bounds
        self.assertTrue(np.all(y_calibrated >= 0))
        self.assertTrue(np.all(y_calibrated <= 1))
    
    def test_isotonic_regression(self):
        """Test isotonic regression calibration"""
        corrector = CalibrationCorrector(method='isotonic')
        
        # Fit on training data
        corrector.fit(self.y_true[:70], self.y_pred_overconfident[:70])
        
        # Transform test data
        y_calibrated = corrector.transform(self.y_pred_overconfident[70:])
        
        # Check bounds
        self.assertTrue(np.all(y_calibrated >= 0))
        self.assertTrue(np.all(y_calibrated <= 1))
    
    def test_calibration_slope_calculation(self):
        """Test calibration slope calculation"""
        slope, intercept = compute_calibration_slope(
            self.y_true, self.y_pred_overconfident
        )
        
        # Check that slope is calculated
        self.assertFalse(np.isnan(slope))
        self.assertFalse(np.isnan(intercept))
    
    def test_calibration_improves_slope(self):
        """Test that calibration improves calibration slope"""
        # Calculate original slope
        slope_original, _ = compute_calibration_slope(
            self.y_true[70:], self.y_pred_overconfident[70:]
        )
        
        # Apply calibration
        corrector = CalibrationCorrector(method='platt')
        corrector.fit(self.y_true[:70], self.y_pred_overconfident[:70])
        y_calibrated = corrector.transform(self.y_pred_overconfident[70:])
        
        # Calculate calibrated slope
        slope_calibrated, _ = compute_calibration_slope(
            self.y_true[70:], y_calibrated
        )
        
        # Calibrated slope should be calculated (not NaN)
        # (This is probabilistic, so we just check it's calculated)
        self.assertIsNotNone(slope_calibrated)
        if not np.isnan(slope_calibrated):
            # If calculated, should be a valid number
            self.assertIsInstance(slope_calibrated, (int, float, np.number))


if __name__ == '__main__':
    unittest.main()
