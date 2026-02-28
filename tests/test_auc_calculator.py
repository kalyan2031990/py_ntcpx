"""
Test suite for AUC calculator with confidence intervals
"""

import unittest
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics.auc_calculator import calculate_auc_with_ci, compare_aucs_delong


class TestAUCCalculator(unittest.TestCase):
    """Test suite for AUC calculator"""
    
    def setUp(self):
        """Create synthetic test data"""
        np.random.seed(42)
        
        # Create well-separated classes for high AUC
        n_samples = 100
        self.y_true_good = np.concatenate([
            np.zeros(50),
            np.ones(50)
        ])
        self.y_pred_good = np.concatenate([
            np.random.uniform(0, 0.3, 50),  # Low predictions for class 0
            np.random.uniform(0.7, 1.0, 50)  # High predictions for class 1
        ])
        
        # Create poorly-separated classes for low AUC
        self.y_true_poor = np.concatenate([
            np.zeros(50),
            np.ones(50)
        ])
        self.y_pred_poor = np.random.uniform(0, 1, 100)  # Random predictions
    
    def test_auc_calculation_bootstrap(self):
        """Test AUC calculation with bootstrap CI"""
        auc_val, ci = calculate_auc_with_ci(
            self.y_true_good, self.y_pred_good,
            method='bootstrap',
            n_bootstraps=100  # Reduced for speed
        )
        
        # Check that AUC is in valid range
        self.assertGreaterEqual(auc_val, 0.0)
        self.assertLessEqual(auc_val, 1.0)
        
        # Check that CI is valid
        self.assertLessEqual(ci[0], auc_val)
        self.assertGreaterEqual(ci[1], auc_val)
        self.assertGreaterEqual(ci[0], 0.0)
        self.assertLessEqual(ci[1], 1.0)
    
    def test_auc_calculation_delong(self):
        """Test AUC calculation with DeLong CI"""
        auc_val, ci = calculate_auc_with_ci(
            self.y_true_good, self.y_pred_good,
            method='delong'
        )
        
        # Check that AUC is in valid range
        self.assertGreaterEqual(auc_val, 0.0)
        self.assertLessEqual(auc_val, 1.0)
        
        # Check that CI is valid
        self.assertLessEqual(ci[0], auc_val)
        self.assertGreaterEqual(ci[1], auc_val)
    
    def test_auc_high_vs_low(self):
        """Test that high AUC is detected correctly"""
        auc_high, _ = calculate_auc_with_ci(
            self.y_true_good, self.y_pred_good,
            method='bootstrap',
            n_bootstraps=100
        )
        
        auc_low, _ = calculate_auc_with_ci(
            self.y_true_poor, self.y_pred_poor,
            method='bootstrap',
            n_bootstraps=100
        )
        
        # High AUC should be greater than low AUC
        self.assertGreater(auc_high, auc_low)
        self.assertGreater(auc_high, 0.7)  # Should be well above chance
        self.assertLess(auc_low, 0.6)  # Should be close to chance
    
    def test_auc_requires_both_classes(self):
        """Test that AUC calculation requires both classes"""
        # Single class
        y_single = np.zeros(100)
        y_pred_single = np.random.uniform(0, 1, 100)
        
        with self.assertRaises(ValueError):
            calculate_auc_with_ci(y_single, y_pred_single)
    
    def test_compare_aucs_delong(self):
        """Test DeLong test for comparing two AUCs"""
        # Create two different prediction sets
        y_pred1 = self.y_pred_good
        y_pred2 = self.y_pred_poor
        
        z_stat, p_value = compare_aucs_delong(
            self.y_true_good, y_pred1, y_pred2
        )
        
        # Check that results are valid
        self.assertIsInstance(z_stat, (int, float))
        self.assertIsInstance(p_value, (int, float))
        self.assertGreaterEqual(p_value, 0.0)
        self.assertLessEqual(p_value, 1.0)
    
    def test_auc_ci_coverage(self):
        """Test that CI covers the point estimate"""
        auc_val, ci = calculate_auc_with_ci(
            self.y_true_good, self.y_pred_good,
            method='bootstrap',
            n_bootstraps=200
        )
        
        # CI should contain the point estimate
        self.assertLessEqual(ci[0], auc_val)
        self.assertGreaterEqual(ci[1], auc_val)
        
        # CI should be reasonable width (not too wide)
        ci_width = ci[1] - ci[0]
        self.assertLess(ci_width, 0.5)  # Should be reasonable


if __name__ == '__main__':
    unittest.main()
