"""
Test suite for OverfitResistantMLModels

Tests EPV validation, model creation, and nested CV
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.machine_learning.ml_models import OverfitResistantMLModels


class TestOverfitResistantMLModels(unittest.TestCase):
    """Test suite for OverfitResistantMLModels"""
    
    def setUp(self):
        """Create synthetic test data"""
        np.random.seed(42)
        
        # Create synthetic features and outcomes
        # Use fewer features to ensure EPV >= 5
        n_samples = 54
        n_features = 5  # Reduced from 10 to ensure EPV >= 5
        
        self.X = np.random.randn(n_samples, n_features)
        self.y = np.random.binomial(1, 0.3, n_samples)
        
        # Ensure both classes present and enough events for EPV >= 5
        while len(np.unique(self.y)) < 2 or np.sum(self.y) < 25:  # EPV = 25/5 = 5
            self.y = np.random.binomial(1, 0.5, n_samples)  # Higher probability for more events
    
    def test_epv_calculation(self):
        """Test EPV calculation and warnings"""
        # Use parameters that ensure EPV >= 5
        n_features = 5
        n_samples = 54
        n_events = 30  # EPV = 30/5 = 6 (>= 5, < 10, will trigger warning)
        
        ml_model = OverfitResistantMLModels(
            n_features=n_features,
            n_samples=n_samples,
            n_events=n_events,
            random_seed=42
        )
        
        expected_epv = n_events / n_features
        self.assertAlmostEqual(ml_model.epv, expected_epv, places=2)
    
    def test_epv_error_very_low_epv(self):
        """Test that very low EPV (< 5) raises ValueError"""
        # Create scenario with EPV < 5
        n_features = 20
        n_samples = 30
        n_events = 4  # EPV = 4/20 = 0.2 (very low, < 5)
        
        with self.assertRaises(ValueError) as context:
            OverfitResistantMLModels(
                n_features=n_features,
                n_samples=n_samples,
                n_events=n_events,
                random_seed=42
            )
        
        self.assertIn("EPV too low", str(context.exception))
        self.assertIn("Minimum EPV = 5", str(context.exception))
    
    def test_epv_warning_low_epv(self):
        """Test that low EPV (5 <= EPV < 10) triggers warning"""
        # Create scenario with EPV between 5 and 10
        n_features = 1  # Single feature
        n_samples = 30
        n_events = 7  # EPV = 7/1 = 7 (low but >= 5, < 10)
        
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ml_model = OverfitResistantMLModels(
                n_features=n_features,
                n_samples=n_samples,
                n_events=n_events,
                random_seed=42
            )
            
            # Check that warning was issued
            self.assertTrue(len(w) > 0)
            self.assertTrue("LOW EPV" in str(w[0].message))
    
    def test_ann_model_creation(self):
        """Test ANN model creation"""
        ml_model = OverfitResistantMLModels(
            n_features=self.X.shape[1],
            n_samples=len(self.X),
            n_events=int(np.sum(self.y)),
            random_seed=42
        )
        
        ann_model = ml_model.create_ann_model()
        
        # Check that model is a Pipeline
        from sklearn.pipeline import Pipeline
        self.assertIsInstance(ann_model, Pipeline)
        
        # Check that it can be fit
        ann_model.fit(self.X, self.y)
        
        # Check that it can predict
        predictions = ann_model.predict_proba(self.X)
        self.assertEqual(predictions.shape[0], len(self.X))
        self.assertEqual(predictions.shape[1], 2)
    
    def test_xgboost_model_creation(self):
        """Test XGBoost model creation"""
        try:
            import xgboost as xgb
        except ImportError:
            self.skipTest("XGBoost not available")
        
        ml_model = OverfitResistantMLModels(
            n_features=self.X.shape[1],
            n_samples=len(self.X),
            n_events=int(np.sum(self.y)),
            random_seed=42
        )
        
        xgb_model = ml_model.create_xgboost_model()
        
        # Check that model is XGBClassifier
        self.assertIsNotNone(xgb_model)
        
        # Check that it can be fit
        xgb_model.fit(self.X, self.y)
        
        # Check that it can predict
        predictions = xgb_model.predict_proba(self.X)
        self.assertEqual(predictions.shape[0], len(self.X))
        self.assertEqual(predictions.shape[1], 2)
    
    def test_complexity_adjustment_small_sample(self):
        """Test automatic complexity adjustment for small samples"""
        # Very small sample, but ensure EPV >= 5
        ml_model = OverfitResistantMLModels(
            n_features=5,  # Reduced features to ensure EPV >= 5
            n_samples=30,  # Very small
            n_events=25,  # EPV = 25/5 = 5 (minimum)
            random_seed=42
        )
        
        # Check that hidden layers were reduced; XGBoost max_depth=2 (was 1) to avoid constant predictions
        self.assertEqual(ml_model.ANN_CONFIG['hidden_layer_sizes'], (8,))
        self.assertEqual(ml_model.XGBOOST_CONFIG['max_depth'], 2)
        self.assertEqual(ml_model.XGBOOST_CONFIG['n_estimators'], 30)
    
    def test_nested_cv(self):
        """Test nested cross-validation"""
        ml_model = OverfitResistantMLModels(
            n_features=self.X.shape[1],
            n_samples=len(self.X),
            n_events=int(np.sum(self.y)),
            random_seed=42
        )
        
        # Run nested CV
        results = ml_model.train_with_nested_cv(
            self.X, self.y, model_type='ann'
        )
        
        # Check results structure
        self.assertIn('nested_cv_auc_mean', results)
        self.assertIn('nested_cv_auc_std', results)
        self.assertIn('nested_cv_auc_scores', results)
        self.assertIn('epv', results)
        
        # Check that AUC is in valid range
        self.assertGreaterEqual(results['nested_cv_auc_mean'], 0.0)
        self.assertLessEqual(results['nested_cv_auc_mean'], 1.0)
        
        # Check EPV
        self.assertGreater(results['epv'], 0)


if __name__ == '__main__':
    unittest.main()
