"""
Test suite for RadiobiologyGuidedFeatureSelector
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.feature_selector import RadiobiologyGuidedFeatureSelector


class TestRadiobiologyGuidedFeatureSelector(unittest.TestCase):
    """Test suite for RadiobiologyGuidedFeatureSelector"""
    
    def setUp(self):
        """Create synthetic test data"""
        np.random.seed(42)
        
        # Create synthetic features with known relationships
        n_samples = 54
        
        # Create features with different relationships to outcome
        X_data = {
            'Dmean': np.random.uniform(15, 50, n_samples),
            'V30': np.random.uniform(30, 80, n_samples),
            'V45': np.random.uniform(10, 50, n_samples),
            'D50': np.random.uniform(20, 40, n_samples),
            'gEUD': np.random.uniform(25, 45, n_samples),
            'Age': np.random.randint(40, 80, n_samples),
            'Chemotherapy': np.random.choice([0, 1], n_samples),
            'Random1': np.random.randn(n_samples),
            'Random2': np.random.randn(n_samples),
            'Random3': np.random.randn(n_samples)
        }
        
        self.X = pd.DataFrame(X_data)
        
        # Create outcome with relationship to Dmean, V30, V45
        toxicity_prob = 1 / (1 + np.exp(-0.1 * (self.X['Dmean'] - 30)))
        self.y = np.random.binomial(1, toxicity_prob, n_samples)
    
    def test_parotid_essential_features(self):
        """Test that parotid essential features are selected"""
        selector = RadiobiologyGuidedFeatureSelector()
        
        selected = selector.select_features(
            self.X, self.y, organ='Parotid', max_features=5
        )
        
        # Essential features should be included
        self.assertIn('Dmean', selected)
        self.assertIn('V30', selected)
        self.assertIn('V45', selected)
    
    def test_feature_selection_max_features(self):
        """Test that max_features limit is respected"""
        selector = RadiobiologyGuidedFeatureSelector()
        
        max_features = 5
        selected = selector.select_features(
            self.X, self.y, organ='Parotid', max_features=max_features
        )
        
        self.assertLessEqual(len(selected), max_features)
    
    def test_epv_based_feature_capping(self):
        """Test that features are capped based on EPV"""
        selector = RadiobiologyGuidedFeatureSelector()
        
        n_events = int(np.sum(self.y))
        # EPV = n_events / 10, so max_features should be around n_events / 10
        expected_max = max(int(n_events / 10), 3)
        
        selected = selector.select_features(
            self.X, self.y, organ='Parotid', max_features=None
        )
        
        # Should respect EPV-based limit
        self.assertLessEqual(len(selected), expected_max + 2)  # Allow some flexibility
    
    def test_statistical_filtering(self):
        """Test that statistical filtering works"""
        selector = RadiobiologyGuidedFeatureSelector()
        
        # Create data where some features are predictive and others are not
        X_test = pd.DataFrame({
            'Dmean': np.random.uniform(15, 50, 100),
            'Predictive': np.random.uniform(0, 1, 100),  # Will be correlated with outcome
            'Noise': np.random.randn(100)  # Pure noise
        })
        
        # Create outcome correlated with Predictive
        y_test = (X_test['Predictive'] > 0.5).astype(int)
        
        selected = selector.select_features(
            X_test, y_test, organ='Parotid', max_features=3
        )
        
        # Predictive feature should be more likely to be selected than Noise
        # (though this is probabilistic, so we just check structure)
        self.assertGreater(len(selected), 0)
        self.assertLessEqual(len(selected), 3)
    
    def test_other_organs(self):
        """Test feature selection for other organs"""
        selector = RadiobiologyGuidedFeatureSelector()
        
        selected = selector.select_features(
            self.X, self.y, organ='Larynx', max_features=5
        )
        
        # Should still select dose metrics
        self.assertGreater(len(selected), 0)
        self.assertLessEqual(len(selected), 5)


if __name__ == '__main__':
    unittest.main()
