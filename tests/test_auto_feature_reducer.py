"""
Test suite for Auto Feature Reducer (Phase 4.1)
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.auto_feature_reducer import AutoFeatureReducer


class TestAutoFeatureReducer(unittest.TestCase):
    """Test suite for AutoFeatureReducer"""
    
    def setUp(self):
        """Create synthetic test data"""
        np.random.seed(42)
        
        # Create data with low EPV scenario that can be reduced
        # Need enough events to reach EPV >= 5 after reduction
        n_samples = 30
        n_features = 20
        n_events = 10  # EPV = 10/20 = 0.5 (low), but can reduce to 2 features for EPV = 5
        
        self.X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'Feature_{i}' for i in range(n_features)]
        )
        
        # Add essential features
        self.X['Dmean'] = np.random.uniform(15, 50, n_samples)
        self.X['V30'] = np.random.uniform(30, 80, n_samples)
        self.X['V45'] = np.random.uniform(10, 50, n_samples)
        
        # Create y with ~10 events
        self.y = np.random.binomial(1, 0.33, n_samples)  # ~10 events
        # Ensure we have enough events
        while np.sum(self.y) < 10:
            self.y = np.random.binomial(1, 0.5, n_samples)
    
    def test_auto_reduction_low_epv(self):
        """Test automatic feature reduction when EPV < 5"""
        reducer = AutoFeatureReducer(min_epv=5.0)
        
        X_reduced, selected_features, final_epv = reducer.reduce_features(
            self.X, self.y, organ='Parotid'
        )
        
        # Should reduce features to improve EPV
        n_events = int(np.sum(self.y))
        original_epv = n_events / len(self.X.columns)
        
        # After reduction, EPV should improve (or stay same if already adequate)
        self.assertLessEqual(len(selected_features), len(self.X.columns))
        
        # If original EPV < 5 and we have enough events, should improve EPV
        if original_epv < 5.0 and n_events >= 5:
            # EPV should improve (may not reach 5.0 if events are very few)
            self.assertGreaterEqual(final_epv, original_epv)
        
        # Essential features should be preserved if available
        if 'Dmean' in self.X.columns:
            self.assertIn('Dmean', selected_features)
        if 'V30' in self.X.columns:
            self.assertIn('V30', selected_features)
        if 'V45' in self.X.columns:
            self.assertIn('V45', selected_features)
        
        # Final EPV should be calculated correctly
        expected_epv = n_events / len(selected_features) if len(selected_features) > 0 else 0
        self.assertAlmostEqual(final_epv, expected_epv, places=2)
    
    def test_no_reduction_adequate_epv(self):
        """Test that no reduction occurs when EPV >= 5"""
        # Create data with adequate EPV
        n_samples = 50
        n_features = 5
        n_events = 30  # EPV = 30/5 = 6 (adequate)
        
        X_adequate = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'Feature_{i}' for i in range(n_features)]
        )
        y_adequate = np.random.binomial(1, 0.6, n_samples)
        
        reducer = AutoFeatureReducer(min_epv=5.0)
        X_reduced, selected_features, final_epv = reducer.reduce_features(
            X_adequate, y_adequate, organ='Parotid'
        )
        
        # Should not reduce features
        self.assertEqual(len(selected_features), len(X_adequate.columns))
        self.assertGreaterEqual(final_epv, 5.0)


if __name__ == '__main__':
    unittest.main()
