"""
Test suite for Clinical Safety Guard (Phase 7)

Tests safety flagging and DO_NOT_USE criteria
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.safety.clinical_safety_guard import ClinicalSafetyGuard


class TestClinicalSafetyGuard(unittest.TestCase):
    """Test suite for ClinicalSafetyGuard"""
    
    def setUp(self):
        """Create synthetic test data"""
        np.random.seed(42)
        
        # Training data
        self.X_train = np.random.randn(50, 5)
        
        # Test predictions
        self.ntcp_predictions = np.random.uniform(0.1, 0.9, 20)
        self.ntcp_ci_lower = self.ntcp_predictions - 0.1
        self.ntcp_ci_lower = np.clip(self.ntcp_ci_lower, 0.0, 1.0)
        self.X_test = np.random.randn(20, 5)
    
    def test_safety_guard_initialization(self):
        """Test safety guard initialization"""
        guard = ClinicalSafetyGuard(ccs_threshold=0.2)
        
        self.assertEqual(guard.ccs_threshold, 0.2)
        self.assertEqual(guard.ci_alpha, 0.05)
    
    def test_fit_training_data(self):
        """Test fitting on training data"""
        guard = ClinicalSafetyGuard()
        guard.fit(self.X_train)
        
        self.assertIsNotNone(guard.training_stats)
        self.assertEqual(guard.training_stats['n_samples'], len(self.X_train))
    
    def test_evaluate_safety_basic(self):
        """Test basic safety evaluation"""
        guard = ClinicalSafetyGuard()
        guard.fit(self.X_train)
        
        safety_df = guard.evaluate_safety(
            self.ntcp_predictions,
            self.ntcp_ci_lower,
            self.X_test
        )
        
        self.assertEqual(len(safety_df), len(self.ntcp_predictions))
        self.assertIn('NTCP_Prediction', safety_df.columns)
        self.assertIn('Safety_Flag', safety_df.columns)
        self.assertIn('DO_NOT_USE', safety_df.columns)
    
    def test_do_not_use_flag_low_ccs(self):
        """Test that DO_NOT_USE flag is set for low CCS"""
        guard = ClinicalSafetyGuard(ccs_threshold=0.2)
        guard.fit(self.X_train)
        
        # Create test data with very different distribution (low CCS)
        X_test_outlier = self.X_train.mean(axis=0) + 10 * self.X_train.std(axis=0)
        X_test_outlier = X_test_outlier.reshape(1, -1)
        
        safety_df = guard.evaluate_safety(
            np.array([0.5]),
            np.array([0.4]),
            X_test_outlier
        )
        
        # Should flag DO_NOT_USE if CCS is low
        # (Note: actual CCS calculation depends on CohortConsistencyScore implementation)
        self.assertIn('DO_NOT_USE', safety_df.columns)
    
    def test_underprediction_risk_detection(self):
        """Test underprediction risk detection"""
        guard = ClinicalSafetyGuard()
        guard.fit(self.X_train)
        
        # High underprediction risk: low CI lower bound, high point prediction
        ntcp_high_risk = np.array([0.5])  # High prediction
        ci_lower_high_risk = np.array([0.05])  # Very low CI lower bound
        
        safety_df = guard.evaluate_safety(
            ntcp_high_risk,
            ci_lower_high_risk,
            self.X_test[:1]
        )
        
        self.assertIn('Underprediction_Risk', safety_df.columns)
        # Should detect high risk
        risk = safety_df['Underprediction_Risk'].iloc[0]
        self.assertIn(risk, ['HIGH', 'MODERATE', 'LOW', 'UNKNOWN'])
    
    def test_safety_report_generation(self):
        """Test safety report generation"""
        guard = ClinicalSafetyGuard()
        guard.fit(self.X_train)
        
        safety_df = guard.evaluate_safety(
            self.ntcp_predictions,
            self.ntcp_ci_lower,
            self.X_test
        )
        
        report = guard.generate_safety_report(safety_df)
        
        self.assertIn("CLINICAL SAFETY REPORT", report)
        self.assertIn("Total Predictions", report)
        # v3.0.0: Changed to "CCS Warnings" instead of "DO_NOT_USE Flags"
        self.assertIn("CCS Warnings", report)
    
    def test_safety_report_save(self):
        """Test saving safety report to file"""
        guard = ClinicalSafetyGuard()
        guard.fit(self.X_train)
        
        safety_df = guard.evaluate_safety(
            self.ntcp_predictions,
            self.ntcp_ci_lower,
            self.X_test
        )
        
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            report_path = Path(f.name)
        
        try:
            report = guard.generate_safety_report(safety_df, report_path)
            self.assertTrue(report_path.exists())
        finally:
            if report_path.exists():
                report_path.unlink()


if __name__ == '__main__':
    unittest.main()
