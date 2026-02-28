#!/usr/bin/env python3
"""
Unit tests for NTCPEvaluator and v1.1.0 evaluation/QA plumbing.
"""

import unittest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from ntcp_models import EPVError, check_epv  # type: ignore
from src.metrics import NTCPEvaluator  # type: ignore


class TestNTCPEvaluator(unittest.TestCase):
    """Tests for unified NTCP evaluator and EPV logic."""

    def test_epv_error_raised_when_too_low(self):
        """EPV < 10 should raise EPVError for Tier 3 logistic when strict."""
        n_events = 5
        n_features = 2  # EPV = 2.5 < 10
        with self.assertRaises(EPVError):
            check_epv(n_events=n_events, n_features=n_features, min_epv=10.0, model_name="Tier3 Logistic")

    def test_overfitting_flag_gap_based(self):
        """Models with gap > 0.10 should be flagged regardless of absolute AUC."""
        evaluator = NTCPEvaluator(n_bootstrap=10, random_state=0)
        # Construct a deterministic example where apparent AUC is high
        # and CV AUC is close to chance, ensuring a large gap.
        y_true = np.array([0] * 50 + [1] * 50)
        # Apparent model separates classes almost perfectly
        y_pred_apparent = np.concatenate([
            np.linspace(0.0, 0.2, 50),
            np.linspace(0.8, 1.0, 50),
        ])
        # CV predictions are nearly uninformative
        y_pred_cv = np.full_like(y_pred_apparent, 0.5)

        m = evaluator.evaluate(
            y_true=y_true,
            y_pred_apparent=y_pred_apparent,
            model_name="TestModel",
            tier="T3",
            organ="Parotid",
            y_pred_cv=y_pred_cv,
            cv_strategy="5-fold",
        )
        self.assertTrue(m.overfitting_flag)
        self.assertIsNotNone(m.overfitting_gap)
        self.assertGreater(m.overfitting_gap, 0.10)


if __name__ == "__main__":
    unittest.main()

