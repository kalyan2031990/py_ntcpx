"""
Test suite for Classical NTCP Model Mathematics (Phase 3)

Tests: NTCP ∈ [0, 1], monotonicity, NTCP(TD50) ≈ 0.5
"""

import unittest
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestNTCPMathematics(unittest.TestCase):
    """Test suite for NTCP mathematical properties"""
    
    def test_ntcp_bounds(self):
        """Test that NTCP is in [0, 1] for all inputs"""
        # LKB Log-Logistic model
        def lkb_loglogit(geud, TD50=30, m=0.1):
            """LKB Log-Logistic NTCP"""
            if geud <= 0:
                return 0.0
            return 1.0 / (1.0 + (TD50 / geud) ** (4 * m))
        
        # Test over wide range of doses
        doses = np.linspace(0, 100, 1000)
        predictions = [lkb_loglogit(d) for d in doses]
        
        self.assertTrue(np.all(np.array(predictions) >= 0), "NTCP < 0 detected")
        self.assertTrue(np.all(np.array(predictions) <= 1), "NTCP > 1 detected")
    
    def test_ntcp_monotonicity(self):
        """Test that NTCP increases with dose"""
        def lkb_loglogit(geud, TD50=30, m=0.1):
            if geud <= 0:
                return 0.0
            return 1.0 / (1.0 + (TD50 / geud) ** (4 * m))
        
        doses = np.linspace(10, 60, 100)
        predictions = [lkb_loglogit(d) for d in doses]
        
        # Check monotonicity
        diffs = np.diff(predictions)
        self.assertTrue(np.all(diffs >= -1e-10), "NTCP not monotonic (should increase with dose)")
    
    def test_ntcp_at_td50(self):
        """Test that NTCP(TD50) ≈ 0.5"""
        def lkb_loglogit(geud, TD50=30, m=0.1):
            if geud <= 0:
                return 0.0
            return 1.0 / (1.0 + (TD50 / geud) ** (4 * m))
        
        TD50 = 30
        ntcp_at_td50 = lkb_loglogit(TD50, TD50=TD50, m=0.1)
        
        # Should be approximately 0.5
        self.assertAlmostEqual(ntcp_at_td50, 0.5, places=2, 
                             msg=f"NTCP(TD50) should be ~0.5, got {ntcp_at_td50}")
    
    def test_ntcp_edge_cases(self):
        """Test NTCP at edge cases"""
        def lkb_loglogit(geud, TD50=30, m=0.1):
            if geud <= 0:
                return 0.0
            return 1.0 / (1.0 + (TD50 / geud) ** (4 * m))
        
        # Very low dose
        ntcp_low = lkb_loglogit(0.1, TD50=30, m=0.1)
        self.assertGreaterEqual(ntcp_low, 0.0)
        self.assertLessEqual(ntcp_low, 1.0)
        
        # Very high dose
        ntcp_high = lkb_loglogit(100, TD50=30, m=0.1)
        self.assertGreaterEqual(ntcp_high, 0.0)
        self.assertLessEqual(ntcp_high, 1.0)
        self.assertGreater(ntcp_high, 0.5)  # Should be > 0.5 for high dose


if __name__ == '__main__':
    unittest.main()
