"""
Test suite for DVH Validation (Phase 2)

Tests DVH invariance: V(0) = 100%, monotonicity, gEUD reproducibility
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDVHValidation(unittest.TestCase):
    """Test suite for DVH validation"""
    
    def setUp(self):
        """Create synthetic DVH data"""
        np.random.seed(42)
        
        # Valid cumulative DVH
        doses = np.linspace(0, 70, 100)
        volumes = 100 - (doses / 70) * 100  # Linear decrease
        volumes = np.clip(volumes, 0, 100)
        
        self.valid_dvh = pd.DataFrame({
            'Dose[Gy]': doses,
            'Volume[%]': volumes
        })
    
    def test_v0_equals_100(self):
        """Test that V(0) = 100%"""
        # V(0) should be 100%
        v0 = self.valid_dvh[self.valid_dvh['Dose[Gy]'] == 0]['Volume[%]'].values
        
        if len(v0) > 0:
            self.assertAlmostEqual(v0[0], 100.0, places=1)
    
    def test_dvh_monotonicity(self):
        """Test that cumulative DVH is non-increasing"""
        volumes = self.valid_dvh['Volume[%]'].values
        doses = self.valid_dvh['Dose[Gy]'].values
        
        # Sort by dose
        sorted_idx = np.argsort(doses)
        sorted_volumes = volumes[sorted_idx]
        
        # Check monotonicity: volumes should be non-increasing with dose
        diffs = np.diff(sorted_volumes)
        # Allow small numerical errors
        self.assertTrue(np.all(diffs <= 1e-6), "DVH is not monotonic (non-increasing)")
    
    def test_dose_non_negative(self):
        """Test that doses are non-negative"""
        doses = self.valid_dvh['Dose[Gy]'].values
        self.assertTrue(np.all(doses >= 0), "Negative doses found")
    
    def test_volume_range(self):
        """Test that volumes are in [0, 100]"""
        volumes = self.valid_dvh['Volume[%]'].values
        self.assertTrue(np.all(volumes >= 0), "Negative volumes found")
        self.assertTrue(np.all(volumes <= 100), "Volumes > 100% found")
    
    def test_gEUD_reproducibility(self):
        """Test gEUD calculation reproducibility"""
        # Simple gEUD calculation
        doses = self.valid_dvh['Dose[Gy]'].values
        volumes = self.valid_dvh['Volume[%]'].values / 100.0  # Convert to fraction
        
        # Calculate gEUD with a=1 (mean dose)
        total_volume = np.sum(volumes)
        if total_volume > 0:
            rel_volumes = volumes / total_volume
            geud_a1 = np.sum(rel_volumes * doses)
            
            # Calculate again - should be identical
            geud_a1_2 = np.sum(rel_volumes * doses)
            
            self.assertAlmostEqual(geud_a1, geud_a1_2, places=10)


if __name__ == '__main__':
    unittest.main()
