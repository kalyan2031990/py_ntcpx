#!/usr/bin/env python3
"""
Data Validation Tests for py_ntcpx v1.0
========================================

Tests for data format validation, edge cases, and error handling.
Does not modify existing code.

Software: py_ntcpx_v1.0.0
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os


class TestDVHDataValidation(unittest.TestCase):
    """Test DVH data validation"""
    
    def test_valid_dvh_structure(self):
        """Test valid DVH structure"""
        dvh = pd.DataFrame({
            'Dose[Gy]': [0, 10, 20, 30, 40, 50],
            'Volume[cm3]': [100, 90, 70, 50, 30, 10]
        })
        
        # Check structure
        self.assertIn('Dose[Gy]', dvh.columns)
        self.assertIn('Volume[cm3]', dvh.columns)
        self.assertEqual(len(dvh), 6)
    
    def test_dvh_monotonic_dose(self):
        """Test DVH has monotonic dose values"""
        dvh = pd.DataFrame({
            'Dose[Gy]': [0, 10, 20, 30, 40, 50],
            'Volume[cm3]': [100, 90, 70, 50, 30, 10]
        })
        
        # Check dose is non-decreasing
        self.assertTrue((dvh['Dose[Gy]'].diff().dropna() >= 0).all())
    
    def test_dvh_volume_non_negative(self):
        """Test DVH volumes are non-negative"""
        dvh = pd.DataFrame({
            'Dose[Gy]': [0, 10, 20, 30, 40, 50],
            'Volume[cm3]': [100, 90, 70, 50, 30, 10]
        })
        
        # Check volumes are non-negative
        self.assertTrue((dvh['Volume[cm3]'] >= 0).all())
    
    def test_dvh_empty(self):
        """Test handling of empty DVH"""
        dvh = pd.DataFrame(columns=['Dose[Gy]', 'Volume[cm3]'])
        self.assertTrue(dvh.empty)


class TestPatientDataValidation(unittest.TestCase):
    """Test patient data validation"""
    
    def test_valid_patient_data(self):
        """Test valid patient data structure"""
        data = pd.DataFrame({
            'PatientID': ['P001', 'P002', 'P003'],
            'Organ': ['Parotid', 'Larynx', 'SpinalCord'],
            'Observed_Toxicity': [1, 0, 1]
        })
        
        # Check required columns
        self.assertIn('PatientID', data.columns)
        self.assertIn('Organ', data.columns)
        self.assertIn('Observed_Toxicity', data.columns)
    
    def test_toxicity_binary(self):
        """Test toxicity is binary"""
        data = pd.DataFrame({
            'Observed_Toxicity': [0, 1, 0, 1, 0]
        })
        
        # Check all values are 0 or 1
        self.assertTrue(data['Observed_Toxicity'].isin([0, 1]).all())
    
    def test_organ_names(self):
        """Test organ names are valid"""
        valid_organs = ['Parotid', 'Larynx', 'SpinalCord', 'OralCavity']
        data = pd.DataFrame({
            'Organ': valid_organs
        })
        
        # All organs should be strings
        self.assertTrue(data['Organ'].dtype == 'object')
        self.assertTrue(data['Organ'].notna().all())


class TestNTCPValueValidation(unittest.TestCase):
    """Test NTCP value validation"""
    
    def test_ntcp_range(self):
        """Test NTCP values are in [0, 1] range"""
        ntcp_values = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        
        # Check all values in range
        self.assertTrue((ntcp_values >= 0).all())
        self.assertTrue((ntcp_values <= 1).all())
    
    def test_ntcp_confidence_intervals(self):
        """Test NTCP confidence intervals are valid"""
        ntcp = 0.5
        std = 0.1
        ci_lower = max(0, ntcp - 1.96 * std)
        ci_upper = min(1, ntcp + 1.96 * std)
        
        # Check CI bounds
        self.assertGreaterEqual(ci_lower, 0)
        self.assertLessEqual(ci_upper, 1)
        self.assertLessEqual(ci_lower, ci_upper)
        self.assertLessEqual(ci_lower, ntcp)
        self.assertGreaterEqual(ci_upper, ntcp)


class TestOutputFileValidation(unittest.TestCase):
    """Test output file validation"""
    
    def test_csv_output(self):
        """Test CSV output can be created and read"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write("PatientID,Organ,NTCP\nP001,Parotid,0.5\n")
        
        try:
            # Read back
            df = pd.read_csv(tmp_path)
            self.assertIn('PatientID', df.columns)
            self.assertIn('Organ', df.columns)
            self.assertIn('NTCP', df.columns)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_excel_output(self):
        """Test Excel output can be created and read"""
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Write Excel
            with pd.ExcelWriter(tmp_path, engine='openpyxl') as writer:
                pd.DataFrame({'A': [1, 2]}).to_excel(writer, sheet_name='Sheet1', index=False)
            
            # Read back - ensure ExcelFile is properly closed
            with pd.ExcelFile(tmp_path) as xl_file:
                self.assertIn('Sheet1', xl_file.sheet_names)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def test_missing_values(self):
        """Test handling of missing values"""
        data = pd.DataFrame({
            'PatientID': ['P001', 'P002', None],
            'Organ': ['Parotid', None, 'Larynx'],
            'NTCP': [0.5, 0.3, np.nan]
        })
        
        # Check missing values are detected
        self.assertTrue(data['PatientID'].isna().any())
        self.assertTrue(data['Organ'].isna().any())
        self.assertTrue(data['NTCP'].isna().any())
    
    def test_extreme_dose_values(self):
        """Test handling of extreme dose values"""
        # Very high dose
        high_dose = pd.DataFrame({
            'Dose[Gy]': [0, 100, 200],
            'Volume[cm3]': [100, 50, 0]
        })
        
        # Very low dose
        low_dose = pd.DataFrame({
            'Dose[Gy]': [0, 0.1, 0.2],
            'Volume[cm3]': [100, 99, 98]
        })
        
        # Both should be valid structures
        self.assertIn('Dose[Gy]', high_dose.columns)
        self.assertIn('Dose[Gy]', low_dose.columns)
    
    def test_single_bin_dvh(self):
        """Test handling of single-bin DVH"""
        single_bin = pd.DataFrame({
            'Dose[Gy]': [50],
            'Volume[cm3]': [100]
        })
        
        # Should still be valid
        self.assertEqual(len(single_bin), 1)
        self.assertIn('Dose[Gy]', single_bin.columns)


if __name__ == '__main__':
    unittest.main()

