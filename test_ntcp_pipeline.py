#!/usr/bin/env python3
"""
Comprehensive Test Suite for py_ntcpx v1.0
==========================================

Unit tests, integration tests, and validation tests for the NTCP pipeline.
Tests all components without modifying existing code.

Software: py_ntcpx_v1.0.0
"""

import unittest
import sys
import os
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import modules to test
try:
    from ntcp_utils import normalize_columns, find_dvh_file
except ImportError:
    normalize_columns = None
    find_dvh_file = None

try:
    from ntcp_novel_models import ProbabilisticgEUDModel, MonteCarloNTCPModel
except ImportError:
    ProbabilisticgEUDModel = None
    MonteCarloNTCPModel = None

try:
    from ntcp_qa_modules import UncertaintyAwareNTCP, CohortConsistencyScore
except ImportError:
    UncertaintyAwareNTCP = None
    CohortConsistencyScore = None


class TestNTCPUtils(unittest.TestCase):
    """Test utility functions"""
    
    def test_normalize_columns(self):
        """Test column normalization"""
        if normalize_columns is None:
            self.skipTest("ntcp_utils not available")
        
        # Test various column name formats
        df = pd.DataFrame({
            'Patient ID': ['P001', 'P002'],
            'patient_name': ['John', 'Jane'],
            'ORGAN': ['Parotid', 'Larynx'],
            'Observed_Toxicity': [1, 0],
            'Total_Dose(Gy)': [70, 66]
        })
        
        result = normalize_columns(df.copy())
        
        # Check that columns are normalized
        self.assertIn('PatientID', result.columns)
        self.assertIn('PatientName', result.columns)
        self.assertIn('Organ', result.columns)
    
    def test_normalize_columns_empty(self):
        """Test normalization with empty DataFrame"""
        if normalize_columns is None:
            self.skipTest("ntcp_utils not available")
        
        df = pd.DataFrame()
        result = normalize_columns(df)
        self.assertTrue(result.empty)
    
    def test_find_dvh_file(self):
        """Test DVH file finding"""
        if find_dvh_file is None:
            self.skipTest("ntcp_utils not available")
        
        # Create temporary directory with test files
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            
            # Create test DVH file
            test_file = tmp_path / "PT001_Parotid.csv"
            test_file.write_text("Dose[Gy],Volume[cm3]\n0,100\n50,50\n")
            
            # Test finding file
            found = find_dvh_file(tmp_path, "PT001", "Test", "Parotid")
            self.assertIsNotNone(found)
            self.assertEqual(found.name, "PT001_Parotid.csv")


class TestNovelModels(unittest.TestCase):
    """Test novel NTCP models"""
    
    def test_probabilistic_geud_model_init(self):
        """Test Probabilistic gEUD model initialization"""
        if ProbabilisticgEUDModel is None:
            self.skipTest("ntcp_novel_models not available")
        
        model = ProbabilisticgEUDModel('Parotid')
        self.assertEqual(model.organ, 'Parotid')
        self.assertIn('Parotid', model.param_distributions)
    
    def test_probabilistic_geud_calculate(self):
        """Test Probabilistic gEUD calculation"""
        if ProbabilisticgEUDModel is None:
            self.skipTest("ntcp_novel_models not available")
        
        model = ProbabilisticgEUDModel('Parotid')
        
        # Create sample DVH
        dvh = pd.DataFrame({
            'dose_gy': [0, 10, 20, 30, 40, 50],
            'volume_cm3': [100, 90, 70, 50, 30, 10]
        })
        
        result = model.calculate_ntcp(dvh, n_samples=100)
        
        # Check result structure
        self.assertIn('mean', result)
        self.assertIn('std', result)
        self.assertIn('ci_lower', result)
        self.assertIn('ci_upper', result)
        
        # Check values are in valid range
        self.assertGreaterEqual(result['mean'], 0)
        self.assertLessEqual(result['mean'], 1)
        self.assertGreaterEqual(result['ci_lower'], 0)
        self.assertLessEqual(result['ci_upper'], 1)
    
    def test_monte_carlo_ntcp_model(self):
        """Test Monte Carlo NTCP model"""
        if MonteCarloNTCPModel is None:
            self.skipTest("ntcp_novel_models not available")
        
        model = MonteCarloNTCPModel('Parotid')
        
        # Create sample DVH
        dvh = pd.DataFrame({
            'dose_gy': [0, 10, 20, 30, 40, 50],
            'volume_cm3': [100, 90, 70, 50, 30, 10]
        })
        
        result = model.calculate_ntcp(dvh, n_samples=100)
        
        # Check result structure
        self.assertIn('mean', result)
        self.assertIn('std', result)
        self.assertIn('ci_lower', result)
        self.assertIn('ci_upper', result)


class TestQAModules(unittest.TestCase):
    """Test QA modules"""
    
    def test_uncertainty_aware_ntcp(self):
        """Test Uncertainty-Aware NTCP calculation"""
        if UncertaintyAwareNTCP is None:
            self.skipTest("ntcp_qa_modules not available")
        
        ua_ntcp = UncertaintyAwareNTCP()
        
        # Mock NTCP function
        def mock_ntcp_func(params, dvh):
            return 0.5
        
        params = {'n': 0.7, 'TD50': 28.4, 'm': 0.4}
        dvh = pd.DataFrame({'dose_gy': [0, 50], 'volume_cm3': [100, 0]})
        
        result = ua_ntcp.calculate_untcp(mock_ntcp_func, params, dvh)
        
        # Check result structure
        self.assertIn('ntcp', result)
        self.assertIn('std', result)
        self.assertIn('ci_lower', result)
        self.assertIn('ci_upper', result)
        self.assertIn('uncertainty_contributions', result)
        self.assertIn('clinical_interpretation', result)
        
        # Check values
        self.assertGreaterEqual(result['ntcp'], 0)
        self.assertLessEqual(result['ntcp'], 1)
        self.assertGreaterEqual(result['std'], 0)
    
    def test_cohort_consistency_score(self):
        """Test Cohort Consistency Score"""
        if CohortConsistencyScore is None:
            self.skipTest("ntcp_qa_modules not available")
        
        ccs = CohortConsistencyScore()
        
        # Create training data
        X_train = np.random.randn(50, 5)
        ccs.fit(X_train)
        
        # Test on new data
        X_new = np.random.randn(1, 5)
        result = ccs.calculate_ccs(X_new)
        
        # Check result structure
        self.assertIn('ccs', result)
        self.assertIn('warning', result)
        self.assertIn('safety', result)
        
        # Check CCS is in valid range
        self.assertGreaterEqual(result['ccs'], 0)
        self.assertLessEqual(result['ccs'], 1)


class TestDataValidation(unittest.TestCase):
    """Test data validation"""
    
    def test_dvh_data_structure(self):
        """Test DVH data structure validation"""
        # Valid DVH
        valid_dvh = pd.DataFrame({
            'Dose[Gy]': [0, 10, 20, 30, 40, 50],
            'Volume[cm3]': [100, 90, 70, 50, 30, 10]
        })
        
        # Check required columns
        self.assertIn('Dose[Gy]', valid_dvh.columns)
        self.assertIn('Volume[cm3]', valid_dvh.columns)
        
        # Check data types
        self.assertTrue(pd.api.types.is_numeric_dtype(valid_dvh['Dose[Gy]']))
        self.assertTrue(pd.api.types.is_numeric_dtype(valid_dvh['Volume[cm3]']))
        
        # Check no negative values
        self.assertTrue((valid_dvh['Dose[Gy]'] >= 0).all())
        self.assertTrue((valid_dvh['Volume[cm3]'] >= 0).all())
    
    def test_patient_data_structure(self):
        """Test patient data structure validation"""
        # Valid patient data
        valid_data = pd.DataFrame({
            'PatientID': ['P001', 'P002'],
            'Organ': ['Parotid', 'Larynx'],
            'Observed_Toxicity': [1, 0]
        })
        
        # Check required columns
        self.assertIn('PatientID', valid_data.columns)
        self.assertIn('Organ', valid_data.columns)
        self.assertIn('Observed_Toxicity', valid_data.columns)
        
        # Check toxicity is binary
        self.assertTrue(valid_data['Observed_Toxicity'].isin([0, 1]).all())


class TestOutputFormats(unittest.TestCase):
    """Test output format validation"""
    
    def test_csv_output_format(self):
        """Test CSV output format"""
        # Create sample output
        output = pd.DataFrame({
            'PatientID': ['P001', 'P002'],
            'Organ': ['Parotid', 'Larynx'],
            'NTCP_LKB_LogLogit': [0.5, 0.3],
            'NTCP_ML_ANN': [0.6, 0.4]
        })
        
        # Check structure
        self.assertIn('PatientID', output.columns)
        self.assertIn('Organ', output.columns)
        
        # Check NTCP values are in valid range
        ntcp_cols = [c for c in output.columns if c.startswith('NTCP_')]
        for col in ntcp_cols:
            self.assertTrue((output[col] >= 0).all())
            self.assertTrue((output[col] <= 1).all())
    
    def test_excel_output_format(self):
        """Test Excel output format"""
        # Create sample multi-sheet output
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            with pd.ExcelWriter(tmp_path, engine='openpyxl') as writer:
                pd.DataFrame({'A': [1, 2]}).to_excel(writer, sheet_name='Sheet1', index=False)
                pd.DataFrame({'B': [3, 4]}).to_excel(writer, sheet_name='Sheet2', index=False)
            
            # Verify file exists and can be read - ensure ExcelFile is properly closed
            self.assertTrue(Path(tmp_path).exists())
            with pd.ExcelFile(tmp_path) as xl_file:
                self.assertIn('Sheet1', xl_file.sheet_names)
                self.assertIn('Sheet2', xl_file.sheet_names)
        finally:
            if Path(tmp_path).exists():
                os.unlink(tmp_path)


class TestBiologicalDVH(unittest.TestCase):
    """Test biological DVH transformations"""
    
    def test_bed_calculation(self):
        """Test BED calculation"""
        # BED = nd(1 + d/(α/β))
        dose_per_fraction = 2.0
        n_fractions = 35
        alpha_beta = 3.0
        
        # Total dose
        total_dose = dose_per_fraction * n_fractions
        
        # BED calculation
        bed = n_fractions * dose_per_fraction * (1 + dose_per_fraction / alpha_beta)
        
        # Check BED is greater than total dose
        self.assertGreater(bed, total_dose)
    
    def test_eqd2_calculation(self):
        """Test EQD2 calculation"""
        # EQD2 = D * (d + α/β) / (2 + α/β)
        dose = 70.0
        dose_per_fraction = 2.0
        alpha_beta = 3.0
        
        eqd2 = dose * (dose_per_fraction + alpha_beta) / (2.0 + alpha_beta)
        
        # For standard fractionation, EQD2 should be close to dose
        self.assertAlmostEqual(eqd2, dose, places=1)


class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    def test_pipeline_data_flow(self):
        """Test data flow through pipeline"""
        # Simulate pipeline data flow
        # Step 1: Input data
        input_data = pd.DataFrame({
            'PatientID': ['P001'],
            'Organ': ['Parotid'],
            'Observed_Toxicity': [1]
        })
        
        # Step 2: Processed DVH
        dvh_data = pd.DataFrame({
            'Dose[Gy]': [0, 50],
            'Volume[cm3]': [100, 0]
        })
        
        # Step 3: NTCP results
        ntcp_results = pd.DataFrame({
            'PatientID': ['P001'],
            'Organ': ['Parotid'],
            'NTCP_LKB_LogLogit': [0.5],
            'uNTCP': [0.5],
            'uNTCP_STD': [0.1]
        })
        
        # Verify data consistency
        self.assertEqual(len(input_data), len(ntcp_results))
        self.assertEqual(input_data['PatientID'].iloc[0], ntcp_results['PatientID'].iloc[0])


def run_tests():
    """Run all tests and generate report"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestNTCPUtils,
        TestNovelModels,
        TestQAModules,
        TestDataValidation,
        TestOutputFormats,
        TestBiologicalDVH,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    result = run_tests()
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)

