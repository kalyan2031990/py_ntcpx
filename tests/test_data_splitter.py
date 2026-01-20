"""
Test suite for PatientDataSplitter

Tests patient-level splitting and leakage detection using synthetic data
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.validation.data_splitter import PatientDataSplitter


class TestPatientDataSplitter(unittest.TestCase):
    """Test suite for PatientDataSplitter"""
    
    def setUp(self):
        """Create synthetic test data"""
        np.random.seed(42)
        
        # Create synthetic patient data
        # Each patient may have multiple rows (e.g., multiple organs)
        n_patients = 54
        n_organs_per_patient = 2
        
        data = []
        for i in range(n_patients):
            patient_id = f"SYN_{i:03d}"
            
            # Realistic dose distribution
            dmean = np.random.uniform(15, 50)
            
            # Toxicity probability increases with dose
            toxicity_prob = 1 / (1 + np.exp(-0.1 * (dmean - 30)))
            
            for organ in ['Parotid_L', 'Parotid_R']:
                toxicity = np.random.binomial(1, toxicity_prob)
                data.append({
                    'PrimaryPatientID': patient_id,
                    'Organ': organ,
                    'Dmean': dmean + np.random.normal(0, 2),
                    'V30': np.random.uniform(30, 80),
                    'V45': np.random.uniform(10, 50),
                    'Observed_Toxicity': toxicity
                })
        
        self.test_df = pd.DataFrame(data)
    
    def test_train_test_no_overlap(self):
        """Verify no patient overlap between train and test"""
        splitter = PatientDataSplitter(random_seed=42, test_size=0.2)
        train_df, test_df = splitter.create_splits(
            self.test_df,
            patient_id_col='PrimaryPatientID',
            outcome_col='Observed_Toxicity'
        )
        
        train_patients = set(train_df['PrimaryPatientID'].unique())
        test_patients = set(test_df['PrimaryPatientID'].unique())
        overlap = train_patients & test_patients
        
        self.assertEqual(len(overlap), 0, "Data leakage detected: patients in both sets!")
    
    def test_patient_level_split(self):
        """Verify split is at patient level, not row level"""
        splitter = PatientDataSplitter(random_seed=42, test_size=0.2)
        train_df, test_df = splitter.create_splits(
            self.test_df,
            patient_id_col='PrimaryPatientID',
            outcome_col='Observed_Toxicity'
        )
        
        # Check that all rows for a patient are in the same set
        train_patients = set(train_df['PrimaryPatientID'].unique())
        test_patients = set(test_df['PrimaryPatientID'].unique())
        
        # Verify no patient has rows in both sets
        for patient_id in train_patients:
            self.assertNotIn(patient_id, test_patients, 
                           f"Patient {patient_id} appears in both sets!")
        
        # Verify total patients preserved
        total_train_patients = len(train_patients)
        total_test_patients = len(test_patients)
        total_unique_patients = len(self.test_df['PrimaryPatientID'].unique())
        
        self.assertEqual(
            total_train_patients + total_test_patients,
            total_unique_patients,
            "Total patients not preserved in split!"
        )
    
    def test_leakage_detection(self):
        """Test leakage detection function"""
        splitter = PatientDataSplitter(random_seed=42, test_size=0.2)
        train_df, test_df = splitter.create_splits(
            self.test_df,
            patient_id_col='PrimaryPatientID',
            outcome_col='Observed_Toxicity'
        )
        
        # Should pass validation
        self.assertTrue(
            splitter.validate_no_leakage(train_df, test_df, patient_id_col='PrimaryPatientID')
        )
        
        # Create artificial leakage
        leaked_patient = train_df['PrimaryPatientID'].iloc[0]
        test_df_with_leak = test_df.copy()
        test_df_with_leak = pd.concat([
            test_df_with_leak,
            train_df[train_df['PrimaryPatientID'] == leaked_patient]
        ])
        
        # Should raise ValueError
        with self.assertRaises(ValueError):
            splitter.validate_no_leakage(
                train_df, 
                test_df_with_leak, 
                patient_id_col='PrimaryPatientID'
            )
    
    def test_stratification(self):
        """Test that stratification maintains outcome distribution"""
        splitter = PatientDataSplitter(random_seed=42, test_size=0.2)
        train_df, test_df = splitter.create_splits(
            self.test_df,
            patient_id_col='PrimaryPatientID',
            outcome_col='Observed_Toxicity'
        )
        
        # Calculate outcome rates (per patient)
        train_patients = self.test_df[
            self.test_df['PrimaryPatientID'].isin(train_df['PrimaryPatientID'].unique())
        ]
        test_patients = self.test_df[
            self.test_df['PrimaryPatientID'].isin(test_df['PrimaryPatientID'].unique())
        ]
        
        train_outcome_rate = train_patients.groupby('PrimaryPatientID')['Observed_Toxicity'].max().mean()
        test_outcome_rate = test_patients.groupby('PrimaryPatientID')['Observed_Toxicity'].max().mean()
        overall_outcome_rate = self.test_df.groupby('PrimaryPatientID')['Observed_Toxicity'].max().mean()
        
        # Check that outcome rates are similar (within 20% relative difference)
        self.assertLess(
            abs(train_outcome_rate - overall_outcome_rate) / overall_outcome_rate,
            0.2,
            "Train outcome rate differs too much from overall"
        )
        self.assertLess(
            abs(test_outcome_rate - overall_outcome_rate) / overall_outcome_rate,
            0.2,
            "Test outcome rate differs too much from overall"
        )
    
    def test_reproducibility(self):
        """Test that same seed gives identical results"""
        splitter1 = PatientDataSplitter(random_seed=42, test_size=0.2)
        train_df1, test_df1 = splitter1.create_splits(
            self.test_df,
            patient_id_col='PrimaryPatientID',
            outcome_col='Observed_Toxicity'
        )
        
        splitter2 = PatientDataSplitter(random_seed=42, test_size=0.2)
        train_df2, test_df2 = splitter2.create_splits(
            self.test_df,
            patient_id_col='PrimaryPatientID',
            outcome_col='Observed_Toxicity'
        )
        
        train_patients1 = set(train_df1['PrimaryPatientID'].unique())
        train_patients2 = set(train_df2['PrimaryPatientID'].unique())
        test_patients1 = set(test_df1['PrimaryPatientID'].unique())
        test_patients2 = set(test_df2['PrimaryPatientID'].unique())
        
        self.assertEqual(train_patients1, train_patients2, "Train sets differ with same seed!")
        self.assertEqual(test_patients1, test_patients2, "Test sets differ with same seed!")


if __name__ == '__main__':
    unittest.main()
