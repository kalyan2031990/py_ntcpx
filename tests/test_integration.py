"""
Integration tests for end-to-end pipeline

Tests complete workflow from data loading to model evaluation
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import tempfile
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.validation.data_splitter import PatientDataSplitter
from src.models.machine_learning.ml_models import OverfitResistantMLModels
from src.features.feature_selector import RadiobiologyGuidedFeatureSelector
from src.metrics.auc_calculator import calculate_auc_with_ci
from src.reporting.leakage_detector import DataLeakageDetector


class TestEndToEndPipeline(unittest.TestCase):
    """Integration tests for complete pipeline"""
    
    def setUp(self):
        """Create synthetic test data"""
        np.random.seed(42)
        
        # Create realistic patient data
        n_patients = 54
        n_organs_per_patient = 2
        
        data = []
        for i in range(n_patients):
            patient_id = f"SYN_{i:03d}"
            dmean = np.random.uniform(15, 50)
            toxicity_prob = 1 / (1 + np.exp(-0.1 * (dmean - 30)))
            
            for organ in ['Parotid_L', 'Parotid_R']:
                toxicity = np.random.binomial(1, toxicity_prob)
                data.append({
                    'PrimaryPatientID': patient_id,
                    'Organ': organ,
                    'Dmean': dmean + np.random.normal(0, 2),
                    'V30': np.random.uniform(30, 80),
                    'V45': np.random.uniform(10, 50),
                    'gEUD': np.random.uniform(25, 45),
                    'mean_dose': dmean,
                    'Observed_Toxicity': toxicity
                })
        
        self.patient_df = pd.DataFrame(data)
    
    def test_patient_level_split_integration(self):
        """Test patient-level splitting in complete workflow"""
        # Split data
        splitter = PatientDataSplitter(random_seed=42, test_size=0.2)
        train_df, test_df = splitter.create_splits(
            self.patient_df,
            patient_id_col='PrimaryPatientID',
            outcome_col='Observed_Toxicity'
        )
        
        # Verify no leakage
        leakage_detector = DataLeakageDetector()
        leakage_check = leakage_detector.check_patient_overlap(
            train_df, test_df, 'PrimaryPatientID'
        )
        self.assertTrue(leakage_check, "Data leakage detected!")
        
        # Verify split sizes
        train_patients = len(train_df['PrimaryPatientID'].unique())
        test_patients = len(test_df['PrimaryPatientID'].unique())
        total_patients = len(self.patient_df['PrimaryPatientID'].unique())
        
        self.assertEqual(train_patients + test_patients, total_patients)
        self.assertGreater(train_patients, 0)
        self.assertGreater(test_patients, 0)
    
    def test_feature_selection_integration(self):
        """Test feature selection in complete workflow"""
        # Split data
        splitter = PatientDataSplitter(random_seed=42, test_size=0.2)
        train_df, test_df = splitter.create_splits(
            self.patient_df,
            patient_id_col='PrimaryPatientID',
            outcome_col='Observed_Toxicity'
        )
        
        # Prepare features
        feature_cols = ['Dmean', 'V30', 'V45', 'gEUD', 'mean_dose']
        X_train = train_df[feature_cols].values
        y_train = train_df['Observed_Toxicity'].values
        
        # Select features
        selector = RadiobiologyGuidedFeatureSelector()
        X_train_df = pd.DataFrame(X_train, columns=feature_cols)
        selected_features = selector.select_features(
            X_train_df, y_train, organ='Parotid'
        )
        
        # Verify essential features selected
        self.assertIn('Dmean', selected_features)
        self.assertIn('V30', selected_features)
        self.assertIn('V45', selected_features)
        self.assertLessEqual(len(selected_features), len(feature_cols))
    
    def test_ml_training_integration(self):
        """Test ML model training in complete workflow"""
        # Split data
        splitter = PatientDataSplitter(random_seed=42, test_size=0.2)
        train_df, test_df = splitter.create_splits(
            self.patient_df,
            patient_id_col='PrimaryPatientID',
            outcome_col='Observed_Toxicity'
        )
        
        # Prepare features
        feature_cols = ['Dmean', 'V30', 'V45']
        X_train = train_df[feature_cols].values
        y_train = train_df['Observed_Toxicity'].values
        X_test = test_df[feature_cols].values
        y_test = test_df['Observed_Toxicity'].values
        
        # Train model
        ml_model = OverfitResistantMLModels(
            n_features=X_train.shape[1],
            n_samples=len(X_train),
            n_events=int(np.sum(y_train)),
            random_seed=42
        )
        
        ann_model = ml_model.create_ann_model()
        ann_model.fit(X_train, y_train)
        
        # Predict
        y_pred = ann_model.predict_proba(X_test)[:, 1]
        
        # Calculate AUC with CI
        auc_val, auc_ci = calculate_auc_with_ci(y_test, y_pred)
        
        # Verify results
        self.assertGreaterEqual(auc_val, 0.0)
        self.assertLessEqual(auc_val, 1.0)
        self.assertLessEqual(auc_ci[0], auc_val)
        self.assertGreaterEqual(auc_ci[1], auc_val)
    
    def test_complete_workflow(self):
        """Test complete workflow from split to evaluation"""
        # 1. Split data
        splitter = PatientDataSplitter(random_seed=42, test_size=0.2)
        train_df, test_df = splitter.create_splits(
            self.patient_df,
            patient_id_col='PrimaryPatientID',
            outcome_col='Observed_Toxicity'
        )
        
        # 2. Check for leakage
        leakage_detector = DataLeakageDetector()
        leakage_detector.check_patient_overlap(train_df, test_df, 'PrimaryPatientID')
        leakage_report = leakage_detector.generate_report()
        self.assertTrue(leakage_report['passed'], "Leakage detected in workflow!")
        
        # 3. Feature selection
        feature_cols = ['Dmean', 'V30', 'V45', 'gEUD', 'mean_dose']
        X_train = train_df[feature_cols].values
        y_train = train_df['Observed_Toxicity'].values
        X_test = test_df[feature_cols].values
        y_test = test_df['Observed_Toxicity'].values
        
        selector = RadiobiologyGuidedFeatureSelector()
        X_train_df = pd.DataFrame(X_train, columns=feature_cols)
        selected_features = selector.select_features(
            X_train_df, y_train, organ='Parotid'
        )
        
        # 4. Train model
        selected_indices = [i for i, f in enumerate(feature_cols) if f in selected_features]
        X_train_selected = X_train[:, selected_indices]
        X_test_selected = X_test[:, selected_indices]
        
        ml_model = OverfitResistantMLModels(
            n_features=X_train_selected.shape[1],
            n_samples=len(X_train_selected),
            n_events=int(np.sum(y_train)),
            random_seed=42
        )
        
        ann_model = ml_model.create_ann_model()
        ann_model.fit(X_train_selected, y_train)
        
        # 5. Evaluate
        y_pred = ann_model.predict_proba(X_test_selected)[:, 1]
        auc_val, auc_ci = calculate_auc_with_ci(y_test, y_pred)
        
        # 6. Verify all steps completed
        self.assertGreater(len(selected_features), 0)
        self.assertGreater(auc_val, 0.0)
        self.assertLess(auc_val, 1.0)


if __name__ == '__main__':
    unittest.main()
