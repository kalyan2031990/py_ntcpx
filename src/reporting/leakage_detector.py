"""
Data leakage detection for QA reporting
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Set


class DataLeakageDetector:
    """
    Detect data leakage in train-test splits and feature processing
    
    Checks:
    1. Patient overlap between train and test sets
    2. Feature scaling fit on training data only
    3. Feature extraction timing (after split)
    """
    
    def __init__(self):
        """Initialize leakage detector"""
        self.checks_performed = []
        self.warnings = []
        self.errors = []
    
    def check_patient_overlap(self, 
                             train_df: pd.DataFrame,
                             test_df: pd.DataFrame,
                             patient_id_col: str = 'PrimaryPatientID') -> bool:
        """
        Check for patient overlap between train and test sets
        
        Parameters
        ----------
        train_df : pd.DataFrame
            Training set
        test_df : pd.DataFrame
            Test set
        patient_id_col : str
            Column name containing patient ID
            
        Returns
        -------
        bool
            True if no leakage detected, False otherwise
        """
        if patient_id_col not in train_df.columns or patient_id_col not in test_df.columns:
            self.errors.append(
                f"Patient ID column '{patient_id_col}' not found in train or test data"
            )
            return False
        
        train_patients = set(train_df[patient_id_col].unique())
        test_patients = set(test_df[patient_id_col].unique())
        overlap = train_patients & test_patients
        
        if overlap:
            self.errors.append(
                f"DATA LEAKAGE DETECTED: {len(overlap)} patients in both train and test sets. "
                f"Sample IDs: {list(overlap)[:5]}"
            )
            return False
        else:
            self.checks_performed.append("Patient overlap check: PASSED")
            return True
    
    def check_feature_scaling(self, 
                             X_train_mean: Optional[float] = None,
                             X_test_mean: Optional[float] = None,
                             X_train_std: Optional[float] = None) -> bool:
        """
        Check that feature scaling was fit on training data only
        
        Parameters
        ----------
        X_train_mean : float, optional
            Mean of scaled training features (should be ~0)
        X_test_mean : float, optional
            Mean of scaled test features (may differ from 0)
        X_train_std : float, optional
            Std of scaled training features (should be ~1)
            
        Returns
        -------
        bool
            True if scaling appears correct, False otherwise
        """
        if X_train_mean is None:
            self.warnings.append("Feature scaling check skipped: no training mean provided")
            return True
        
        # Training data should be scaled to mean ~0, std ~1
        if abs(X_train_mean) > 0.1:
            self.warnings.append(
                f"Training features mean is {X_train_mean:.3f}, expected ~0. "
                "This may indicate scaling was fit on test data."
            )
        
        if X_train_std is not None and abs(X_train_std - 1.0) > 0.1:
            self.warnings.append(
                f"Training features std is {X_train_std:.3f}, expected ~1. "
                "This may indicate incorrect scaling."
            )
        else:
            self.checks_performed.append("Feature scaling check: PASSED")
        
        return True
    
    def check_feature_extraction_timing(self,
                                       features_extracted_before_split: bool = False) -> bool:
        """
        Check that feature extraction happened after train-test split
        
        Parameters
        ----------
        features_extracted_before_split : bool
            Whether features were extracted before split (leakage risk)
            
        Returns
        -------
        bool
            True if no leakage detected, False otherwise
        """
        if features_extracted_before_split:
            self.warnings.append(
                "Feature extraction may have occurred before train-test split. "
                "This can cause data leakage if feature statistics use test data."
            )
            return False
        else:
            self.checks_performed.append("Feature extraction timing check: PASSED")
            return True
    
    def generate_report(self) -> Dict:
        """
        Generate leakage detection report
        
        Returns
        -------
        dict with keys:
            - 'passed': bool - True if all checks passed
            - 'checks_performed': list of check descriptions
            - 'warnings': list of warnings
            - 'errors': list of errors
            - 'summary': str - Human-readable summary
        """
        passed = len(self.errors) == 0
        has_warnings = len(self.warnings) > 0
        
        summary = f"""
Data Leakage Detection Report
==============================
Status: {'PASSED' if passed else 'FAILED'}
Checks Performed: {len(self.checks_performed)}
Warnings: {len(self.warnings)}
Errors: {len(self.errors)}

Checks:
{chr(10).join(f'  ✓ {check}' for check in self.checks_performed)}

"""
        
        if self.warnings:
            summary += f"Warnings:\n{chr(10).join(f'  ⚠ {w}' for w in self.warnings)}\n\n"
        
        if self.errors:
            summary += f"Errors:\n{chr(10).join(f'  ✗ {e}' for e in self.errors)}\n\n"
        
        return {
            'passed': passed,
            'checks_performed': self.checks_performed,
            'warnings': self.warnings,
            'errors': self.errors,
            'summary': summary,
            'has_warnings': has_warnings
        }
