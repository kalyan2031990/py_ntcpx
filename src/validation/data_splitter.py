"""
Patient-level data splitting with leakage prevention

CRITICAL: Split by PATIENT ID, not by rows
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
import warnings
from sklearn.model_selection import train_test_split


class PatientDataSplitter:
    """
    CRITICAL: Patient-level splitting with no data leakage
    
    Ensures:
    - Split by PATIENT ID, not by rows
    - Stratify by toxicity outcome AND institution if multi-site
    - NEVER allow same patient in both train and test
    """
    
    def __init__(self, random_seed: int = 42, test_size: float = 0.2):
        """
        Initialize patient data splitter
        
        Parameters
        ----------
        random_seed : int
            Random seed for reproducibility
        test_size : float
            Proportion of patients in test set (0.0 to 1.0)
        """
        self.random_seed = random_seed
        self.test_size = test_size
        self._fitted = False
        self._train_patients = None
        self._test_patients = None
        
    def create_splits(self, 
                     patient_df: pd.DataFrame, 
                     patient_id_col: str = 'PrimaryPatientID',
                     outcome_col: str = 'Observed_Toxicity',
                     stratify_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create patient-level train/test splits with stratification.
        
        IMPORTANT: 
        - Split by PATIENT ID, not by rows
        - Stratify by toxicity outcome AND institution if multi-site
        - NEVER allow same patient in both train and test
        
        Parameters
        ----------
        patient_df : pd.DataFrame
            DataFrame with patient data (may have multiple rows per patient)
        patient_id_col : str
            Column name containing patient ID
        outcome_col : str
            Column name containing outcome (toxicity)
        stratify_cols : list of str, optional
            Additional columns for stratification (e.g., institution)
            
        Returns
        -------
        train_df : pd.DataFrame
            Training set (patient-level split)
        test_df : pd.DataFrame
            Test set (patient-level split)
        """
        np.random.seed(self.random_seed)
        
        # Validate required columns
        if patient_id_col not in patient_df.columns:
            raise ValueError(f"Patient ID column '{patient_id_col}' not found in DataFrame")
        if outcome_col not in patient_df.columns:
            raise ValueError(f"Outcome column '{outcome_col}' not found in DataFrame")
        
        # Get unique patients
        unique_patients = patient_df[patient_id_col].unique()
        
        # Create stratification key
        if stratify_cols:
            # Group by patient to get per-patient stratification values
            patient_strata = patient_df.groupby(patient_id_col)[stratify_cols + [outcome_col]].first()
            
            # Create composite stratification key
            strata_values = []
            for _, row in patient_strata.iterrows():
                key_parts = [str(row[col]) for col in stratify_cols + [outcome_col]]
                strata_values.append('_'.join(key_parts))
            
            strata_key = pd.Series(strata_values, index=patient_strata.index)
        else:
            # Stratify by outcome only
            patient_outcomes = patient_df.groupby(patient_id_col)[outcome_col].max()
            strata_key = patient_outcomes
        
        # Check if stratification is possible
        can_stratify = len(strata_key.unique()) > 1
        
        # Stratified split
        if can_stratify:
            try:
                train_patients, test_patients = train_test_split(
                    unique_patients,
                    test_size=self.test_size,
                    random_state=self.random_seed,
                    stratify=strata_key[unique_patients]
                )
            except ValueError as e:
                # If stratification fails (e.g., too few samples per stratum), use random split
                warnings.warn(f"Stratified split failed: {e}. Using random split instead.")
                train_patients, test_patients = train_test_split(
                    unique_patients,
                    test_size=self.test_size,
                    random_state=self.random_seed,
                    stratify=None
                )
        else:
            # Random split if stratification not possible
            train_patients, test_patients = train_test_split(
                unique_patients,
                test_size=self.test_size,
                random_state=self.random_seed,
                stratify=None
            )
        
        # Create train and test DataFrames
        train_df = patient_df[patient_df[patient_id_col].isin(train_patients)].copy()
        test_df = patient_df[patient_df[patient_id_col].isin(test_patients)].copy()
        
        # Store for validation
        self._fitted = True
        self._train_patients = set(train_patients)
        self._test_patients = set(test_patients)
        
        return train_df, test_df
    
    def validate_no_leakage(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                           patient_id_col: str = 'PrimaryPatientID') -> bool:
        """
        Verify no patient overlap between train and test
        
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
            True if no leakage detected
            
        Raises
        ------
        ValueError
            If data leakage is detected
        """
        if not self._fitted:
            raise ValueError("Splitter must be fitted (call create_splits first)")
        
        train_patients = set(train_df[patient_id_col].unique())
        test_patients = set(test_df[patient_id_col].unique())
        overlap = train_patients & test_patients
        
        if overlap:
            raise ValueError(
                f"DATA LEAKAGE DETECTED: {len(overlap)} patients in both sets: {list(overlap)[:10]}"
            )
        return True
    
    def get_train_patients(self) -> set:
        """Get set of training patient IDs"""
        if not self._fitted:
            raise ValueError("Splitter must be fitted (call create_splits first)")
        return self._train_patients.copy()
    
    def get_test_patients(self) -> set:
        """Get set of test patient IDs"""
        if not self._fitted:
            raise ValueError("Splitter must be fitted (call create_splits first)")
        return self._test_patients.copy()
