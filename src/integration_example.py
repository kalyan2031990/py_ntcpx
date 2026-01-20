"""
Example integration of new components into existing pipeline

This shows how to use PatientDataSplitter, OverfitResistantMLModels, etc.
with the existing code3_ntcp_analysis_ml.py structure
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Import new components
from src.validation.data_splitter import PatientDataSplitter
from src.models.machine_learning.ml_models import OverfitResistantMLModels
from src.features.feature_selector import RadiobiologyGuidedFeatureSelector
from src.metrics.auc_calculator import calculate_auc_with_ci
from src.reporting.leakage_detector import DataLeakageDetector


def integrate_patient_level_split(patient_df: pd.DataFrame, 
                                  feature_cols: list,
                                  outcome_col: str = 'Observed_Toxicity',
                                  organ: str = 'Parotid',
                                  random_seed: int = 42):
    """
    Example: Integrate patient-level splitting into existing pipeline
    
    This replaces the current train_test_split call in code3 with
    patient-level splitting to prevent data leakage.
    
    Parameters
    ----------
    patient_df : pd.DataFrame
        DataFrame with patient data (may have multiple rows per patient)
    feature_cols : list
        List of feature column names
    outcome_col : str
        Outcome column name
    organ : str
        Organ name for feature selection
    random_seed : int
        Random seed for reproducibility
        
    Returns
    -------
    dict with keys:
        - 'X_train': Training features
        - 'X_test': Test features
        - 'y_train': Training outcomes
        - 'y_test': Test outcomes
        - 'train_df': Training DataFrame
        - 'test_df': Test DataFrame
        - 'leakage_report': Leakage detection report
    """
    # Create patient-level splitter
    splitter = PatientDataSplitter(random_seed=random_seed, test_size=0.2)
    
    # Split at patient level
    train_df, test_df = splitter.create_splits(
        patient_df,
        patient_id_col='PrimaryPatientID',
        outcome_col=outcome_col
    )
    
    # Check for leakage
    leakage_detector = DataLeakageDetector()
    leakage_detector.check_patient_overlap(train_df, test_df, 'PrimaryPatientID')
    leakage_report = leakage_detector.generate_report()
    
    if not leakage_report['passed']:
        print("WARNING: Data leakage detected!")
        print(leakage_report['summary'])
    
    # Extract features and outcomes (AFTER split)
    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values
    y_train = train_df[outcome_col].values
    y_test = test_df[outcome_col].values
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'train_df': train_df,
        'test_df': test_df,
        'leakage_report': leakage_report
    }


def integrate_feature_selection(X_train: np.ndarray,
                                y_train: np.ndarray,
                                feature_names: list,
                                organ: str = 'Parotid'):
    """
    Example: Integrate domain-guided feature selection
    
    This selects features based on EPV rule and domain knowledge
    to prevent overfitting.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training outcomes
    feature_names : list
        List of feature names
    organ : str
        Organ name for domain knowledge
        
    Returns
    -------
    dict with keys:
        - 'X_train_selected': Selected training features
        - 'selected_features': List of selected feature names
        - 'feature_selector': Feature selector object
    """
    # Create feature selector
    selector = RadiobiologyGuidedFeatureSelector()
    
    # Convert to DataFrame for selection
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    
    # Select features
    selected_features = selector.select_features(
        X_train_df,
        y_train,
        organ=organ
    )
    
    # Get selected feature indices
    selected_indices = [i for i, name in enumerate(feature_names) if name in selected_features]
    
    # Extract selected features
    X_train_selected = X_train[:, selected_indices]
    
    print(f"Selected {len(selected_features)} features: {selected_features}")
    
    return {
        'X_train_selected': X_train_selected,
        'selected_features': selected_features,
        'selected_indices': selected_indices,
        'feature_selector': selector
    }


def integrate_conservative_ml(X_train: np.ndarray,
                              y_train: np.ndarray,
                              X_test: np.ndarray,
                              y_test: np.ndarray,
                              model_type: str = 'ann',
                              random_seed: int = 42):
    """
    Example: Integrate conservative ML models with nested CV
    
    This uses OverfitResistantMLModels with proper EPV validation
    and nested cross-validation.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training outcomes
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test outcomes
    model_type : str
        'ann' or 'xgboost'
    random_seed : int
        Random seed for reproducibility
        
    Returns
    -------
    dict with keys:
        - 'model': Trained model
        - 'test_auc': Test AUC with CI
        - 'nested_cv_results': Nested CV results
        - 'epv': Events per variable
    """
    # Calculate EPV
    n_features = X_train.shape[1]
    n_samples = len(y_train)
    n_events = int(np.sum(y_train))
    
    # Create ML model with EPV awareness
    ml_model = OverfitResistantMLModels(
        n_features=n_features,
        n_samples=n_samples,
        n_events=n_events,
        random_seed=random_seed
    )
    
    print(f"EPV: {ml_model.epv:.2f} events per variable")
    
    # Train model
    if model_type.lower() == 'ann':
        model = ml_model.create_ann_model()
    elif model_type.lower() == 'xgboost':
        model = ml_model.create_xgboost_model()
        if model is None:
            raise ValueError("XGBoost not available")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Fit on training data
    model.fit(X_train, y_train)
    
    # Predict on test data
    y_pred_test = model.predict_proba(X_test)[:, 1]
    
    # Calculate AUC with CI
    test_auc, test_auc_ci = calculate_auc_with_ci(y_test, y_pred_test)
    
    # Nested CV on training data
    nested_cv_results = ml_model.train_with_nested_cv(X_train, y_train, model_type=model_type)
    
    print(f"Test AUC: {test_auc:.3f} (95% CI: {test_auc_ci[0]:.3f}-{test_auc_ci[1]:.3f})")
    print(f"Nested CV AUC: {nested_cv_results['nested_cv_auc_mean']:.3f} ± {nested_cv_results['nested_cv_auc_std']:.3f}")
    
    return {
        'model': model,
        'test_auc': test_auc,
        'test_auc_ci': test_auc_ci,
        'nested_cv_results': nested_cv_results,
        'epv': ml_model.epv,
        'sample_size_adequate': ml_model.epv >= 10
    }


# Example usage
if __name__ == '__main__':
    print("This is an integration example showing how to use the new components.")
    print("See the implementation documentation for full integration guide.")
