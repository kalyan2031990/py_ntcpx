#!/usr/bin/env python3
"""
Tiered NTCP Analysis - Four-Tier Framework
==========================================
Extends py_ntcpx with four-tier NTCP framework:
- Tier 1: Legacy-A (QUANTEC LKB / RS, fixed)
- Tier 2: Legacy-B (MLE-refitted LKB / RS)
- Tier 3: Modern Classical (de Vette multivariable NTCP)
- Tier 4: AI (ANN, XGBoost, SHAP, uNTCP, CCS) - already exists

This script appends new tiers to existing results without modifying them.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
import warnings
warnings.filterwarnings('ignore')

# Windows-safe encoding
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except (AttributeError, ValueError):
    pass

# Import tier modules
try:
    from ntcp_models.legacy_fixed import LegacyFixedNTCP
    from ntcp_models.legacy_mle import LegacyMLENTCP
    from ntcp_models.modern_logistic import ModernLogisticNTCP
except ImportError:
    # Fallback if running from different directory
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from ntcp_models.legacy_fixed import LegacyFixedNTCP
    from ntcp_models.legacy_mle import LegacyMLENTCP
    from ntcp_models.modern_logistic import ModernLogisticNTCP

# Import existing modules
from code3_ntcp_analysis_ml import DVHProcessor, NTCPCalculator
try:
    from ntcp_qa_modules import CohortConsistencyScore
except ImportError:
    print("Warning: CohortConsistencyScore not available")
    CohortConsistencyScore = None

# Import plotting utilities
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d


def load_existing_results(code3_output_dir):
    """
    Load existing NTCP results from code3 output
    
    Args:
        code3_output_dir: Path to code3 output directory
    
    Returns:
        DataFrame with existing results
    """
    code3_path = Path(code3_output_dir)
    
    # Try to find results file
    results_file = None
    for pattern in ['ntcp_results.xlsx', 'enhanced_ntcp_calculations.csv', '*.xlsx']:
        candidates = list(code3_path.glob(pattern))
        if candidates:
            results_file = candidates[0]
            break
    
    if results_file is None:
        raise FileNotFoundError(f"Could not find NTCP results in {code3_output_dir}")
    
    print(f"Loading existing results from: {results_file}")
    
    if results_file.suffix == '.xlsx':
        # Try 'Complete Results' sheet first
        try:
            df = pd.read_excel(results_file, sheet_name='Complete Results')
        except:
            df = pd.read_excel(results_file, sheet_name=0)
    else:
        df = pd.read_csv(results_file)
    
    print(f"Loaded {len(df)} patient-organ combinations")
    return df


def load_dvh_for_patient(dvh_dir, primary_patient_id, organ):
    """Load DVH file for a patient-organ combination"""
    dvh_processor = DVHProcessor(dvh_dir)
    dvh, _ = dvh_processor.load_dvh_file(primary_patient_id, organ, dvh_id=primary_patient_id)
    return dvh


def add_tier2_mle_predictions(results_df, dvh_dir, output_dir):
    """
    Add Tier 2: MLE-refitted LKB & RS predictions
    
    Args:
        results_df: Existing results DataFrame
        dvh_dir: Directory with DVH CSV files
        output_dir: Output directory for saving parameters
    
    Returns:
        DataFrame with added Tier 2 predictions
    """
    print("\n" + "="*60)
    print("TIER 2: MLE-Refitted LKB & RS Models")
    print("="*60)
    
    mle_calculator = LegacyMLENTCP()
    dvh_processor = DVHProcessor(dvh_dir)
    
    # Process each organ separately
    for organ in sorted(results_df['Organ'].unique()):
        organ_data = results_df[results_df['Organ'] == organ].copy()
        
        if len(organ_data) < 10:
            print(f"\n  Skipping {organ}: insufficient data (n={len(organ_data)})")
            continue
        
        print(f"\n  Processing {organ} (n={len(organ_data)})...")
        
        # Prepare data for MLE fitting
        geud_values = organ_data['gEUD'].values
        outcomes = organ_data['Observed_Toxicity'].values.astype(int)
        
        # Prepare dose_metrics for probit model
        dose_metrics_list = []
        for _, row in organ_data.iterrows():
            dose_metrics_list.append({
                'gEUD': row['gEUD'],
                'v_effective': row.get('v_effective', 1.0),
                'max_dose': row.get('max_dose', row['gEUD'])
            })
        
        # Fit MLE models
        print(f"    Fitting LKB Probit MLE...")
        probit_params = mle_calculator.fit_lkb_probit_mle(geud_values, dose_metrics_list, outcomes)
        
        print(f"    Fitting LKB LogLogit MLE...")
        loglogit_params = mle_calculator.fit_lkb_loglogit_mle(geud_values, outcomes)
        
        # Store fitted parameters
        mle_calculator.fitted_params[organ] = {
            'LKB_Probit_MLE': probit_params,
            'LKB_LogLogit_MLE': loglogit_params
        }
        
        # Calculate predictions for all patients in organ
        organ_indices = organ_data.index
        ntcp_lkb_probit_mle = []
        ntcp_lkb_loglogit_mle = []
        
        for idx in organ_indices:
            row = results_df.loc[idx]
            dose_metrics = {
                'gEUD': row['gEUD'],
                'v_effective': row.get('v_effective', 1.0),
                'max_dose': row.get('max_dose', row['gEUD'])
            }
            
            # LKB Probit MLE
            if probit_params and probit_params.get('converged', False):
                ntcp_probit = mle_calculator.calculator.ntcp_lkb_probit(
                    dose_metrics,
                    probit_params['TD50'],
                    probit_params['m'],
                    probit_params['n']
                )
                ntcp_lkb_probit_mle.append(ntcp_probit)
            else:
                ntcp_lkb_probit_mle.append(np.nan)
            
            # LKB LogLogit MLE
            if loglogit_params and loglogit_params.get('converged', False):
                ntcp_loglogit = mle_calculator.calculator.ntcp_lkb_loglogit(
                    row['gEUD'],
                    loglogit_params['TD50'],
                    loglogit_params['gamma50']
                )
                ntcp_lkb_loglogit_mle.append(ntcp_loglogit)
            else:
                ntcp_lkb_loglogit_mle.append(np.nan)
        
        # Add to results DataFrame
        results_df.loc[organ_indices, 'NTCP_LKB_Probit_MLE'] = ntcp_lkb_probit_mle
        results_df.loc[organ_indices, 'NTCP_LKB_LogLogit_MLE'] = ntcp_lkb_loglogit_mle
        
        print(f"    Added MLE predictions for {organ}")
    
    # Save MLE parameters
    mle_calculator.save_parameters(output_dir)
    
    return results_df


def add_tier3_logistic_predictions(results_df, clinical_file, output_dir):
    """
    Add Tier 3: Modern multivariable logistic NTCP
    
    Args:
        results_df: Existing results DataFrame
        clinical_file: Optional path to clinical factors file
        output_dir: Output directory
    
    Returns:
        DataFrame with added Tier 3 predictions
    """
    print("\n" + "="*60)
    print("TIER 3: Modern Multivariable Logistic NTCP")
    print("="*60)
    
    # Load clinical data if available
    clinical_data = None
    if clinical_file and Path(clinical_file).exists():
        try:
            clinical_data = pd.read_excel(clinical_file)
            print(f"  Loaded clinical data from: {clinical_file}")
        except:
            print(f"  Warning: Could not load clinical data, using DVH-only model")
    
    logistic_model = ModernLogisticNTCP()
    
    # Process each organ separately
    for organ in sorted(results_df['Organ'].unique()):
        organ_data = results_df[results_df['Organ'] == organ].copy()
        
        if len(organ_data) < 20:
            print(f"\n  Skipping {organ}: insufficient data for logistic regression (n={len(organ_data)})")
            continue
        
        print(f"\n  Training logistic model for {organ} (n={len(organ_data)})...")
        
        # Train model
        train_results = logistic_model.train_model(organ_data, organ, clinical_data)
        
        if train_results is None:
            print(f"    Training failed for {organ}")
            continue
        
        # Predict NTCP
        predictions = logistic_model.predict_ntcp(organ_data, organ, clinical_data)
        
        # Add to results DataFrame
        organ_indices = organ_data.index
        results_df.loc[organ_indices, 'NTCP_LOGISTIC'] = predictions
        
        print(f"    Added logistic predictions for {organ}")
    
    return results_df


def calculate_ccs_for_tiers(results_df):
    """
    Calculate CCS (Cohort Consistency Score) for all tiers
    
    Args:
        results_df: Results DataFrame
    
    Returns:
        DataFrame with CCS columns for each tier
    """
    print("\n" + "="*60)
    print("Calculating CCS for All Tiers")
    print("="*60)
    
    if CohortConsistencyScore is None:
        print("  Warning: CohortConsistencyScore not available, skipping CCS calculation")
        return results_df
    
    ccs_calculator = CohortConsistencyScore()
    
    # Process each organ separately
    for organ in sorted(results_df['Organ'].unique()):
        organ_data = results_df[results_df['Organ'] == organ].copy()
        
        if len(organ_data) < 10:
            continue
        
        print(f"\n  Calculating CCS for {organ}...")
        
        # Prepare features (use DVH metrics)
        feature_cols = ['mean_dose', 'gEUD', 'V30', 'V50', 'D20', 'max_dose']
        available_features = [f for f in feature_cols if f in organ_data.columns]
        
        if len(available_features) == 0:
            print(f"    Warning: No features available for CCS calculation")
            continue
        
        X_features = organ_data[available_features].fillna(0).values
        
        # Fit CCS on training data
        try:
            ccs_calculator.fit(X_features)
        except Exception as e:
            print(f"    Warning: CCS fitting failed: {e}")
            continue
        
        # Calculate CCS for each patient
        organ_indices = organ_data.index
        ccs_values = []
        
        for idx in organ_indices:
            row = results_df.loc[idx]
            patient_features = np.array([row.get(f, 0) for f in available_features])
            
            try:
                ccs_result = ccs_calculator.calculate_ccs(patient_features)
                ccs_values.append(ccs_result['ccs'])
            except:
                ccs_values.append(np.nan)
        
        # Add CCS columns for different tiers
        results_df.loc[organ_indices, 'CCS_QUANTEC'] = ccs_values
        results_df.loc[organ_indices, 'CCS_MLE'] = ccs_values
        results_df.loc[organ_indices, 'CCS_Logistic'] = ccs_values
        results_df.loc[organ_indices, 'CCS_ML'] = ccs_values  # For ML models
        
        print(f"    Added CCS for {organ}")
    
    return results_df


def plot_dose_response_curves(results_df, output_dir):
    """
    Plot dose-response curves for all tiers using gEUD
    
    Args:
        results_df: Results DataFrame
        output_dir: Output directory
    """
    print("\n" + "="*60)
    print("Generating Dose-Response Curves")
    print("="*60)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process each organ separately
    for organ in sorted(results_df['Organ'].unique()):
        organ_data = results_df[results_df['Organ'] == organ].copy()
        
        if len(organ_data) < 5:
            continue
        
        print(f"  Plotting dose-response for {organ}...")
        
        # Get gEUD range
        geud_min = organ_data['gEUD'].min()
        geud_max = organ_data['gEUD'].max()
        geud_range = np.linspace(geud_min, geud_max, 200)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Tier 1: QUANTEC LKB
        if 'NTCP_LKB_LogLogit' in organ_data.columns:
            ntcp_quantec = []
            for geud in geud_range:
                # Use median parameters from literature
                ntcp_calc = NTCPCalculator()
                if organ in ntcp_calc.literature_params:
                    params = ntcp_calc.literature_params[organ]['LKB_LogLogit']
                    ntcp = ntcp_calc.ntcp_lkb_loglogit(geud, params['TD50'], params['gamma50'])
                    ntcp_quantec.append(ntcp)
                else:
                    ntcp_quantec.append(np.nan)
            
            if not all(np.isnan(ntcp_quantec)):
                ax.plot(geud_range, ntcp_quantec, 'b-', label='QUANTEC LKB', linewidth=2)
        
        # Tier 2: Local-LKB (LOESS)
        if 'NTCP_LKB_LOCAL' in organ_data.columns:
            valid_mask = ~organ_data['NTCP_LKB_LOCAL'].isna()
            if valid_mask.sum() > 5:
                geud_valid = organ_data.loc[valid_mask, 'gEUD'].values
                ntcp_valid = organ_data.loc[valid_mask, 'NTCP_LKB_LOCAL'].values
                
                sort_idx = np.argsort(geud_valid)
                geud_sorted = geud_valid[sort_idx]
                ntcp_sorted = ntcp_valid[sort_idx]
                
                if len(geud_sorted) > 10:
                    ntcp_smooth = gaussian_filter1d(ntcp_sorted, sigma=2.0)
                    ax.plot(geud_sorted, ntcp_smooth, 'g--', label='Local-LKB (LOESS)', linewidth=2)
        
        
        # Tier 2: Local-LKB (cohort-calibrated)
        if 'NTCP_LKB_LOCAL' in organ_data.columns:
            valid_mask = ~organ_data['NTCP_LKB_LOCAL'].isna()
            if valid_mask.sum() > 5:
                geud_valid = organ_data.loc[valid_mask, 'gEUD'].values
                ntcp_valid = organ_data.loc[valid_mask, 'NTCP_LKB_LOCAL'].values
                
                sort_idx = np.argsort(geud_valid)
                geud_sorted = geud_valid[sort_idx]
                ntcp_sorted = ntcp_valid[sort_idx]
                
                if len(geud_sorted) > 10:
                    ntcp_smooth = gaussian_filter1d(ntcp_sorted, sigma=2.0)
                    ax.plot(geud_sorted, ntcp_smooth, 'g--', label='Local-LKB (LOESS)', linewidth=2)
        
        
        # Tier 2: MLE-LKB (use fitted parameters if available)
        # This would require loading MLE parameters - simplified for now
        if 'NTCP_LKB_LogLogit_MLE' in organ_data.columns:
            # Use LOESS smoothing on actual predictions
            valid_mask = ~organ_data['NTCP_LKB_LogLogit_MLE'].isna()
            if valid_mask.sum() > 5:
                geud_valid = organ_data.loc[valid_mask, 'gEUD'].values
                ntcp_valid = organ_data.loc[valid_mask, 'NTCP_LKB_LogLogit_MLE'].values
                
                # Sort by gEUD
                sort_idx = np.argsort(geud_valid)
                geud_sorted = geud_valid[sort_idx]
                ntcp_sorted = ntcp_valid[sort_idx]
                
                # LOESS smoothing
                if len(geud_sorted) > 10:
                    ntcp_smooth = gaussian_filter1d(ntcp_sorted, sigma=2.0)
                    ax.plot(geud_sorted, ntcp_smooth, 'g--', label='MLE-LKB (LOESS)', linewidth=2)
        
        # Tier 3: Logistic (LOESS)
        if 'NTCP_LOGISTIC' in organ_data.columns:
            valid_mask = ~organ_data['NTCP_LOGISTIC'].isna()
            if valid_mask.sum() > 5:
                geud_valid = organ_data.loc[valid_mask, 'gEUD'].values
                ntcp_valid = organ_data.loc[valid_mask, 'NTCP_LOGISTIC'].values
                
                sort_idx = np.argsort(geud_valid)
                geud_sorted = geud_valid[sort_idx]
                ntcp_sorted = ntcp_valid[sort_idx]
                
                if len(geud_sorted) > 10:
                    ntcp_smooth = gaussian_filter1d(ntcp_sorted, sigma=2.0)
                    ax.plot(geud_sorted, ntcp_smooth, 'r:', label='Logistic NTCP (LOESS)', linewidth=2)
        
        # Tier 4: ANN (LOESS)
        if 'ML_ANN' in organ_data.columns or 'NTCP_ML_ANN' in organ_data.columns:
            ann_col = 'ML_ANN' if 'ML_ANN' in organ_data.columns else 'NTCP_ML_ANN'
            valid_mask = ~organ_data[ann_col].isna()
            if valid_mask.sum() > 5:
                geud_valid = organ_data.loc[valid_mask, 'gEUD'].values
                ntcp_valid = organ_data.loc[valid_mask, ann_col].values
                
                sort_idx = np.argsort(geud_valid)
                geud_sorted = geud_valid[sort_idx]
                ntcp_sorted = ntcp_valid[sort_idx]
                
                if len(geud_sorted) > 10:
                    ntcp_smooth = gaussian_filter1d(ntcp_sorted, sigma=2.0)
                    ax.plot(geud_sorted, ntcp_smooth, 'm-.', label='ANN (LOESS)', linewidth=2)
        
        # Tier 4: XGBoost (LOESS)
        if 'ML_XGBoost' in organ_data.columns or 'NTCP_ML_XGBoost' in organ_data.columns:
            xgb_col = 'ML_XGBoost' if 'ML_XGBoost' in organ_data.columns else 'NTCP_ML_XGBoost'
            valid_mask = ~organ_data[xgb_col].isna()
            if valid_mask.sum() > 5:
                geud_valid = organ_data.loc[valid_mask, 'gEUD'].values
                ntcp_valid = organ_data.loc[valid_mask, xgb_col].values
                
                sort_idx = np.argsort(geud_valid)
                geud_sorted = geud_valid[sort_idx]
                ntcp_sorted = ntcp_valid[sort_idx]
                
                if len(geud_sorted) > 10:
                    ntcp_smooth = gaussian_filter1d(ntcp_sorted, sigma=2.0)
                    ax.plot(geud_sorted, ntcp_smooth, 'c-.', label='XGBoost (LOESS)', linewidth=2)
        
        # Tier 4: RandomForest (LOESS)
        if 'ML_RandomForest' in organ_data.columns or 'NTCP_ML_RandomForest' in organ_data.columns:
            rf_col = 'ML_RandomForest' if 'ML_RandomForest' in organ_data.columns else 'NTCP_ML_RandomForest'
            valid_mask = ~organ_data[rf_col].isna()
            if valid_mask.sum() > 5:
                geud_valid = organ_data.loc[valid_mask, 'gEUD'].values
                ntcp_valid = organ_data.loc[valid_mask, rf_col].values
                
                sort_idx = np.argsort(geud_valid)
                geud_sorted = geud_valid[sort_idx]
                ntcp_sorted = ntcp_valid[sort_idx]
                
                if len(geud_sorted) > 10:
                    ntcp_smooth = gaussian_filter1d(ntcp_sorted, sigma=2.0)
                    ax.plot(geud_sorted, ntcp_smooth, color='orange', linestyle='-.', label='Random Forest (LOESS)', linewidth=2)
        
        ax.set_xlabel('gEUD (Gy)', fontsize=12)
        ax.set_ylabel('NTCP', fontsize=12)
        ax.set_title(f'Dose-Response Curves: {organ}', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        plot_file = output_path / f'dose_response_{organ}.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    Saved: {plot_file}")
    
    # Combined plot for all organs
    fig, axes = plt.subplots(1, len(sorted(results_df['Organ'].unique())), 
                            figsize=(6*len(sorted(results_df['Organ'].unique())), 6))
    if len(sorted(results_df['Organ'].unique())) == 1:
        axes = [axes]
    
    for idx, organ in enumerate(sorted(results_df['Organ'].unique())):
        ax = axes[idx]
        organ_data = results_df[results_df['Organ'] == organ].copy()
        
        # Plot all available models
        if 'NTCP_LKB_LogLogit' in organ_data.columns:
            geud_valid = organ_data['gEUD'].values
            ntcp_valid = organ_data['NTCP_LKB_LogLogit'].values
            valid_mask = ~np.isnan(ntcp_valid)
            if valid_mask.sum() > 0:
                ax.scatter(geud_valid[valid_mask], ntcp_valid[valid_mask], 
                          alpha=0.5, label='QUANTEC-LKB', s=20)
        
        if 'NTCP_LKB_LOCAL' in organ_data.columns:
            geud_valid = organ_data['gEUD'].values
            ntcp_valid = organ_data['NTCP_LKB_LOCAL'].values
            valid_mask = ~np.isnan(ntcp_valid)
            if valid_mask.sum() > 0:
                ax.scatter(geud_valid[valid_mask], ntcp_valid[valid_mask], 
                          alpha=0.5, label='Local-LKB', s=20)
        
        
        if 'NTCP_LOGISTIC' in organ_data.columns:
            geud_valid = organ_data['gEUD'].values
            ntcp_valid = organ_data['NTCP_LOGISTIC'].values
            valid_mask = ~np.isnan(ntcp_valid)
            if valid_mask.sum() > 0:
                ax.scatter(geud_valid[valid_mask], ntcp_valid[valid_mask], 
                          alpha=0.5, label='Logistic', s=20)
        
        ax.set_xlabel('gEUD (Gy)', fontsize=10)
        ax.set_ylabel('NTCP', fontsize=10)
        ax.set_title(organ, fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
    
    plt.tight_layout()
    combined_plot = output_path / 'dose_response_tiers.png'
    plt.savefig(combined_plot, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n  Saved combined plot: {combined_plot}")


def create_ml_qa_validation(results_df, output_dir, code3_output_dir=None):
    """
    Create ML QA validation output.
    
    Reports Apparent_AUC (in-sample AUC on full data). For ML models, if code3
    has written ml_cv_metrics.xlsx, merges CV_AUC and Test_AUC and sets
    Overfitting_Gap = Apparent_AUC - CV_AUC (realistic overfitting check).
    
    Args:
        results_df: Results DataFrame
        output_dir: Output directory
        code3_output_dir: Optional path to code3 output (to load ml_cv_metrics.xlsx)
    """
    print("\n" + "="*60)
    print("Creating ML QA Validation")
    print("="*60)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    from sklearn.metrics import roc_auc_score, brier_score_loss
    
    def calibration_metrics(y_true, y_pred, n_bins=10):
        """Compute ECE and MCE (expected/maximum calibration error)."""
        try:
            bins = np.linspace(0, 1, n_bins + 1)
            ece, mce = 0.0, 0.0
            n = len(y_true)
            if n < 10:
                return np.nan, np.nan
            for i in range(n_bins):
                lo, hi = bins[i], bins[i + 1]
                mask = (y_pred >= lo) & (y_pred < hi) if i < n_bins - 1 else (y_pred >= lo) & (y_pred <= hi)
                if mask.sum() == 0:
                    continue
                acc = y_true[mask].mean()
                conf = y_pred[mask].mean()
                ece += np.abs(acc - conf) * mask.sum() / n
                mce = max(mce, np.abs(acc - conf))
            return round(ece, 4), round(mce, 4)
        except Exception:
            return np.nan, np.nan
    
    qa_data = []
    
    # Process each organ
    for organ in sorted(results_df['Organ'].unique()):
        organ_data = results_df[results_df['Organ'] == organ].copy()
        
        if len(organ_data) < 10:
            continue
        
        y_true = organ_data['Observed_Toxicity'].values.astype(int)
        
        # Evaluate each model
        models_to_eval = [
            ('QUANTEC-LKB', 'NTCP_LKB_LogLogit'),
            ('Local-LKB', 'NTCP_LKB_LOCAL'),
            ('MLE_LKB', 'NTCP_LKB_LogLogit_MLE'),
            ('Logistic', 'NTCP_LOGISTIC'),
            ('ANN', 'NTCP_ML_ANN'),
            ('XGBoost', 'NTCP_ML_XGBoost'),
            ('RandomForest', 'NTCP_ML_RandomForest')
        ]
        
        for model_name, col_name in models_to_eval:
            if col_name not in organ_data.columns:
                continue
            
            y_pred = organ_data[col_name].values
            valid_mask = ~np.isnan(y_pred)
            
            if valid_mask.sum() < 5:
                continue
            
            y_true_valid = y_true[valid_mask]
            y_pred_valid = y_pred[valid_mask]
            
            try:
                auc_val = roc_auc_score(y_true_valid, y_pred_valid)
                brier = brier_score_loss(y_true_valid, y_pred_valid)
                ece, mce = calibration_metrics(y_true_valid, y_pred_valid)
                qa_data.append({
                    'Organ': organ,
                    'Model': model_name,
                    'Apparent_AUC': auc_val,
                    'Overfitting_Gap': np.nan,
                    'Brier_Score': brier,
                    'ECE': ece,
                    'MCE': mce,
                    'N_Samples': valid_mask.sum()
                })
            except Exception:
                pass
    
    if qa_data:
        qa_df = pd.DataFrame(qa_data)
        # Merge code3 CV metrics if available (ML models only)
        code3_path = Path(code3_output_dir) if code3_output_dir else None
        cv_file = code3_path / 'ml_cv_metrics.xlsx' if code3_path else None
        if cv_file and cv_file.exists():
            try:
                cv_df = pd.read_excel(cv_file)
                merge_cols = [c for c in ['CV_AUC_mean', 'CV_AUC_std', 'Test_AUC', 'Constant_Predictor'] if c in cv_df.columns]
                if merge_cols:
                    qa_df = qa_df.merge(
                        cv_df[['Organ', 'Model'] + merge_cols],
                        on=['Organ', 'Model'],
                        how='left'
                    )
                    if 'CV_AUC_mean' in qa_df.columns:
                        qa_df['Overfitting_Gap'] = qa_df.apply(
                            lambda r: (r['Apparent_AUC'] - r['CV_AUC_mean']) if pd.notna(r.get('CV_AUC_mean')) else np.nan,
                            axis=1
                        )
                    # Flag high overfitting (gap > 0.1)
                    if 'Overfitting_Gap' in qa_df.columns:
                        qa_df['Overfitting_Flag'] = qa_df.apply(
                            lambda r: 'High' if pd.notna(r.get('Overfitting_Gap')) and r['Overfitting_Gap'] > 0.1 else '',
                            axis=1
                        )
            except Exception as e:
                print(f"  Warning: Could not merge ml_cv_metrics.xlsx: {e}")
        if 'Overfitting_Flag' not in qa_df.columns:
            qa_df['Overfitting_Flag'] = ''
        # Small-sample note for manuscript/report
        n_total = qa_df['N_Samples'].max() if 'N_Samples' in qa_df.columns else 0
        if n_total > 0 and n_total < 100:
            note = f"Small sample (n={int(n_total)}). Prefer CV_AUC over Apparent_AUC for ML. Results exploratory."
        else:
            note = "Prefer CV_AUC over Apparent_AUC for ML models."
        qa_file = output_path / 'ml_validation.xlsx'
        
        with pd.ExcelWriter(qa_file, engine='openpyxl') as writer:
            qa_df.to_excel(writer, sheet_name='ML_QA', index=False)
            pd.DataFrame([{'Note': note}]).to_excel(writer, sheet_name='Note', index=False)
        
        print(f"  Saved ML QA validation: {qa_file}")
    else:
        print("  Warning: No ML QA data generated")


def create_master_excel_report(results_df, output_dir):
    """
    Create comprehensive Excel master report with all tiers
    
    Args:
        results_df: Results DataFrame with all tiers
        output_dir: Output directory
    """
    print("\n" + "="*60)
    print("Creating 4-Tier Master Excel Report")
    print("="*60)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    master_file = output_path / 'NTCP_4Tier_Master.xlsx'
    
    with pd.ExcelWriter(master_file, engine='openpyxl') as writer:
        # Sheet 1: Patient NTCPs (all tiers)
        ntcp_cols = ['PrimaryPatientID', 'AnonPatientID', 'Organ', 'Observed_Toxicity']
        ntcp_pred_cols = [col for col in results_df.columns if 'NTCP_' in col or col.startswith('ML_')]
        patient_ntcp_df = results_df[ntcp_cols + ntcp_pred_cols].copy()
        patient_ntcp_df.to_excel(writer, sheet_name='Patient NTCPs', index=False)
        
        # Sheet 2: Model parameters (QUANTEC + MLE + Logistic + Local)
        params_data = []
        
        # Load local classical parameters from JSON if available
        local_params_file = Path(output_dir).parent / 'code3_output' / 'local_classical_parameters.json'
        if not local_params_file.exists():
            local_params_file = output_path / 'local_classical_parameters.json'
        
        local_params = {}
        if local_params_file.exists():
            try:
                import json
                with open(local_params_file, 'r') as f:
                    local_params = json.load(f)
            except:
                pass
        
        # Add QUANTEC parameters (placeholder - would load from code3 output)
        params_data.append({'Organ': 'See code3 output', 'Model': 'QUANTEC_LKB', 'Source': 'Literature'})
        
        # Add local classical parameters
        for organ, params in local_params.items():
            params_data.append({
                'Organ': organ,
                'Model': 'Local-LKB',
                'TD50': params.get('TD50_local', np.nan),
                'm': params.get('m_local', np.nan),
                'n': 1.0,
                'Source': 'Local calibration'
            })
            params_data.append({
                'Organ': organ,
                'Model': 'Local-RS',
                'D50': params.get('D50_rs_local', np.nan),
                'gamma': params.get('gamma_local', np.nan),
                'Source': 'Local calibration'
            })
        
        if params_data:
            params_df = pd.DataFrame(params_data)
            params_df.to_excel(writer, sheet_name='Model Parameters', index=False)
        else:
            # Fallback placeholder
            params_df = pd.DataFrame([{'Organ': 'See code3 output', 'Model': 'QUANTEC_LKB', 'Note': 'See code3 output'}])
            params_df.to_excel(writer, sheet_name='Model Parameters', index=False)
        
        # Sheet 3: AUC, Brier, CCS
        from sklearn.metrics import roc_auc_score, brier_score_loss
        
        metrics_data = []
        for organ in sorted(results_df['Organ'].unique()):
            organ_data = results_df[results_df['Organ'] == organ].copy()
            y_true = organ_data['Observed_Toxicity'].values.astype(int)
            
            for col in ntcp_pred_cols:
                if col not in organ_data.columns:
                    continue
                y_pred = organ_data[col].values
                valid_mask = ~np.isnan(y_pred)
                if valid_mask.sum() < 5:
                    continue
                
                try:
                    auc = roc_auc_score(y_true[valid_mask], y_pred[valid_mask])
                    brier = brier_score_loss(y_true[valid_mask], y_pred[valid_mask])
                    
                    metrics_data.append({
                        'Organ': organ,
                        'Model': col,
                        'AUC': auc,
                        'Brier_Score': brier,
                        'N': valid_mask.sum()
                    })
                except:
                    pass
        
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            metrics_df.to_excel(writer, sheet_name='AUC_Brier_CCS', index=False)
        
        # Sheet 4: uNTCP summary (if available)
        if 'uNTCP' in results_df.columns:
            untcp_cols = ['PrimaryPatientID', 'Organ', 'uNTCP', 'uNTCP_STD', 'uNTCP_CI_L', 'uNTCP_CI_U']
            available_untcp_cols = [col for col in untcp_cols if col in results_df.columns]
            if available_untcp_cols:
                untcp_df = results_df[available_untcp_cols].copy()
                untcp_df.to_excel(writer, sheet_name='uNTCP Summary', index=False)
        
        # Sheet 5: ML QA
        try:
            qa_file = output_path / 'ml_validation.xlsx'
            if qa_file.exists():
                qa_df = pd.read_excel(qa_file)
                qa_df.to_excel(writer, sheet_name='ML QA', index=False)
        except:
            pass
        
        # Sheet 6: Dose-response stats
        dose_response_data = []
        for organ in sorted(results_df['Organ'].unique()):
            organ_data = results_df[results_df['Organ'] == organ].copy()
            dose_response_data.append({
                'Organ': organ,
                'gEUD_Mean': organ_data['gEUD'].mean(),
                'gEUD_Std': organ_data['gEUD'].std(),
                'gEUD_Min': organ_data['gEUD'].min(),
                'gEUD_Max': organ_data['gEUD'].max(),
                'N': len(organ_data)
            })
        
        if dose_response_data:
            dose_response_df = pd.DataFrame(dose_response_data)
            dose_response_df.to_excel(writer, sheet_name='Dose-Response Stats', index=False)

        # Sheet 7: Patient NTCPs (clean, non-redundant view)
        clean_id_cols = ['PrimaryPatientID', 'AnonPatientID', 'Organ', 'Observed_Toxicity']
        clean_id_cols = [c for c in clean_id_cols if c in results_df.columns]

        # Canonical prediction columns grouped by tier / model family
        preferred_pred_cols = [
            # Tier 1 – Legacy-A (literature LKB / RS)
            'NTCP_LKB_LogLogit',
            'NTCP_LKB_Probit',
            'NTCP_RS_Poisson',
            # Tier 2 – Legacy-B (MLE-refitted)
            'NTCP_LKB_Probit_MLE',
            'NTCP_LKB_LogLogit_MLE',
            # Tier 3 – Modern Classical (multivariable logistic)
            'NTCP_LOGISTIC',
            # Tier 4 – AI (ANN, XGBoost, Random Forest)
            'NTCP_ML_ANN',
            'NTCP_ML_XGBoost',
            'NTCP_ML_RandomForest',
            # Probabilistic / Monte Carlo / uNTCP
            'ProbNTCP_Mean',
            'MC_NTCP_Mean',
            'uNTCP',
            # CCS per tier
            'CCS_QUANTEC',
            'CCS_MLE',
            'CCS_Logistic',
            'CCS_ML',
        ]
        clean_pred_cols = [c for c in preferred_pred_cols if c in results_df.columns]

        if clean_id_cols and clean_pred_cols:
            clean_df = results_df[clean_id_cols + clean_pred_cols].copy()
            clean_df.to_excel(writer, sheet_name='Patient NTCPs (clean)', index=False)

    print(f"  Saved master report: {master_file}")


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description='Tiered NTCP Analysis - Four-Tier Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--code3_output',
        type=str,
        required=True,
        help='Path to code3 output directory (contains existing NTCP results)'
    )
    
    parser.add_argument(
        '--dvh_dir',
        type=str,
        required=True,
        help='Directory containing DVH CSV files'
    )
    
    parser.add_argument(
        '--clinical_file',
        type=str,
        default=None,
        help='Optional Excel file with clinical factors'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='output',
        help='Output directory for tiered analysis results'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("Tiered NTCP Analysis - Four-Tier Framework")
    print("="*60)
    print(f"Code3 output: {args.code3_output}")
    print(f"DVH directory: {args.dvh_dir}")
    print(f"Output directory: {args.output_dir}")
    print("="*60)
    
    # Load existing results
    results_df = load_existing_results(args.code3_output)
    
    # Add Tier 2: MLE-refitted models
    results_df = add_tier2_mle_predictions(
        results_df, args.dvh_dir, args.output_dir
    )
    
    # Add Tier 3: Modern logistic regression
    results_df = add_tier3_logistic_predictions(
        results_df, args.clinical_file, args.output_dir
    )
    
    # Calculate CCS for all tiers
    results_df = calculate_ccs_for_tiers(results_df)
    
    # Generate dose-response curves
    plot_dose_response_curves(results_df, args.output_dir)
    
    # Create ML QA validation
    create_ml_qa_validation(results_df, args.output_dir, args.code3_output)
    
    # Create master Excel report
    create_master_excel_report(results_df, args.output_dir)
    
    # Save updated results
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results_file = output_path / 'tiered_ntcp_results.xlsx'
    with pd.ExcelWriter(results_file, engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='Complete Results', index=False)
    
    print(f"\n{'='*60}")
    print("Tiered NTCP Analysis Complete!")
    print(f"{'='*60}")
    print(f"Results saved to: {output_path.absolute()}")
    print(f"  - tiered_ntcp_results.xlsx")
    print(f"  - NTCP_4Tier_Master.xlsx")
    print(f"  - ml_validation.xlsx")
    print(f"  - dose_response_tiers.png")
    print(f"  - model_parameters_mle.json")


if __name__ == '__main__':
    main()

