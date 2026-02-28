#!/usr/bin/env python3
"""
shap_code7.py - True-Model SHAP Analysis (Clinical Grade)
==========================================================

Explains the exact ML models trained in Step-3 using saved models.
Ensures SHAP explains the same models that produced the reported AUC.

Software: py_ntcpx_v1.0.0
"""

import os
import json
import joblib
import argparse
import sys
from pathlib import Path

import shap
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Windows-safe encoding configuration
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except (AttributeError, ValueError):
    pass


class ModelPackage:
    """Load and manage Step-3 model package"""
    
    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        self.models = {}
        self.scalers = {}
        self.feature_registries = {}
        self.feature_matrices = {}
    
    def load_organ_models(self, organ):
        """Load models for a specific organ"""
        organ_prefix = f"{organ}_"
        
        # Try organ-specific files first
        ann_path = self.model_dir / f"{organ}_ANN_model.pkl"
        xgb_path = self.model_dir / f"{organ}_XGBoost_model.pkl"
        rf_path = self.model_dir / f"{organ}_RandomForest_model.pkl"
        scaler_path = self.model_dir / f"{organ}_scaler.pkl"
        registry_path = self.model_dir / f"{organ}_feature_registry.json"
        matrix_path = self.model_dir / f"{organ}_feature_matrix.csv"
        
        # Fallback to default files
        if not ann_path.exists():
            ann_path = self.model_dir / "ANN_model.pkl"
        if not xgb_path.exists():
            xgb_path = self.model_dir / "XGBoost_model.pkl"
        if not rf_path.exists():
            rf_path = self.model_dir / "RandomForest_model.pkl"
        if not scaler_path.exists():
            scaler_path = self.model_dir / "scaler.pkl"
        if not registry_path.exists():
            registry_path = self.model_dir / "feature_registry.json"
        if not matrix_path.exists():
            matrix_path = self.model_dir / "feature_matrix.csv"
        
        models = {}
        scaler = None
        registry = None
        matrix = None
        
        if ann_path.exists():
            try:
                models['ANN'] = joblib.load(ann_path)
            except Exception as e:
                print(f"  [WARNING] Failed to load ANN model: {e}")
        
        if xgb_path.exists():
            try:
                models['XGBoost'] = joblib.load(xgb_path)
            except Exception as e:
                print(f"  [WARNING] Failed to load XGBoost model: {e}")
        
        if rf_path.exists():
            try:
                models['RandomForest'] = joblib.load(rf_path)
            except Exception as e:
                print(f"  [WARNING] Failed to load RandomForest model: {e}")
        
        if scaler_path.exists():
            try:
                scaler = joblib.load(scaler_path)
            except Exception as e:
                print(f"  [WARNING] Failed to load scaler: {e}")
        
        if registry_path.exists():
            try:
                with open(registry_path, 'r') as f:
                    registry = json.load(f)
            except Exception as e:
                print(f"  [WARNING] Failed to load feature registry: {e}")
        
        if matrix_path.exists():
            try:
                matrix = pd.read_csv(matrix_path)
            except Exception as e:
                print(f"  [WARNING] Failed to load feature matrix: {e}")
        
        return models, scaler, registry, matrix
    
    def load_training_matrix(self, organ):
        """Load training feature matrix for an organ (identity-safe)"""
        models, scaler, registry, matrix = self.load_organ_models(organ)
        
        if matrix is None:
            return None, None, None
        
        # Get feature names from registry or matrix columns
        if registry and 'all_features' in registry:
            feature_names = registry['all_features']
        else:
            feature_names = [c for c in matrix.columns if c not in ['PrimaryPatientID', 'AnonPatientID', 'Organ', 'Observed_Toxicity']]
        
        # Identity safety: ensure no identifiers in features
        assert "PrimaryPatientID" not in feature_names, "PrimaryPatientID found in feature names"
        assert "AnonPatientID" not in feature_names, "AnonPatientID found in feature names"
        assert "Organ" not in feature_names, "Organ found in feature names"
        
        # Extract features only
        X = matrix[feature_names].copy()
        
        # Extract target if available
        y = None
        if 'Observed_Toxicity' in matrix.columns:
            y = matrix['Observed_Toxicity'].copy()
        
        return X, y, feature_names


def run_shap_analysis(code3_dir, output_dir):
    """Run SHAP analysis for all organs and models"""
    
    model_dir = Path(code3_dir) / "models"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    print("=" * 60)
    print("Step 7: True-Model SHAP Analysis (Clinical Grade)")
    print("=" * 60)
    print(f"Model directory: {model_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    # Initialize model package loader
    package = ModelPackage(model_dir)
    
    # Find all organs from model files
    organ_files = list(model_dir.glob("*_ANN_model.pkl"))
    organs = set()
    
    for f in organ_files:
        # Extract organ name (remove _ANN_model.pkl suffix)
        organ = f.stem.replace("_ANN_model", "")
        if organ and organ != "ANN_model":
            organs.add(organ)
    
    # If no organ-specific files, try default files
    if len(organs) == 0:
        default_ann = model_dir / "ANN_model.pkl"
        if default_ann.exists():
            # Try to infer organ from feature matrix or use "Default"
            default_matrix = model_dir / "feature_matrix.csv"
            if default_matrix.exists():
                try:
                    df = pd.read_csv(default_matrix)
                    if 'Organ' in df.columns:
                        organs = set(df['Organ'].unique())
                    else:
                        organs = {"Default"}
                except:
                    organs = {"Default"}
            else:
                organs = {"Default"}
    
    if len(organs) == 0:
        raise ValueError("No models found in model directory")
    
    print(f"\nFound {len(organs)} organ(s): {', '.join(sorted(organs))}")
    
    all_lime_summary_rows = []
    # Process each organ
    for organ in sorted(organs):
        print(f"\n{'='*60}")
        print(f"Processing {organ}...")
        print(f"{'='*60}")
        
        # Load models and data
        models, scaler, registry, matrix = package.load_organ_models(organ)
        
        if not models:
            print(f"  [WARNING] No models found for {organ}, skipping...")
            continue
        
        # Load training matrix (identity-safe)
        X, y, feature_names = package.load_training_matrix(organ)
        
        if X is None or len(X) == 0:
            print(f"  [WARNING] No feature matrix found for {organ}, skipping...")
            continue
        
        print(f"  Loaded {len(X)} samples with {len(feature_names)} features")
        
        # Store bootstrap results for comparison between models
        previous_bootstrap_results = None
        
        # Process each model (ANN, XGBoost, RandomForest, etc.)
        for model_name, model in models.items():
            print(f"\n  Processing {model_name}...")
            
            # Create output directory
            organ_outdir = output_dir / organ / model_name
            organ_outdir.mkdir(parents=True, exist_ok=True)
            
            try:
                # Prepare data for SHAP
                if model_name == "ANN":
                    # ANN requires scaled input
                    if scaler is None:
                        print(f"    [WARNING] No scaler found for ANN, skipping...")
                        continue
                    
                    X_scaled = scaler.transform(X)
                    
                    # Create prediction function
                    def ann_predict_proba(X_scaled_input):
                        return model.predict_proba(X_scaled_input)[:, 1]
                    
                    # Use KernelExplainer for ANN
                    background = X_scaled[:min(30, len(X_scaled))]
                    explainer = shap.KernelExplainer(ann_predict_proba, background)
                    shap_values = explainer.shap_values(X_scaled)
                    
                    # Convert to numpy array if needed
                    if isinstance(shap_values, list):
                        shap_values = shap_values[0] if len(shap_values) > 0 else shap_values
                    shap_values = np.array(shap_values)
                    
                    # Use original X for plotting (not scaled)
                    X_plot = X
                
                elif model_name in ["XGBoost", "RandomForest"]:
                    # Tree-based models: use TreeExplainer as in v3.0.0 layout,
                    # with a small fix for XGBoost base_score serialization.
                    if model_name == "XGBoost" and hasattr(model, 'base_score') and isinstance(model.base_score, str):
                        try:
                            model.base_score = float(model.base_score.strip('[]'))
                        except (ValueError, AttributeError):
                            model.base_score = 0.5

                    # Use model's expected features so SHAP and X match (fixes RF shape mismatch)
                    model_feature_names = None
                    if hasattr(model, 'feature_names_in_'):
                        model_feature_names = list(model.feature_names_in_)
                    elif model_name == "XGBoost" and hasattr(model, 'get_booster'):
                        try:
                            model_feature_names = model.get_booster().feature_names
                        except Exception:
                            model_feature_names = None
                    if model_feature_names is not None and len(model_feature_names) > 0:
                        # Subset X to columns the model was trained on (preserve order)
                        in_both = [c for c in model_feature_names if c in X.columns]
                        if len(in_both) == len(model_feature_names):
                            X = X[model_feature_names].copy()
                        elif len(in_both) > 0:
                            X = X[in_both].copy()
                            feature_names = in_both

                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X)

                    # Handle list output (e.g. multi-class)
                    if isinstance(shap_values, list):
                        # For binary classification keep the positive class if present
                        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

                    shap_values = np.array(shap_values)
                    if len(shap_values.shape) > 2:
                        shap_values = shap_values.reshape(shap_values.shape[0], -1)

                    X_plot = X
                
                else:
                    print(f"    [WARNING] Unknown model type: {model_name}, skipping...")
                    continue
                
                # Ensure shap_values is 2D
                if len(shap_values.shape) > 2:
                    shap_values = shap_values.reshape(shap_values.shape[0], -1)
                
                # Align SHAP and X to same number of features (fixes DimensionError for RandomForest)
                n_x = X_plot.shape[1] if hasattr(X_plot, 'shape') else len(feature_names)
                n_shap = shap_values.shape[1]
                if n_shap != n_x:
                    n_use = min(n_shap, n_x)
                    shap_values = shap_values[:, :n_use]
                    if hasattr(X_plot, 'iloc'):
                        X_plot = X_plot.iloc[:, :n_use].copy()
                    elif hasattr(X_plot, 'shape'):
                        X_plot = X_plot[:, :n_use]
                    feature_names = feature_names[:n_use]
                    print(f"    Aligned SHAP/X to {n_use} features (was shap={n_shap}, X={n_x})")
                
                print(f"    Computed SHAP values: shape {shap_values.shape}")
                
                # Register for LIME early so RandomForest gets LIME even if plots fail later
                if 'lime_models' not in locals():
                    lime_models = {}
                    lime_data = {}
                lime_models[model_name] = model
                lime_data[model_name] = {
                    'X_train': X_plot,
                    'X_scaled': X_scaled if model_name == "ANN" else X_plot,
                    'scaler': scaler if model_name == "ANN" else None,
                    'feature_names': feature_names
                }
                
                # v3.0.0: Improved bootstrap SHAP with stability warnings for ANN
                feature_stability = []
                if len(X_plot) < 100:
                    print(f"    Computing bootstrap SHAP for stability assessment (n={len(X_plot)})...")
                    try:
                        # Check if we have CCS information for low consistency warning
                        low_consistency = False
                        try:
                            from ntcp_qa_modules import CohortConsistencyScore
                            ccs_calc = CohortConsistencyScore(n_samples=len(X_plot))
                            ccs_calc.fit(X_plot.values if isinstance(X_plot, pd.DataFrame) else X_plot)
                            # If threshold is very low (0.0 or 0.1), cohort consistency is low
                            if ccs_calc.ccs_threshold < 0.2:
                                low_consistency = True
                        except:
                            pass  # CCS check is optional
                        
                        bootstrap_results = calculate_bootstrap_shap(
                            model, X_plot, model_type=model_name.lower(), n_bootstrap=min(100, len(X_plot) * 2)
                        )
                        
                        # Save bootstrap stability report
                        stability_df = pd.DataFrame(bootstrap_results['feature_stability'])
                        stability_df.to_excel(organ_outdir / "shap_stability_report.xlsx", index=False)
                        print(f"    Saved bootstrap stability report")
                        
                        feature_stability = bootstrap_results.get('feature_stability', [])
                        
                        # Flag inconsistent features if comparing with another model
                        if 'previous_bootstrap_results' in locals() and previous_bootstrap_results is not None:
                            try:
                                inconsistent = flag_inconsistent_importance(
                                    previous_bootstrap_results, bootstrap_results, threshold=0.7
                                )
                                if inconsistent:
                                    inconsistent_df = pd.DataFrame(inconsistent)
                                    inconsistent_df.to_excel(organ_outdir / "inconsistent_features.xlsx", index=False)
                                    print(f"    Found {len(inconsistent)} inconsistent features between models")
                            except (TypeError, KeyError) as e:
                                # Skip comparison if results structure is incompatible
                                pass
                        
                        # Store for comparison
                        previous_bootstrap_results = bootstrap_results
                    except Exception as e:
                        # v3.0.0: Clear warning for ANN in low-consistency scenarios
                        if model_name == "ANN":
                            print(f"    WARNING - SHAP stability analysis skipped for ANN due to low cohort consistency. Global interpretations may be unstable.")
                        else:
                            print(f"    Warning: Bootstrap SHAP failed: {e}")
                        import traceback
                        traceback.print_exc()
                        feature_stability = []
                
                # === Save SHAP values table first (so RandomForest gets it even if plots fail) ===
                shap_df = pd.DataFrame(shap_values, columns=feature_names)
                shap_df.to_excel(organ_outdir / "shap_table.xlsx", index=False)
                print(f"    Saved SHAP values table")
                
                # === SHAP API: wrap into Explanation object and save plots (try/except so one failure doesn't skip LIME) ===
                try:
                    explanation = shap.Explanation(
                        values=shap_values,
                        data=X_plot.values if hasattr(X_plot, 'values') else X_plot,
                        feature_names=feature_names
                    )
                    
                    # Beeswarm plot
                    plt.figure(figsize=(10, 8))
                    shap.plots.beeswarm(explanation, show=False)
                    plt.title(f"SHAP Beeswarm — {organ} [{model_name}]", fontsize=14, fontweight='bold')
                    plt.tight_layout()
                    plt.savefig(organ_outdir / "shap_beeswarm.png", dpi=600, bbox_inches='tight')
                    plt.close()
                    print(f"    Saved beeswarm plot")
                    
                    # Global importance bar plot
                    plt.figure(figsize=(8, 6))
                    shap.plots.bar(explanation, show=False)
                    plt.title(f"Global Feature Importance — {organ} [{model_name}]", fontsize=14, fontweight='bold')
                    plt.tight_layout()
                    plt.savefig(organ_outdir / "shap_bar.png", dpi=600, bbox_inches='tight')
                    plt.close()
                    print(f"    Saved bar plot")
                except Exception as plot_err:
                    print(f"    Warning: SHAP plots failed ({plot_err}); shap_table.xlsx was saved.")
                    import traceback
                    traceback.print_exc()
                
                # (lime_models/lime_data already registered above)
            
            except Exception as e:
                print(f"    [ERROR] Failed to process {model_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # v3.0.0: Generate LIME explanations for representative patients and save patient-level summary table
        if 'lime_models' in locals() and len(lime_models) > 0:
            print(f"\n  Generating LIME explanations for representative patients...")
            organ_lime_rows = []
            try:
                X_all = lime_data[list(lime_models.keys())[0]]['X_train']
                y_pred_all = {}
                for model_name, model in lime_models.items():
                    data = lime_data[model_name]
                    if model_name == "ANN":
                        y_pred = model.predict_proba(data['X_scaled'])[:, 1]
                    else:
                        y_pred = model.predict_proba(X_all.values if isinstance(X_all, pd.DataFrame) else X_all)[:, 1]
                    y_pred_all[model_name] = y_pred
                
                if len(y_pred_all) > 0:
                    first_model_preds = list(y_pred_all.values())[0]
                    highest_idx = int(np.argmax(first_model_preds))
                    median_idx = int(np.argsort(first_model_preds)[len(first_model_preds) // 2])
                    lowest_idx = int(np.argmin(first_model_preds))
                    representative_indices = [highest_idx, median_idx, lowest_idx]
                
                    for model_name, model in lime_models.items():
                        data = lime_data[model_name]
                        preds = y_pred_all[model_name]
                        pred_list = [float(preds[j]) for j in representative_indices]
                        rows = generate_lime_explanations(
                            model=model,
                            X_train=data['X_train'],
                            X_test=data['X_train'].iloc[representative_indices] if isinstance(data['X_train'], pd.DataFrame) else data['X_train'].iloc[representative_indices],
                            feature_names=data['feature_names'],
                            output_dir=output_dir / organ / model_name,
                            patient_indices=representative_indices,
                            scaler=data['scaler'],
                            model_type=model_name,
                            organ=organ,
                            model_name=model_name,
                            pred_proba_list=pred_list,
                        )
                        organ_lime_rows.extend(rows)
                        if rows:
                            pd.DataFrame(rows).to_excel(output_dir / organ / model_name / "LIME_patient_summary.xlsx", index=False)
                            print(f"    Saved LIME_patient_summary.xlsx for {organ} / {model_name}")
                
                    print(f"    Generated LIME explanations for {len(representative_indices)} representative patients")
            except Exception as e:
                print(f"    Warning: LIME generation failed: {e}")
                import traceback
                traceback.print_exc()
            
            # Accumulate into global summary across all organs/models
            if organ_lime_rows:
                all_lime_summary_rows.extend(organ_lime_rows)
    
    if all_lime_summary_rows:
        pd.DataFrame(all_lime_summary_rows).to_excel(output_dir / "LIME_summary_all_organs.xlsx", index=False)
        print(f"\n  Saved LIME_summary_all_organs.xlsx ({len(all_lime_summary_rows)} rows) at {output_dir / 'LIME_summary_all_organs.xlsx'}")
    
    print(f"\n{'='*60}")
    print("[OK] Step 7: True-Model SHAP completed")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"{'='*60}")


def generate_lime_explanations(model, X_train, X_test, feature_names, output_dir, patient_indices, scaler=None, model_type='xgboost', organ=None, model_name=None, pred_proba_list=None):
    """
    Generate LIME (Local Interpretable Model-agnostic Explanations) for specific patients.
    Optionally returns a list of summary rows for LIME_patient_summary.xlsx.

    Parameters
    ----------
    organ : str, optional
        Organ name (for summary table).
    model_name : str, optional
        Model name (for summary table).
    pred_proba_list : list of float, optional
        Predicted NTCP (prob) for each patient in patient_indices.

    Returns
    -------
    list of dict
        Summary rows for LIME patient-level table (Organ, Model, Patient_Index, Predicted_NTCP, Top_Positive_Features, Top_Negative_Features, LIME_HTML_Path, LIME_PNG_Path).
    """
    summary_rows = []
    try:
        from lime import lime_tabular
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError:
        print(f"    [WARNING] LIME not available. Install with: pip install lime")
        return summary_rows

    try:
        if isinstance(X_train, pd.DataFrame):
            X_train_array = X_train.values
        else:
            X_train_array = X_train

        if isinstance(X_test, pd.DataFrame):
            X_test_array = X_test.values
        else:
            X_test_array = X_test

        explainer = LimeTabularExplainer(
            X_train_array,
            feature_names=feature_names,
            mode='classification',
            training_labels=None,
            discretize_continuous=True
        )

        def predict_proba_wrapper(X_input):
            if model_type == "ANN" and scaler is not None:
                X_scaled = scaler.transform(X_input)
                return model.predict_proba(X_scaled)
            else:
                return model.predict_proba(X_input)

        for i, (patient_idx, instance) in enumerate(zip(patient_indices, X_test_array)):
            try:
                explanation = explainer.explain_instance(
                    instance,
                    predict_proba_wrapper,
                    num_features=min(10, len(feature_names)),
                    top_labels=1
                )

                html_file = output_dir / f"lime_explanation_{patient_idx}.html"
                explanation.save_to_file(str(html_file))

                png_file = None
                try:
                    fig = explanation.as_pyplot_figure()
                    png_file = output_dir / f"lime_explanation_{patient_idx}.png"
                    fig.savefig(png_file, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                except Exception as e:
                    print(f"      Warning: Could not save PNG for patient {patient_idx}: {e}")

                # Build summary row: top positive/negative features from LIME
                pred_ntcp = pred_proba_list[i] if pred_proba_list is not None and i < len(pred_proba_list) else np.nan
                try:
                    # as_list(label) returns list of (feature_name, weight); positive = risk-increasing
                    lime_list = explanation.as_list(label=1)
                    pos = [f"{f}: {w:.3f}" for f, w in lime_list if w > 0][:5]
                    neg = [f"{f}: {w:.3f}" for f, w in lime_list if w < 0][:5]
                    top_pos = "; ".join(pos) if pos else ""
                    top_neg = "; ".join(neg) if neg else ""
                except Exception:
                    top_pos = ""
                    top_neg = ""

                summary_rows.append({
                    "Organ": organ or "",
                    "Model": model_name or model_type,
                    "Patient_Index": patient_idx,
                    "Predicted_NTCP": pred_ntcp,
                    "Top_Positive_Features_Risk": top_pos,
                    "Top_Negative_Features_Protective": top_neg,
                    "LIME_HTML_Path": str(html_file.name),
                    "LIME_PNG_Path": str(png_file.name) if png_file and png_file.exists() else "",
                })
                print(f"      Generated LIME explanation for patient {patient_idx}")
            except Exception as e:
                print(f"      Warning: LIME explanation failed for patient {patient_idx}: {e}")
                continue

    except Exception as e:
        print(f"    [ERROR] LIME generation failed: {e}")
        import traceback
        traceback.print_exc()
    return summary_rows


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description="Step 7: True-Model SHAP Analysis (Clinical Grade) + LIME (v1.0.0)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--code3_dir",
        type=str,
        required=True,
        help="Path to code3 output directory"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Output directory for SHAP results"
    )
    
    args = parser.parse_args()
    
    run_shap_analysis(args.code3_dir, args.outdir)


def calculate_bootstrap_shap(model, X, model_type='xgboost', n_bootstrap=100):
    """
    Calculate SHAP values with bootstrapping for stability assessment
    
    Parameters
    ----------
    model : trained model
        ML model (ANN, XGBoost, or RandomForest)
    X : pd.DataFrame
        Feature matrix
    model_type : str
        Type of model ('ann' or 'xgboost')
    n_bootstrap : int
        Number of bootstrap samples
        
    Returns
    -------
    dict
        Dictionary with mean SHAP, std SHAP, and feature stability
    """
    shap_values_list = []
    feature_importance_rankings = []
    
    print(f"      Running {n_bootstrap} bootstrap iterations...")
    
    for i in range(n_bootstrap):
        if (i + 1) % 20 == 0:
            print(f"        Bootstrap {i+1}/{n_bootstrap}...")
        
        # Bootstrap sample
        indices = np.random.choice(len(X), size=len(X), replace=True)
        X_boot = X.iloc[indices]
        
        try:
            # Calculate SHAP values
            if model_type.lower() in ('xgboost', 'randomforest'):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_boot)
                # Handle multi-class output
                if isinstance(shap_values, list):
                    shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                # For ANN or other models
                # Use smaller background for speed
                background = X_boot[:min(20, len(X_boot))]
                
                def predict_proba_wrapper(X_input):
                    return model.predict_proba(X_input)[:, 1]
                
                explainer = shap.KernelExplainer(predict_proba_wrapper, background)
                shap_values = explainer.shap_values(X_boot[:min(50, len(X_boot))])  # Limit samples for speed
                
                # Handle list output
                if isinstance(shap_values, list):
                    shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            
            shap_values = np.array(shap_values)
            
            # Ensure 2D
            if len(shap_values.shape) > 2:
                shap_values = shap_values.reshape(shap_values.shape[0], -1)
            
            shap_values_list.append(shap_values)
            
            # Calculate feature importance ranking for this bootstrap
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            ranking = np.argsort(-mean_abs_shap)  # Descending order
            feature_importance_rankings.append(ranking)
            
        except Exception as e:
            # Skip this bootstrap if it fails
            continue
    
    if len(shap_values_list) == 0:
        raise ValueError("All bootstrap iterations failed")
    
    # Calculate consensus
    shap_array = np.array(shap_values_list)
    mean_shap = np.mean(shap_array, axis=0)
    std_shap = np.std(shap_array, axis=0)
    
    # Calculate feature stability
    feature_stability = calculate_feature_stability(feature_importance_rankings, X.columns)
    
    return {
        'mean_shap': mean_shap,
        'std_shap': std_shap,
        'feature_stability': feature_stability,
        'bootstrap_samples': len(shap_values_list)
    }


def calculate_feature_stability(rankings, feature_names):
    """
    Calculate how stable feature rankings are across bootstrap samples
    
    Parameters
    ----------
    rankings : list of np.ndarray
        List of feature ranking arrays (one per bootstrap)
    feature_names : list of str
        Feature names
        
    Returns
    -------
    list of dict
        Stability report for each feature
    """
    n_features = len(feature_names)
    n_bootstrap = len(rankings)
    
    if n_bootstrap == 0:
        return []
    
    # Calculate mean rank for each feature
    mean_ranks = np.zeros(n_features)
    for ranking in rankings:
        for rank, feature_idx in enumerate(ranking):
            if feature_idx < n_features:
                mean_ranks[feature_idx] += rank
    
    mean_ranks = mean_ranks / n_bootstrap
    
    # Calculate rank stability (lower std = more stable)
    rank_stds = np.zeros(n_features)
    for feature_idx in range(n_features):
        ranks = []
        for ranking in rankings:
            if feature_idx < len(ranking):
                rank = np.where(ranking == feature_idx)[0]
                if len(rank) > 0:
                    ranks.append(rank[0])
        if len(ranks) > 0:
            rank_stds[feature_idx] = np.std(ranks)
        else:
            rank_stds[feature_idx] = n_features  # High penalty if never ranked
    
    # Create stability report
    stability_report = []
    for i, feature in enumerate(feature_names):
        stability_report.append({
            'feature': feature,
            'mean_rank': mean_ranks[i],
            'rank_std': rank_stds[i],
            'stability_category': 'High' if rank_stds[i] < 2 else 'Medium' if rank_stds[i] < 5 else 'Low'
        })
    
    return sorted(stability_report, key=lambda x: x['mean_rank'])


def flag_inconsistent_importance(shap_results_ann, shap_results_xgb, threshold=0.7):
    """
    Flag features with inconsistent importance between models
    
    Parameters
    ----------
    shap_results_ann : dict
        SHAP results from ANN model
    shap_results_xgb : dict
        SHAP results from XGBoost model
    threshold : float
        Threshold for flagging inconsistency (default: 0.7)
        
    Returns
    -------
    list of dict
        List of inconsistent features with details
    """
    # v3.0.0: Safety checks for None or missing data
    if shap_results_ann is None or shap_results_xgb is None:
        return []
    
    if 'feature_stability' not in shap_results_ann or 'feature_stability' not in shap_results_xgb:
        return []
    
    if not shap_results_ann['feature_stability'] or not shap_results_xgb['feature_stability']:
        return []
    
    ann_importance = {item['feature']: item['mean_rank'] for item in shap_results_ann['feature_stability']}
    xgb_importance = {item['feature']: item['mean_rank'] for item in shap_results_xgb['feature_stability']}
    
    inconsistent_features = []
    
    all_features = set(ann_importance.keys()) | set(xgb_importance.keys())
    
    for feature in all_features:
        ann_rank = ann_importance.get(feature, 100)  # High rank if not in top features
        xgb_rank = xgb_importance.get(feature, 100)
        
        rank_diff = abs(ann_rank - xgb_rank)
        
        if rank_diff > len(all_features) * threshold:
            inconsistent_features.append({
                'feature': feature,
                'ann_rank': ann_rank,
                'xgb_rank': xgb_rank,
                'rank_difference': rank_diff,
                'severity': 'High' if rank_diff > 10 else 'Medium'
            })
    
    return inconsistent_features


if __name__ == "__main__":
    main()

