#!/usr/bin/env python3
"""
shap_code7.py - True-Model SHAP Analysis (Clinical Grade)
==========================================================

Explains the exact ML models trained in Step-3 using saved models.
Ensures SHAP explains the same models that produced the reported AUC.

Software: py_ntcpx v1.0
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
        scaler_path = self.model_dir / f"{organ}_scaler.pkl"
        registry_path = self.model_dir / f"{organ}_feature_registry.json"
        matrix_path = self.model_dir / f"{organ}_feature_matrix.csv"
        
        # Fallback to default files
        if not ann_path.exists():
            ann_path = self.model_dir / "ANN_model.pkl"
        if not xgb_path.exists():
            xgb_path = self.model_dir / "XGBoost_model.pkl"
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
        
        # Process each model
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
                
                elif model_name == "XGBoost":
                    # XGBoost can use TreeExplainer (faster and exact)
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X)
                    
                    # Handle multi-class output
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                    shap_values = np.array(shap_values)
                    
                    X_plot = X
                
                else:
                    print(f"    [WARNING] Unknown model type: {model_name}, skipping...")
                    continue
                
                # Ensure shap_values is 2D
                if len(shap_values.shape) > 2:
                    shap_values = shap_values.reshape(shap_values.shape[0], -1)
                
                print(f"    Computed SHAP values: shape {shap_values.shape}")
                
                # === SHAP API FIX: wrap into Explanation object ===
                explanation = shap.Explanation(
                    values=shap_values,
                    data=X_plot.values,
                    feature_names=X_plot.columns.tolist()
                )
                
                # === Beeswarm plot ===
                plt.figure(figsize=(10, 8))
                shap.plots.beeswarm(explanation, show=False)
                plt.title(f"SHAP Beeswarm — {organ} [{model_name}]", fontsize=14, fontweight='bold')
                plt.tight_layout()
                plt.savefig(organ_outdir / "shap_beeswarm.png", dpi=600, bbox_inches='tight')
                plt.close()
                print(f"    Saved beeswarm plot")
                
                # === Global importance bar plot (clinical standard) ===
                plt.figure(figsize=(8, 6))
                shap.plots.bar(explanation, show=False)
                plt.title(f"Global Feature Importance — {organ} [{model_name}]", fontsize=14, fontweight='bold')
                plt.tight_layout()
                plt.savefig(organ_outdir / "shap_bar.png", dpi=600, bbox_inches='tight')
                plt.close()
                print(f"    Saved bar plot")
                
                # === Save SHAP values table ===
                shap_df = pd.DataFrame(shap_values, columns=feature_names)
                shap_df.to_excel(organ_outdir / "shap_table.xlsx", index=False)
                print(f"    Saved SHAP values table")
                
            except Exception as e:
                print(f"    [ERROR] Failed to process {model_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"\n{'='*60}")
    print("[OK] Step 7: True-Model SHAP completed")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"{'='*60}")


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description="Step 7: True-Model SHAP Analysis (Clinical Grade)",
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


if __name__ == "__main__":
    main()

