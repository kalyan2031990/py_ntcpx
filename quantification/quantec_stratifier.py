#!/usr/bin/env python3
"""
quantec_stratifier.py - QUANTEC Stratification Engine
======================================================

Assigns each patient-organ to a QUANTEC dose-risk bin and computes per-bin:
- Observed toxicity rate
- Mean predicted NTCP
- Absolute error

Generates quantec_validation_[Organ].xlsx for each organ.

Software: py_ntcpx_v1.0.0 - Clinical Governance Upgrade
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np

# Windows-safe encoding configuration
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except (AttributeError, ValueError):
    pass

# Import Windows-safe utilities
try:
    from windows_safe_utils import safe_print, safe_encode_unicode
except ImportError:
    def safe_print(*args, **kwargs):
        print(*args, **kwargs)
    def safe_encode_unicode(text):
        return str(text)


class QUANTECStratifier:
    """QUANTEC dose-risk binning and validation"""
    
    def __init__(self, bins_config_path: Path, output_dir: Path):
        """
        Initialize QUANTEC stratifier
        
        Args:
            bins_config_path: Path to quantec_bins.json
            output_dir: Output directory for validation tables
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load bin definitions
        with open(bins_config_path, 'r', encoding='utf-8') as f:
            self.bins_config = json.load(f)
    
    def assign_to_bin(self, row: pd.Series, organ: str) -> Optional[str]:
        """
        Assign patient-organ to QUANTEC bin based on dose metrics
        
        Args:
            row: DataFrame row with dose metrics
            organ: Organ name
        
        Returns:
            Bin label or None if no matching bin
        """
        if organ not in self.bins_config:
            return None
        
        bins = self.bins_config[organ]
        
        # Evaluate each bin condition
        for bin_def in bins:
            condition = bin_def['condition']
            try:
                # Evaluate condition using row data as context
                # Replace metric names with row values
                condition_eval = condition
                for col in row.index:
                    if col in condition_eval:
                        value = row[col]
                        if pd.isna(value):
                            continue
                        # Replace column name with value in condition
                        condition_eval = condition_eval.replace(col, str(float(value)))
                
                # Evaluate condition
                # Safe eval: only allow comparisons and arithmetic
                # Check for unsafe patterns
                unsafe_patterns = ['__', 'import', 'exec', 'eval', 'open', 'file']
                if any(pattern in condition_eval for pattern in unsafe_patterns):
                    continue
                
                result = eval(condition_eval, {"__builtins__": {}}, {
                    'mean_dose': float(row.get('mean_dose', 0)) if not pd.isna(row.get('mean_dose')) else np.nan,
                    'max_dose': float(row.get('max_dose', 0)) if not pd.isna(row.get('max_dose')) else np.nan,
                    'min_dose': float(row.get('min_dose', 0)) if not pd.isna(row.get('min_dose')) else np.nan,
                    'gEUD': float(row.get('gEUD', 0)) if not pd.isna(row.get('gEUD')) else np.nan,
                })
                
                if result:
                    return bin_def['label']
            except Exception as e:
                # Skip this bin if evaluation fails
                continue
        
        return None
    
    def stratify_organ(self, organ_data: pd.DataFrame, organ: str) -> pd.DataFrame:
        """
        Stratify organ data by QUANTEC bins
        
        Args:
            organ_data: DataFrame with organ-specific data
            organ: Organ name
        
        Returns:
            DataFrame with bin assignments
        """
        if organ not in self.bins_config:
            safe_print(f"Warning: No QUANTEC bins defined for {organ}")
            return organ_data
        
        # Assign bins
        organ_data = organ_data.copy()
        organ_data['QUANTEC_Bin'] = organ_data.apply(
            lambda row: self.assign_to_bin(row, organ), axis=1
        )
        
        return organ_data
    
    def compute_bin_metrics(self, organ_data: pd.DataFrame, organ: str) -> pd.DataFrame:
        """
        Compute per-bin validation metrics
        
        Args:
            organ_data: DataFrame with bin assignments
            organ: Organ name
        
        Returns:
            DataFrame with per-bin metrics
        """
        if 'QUANTEC_Bin' not in organ_data.columns:
            return pd.DataFrame()
        
        bin_metrics = []
        
        # Get all NTCP columns
        ntcp_cols = [col for col in organ_data.columns if col.startswith('NTCP_')]
        
        for bin_label in organ_data['QUANTEC_Bin'].dropna().unique():
            bin_data = organ_data[organ_data['QUANTEC_Bin'] == bin_label]
            
            if len(bin_data) == 0:
                continue
            
            n_patients = len(bin_data)
            
            # Observed toxicity
            if 'Observed_Toxicity' in bin_data.columns:
                observed_rate = bin_data['Observed_Toxicity'].mean()
                n_events = int(bin_data['Observed_Toxicity'].sum())
            else:
                observed_rate = np.nan
                n_events = 0
            
            bin_result = {
                'Organ': organ,
                'QUANTEC_Bin': bin_label,
                'N_Patients': n_patients,
                'N_Events': n_events,
                'Observed_Toxicity_Rate': observed_rate,
                'Observed_Toxicity_Percent': f"{observed_rate * 100:.1f}%" if not pd.isna(observed_rate) else "N/A"
            }
            
            # Mean predicted NTCP for each model
            for ntcp_col in ntcp_cols:
                model_name = ntcp_col.replace('NTCP_', '')
                valid_data = bin_data[ntcp_col].dropna()
                
                if len(valid_data) > 0:
                    mean_ntcp = valid_data.mean()
                    std_ntcp = valid_data.std()
                    
                    bin_result[f'Mean_NTCP_{model_name}'] = mean_ntcp
                    bin_result[f'Std_NTCP_{model_name}'] = std_ntcp
                    
                    # Absolute error (observed - predicted)
                    if not pd.isna(observed_rate):
                        abs_error = abs(observed_rate - mean_ntcp)
                        bin_result[f'AbsError_{model_name}'] = abs_error
                    else:
                        bin_result[f'AbsError_{model_name}'] = np.nan
                else:
                    bin_result[f'Mean_NTCP_{model_name}'] = np.nan
                    bin_result[f'Std_NTCP_{model_name}'] = np.nan
                    bin_result[f'AbsError_{model_name}'] = np.nan
            
            bin_metrics.append(bin_result)
        
        return pd.DataFrame(bin_metrics)
    
    def generate_validation_table(self, ntcp_results_path: Path) -> Dict[str, Path]:
        """
        Generate QUANTEC validation tables for all organs
        
        Args:
            ntcp_results_path: Path to NTCP results Excel file
        
        Returns:
            Dict mapping organ names to output file paths
        """
        # Load NTCP results
        try:
            with pd.ExcelFile(ntcp_results_path) as xl:
                # Try to find the right sheet
                sheet_name = None
                for name in xl.sheet_names:
                    df_test = pd.read_excel(xl, sheet_name=name, nrows=5)
                    if 'Organ' in df_test.columns and 'PrimaryPatientID' in df_test.columns:
                        sheet_name = name
                        break
                
                if sheet_name is None:
                    sheet_name = xl.sheet_names[0]
                
                results_df = pd.read_excel(xl, sheet_name=sheet_name)
        except Exception as e:
            safe_print(f"ERROR: Failed to load NTCP results: {e}")
            return {}
        
        # Validate required columns
        required_cols = ['Organ', 'PrimaryPatientID']
        missing_cols = [col for col in required_cols if col not in results_df.columns]
        if missing_cols:
            safe_print(f"ERROR: Missing required columns: {missing_cols}")
            return {}
        
        # Check for dose metrics
        dose_metric_cols = ['mean_dose', 'MeanDose(Gy)', 'max_dose', 'MaxDose(Gy)']
        has_dose_metrics = any(col in results_df.columns for col in dose_metric_cols)
        
        if not has_dose_metrics:
            safe_print("ERROR: No dose metric columns found (need mean_dose or MeanDose(Gy))")
            return {}
        
        # Normalize column names
        if 'MeanDose(Gy)' in results_df.columns and 'mean_dose' not in results_df.columns:
            results_df['mean_dose'] = results_df['MeanDose(Gy)']
        if 'MaxDose(Gy)' in results_df.columns and 'max_dose' not in results_df.columns:
            results_df['max_dose'] = results_df['MaxDose(Gy)']
        
        output_files = {}
        
        # Process each organ
        for organ in sorted(results_df['Organ'].unique()):
            organ_data = results_df[results_df['Organ'] == organ].copy()
            
            if len(organ_data) == 0:
                continue
            
            safe_print(f"\nProcessing {organ}...")
            
            # Stratify by QUANTEC bins
            organ_data_stratified = self.stratify_organ(organ_data, organ)
            
            # Count unassigned
            unassigned = organ_data_stratified['QUANTEC_Bin'].isna().sum()
            if unassigned > 0:
                safe_print(f"  Warning: {unassigned} patients not assigned to any bin")
            
            # Compute bin metrics
            bin_metrics_df = self.compute_bin_metrics(organ_data_stratified, organ)
            
            if len(bin_metrics_df) == 0:
                safe_print(f"  Warning: No bin metrics computed for {organ}")
                continue
            
            # Save validation table
            output_file = self.output_dir / f"quantec_validation_{organ}.xlsx"
            
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                # Sheet 1: Bin metrics summary
                bin_metrics_df.to_excel(writer, index=False, sheet_name='Bin_Metrics')
                
                # Sheet 2: Detailed patient-level assignments
                detailed_cols = ['PrimaryPatientID', 'Organ', 'QUANTEC_Bin', 'Observed_Toxicity']
                dose_cols = [col for col in organ_data_stratified.columns if 'dose' in col.lower() or 'Dose' in col]
                ntcp_cols = [col for col in organ_data_stratified.columns if col.startswith('NTCP_')]
                
                detailed_cols.extend(dose_cols)
                detailed_cols.extend(ntcp_cols)
                
                detailed_cols = [col for col in detailed_cols if col in organ_data_stratified.columns]
                
                organ_data_stratified[detailed_cols].to_excel(
                    writer, index=False, sheet_name='Patient_Assignments'
                )
            
            output_files[organ] = output_file
            safe_print(f"  Saved: {output_file}")
            safe_print(f"    Bins: {len(bin_metrics_df)}")
            safe_print(f"    Patients: {len(organ_data_stratified)}")
        
        return output_files


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='QUANTEC Stratification Engine',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--ntcp_results',
        type=str,
        required=True,
        help='Path to NTCP results Excel file (from Step 3)'
    )
    
    parser.add_argument(
        '--bins_config',
        type=str,
        default='quantification/quantec_bins.json',
        help='Path to QUANTEC bins configuration (default: quantification/quantec_bins.json)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='out2/code3_output/quantec_validation',
        help='Output directory for validation tables (default: out2/code3_output/quantec_validation)'
    )
    
    args = parser.parse_args()
    
    ntcp_results_path = Path(args.ntcp_results)
    bins_config_path = Path(args.bins_config)
    output_dir = Path(args.output_dir)
    
    if not ntcp_results_path.exists():
        safe_print(f"ERROR: NTCP results file not found: {ntcp_results_path}")
        return 1
    
    if not bins_config_path.exists():
        safe_print(f"ERROR: Bins configuration not found: {bins_config_path}")
        return 1
    
    safe_print("="*60)
    safe_print("QUANTEC STRATIFICATION ENGINE")
    safe_print("="*60)
    safe_print(f"NTCP results: {ntcp_results_path}")
    safe_print(f"Bins config: {bins_config_path}")
    safe_print(f"Output directory: {output_dir}")
    safe_print("="*60)
    
    stratifier = QUANTECStratifier(bins_config_path, output_dir)
    output_files = stratifier.generate_validation_table(ntcp_results_path)
    
    if output_files:
        safe_print("\n" + "="*60)
        safe_print("QUANTEC VALIDATION COMPLETED")
        safe_print("="*60)
        safe_print(f"Generated {len(output_files)} validation table(s):")
        for organ, file_path in output_files.items():
            safe_print(f"  {organ}: {file_path}")
        return 0
    else:
        safe_print("\n[ERROR] No validation tables generated")
        return 1


if __name__ == "__main__":
    sys.exit(main())

