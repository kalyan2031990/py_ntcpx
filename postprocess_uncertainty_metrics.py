#!/usr/bin/env python3
"""
Post-processing script for Uncertainty Metrics Summary
=======================================================

Adds "% DO_NOT_USE (CCS)" as a post-processing metric.
This is a reporting-only metric, not a new model feature.

Reads existing outputs from code3 (enhanced_ntcp_calculations.csv)
and generates summary statistics per organ for Table 3 in the manuscript.

Author: NTCP Analysis Pipeline
Version: 1.0
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def calculate_uncertainty_metrics_summary(input_file, output_dir):
    """
    Calculate uncertainty metrics summary per organ.
    
    Parameters:
    -----------
    input_file : str or Path
        Path to enhanced_ntcp_calculations.csv from code3
    output_dir : str or Path
        Output directory for summary tables
        
    Returns:
    --------
    pd.DataFrame
        Summary table with metrics per organ
    """
    
    input_path = Path(input_file)
    output_path = Path(output_dir)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Uncertainty Metrics Post-Processing")
    print("=" * 60)
    print(f"Reading input file: {input_path}")
    
    # Read the enhanced NTCP calculations
    try:
        df = pd.read_csv(input_path)
        print(f"Loaded {len(df)} records")
        print(f"Columns: {list(df.columns)}")
    except Exception as e:
        print(f"Error: Failed to read input file: {e}")
        return None
    
    # Verify required columns exist
    required_cols = ['Organ', 'CCS', 'uNTCP', 'uNTCP_CI_L', 'uNTCP_CI_U']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        # Try to continue with available columns
        if 'Organ' not in df.columns:
            print("Error: 'Organ' column is required. Cannot proceed.")
            return None
    
    # Initialize summary data list
    summary_data = []
    
    # Process each organ
    organs = df['Organ'].unique()
    print(f"\nProcessing {len(organs)} organs: {list(organs)}")
    
    for organ in organs:
        organ_data = df[df['Organ'] == organ].copy()
        print(f"\nProcessing {organ} ({len(organ_data)} records)")
        
        # Count total valid predictions (rows with non-null CCS or uNTCP)
        # A prediction is considered valid if it has either CCS or uNTCP
        valid_mask = (
            organ_data['CCS'].notna() | 
            organ_data['uNTCP'].notna()
        )
        total_predictions = valid_mask.sum()
        
        # Calculate mean uNTCP (only for non-null values)
        mean_untcp = organ_data['uNTCP'].mean() if 'uNTCP' in organ_data.columns else np.nan
        
        # Calculate mean CI width (CI_U - CI_L) for valid intervals
        if 'uNTCP_CI_L' in organ_data.columns and 'uNTCP_CI_U' in organ_data.columns:
            ci_valid = (
                organ_data['uNTCP_CI_L'].notna() & 
                organ_data['uNTCP_CI_U'].notna()
            )
            if ci_valid.any():
                ci_widths = (
                    organ_data.loc[ci_valid, 'uNTCP_CI_U'] - 
                    organ_data.loc[ci_valid, 'uNTCP_CI_L']
                )
                mean_ci_width = ci_widths.mean()
            else:
                mean_ci_width = np.nan
        else:
            mean_ci_width = np.nan
        
        # CCS threshold (fixed)
        ccs_threshold = 0.2
        
        # Count DO_NOT_USE (CCS < 0.2)
        if 'CCS' in organ_data.columns:
            # Only count where CCS is not null
            ccs_valid = organ_data['CCS'].notna()
            if ccs_valid.any():
                do_not_use_mask = (organ_data['CCS'] < ccs_threshold) & ccs_valid
                do_not_use_count = do_not_use_mask.sum()
            else:
                do_not_use_count = 0
        else:
            do_not_use_count = 0
        
        # Calculate percentage
        if total_predictions > 0:
            do_not_use_percent = (do_not_use_count / total_predictions) * 100.0
        else:
            do_not_use_percent = np.nan
        
        # Add to summary
        summary_data.append({
            'Organ': organ,
            'Total_predictions': int(total_predictions),
            'Mean_uNTCP': mean_untcp if pd.notna(mean_untcp) else np.nan,
            'Mean_CI_width': mean_ci_width if pd.notna(mean_ci_width) else np.nan,
            'CCS_threshold': ccs_threshold,
            'DO_NOT_USE_count': int(do_not_use_count),
            'DO_NOT_USE_percent': do_not_use_percent if pd.notna(do_not_use_percent) else np.nan
        })
        
        print(f"  Total predictions: {total_predictions}")
        print(f"  Mean uNTCP: {mean_untcp:.4f}" if pd.notna(mean_untcp) else "  Mean uNTCP: N/A")
        print(f"  Mean CI width: {mean_ci_width:.4f}" if pd.notna(mean_ci_width) else "  Mean CI width: N/A")
        print(f"  DO_NOT_USE count: {do_not_use_count} (CCS < {ccs_threshold})")
        print(f"  DO_NOT_USE percent: {do_not_use_percent:.2f}%")
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Round numeric columns for display
    numeric_cols = ['Mean_uNTCP', 'Mean_CI_width', 'DO_NOT_USE_percent']
    for col in numeric_cols:
        if col in summary_df.columns:
            summary_df[col] = summary_df[col].round(4)
    
    # Save to Excel - Use context manager for Windows-safe Excel writing
    excel_file = output_path / 'uncertainty_metrics_summary.xlsx'
    try:
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            summary_df.to_excel(writer, index=False, sheet_name='Uncertainty_Metrics')
        print(f"\n[OK] Summary table saved to: {excel_file}")
    except Exception as e:
        print(f"Warning: Failed to save Excel file: {e}")
        print("Trying CSV instead...")
    
    # Save to CSV
    csv_file = output_path / 'uncertainty_metrics_summary.csv'
    summary_df.to_csv(csv_file, index=False)
    print(f"[OK] Summary table saved to: {csv_file}")
    
    # Create caption file
    caption_file = output_path / 'uncertainty_metrics_caption.txt'
    caption_text = """Predictions with CCS < 0.2 were flagged as DO_NOT_USE and excluded from clinical interpretation. Percentages reflect cohort-level instability rather than model performance."""
    
    with open(caption_file, 'w', encoding='utf-8') as f:
        f.write(caption_text)
    
    print(f"[OK] Caption file saved to: {caption_file}")
    
    print("\n" + "=" * 60)
    print("Post-processing completed successfully!")
    print("=" * 60)
    print(f"\nSummary table:")
    print(summary_df.to_string(index=False))
    
    return summary_df


def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser(
        description='Post-process uncertainty metrics from code3 outputs'
    )
    parser.add_argument(
        '--input_file',
        type=str,
        default='out2/code3_output/enhanced_ntcp_calculations.csv',
        help='Path to enhanced_ntcp_calculations.csv from code3 (default: out2/code3_output/enhanced_ntcp_calculations.csv)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='out2/code5_output/uncertainty_metrics',
        help='Output directory for summary tables (default: out2/code5_output/uncertainty_metrics)'
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        print("Please run code3_ntcp_analysis_ml.py first to generate the input file.")
        return
    
    # Run post-processing
    try:
        summary_df = calculate_uncertainty_metrics_summary(
            args.input_file,
            args.output_dir
        )
        
        if summary_df is not None:
            print("\nNext Steps:")
            print("  - Review uncertainty_metrics_summary.xlsx for Table 3 data")
            print("  - Use uncertainty_metrics_caption.txt for manuscript caption")
            
    except Exception as e:
        print(f"\nError during post-processing: {e}")
        import traceback
        print("\nFull error traceback:")
        traceback.print_exc()


if __name__ == "__main__":
    main()

