"""
Statistical reporting for publication-ready outputs

Generates LaTeX tables and statistical comparisons
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

try:
    from src.metrics.auc_calculator import calculate_auc_with_ci, compare_aucs_delong
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False


class PublicationStatisticalReporter:
    """Generate publication-ready statistical tables"""
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize statistical reporter
        
        Parameters
        ----------
        confidence_level : float
            Confidence level for intervals (default: 0.95 for 95% CI)
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    def create_table2_model_comparison(
        self, 
        results: Dict,
        caption: str = "Model Performance Comparison",
        label: str = "tab:model_comparison"
    ) -> pd.DataFrame:
        """
        Create Table 2: Model Performance Comparison
        
        Format follows IJROBP guidelines
        
        Parameters
        ----------
        results : dict
            Dictionary with model names as keys and metrics as values
            Each value should be a dict with keys: 'auc', 'auc_ci', 'brier', 'brier_ci', 'cal_slope', 'cv_auc_mean', 'cv_auc_std'
        caption : str
            Table caption
        label : str
            LaTeX label for referencing
            
        Returns
        -------
        pd.DataFrame
            Formatted table
        """
        table_data = []
        
        for model_name, metrics in results.items():
            auc_mean = metrics.get('auc', np.nan)
            auc_ci = metrics.get('auc_ci', (np.nan, np.nan))
            brier_mean = metrics.get('brier', np.nan)
            brier_ci = metrics.get('brier_ci', (np.nan, np.nan))
            cal_slope = metrics.get('cal_slope', np.nan)
            cv_auc_mean = metrics.get('cv_auc_mean', np.nan)
            cv_auc_std = metrics.get('cv_auc_std', np.nan)
            
            # Format AUC with CI
            if not np.isnan(auc_mean) and not np.isnan(auc_ci[0]):
                auc_str = f"{auc_mean:.3f} ({auc_ci[0]:.3f}--{auc_ci[1]:.3f})"
            elif not np.isnan(auc_mean):
                auc_str = f"{auc_mean:.3f}"
            else:
                auc_str = "---"
            
            # Format Brier with CI
            if not np.isnan(brier_mean) and not np.isnan(brier_ci[0]):
                brier_str = f"{brier_mean:.3f} ({brier_ci[0]:.3f}--{brier_ci[1]:.3f})"
            elif not np.isnan(brier_mean):
                brier_str = f"{brier_mean:.3f}"
            else:
                brier_str = "---"
            
            # Format calibration slope
            if not np.isnan(cal_slope):
                cal_str = f"{cal_slope:.3f}"
            else:
                cal_str = "---"
            
            # Format CV AUC
            if not np.isnan(cv_auc_mean) and not np.isnan(cv_auc_std):
                cv_str = f"{cv_auc_mean:.3f} $\\pm$ {cv_auc_std:.3f}"
            elif not np.isnan(cv_auc_mean):
                cv_str = f"{cv_auc_mean:.3f}"
            else:
                cv_str = "---"
            
            row = {
                'Model': model_name,
                'AUC (95\\% CI)': auc_str,
                'Brier Score (95\\% CI)': brier_str,
                'Calibration Slope': cal_str,
                'CV AUC (mean $\\pm$ SD)': cv_str
            }
            table_data.append(row)
        
        return pd.DataFrame(table_data)
    
    def create_latex_table(
        self, 
        df: pd.DataFrame, 
        caption: str,
        label: str = "tab:model_comparison",
        position: str = "htbp"
    ) -> str:
        """
        Export table as LaTeX for direct manuscript inclusion
        
        Parameters
        ----------
        df : pd.DataFrame
            Table data
        caption : str
            Table caption
        label : str
            LaTeX label
        position : str
            LaTeX float position (default: "htbp")
            
        Returns
        -------
        str
            LaTeX table code
        """
        # Convert DataFrame to LaTeX
        latex_lines = [
            "\\begin{table}[" + position + "]",
            "\\centering",
            "\\caption{" + caption + "}",
            "\\label{" + label + "}",
            "\\begin{tabular}{" + "l" + "c" * (len(df.columns) - 1) + "}",
            "\\toprule"
        ]
        
        # Header
        header = " & ".join(df.columns) + " \\\\"
        latex_lines.append(header)
        latex_lines.append("\\midrule")
        
        # Rows
        for _, row in df.iterrows():
            row_str = " & ".join([str(val) for val in row.values]) + " \\\\"
            latex_lines.append(row_str)
        
        # Footer
        latex_lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}"
        ])
        
        return "\n".join(latex_lines)
    
    def compare_models_delong(
        self,
        y_true: np.ndarray,
        model_predictions: Dict[str, np.ndarray],
        alpha: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Compare multiple models using DeLong test with Bonferroni correction
        
        Parameters
        ----------
        y_true : np.ndarray
            True binary labels
        model_predictions : dict
            Dictionary with model names as keys and predictions as values
        alpha : float, optional
            Significance level (default: uses self.alpha)
            
        Returns
        -------
        pd.DataFrame
            Comparison table with p-values
        """
        if not METRICS_AVAILABLE:
            raise ImportError("AUC calculator not available")
        
        if alpha is None:
            alpha = self.alpha
        
        model_names = list(model_predictions.keys())
        n_comparisons = len(model_names) * (len(model_names) - 1) // 2
        bonferroni_alpha = alpha / n_comparisons if n_comparisons > 0 else alpha
        
        comparison_data = []
        
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i >= j:
                    continue
                
                y_pred1 = model_predictions[model1]
                y_pred2 = model_predictions[model2]
                
                try:
                    z_stat, p_value = compare_aucs_delong(y_true, y_pred1, y_pred2)
                    p_value_corrected = min(p_value * n_comparisons, 1.0)  # Bonferroni correction
                    significant = p_value_corrected < alpha
                    
                    comparison_data.append({
                        'Model 1': model1,
                        'Model 2': model2,
                        'Z-statistic': f"{z_stat:.3f}",
                        'p-value': f"{p_value:.4f}",
                        'p-value (Bonferroni)': f"{p_value_corrected:.4f}",
                        'Significant': 'Yes' if significant else 'No'
                    })
                except Exception as e:
                    comparison_data.append({
                        'Model 1': model1,
                        'Model 2': model2,
                        'Z-statistic': '---',
                        'p-value': '---',
                        'p-value (Bonferroni)': '---',
                        'Significant': f'Error: {str(e)[:30]}'
                    })
        
        return pd.DataFrame(comparison_data)
