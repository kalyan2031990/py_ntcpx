"""
Publication-ready figure generation (600 DPI)

IJROBP-compliant figures with proper styling
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import seaborn as sns

# Set publication defaults
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.linewidth': 1.0,
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})


class PublicationFigureGenerator:
    """Generate 600 DPI publication-ready figures"""
    
    # IJROBP color scheme
    COLORS = {
        'LKB_LogLogit': '#1f77b4',
        'LKB_Probit': '#ff7f0e', 
        'RS_Poisson': '#2ca02c',
        'ANN': '#d62728',
        'XGBoost': '#9467bd',
        'LKB_QUANTEC': '#1f77b4',
        'RS_QUANTEC': '#2ca02c',
        'LKB_LOCAL': '#4472C4',
    }
    
    def __init__(self, output_dir: Path, dpi: int = 600):
        """
        Initialize figure generator
        
        Parameters
        ----------
        output_dir : Path
            Output directory for figures
        dpi : int
            Resolution in DPI (default: 600)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        
        # Update DPI in rcParams
        plt.rcParams['figure.dpi'] = dpi
        plt.rcParams['savefig.dpi'] = dpi
    
    def plot_roc_curves(
        self, 
        results: Dict, 
        filename: str = "fig2_roc_curves.png",
        figsize: Tuple[float, float] = (8, 8)
    ) -> Path:
        """
        Figure 2: ROC Curves for All Models
        
        Requirements:
        - 600 DPI
        - AUC values in legend
        - 95% CI bands (if available)
        - Diagonal reference line
        
        Parameters
        ----------
        results : dict
            Dictionary with model names as keys and metrics as values
            Each value should have: 'fpr', 'tpr', 'auc', 'auc_ci' (optional)
        filename : str
            Output filename
        figsize : tuple
            Figure size in inches
            
        Returns
        -------
        Path
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        for model_name, metrics in results.items():
            fpr = metrics.get('fpr')
            tpr = metrics.get('tpr')
            auc_val = metrics.get('auc', np.nan)
            auc_ci = metrics.get('auc_ci', (np.nan, np.nan))
            
            if fpr is None or tpr is None:
                continue
            
            # Format label
            if not np.isnan(auc_val) and not np.isnan(auc_ci[0]):
                label = f"{model_name} (AUC = {auc_val:.3f}, 95% CI: {auc_ci[0]:.3f}--{auc_ci[1]:.3f})"
            elif not np.isnan(auc_val):
                label = f"{model_name} (AUC = {auc_val:.3f})"
            else:
                label = model_name
            
            color = self.COLORS.get(model_name, 'gray')
            
            ax.plot(fpr, tpr, 
                   color=color,
                   linewidth=2,
                   label=label)
        
        # Reference line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Reference')
        
        ax.set_xlabel('1 - Specificity (False Positive Rate)', fontsize=11)
        ax.set_ylabel('Sensitivity (True Positive Rate)', fontsize=11)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.legend(loc='lower right', fontsize=9, frameon=True, fancybox=True, shadow=True)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        output_path = self.output_dir / filename
        fig.savefig(output_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return output_path
    
    def plot_calibration_curves(
        self,
        results: Dict,
        filename: str = "fig3_calibration_curves.png",
        figsize: Tuple[float, float] = (8, 8),
        n_bins: int = 10
    ) -> Path:
        """
        Figure 3: Calibration Curves
        
        Parameters
        ----------
        results : dict
            Dictionary with model names as keys
            Each value should have: 'y_true', 'y_pred'
        filename : str
            Output filename
        figsize : tuple
            Figure size
        n_bins : int
            Number of bins for calibration
            
        Returns
        -------
        Path
            Path to saved figure
        """
        from sklearn.calibration import calibration_curve
        
        fig, ax = plt.subplots(figsize=figsize)
        
        for model_name, metrics in results.items():
            y_true = metrics.get('y_true')
            y_pred = metrics.get('y_pred')
            
            if y_true is None or y_pred is None:
                continue
            
            try:
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_true, y_pred, n_bins=n_bins, strategy='uniform'
                )
                
                color = self.COLORS.get(model_name, 'gray')
                ax.plot(mean_predicted_value, fraction_of_positives, 
                       'o-', color=color, linewidth=2, markersize=6,
                       label=model_name)
            except Exception:
                continue
        
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Perfect calibration')
        
        ax.set_xlabel('Mean Predicted Probability', fontsize=11)
        ax.set_ylabel('Fraction of Positives', fontsize=11)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.legend(loc='upper left', fontsize=9, frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        output_path = self.output_dir / filename
        fig.savefig(output_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return output_path
    
    def plot_model_comparison_bar(
        self,
        results: Dict,
        filename: str = "fig4_model_comparison.png",
        figsize: Tuple[float, float] = (10, 6),
        metric: str = 'auc'
    ) -> Path:
        """
        Figure 4: Model Comparison Bar Chart
        
        Parameters
        ----------
        results : dict
            Dictionary with model names and metrics
        filename : str
            Output filename
        figsize : tuple
            Figure size
        metric : str
            Metric to plot ('auc' or 'brier')
            
        Returns
        -------
        Path
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        model_names = []
        values = []
        ci_lower = []
        ci_upper = []
        
        for model_name, metrics in results.items():
            if metric == 'auc':
                value = metrics.get('auc', np.nan)
                ci = metrics.get('auc_ci', (np.nan, np.nan))
            elif metric == 'brier':
                value = metrics.get('brier', np.nan)
                ci = metrics.get('brier_ci', (np.nan, np.nan))
            else:
                continue
            
            if np.isnan(value):
                continue
            
            model_names.append(model_name)
            values.append(value)
            ci_lower.append(ci[0] if not np.isnan(ci[0]) else value)
            ci_upper.append(ci[1] if not np.isnan(ci[1]) else value)
        
        if not model_names:
            plt.close()
            return None
        
        # Create bar chart with error bars
        x_pos = np.arange(len(model_names))
        colors = [self.COLORS.get(name, 'gray') for name in model_names]
        
        bars = ax.bar(x_pos, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Add error bars
        yerr_lower = [v - cl for v, cl in zip(values, ci_lower)]
        yerr_upper = [cu - v for v, cu in zip(values, ci_upper)]
        ax.errorbar(x_pos, values, yerr=[yerr_lower, yerr_upper], 
                   fmt='none', color='black', capsize=5, capthick=1.5, linewidth=1.5)
        
        # Labels
        ax.set_xlabel('Model', fontsize=11, fontweight='bold')
        ylabel = 'AUC' if metric == 'auc' else 'Brier Score'
        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        output_path = self.output_dir / filename
        fig.savefig(output_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return output_path
