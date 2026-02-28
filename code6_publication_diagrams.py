#!/usr/bin/env python3
"""
code6_publication_diagrams.py - Journal-Grade Scientific Figures Generator
===========================================================================

Generates reproducible, journal-grade scientific figures for py_ntcpx v1.0:
- Figure 1: Pipeline Workflow (DAG style)
- Figure 2: Feature-Model Matrix
- Figure 3: NTCP Modeling Spectrum
- Figure 4: Explainable AI (SHAP) Integration

Software: py_ntcpx_v1.0.0
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple

# Windows-safe encoding configuration
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except (AttributeError, ValueError):
    pass

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Publication-quality defaults
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.linewidth': 1.5,
    'figure.dpi': 100,
    'savefig.dpi': 1200,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white'
})

# Color scheme (journal-safe, colorblind-friendly)
COLORS = {
    'input': '#E8F4F8',
    'dvh': '#B8E6F0',
    'bdvh': '#7FC8E8',
    'dosimetric': '#90CDF4',
    'traditional': '#FFD700',
    'probabilistic': '#FFA500',
    'ml': '#90EE90',
    'qa': '#FFB6C1',
    'xai': '#DDA0DD',
    'output': '#F0E68C',
    'arrow': '#333333',
    'text': '#000000'
}


def generate_figure1_workflow(output_dir: Path, dpi: int = 1200) -> None:
    """
    Generate Figure 1: Pipeline Workflow (DAG style)
    
    Horizontal flow with domain lanes:
    - Input Data
    - DVH / bDVH
    - NTCP Models
    - QA & Uncertainty
    - XAI & Outputs
    """
    logger.info("Generating Figure 1: Pipeline Workflow...")
    
    try:
        fig, ax = plt.subplots(figsize=(18, 10))
        ax.set_xlim(0, 18)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Define lane positions (y-coordinates)
        lanes = {
        'input': 8.5,
        'dvh_bdvh': 6.5,
        'dosimetric': 4.5,
        'ntcp': 2.5,
            'qa': 0.5
        }
        
        lane_height = 1.5
        lane_spacing = 0.3
        
        # Lane 1: Input Data
        y = lanes['input']
        ax.add_patch(FancyBboxPatch((0.5, y - 0.6), 3.0, lane_height, 
                                    boxstyle="round,pad=0.1", facecolor=COLORS['input'], 
                                    edgecolor=COLORS['text'], linewidth=1.5))
        ax.text(2.0, y, 'DVH Files\n(TXT)', ha='center', va='center', 
                fontsize=9, weight='bold')
        
        ax.add_patch(FancyBboxPatch((4.0, y - 0.6), 3.0, lane_height,
                                    boxstyle="round,pad=0.1", facecolor=COLORS['input'],
                                    edgecolor=COLORS['text'], linewidth=1.5))
        ax.text(5.5, y, 'Clinical Data\n(Excel)', ha='center', va='center',
                fontsize=9, weight='bold')
        
        # Lane 2: DVH / bDVH
        y = lanes['dvh_bdvh']
        ax.add_patch(FancyBboxPatch((1.0, y - 0.6), 2.5, lane_height,
                                    boxstyle="round,pad=0.1", facecolor=COLORS['dvh'],
                                    edgecolor=COLORS['text'], linewidth=1.5))
        ax.text(2.25, y, 'Physical\nDVH', ha='center', va='center',
                fontsize=9, weight='bold')
        
        ax.add_patch(FancyBboxPatch((4.5, y - 0.6), 2.5, lane_height,
                                    boxstyle="round,pad=0.1", facecolor=COLORS['bdvh'],
                                    edgecolor=COLORS['text'], linewidth=1.5))
        ax.text(5.75, y, 'Biological\nDVH', ha='center', va='center',
                fontsize=9, weight='bold')
        
        # Lane 3: Dosimetric Summary
        y = lanes['dosimetric']
        ax.add_patch(FancyBboxPatch((2.0, y - 0.6), 4.0, lane_height,
                                    boxstyle="round,pad=0.1", facecolor=COLORS['dosimetric'],
                                    edgecolor=COLORS['text'], linewidth=1.5))
        ax.text(4.0, y, 'Dosimetric Metrics\ngEUD, Dmean, Vx, Dx', ha='center', va='center',
                fontsize=9, weight='bold')
        
        # Lane 4: NTCP Models
        y = lanes['ntcp']
        # Traditional models
        ax.add_patch(FancyBboxPatch((0.5, y - 0.6), 2.5, lane_height,
                                    boxstyle="round,pad=0.1", facecolor=COLORS['traditional'],
                                    edgecolor=COLORS['text'], linewidth=1.5))
        ax.text(1.75, y, 'LKB\nRS', ha='center', va='center', fontsize=9, weight='bold')
        
        # Probabilistic models
        ax.add_patch(FancyBboxPatch((3.5, y - 0.6), 2.5, lane_height,
                                    boxstyle="round,pad=0.1", facecolor=COLORS['probabilistic'],
                                    edgecolor=COLORS['text'], linewidth=1.5))
        ax.text(4.75, y, 'Probabilistic\nMonte Carlo', ha='center', va='center',
                fontsize=9, weight='bold')
        
        # ML models
        ax.add_patch(FancyBboxPatch((6.5, y - 0.6), 2.5, lane_height,
                                    boxstyle="round,pad=0.1", facecolor=COLORS['ml'],
                                    edgecolor=COLORS['text'], linewidth=1.5))
        ax.text(7.75, y, 'ANN\nXGBoost', ha='center', va='center',
                fontsize=9, weight='bold')
        
        # Lane 5: QA & Uncertainty
        y = lanes['qa']
        ax.add_patch(FancyBboxPatch((2.0, y - 0.6), 3.5, lane_height,
                                    boxstyle="round,pad=0.1", facecolor=COLORS['qa'],
                                    edgecolor=COLORS['text'], linewidth=1.5))
        ax.text(3.75, y, 'uNTCP\nCCS', ha='center', va='center',
                fontsize=9, weight='bold')
        
        # XAI
        ax.add_patch(FancyBboxPatch((6.0, y - 0.6), 3.0, lane_height,
                                    boxstyle="round,pad=0.1", facecolor=COLORS['xai'],
                                    edgecolor=COLORS['text'], linewidth=1.5))
        ax.text(7.5, y, 'SHAP\nExplanations', ha='center', va='center',
                fontsize=9, weight='bold')
        
        # Output
        ax.add_patch(FancyBboxPatch((9.5, y - 0.6), 3.0, lane_height,
                                    boxstyle="round,pad=0.1", facecolor=COLORS['output'],
                                    edgecolor=COLORS['text'], linewidth=1.5))
        ax.text(11.0, y, 'Publication\nTables', ha='center', va='center',
                fontsize=9, weight='bold')
        
        # Add forward arrows (horizontal flow)
        arrow_props = dict(arrowstyle='->', lw=2.0, color=COLORS['arrow'])
        
        # Input to DVH/bDVH
        ax.annotate('', xy=(2.25, lanes['dvh_bdvh'] + 0.3), 
                    xytext=(2.0, lanes['input'] - 0.3), arrowprops=arrow_props)
        ax.annotate('', xy=(5.75, lanes['dvh_bdvh'] + 0.3),
                    xytext=(5.5, lanes['input'] - 0.3), arrowprops=arrow_props)
        
        # DVH/bDVH to Dosimetric
        ax.annotate('', xy=(4.0, lanes['dosimetric'] + 0.3),
                    xytext=(2.25, lanes['dvh_bdvh'] - 0.3), arrowprops=arrow_props)
        ax.annotate('', xy=(4.0, lanes['dosimetric'] + 0.3),
                    xytext=(5.75, lanes['dvh_bdvh'] - 0.3), arrowprops=arrow_props)
        
        # Dosimetric to NTCP
        ax.annotate('', xy=(1.75, lanes['ntcp'] + 0.3),
                    xytext=(2.0, lanes['dosimetric'] - 0.3), arrowprops=arrow_props)
        ax.annotate('', xy=(4.75, lanes['ntcp'] + 0.3),
                    xytext=(4.0, lanes['dosimetric'] - 0.3), arrowprops=arrow_props)
        ax.annotate('', xy=(7.75, lanes['ntcp'] + 0.3),
                    xytext=(6.0, lanes['dosimetric'] - 0.3), arrowprops=arrow_props)
        
        # NTCP to QA/XAI/Output
        ax.annotate('', xy=(3.75, lanes['qa'] + 0.3),
                    xytext=(4.0, lanes['ntcp'] - 0.3), arrowprops=arrow_props)
        ax.annotate('', xy=(7.5, lanes['qa'] + 0.3),
                    xytext=(7.75, lanes['ntcp'] - 0.3), arrowprops=arrow_props)
        ax.annotate('', xy=(11.0, lanes['qa'] + 0.3),
                    xytext=(9.0, lanes['ntcp'] - 0.3), arrowprops=arrow_props)
        
        # Lane labels (left side)
        ax.text(0.2, lanes['input'], 'Input Data', ha='right', va='center',
                fontsize=10, weight='bold', rotation=90)
        ax.text(0.2, lanes['dvh_bdvh'], 'DVH / bDVH', ha='right', va='center',
                fontsize=10, weight='bold', rotation=90)
        ax.text(0.2, lanes['dosimetric'], 'Dosimetric', ha='right', va='center',
                fontsize=10, weight='bold', rotation=90)
        ax.text(0.2, lanes['ntcp'], 'NTCP Models', ha='right', va='center',
                fontsize=10, weight='bold', rotation=90)
        ax.text(0.2, lanes['qa'], 'QA / XAI / Output', ha='right', va='center',
                fontsize=10, weight='bold', rotation=90)
    
        # Save
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / 'figure_workflow.png', dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.savefig(output_dir / 'figure_workflow.svg', format='svg', bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved: figure_workflow.png, figure_workflow.svg")
    except Exception as e:
        logger.error(f"[FAIL] Publication diagrams failed: {e}")
        raise


def generate_figure2_feature_model_matrix(output_dir: Path, dpi: int = 1200) -> None:
    """
    Generate Figure 2: Feature-Model Matrix
    
    Matrix showing which features are used by which models.
    Rows: Feature categories
    Columns: Models
    """
    logger.info("Generating Figure 2: Feature-Model Matrix...")
    
    # --- define feature_categories outside try ---
    # Feature categories (rows)
    feature_categories = [
        'Physical DVH',
        'Biological DVH',
        'NTCP Parameters',
        'Clinical Factors'
    ]
    
    # Models (columns)
    models = [
        'LKB\n(Logistic)',
        'LKB\n(Probit)',
        'RS',
        'Probabilistic\ngEUD',
        'Monte Carlo\nNTCP',
        'ANN',
        'XGBoost'
    ]
    
    # Define which features are used by which models
    # 1 = used, 0 = not used
    matrix = {
        'Physical DVH': [1, 1, 1, 1, 1, 1, 1],
        'Biological DVH': [0, 0, 0, 1, 1, 1, 1],
        'NTCP Parameters': [1, 1, 1, 1, 1, 0, 0],
        'Clinical Factors': [0, 0, 0, 0, 0, 1, 1]
    }
    
    # Grid dimensions
    cell_width = 1.6
    cell_height = 1.8
    start_x = 2.0
    start_y = 7.5
    
    try:
        # plotting code only
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Draw header (models)
        for i, model in enumerate(models):
            x = start_x + i * cell_width
            ax.add_patch(FancyBboxPatch((x, start_y), cell_width, cell_height,
                                        boxstyle="round,pad=0.1", facecolor=COLORS['traditional'],
                                        edgecolor=COLORS['text'], linewidth=1.5))
            ax.text(x + cell_width/2, start_y + cell_height/2, model, ha='center', va='center',
                    fontsize=8, weight='bold')
        
        # Draw rows (features)
        for j, feature_cat in enumerate(feature_categories):
            y = start_y - (j + 1) * cell_height
            
            # Feature label
            ax.add_patch(FancyBboxPatch((0.2, y), 1.6, cell_height,
                                        boxstyle="round,pad=0.1", facecolor=COLORS['input'],
                                        edgecolor=COLORS['text'], linewidth=1.5))
            ax.text(1.0, y + cell_height/2, feature_cat, ha='center', va='center',
                    fontsize=9, weight='bold')
            
            # Matrix cells
            for i, model in enumerate(models):
                x = start_x + i * cell_width
                used = matrix[feature_cat][i]
                
                if used:
                    color = COLORS['ml']
                else:
                    color = '#F0F0F0'
                
                ax.add_patch(FancyBboxPatch((x, y), cell_width, cell_height,
                                            boxstyle="round,pad=0.1", facecolor=color,
                                            edgecolor=COLORS['text'], linewidth=1.0))
                
                # Mark with X or check (ASCII only)
                if used:
                    ax.text(x + cell_width/2, y + cell_height/2, 'X', ha='center', va='center',
                            fontsize=12, weight='bold', color=COLORS['text'])
        
        # Title
        ax.text(7.0, 9.5, 'Feature-Model Matrix', ha='center', va='center',
                fontsize=14, weight='bold')
        ax.text(7.0, 9.0, 'X = Feature used by model', ha='center', va='center',
                fontsize=9, style='italic')
        
        # Legend
        legend_elements = [
            mpatches.Patch(facecolor=COLORS['ml'], edgecolor=COLORS['text'], label='Feature Used'),
            mpatches.Patch(facecolor='#F0F0F0', edgecolor=COLORS['text'], label='Feature Not Used')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=9, frameon=True)
        
        # Save
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / 'figure_feature_model_matrix.png', dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.savefig(output_dir / 'figure_feature_model_matrix.svg', format='svg', bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved: figure_feature_model_matrix.png, figure_feature_model_matrix.svg")
    except Exception as e:
        logger.error(f"[FAIL] Publication diagrams failed: {e}")
        raise


def generate_figure3_methodology_spectrum(output_dir: Path, dpi: int = 1200) -> None:
    """
    Generate Figure 3: NTCP Modeling Spectrum
    
    Visual spectrum from Deterministic → Probabilistic → ML
    Overlay: uNTCP and CCS (safety filter)
    """
    logger.info("Generating Figure 3: NTCP Modeling Spectrum...")
    
    # Define spectrum parameters outside try
    spectrum_x = 1.0
    spectrum_y = 5.0
    spectrum_width = 14.0
    spectrum_height = 1.5
    segment_width = spectrum_width / 3
    advantages_y = 2.0
    
    try:
        # plotting code only
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 8)
        ax.axis('off')
        
        # Title
        ax.text(8.0, 7.5, 'NTCP Modeling Spectrum', ha='center', va='center',
                fontsize=16, weight='bold')
        ax.text(8.0, 7.0, 'Deterministic -> Probabilistic -> Machine Learning', ha='center', va='center',
                fontsize=12, style='italic')
        
        # Draw spectrum segments
        # Deterministic
        ax.add_patch(FancyBboxPatch((spectrum_x, spectrum_y), segment_width, spectrum_height,
                                    boxstyle="round,pad=0.1", facecolor=COLORS['traditional'],
                                    edgecolor=COLORS['text'], linewidth=2.0))
        ax.text(spectrum_x + segment_width/2, spectrum_y + spectrum_height/2, 'Deterministic\nLKB, RS',
                ha='center', va='center', fontsize=10, weight='bold')
        
        # Probabilistic
        ax.add_patch(FancyBboxPatch((spectrum_x + segment_width, spectrum_y), segment_width, spectrum_height,
                                    boxstyle="round,pad=0.1", facecolor=COLORS['probabilistic'],
                                    edgecolor=COLORS['text'], linewidth=2.0))
        ax.text(spectrum_x + segment_width*1.5, spectrum_y + spectrum_height/2, 'Probabilistic\nMonte Carlo',
                ha='center', va='center', fontsize=10, weight='bold')
        
        # ML
        ax.add_patch(FancyBboxPatch((spectrum_x + segment_width*2, spectrum_y), segment_width, spectrum_height,
                                    boxstyle="round,pad=0.1", facecolor=COLORS['ml'],
                                    edgecolor=COLORS['text'], linewidth=2.0))
        ax.text(spectrum_x + segment_width*2.5, spectrum_y + spectrum_height/2, 'Machine Learning\nANN, XGBoost',
                ha='center', va='center', fontsize=10, weight='bold')
        
        # uNTCP overlay (above spectrum)
        uNTCP_y = spectrum_y + spectrum_height + 0.8
        ax.add_patch(FancyBboxPatch((spectrum_x, uNTCP_y), spectrum_width, 0.8,
                                    boxstyle="round,pad=0.1", facecolor=COLORS['qa'],
                                    edgecolor=COLORS['text'], linewidth=1.5, alpha=0.7))
        ax.text(spectrum_x + spectrum_width/2, uNTCP_y + 0.4, 'uNTCP: Uncertainty-Aware NTCP',
                ha='center', va='center', fontsize=11, weight='bold')
        ax.text(spectrum_x + spectrum_width/2, uNTCP_y + 0.1, 'Applies to all models',
                ha='center', va='center', fontsize=9, style='italic')
        
        # CCS overlay (below spectrum)
        CCS_y = spectrum_y - 1.0
        ax.add_patch(FancyBboxPatch((spectrum_x, CCS_y), spectrum_width, 0.8,
                                    boxstyle="round,pad=0.1", facecolor=COLORS['xai'],
                                    edgecolor=COLORS['text'], linewidth=1.5, alpha=0.7))
        ax.text(spectrum_x + spectrum_width/2, CCS_y + 0.4, 'CCS: Cohort Consistency Score (Safety Filter)',
                ha='center', va='center', fontsize=11, weight='bold')
        ax.text(spectrum_x + spectrum_width/2, CCS_y + 0.1, 'Prevents unsafe extrapolation',
                ha='center', va='center', fontsize=9, style='italic')
        
        # Arrows showing direction
        arrow_props = dict(arrowstyle='->', lw=3.0, color=COLORS['arrow'])
        ax.annotate('', xy=(spectrum_x + spectrum_width, spectrum_y + spectrum_height/2),
                    xytext=(spectrum_x, spectrum_y + spectrum_height/2), arrowprops=arrow_props)
        
        # Key advantages text boxes
        # Deterministic advantages
        ax.add_patch(FancyBboxPatch((spectrum_x + 0.2, advantages_y), segment_width - 0.4, 1.2,
                                    boxstyle="round,pad=0.1", facecolor='#F0F0F0',
                                    edgecolor=COLORS['text'], linewidth=1.0))
        ax.text(spectrum_x + segment_width/2, advantages_y + 0.8, 'Interpretable',
                ha='center', va='center', fontsize=9, weight='bold')
        ax.text(spectrum_x + segment_width/2, advantages_y + 0.4, 'Physics-based',
                ha='center', va='center', fontsize=8)
        ax.text(spectrum_x + segment_width/2, advantages_y + 0.1, 'Literature parameters',
                ha='center', va='center', fontsize=8)
        
        # Probabilistic advantages
        ax.add_patch(FancyBboxPatch((spectrum_x + segment_width + 0.2, advantages_y), segment_width - 0.4, 1.2,
                                    boxstyle="round,pad=0.1", facecolor='#F0F0F0',
                                    edgecolor=COLORS['text'], linewidth=1.0))
        ax.text(spectrum_x + segment_width*1.5, advantages_y + 0.8, 'Uncertainty',
                ha='center', va='center', fontsize=9, weight='bold')
        ax.text(spectrum_x + segment_width*1.5, advantages_y + 0.4, 'Parameter variation',
                ha='center', va='center', fontsize=8)
        ax.text(spectrum_x + segment_width*1.5, advantages_y + 0.1, 'Confidence intervals',
                ha='center', va='center', fontsize=8)
        
        # ML advantages
        ax.add_patch(FancyBboxPatch((spectrum_x + segment_width*2 + 0.2, advantages_y), segment_width - 0.4, 1.2,
                                    boxstyle="round,pad=0.1", facecolor='#F0F0F0',
                                    edgecolor=COLORS['text'], linewidth=1.0))
        ax.text(spectrum_x + segment_width*2.5, advantages_y + 0.8, 'Non-linear',
                ha='center', va='center', fontsize=9, weight='bold')
        ax.text(spectrum_x + segment_width*2.5, advantages_y + 0.4, 'Feature interactions',
                ha='center', va='center', fontsize=8)
        ax.text(spectrum_x + segment_width*2.5, advantages_y + 0.1, 'High discrimination',
                ha='center', va='center', fontsize=8)
        
        # Save
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / 'figure_methodology.png', dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.savefig(output_dir / 'figure_methodology.svg', format='svg', bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved: figure_methodology.png, figure_methodology.svg")
    except Exception as e:
        logger.error(f"[FAIL] Publication diagrams failed: {e}")
        raise


def generate_figure4_xai_shap(output_dir: Path, dpi: int = 1200) -> None:
    """
    Generate Figure 4: Explainable AI (SHAP) Integration
    
    Conceptual flow:
    Model prediction → SHAP explanation → Clinical interpretability
    """
    logger.info("Generating Figure 4: Explainable AI (SHAP) Integration...")
    
    # Define step parameters outside try
    step1_x = 1.5
    step1_y = 5.0
    step_width = 2.5
    step_height = 1.5
    step2_x = 5.5
    step2_y = 5.0
    step3_x = 9.5
    step3_y = 5.0
    feature_y = 2.5
    feature_width = 1.8
    feature_height = 1.2
    start_x = 2.0
    protection_y = 0.5
    
    features = [
        ('gEUD', 0.35, '+'),
        ('Dmean', 0.25, '+'),
        ('V30', 0.20, '+'),
        ('Age', 0.10, '-'),
        ('Other', 0.10, '=')
    ]
    
    try:
        # plotting code only
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 8)
        ax.axis('off')
        
        # Title
        ax.text(7.0, 7.5, 'Explainable AI (SHAP) Integration', ha='center', va='center',
                fontsize=16, weight='bold')
        ax.text(7.0, 7.0, 'Model Prediction -> SHAP Explanation -> Clinical Interpretability',
                ha='center', va='center', fontsize=11, style='italic')
        
        # Step 1: Model Prediction
        ax.add_patch(FancyBboxPatch((step1_x, step1_y), step_width, step_height,
                                    boxstyle="round,pad=0.1", facecolor=COLORS['ml'],
                                    edgecolor=COLORS['text'], linewidth=2.0))
        ax.text(step1_x + step_width/2, step1_y + step_height/2 + 0.3, 'Model Prediction',
                ha='center', va='center', fontsize=11, weight='bold')
        ax.text(step1_x + step_width/2, step1_y + step_height/2 - 0.2, 'NTCP = 0.78',
                ha='center', va='center', fontsize=10, family='monospace')
        
        # Step 2: SHAP Explanation
        ax.add_patch(FancyBboxPatch((step2_x, step2_y), step_width, step_height,
                                    boxstyle="round,pad=0.1", facecolor=COLORS['xai'],
                                    edgecolor=COLORS['text'], linewidth=2.0))
        ax.text(step2_x + step_width/2, step2_y + step_height/2 + 0.3, 'SHAP Explanation',
                ha='center', va='center', fontsize=11, weight='bold')
        ax.text(step2_x + step_width/2, step2_y + step_height/2 - 0.2, 'Feature Impact',
                ha='center', va='center', fontsize=10)
        
        # Step 3: Clinical Interpretability
        ax.add_patch(FancyBboxPatch((step3_x, step3_y), step_width, step_height,
                                    boxstyle="round,pad=0.1", facecolor=COLORS['output'],
                                    edgecolor=COLORS['text'], linewidth=2.0))
        ax.text(step3_x + step_width/2, step3_y + step_height/2 + 0.3, 'Clinical\nInterpretability',
                ha='center', va='center', fontsize=11, weight='bold')
        
        # Arrows
        arrow_props = dict(arrowstyle='->', lw=3.0, color=COLORS['arrow'])
        ax.annotate('', xy=(step2_x, step2_y + step_height/2),
                    xytext=(step1_x + step_width, step1_y + step_height/2), arrowprops=arrow_props)
        ax.annotate('', xy=(step3_x, step3_y + step_height/2),
                    xytext=(step2_x + step_width, step2_y + step_height/2), arrowprops=arrow_props)
        
        # Feature impact breakdown (below steps)
        for i, (feat, impact, direction) in enumerate(features):
            x = start_x + i * (feature_width + 0.3)
            
            # Impact bar
            bar_width = impact * feature_width * 2
            ax.add_patch(FancyBboxPatch((x, feature_y), bar_width, feature_height,
                                        boxstyle="round,pad=0.05", facecolor=COLORS['xai'],
                                        edgecolor=COLORS['text'], linewidth=1.0))
            
            # Feature label
            ax.text(x + bar_width/2, feature_y + feature_height/2 + 0.3, feat,
                    ha='center', va='center', fontsize=9, weight='bold')
            ax.text(x + bar_width/2, feature_y + feature_height/2 - 0.2, f'{direction} {impact:.0%}',
                    ha='center', va='center', fontsize=8)
        
        # Directionality label
        ax.text(7.0, feature_y - 0.5, 'Feature Impact (Directionality)',
                ha='center', va='center', fontsize=10, weight='bold')
        
        # Clinical protection box
        ax.add_patch(FancyBboxPatch((3.0, protection_y), 8.0, 1.0,
                                    boxstyle="round,pad=0.1", facecolor='#FFE4E1',
                                    edgecolor=COLORS['text'], linewidth=1.5))
        ax.text(7.0, protection_y + 0.5, 'Clinical Protection: Not Black-Box Trust',
                ha='center', va='center', fontsize=11, weight='bold')
        ax.text(7.0, protection_y + 0.1, 'Explicit feature contributions enable clinical validation',
                ha='center', va='center', fontsize=9, style='italic')
        
        # Save
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / 'figure_xai_shap.png', dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.savefig(output_dir / 'figure_xai_shap.svg', format='svg', bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved: figure_xai_shap.png, figure_xai_shap.svg")
    except Exception as e:
        logger.error(f"[FAIL] Publication diagrams failed: {e}")
        raise


def generate_figure2a_feature_taxonomy(output_dir: Path, dpi: int = 1200) -> None:
    """
    Generate Figure 2a: Feature Taxonomy Used in Modeling
    
    Four vertical columns showing feature categories:
    1. Physical DVH Features
    2. Biological Features
    3. Clinical Factors
    4. Uncertainty & QA Metrics
    """
    logger.info("Generating Figure 2a: Feature Taxonomy...")
    
    try:
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Title
        ax.text(7.0, 9.5, 'Feature Taxonomy Used in Modeling', ha='center', va='center',
                fontsize=16, weight='bold')
        
        # Column positions
        column_width = 3.0
        column_spacing = 0.5
        start_x = 0.8
        column_y_top = 8.5
        column_y_bottom = 1.0
        
        # Column 1: Physical DVH Features
        col1_x = start_x
        ax.add_patch(FancyBboxPatch((col1_x, column_y_bottom), column_width, 
                                    column_y_top - column_y_bottom,
                                    boxstyle="round,pad=0.1", facecolor=COLORS['dvh'],
                                    edgecolor=COLORS['text'], linewidth=2.0))
        ax.text(col1_x + column_width/2, column_y_top - 0.3, 'Physical DVH Features',
                ha='center', va='center', fontsize=12, weight='bold')
        
        physical_features = [
            'Mean dose',
            'Max dose',
            'D2, D10, D50, D95',
            'V15, V30, V50'
        ]
        for i, feat in enumerate(physical_features):
            y_pos = column_y_top - 0.8 - i * 0.6
            ax.text(col1_x + column_width/2, y_pos, feat, ha='center', va='center',
                    fontsize=9)
        
        # Column 2: Biological Features
        col2_x = start_x + column_width + column_spacing
        ax.add_patch(FancyBboxPatch((col2_x, column_y_bottom), column_width,
                                    column_y_top - column_y_bottom,
                                    boxstyle="round,pad=0.1", facecolor=COLORS['bdvh'],
                                    edgecolor=COLORS['text'], linewidth=2.0))
        ax.text(col2_x + column_width/2, column_y_top - 0.3, 'Biological Features',
                ha='center', va='center', fontsize=12, weight='bold')
        
        biological_features = [
            'gEUD',
            'v_effective',
            'bDVH-derived metrics'
        ]
        for i, feat in enumerate(biological_features):
            y_pos = column_y_top - 0.8 - i * 0.6
            ax.text(col2_x + column_width/2, y_pos, feat, ha='center', va='center',
                    fontsize=9)
        
        # Column 3: Clinical Factors
        col3_x = start_x + (column_width + column_spacing) * 2
        ax.add_patch(FancyBboxPatch((col3_x, column_y_bottom), column_width,
                                    column_y_top - column_y_bottom,
                                    boxstyle="round,pad=0.1", facecolor=COLORS['input'],
                                    edgecolor=COLORS['text'], linewidth=2.0))
        ax.text(col3_x + column_width/2, column_y_top - 0.3, 'Clinical Factors',
                ha='center', va='center', fontsize=12, weight='bold')
        
        clinical_features = [
            'Dose per fraction',
            'Observed toxicity',
            'Organ label'
        ]
        for i, feat in enumerate(clinical_features):
            y_pos = column_y_top - 0.8 - i * 0.6
            ax.text(col3_x + column_width/2, y_pos, feat, ha='center', va='center',
                    fontsize=9)
        
        # Column 4: Uncertainty & QA Metrics
        col4_x = start_x + (column_width + column_spacing) * 3
        ax.add_patch(FancyBboxPatch((col4_x, column_y_bottom), column_width,
                                    column_y_top - column_y_bottom,
                                    boxstyle="round,pad=0.1", facecolor=COLORS['qa'],
                                    edgecolor=COLORS['text'], linewidth=2.0))
        ax.text(col4_x + column_width/2, column_y_top - 0.3, 'Uncertainty & QA Metrics',
                ha='center', va='center', fontsize=12, weight='bold')
        
        qa_features = [
            'uNTCP',
            'CI width',
            'CCS',
            'QA flags'
        ]
        for i, feat in enumerate(qa_features):
            y_pos = column_y_top - 0.8 - i * 0.6
            ax.text(col4_x + column_width/2, y_pos, feat, ha='center', va='center',
                    fontsize=9)
        
        # Save
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / 'figure_feature_taxonomy.png', dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.savefig(output_dir / 'figure_feature_taxonomy.svg', format='svg', bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved: figure_feature_taxonomy.png, figure_feature_taxonomy.svg")
    except Exception as e:
        logger.error(f"[FAIL] Publication diagrams failed: {e}")
        raise


def generate_figure3a_model_spectrum(output_dir: Path, dpi: int = 1200) -> None:
    """
    Generate Figure 3a: NTCP Modeling Spectrum
    
    Horizontal spectrum showing individual model blocks:
    Deterministic → Probabilistic → ML
    With explicit model names and uNTCP/CCS indication
    """
    logger.info("Generating Figure 3a: Model Spectrum...")
    
    try:
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 8)
        ax.axis('off')
        
        # Title
        ax.text(8.0, 7.5, 'NTCP Modeling Spectrum', ha='center', va='center',
                fontsize=16, weight='bold')
        ax.text(8.0, 7.0, 'Deterministic -> Probabilistic -> Machine Learning',
                ha='center', va='center', fontsize=12, style='italic')
        
        # Spectrum bar
        spectrum_x = 1.0
        spectrum_y = 5.0
        spectrum_width = 14.0
        spectrum_height = 1.5
        
        # Individual model blocks
        models = [
            ('LKB\n(Logistic)', COLORS['traditional'], 0),
            ('LKB\n(Probit)', COLORS['traditional'], 1),
            ('RS', COLORS['traditional'], 2),
            ('Probabilistic\ngEUD', COLORS['probabilistic'], 3),
            ('Monte Carlo\nNTCP', COLORS['probabilistic'], 4),
            ('ANN', COLORS['ml'], 5),
            ('XGBoost', COLORS['ml'], 6)
        ]
        
        block_width = spectrum_width / len(models)
        
        for i, (model_name, color, idx) in enumerate(models):
            x = spectrum_x + i * block_width
            ax.add_patch(FancyBboxPatch((x, spectrum_y), block_width * 0.9, spectrum_height,
                                        boxstyle="round,pad=0.05", facecolor=color,
                                        edgecolor=COLORS['text'], linewidth=1.5))
            ax.text(x + block_width * 0.45, spectrum_y + spectrum_height/2, model_name,
                    ha='center', va='center', fontsize=8, weight='bold')
        
        # Category labels below
        cat_width = spectrum_width / 3
        ax.text(spectrum_x + cat_width/2, spectrum_y - 0.5, 'Deterministic',
                ha='center', va='center', fontsize=10, weight='bold')
        ax.text(spectrum_x + cat_width * 1.5, spectrum_y - 0.5, 'Probabilistic',
                ha='center', va='center', fontsize=10, weight='bold')
        ax.text(spectrum_x + cat_width * 2.5, spectrum_y - 0.5, 'Machine Learning',
                ha='center', va='center', fontsize=10, weight='bold')
        
        # uNTCP overlay (above spectrum)
        uNTCP_y = spectrum_y + spectrum_height + 0.8
        ax.add_patch(FancyBboxPatch((spectrum_x, uNTCP_y), spectrum_width, 0.8,
                                    boxstyle="round,pad=0.1", facecolor=COLORS['qa'],
                                    edgecolor=COLORS['text'], linewidth=1.5, alpha=0.7))
        ax.text(spectrum_x + spectrum_width/2, uNTCP_y + 0.4, 'uNTCP: Uncertainty-Aware NTCP',
                ha='center', va='center', fontsize=11, weight='bold')
        ax.text(spectrum_x + spectrum_width/2, uNTCP_y + 0.1, 'Computed for all models',
                ha='center', va='center', fontsize=9, style='italic')
        
        # CCS overlay (below spectrum)
        CCS_y = spectrum_y - 1.0
        ax.add_patch(FancyBboxPatch((spectrum_x, CCS_y), spectrum_width, 0.8,
                                    boxstyle="round,pad=0.1", facecolor=COLORS['xai'],
                                    edgecolor=COLORS['text'], linewidth=1.5, alpha=0.7))
        ax.text(spectrum_x + spectrum_width/2, CCS_y + 0.4, 'CCS: Cohort Consistency Score (Safety Filter)',
                ha='center', va='center', fontsize=11, weight='bold')
        ax.text(spectrum_x + spectrum_width/2, CCS_y + 0.1, 'Computed for all models',
                ha='center', va='center', fontsize=9, style='italic')
        
        # Arrow showing direction
        arrow_props = dict(arrowstyle='->', lw=3.0, color=COLORS['arrow'])
        ax.annotate('', xy=(spectrum_x + spectrum_width, spectrum_y + spectrum_height/2),
                    xytext=(spectrum_x, spectrum_y + spectrum_height/2), arrowprops=arrow_props)
        
        # Increasing flexibility label
        ax.text(spectrum_x + spectrum_width/2, 3.5, 'Increasing Flexibility',
                ha='center', va='center', fontsize=10, weight='bold')
        ax.text(spectrum_x + spectrum_width/2, 3.0, 'Increasing Uncertainty Awareness',
                ha='center', va='center', fontsize=10, weight='bold')
        
        # Save
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / 'figure_model_spectrum.png', dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.savefig(output_dir / 'figure_model_spectrum.svg', format='svg', bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved: figure_model_spectrum.png, figure_model_spectrum.svg")
    except Exception as e:
        logger.error(f"[FAIL] Publication diagrams failed: {e}")
        raise


def generate_figure4a_shap_integration(output_dir: Path, dpi: int = 1200) -> None:
    """
    Generate Figure 4a: Explainable AI (SHAP) Integration
    
    Conceptual flow showing:
    ANN/XGBoost → SHAP explainer → Feature importance + Directionality + Clinical interpretability
    No SHAP values shown - role explanation only
    """
    logger.info("Generating Figure 4a: SHAP Integration...")
    
    try:
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 8)
        ax.axis('off')
        
        # Title
        ax.text(7.0, 7.5, 'Explainable AI (SHAP) Integration', ha='center', va='center',
                fontsize=16, weight='bold')
        ax.text(7.0, 7.0, 'Post-hoc Explanation Layer (Not a Predictive Model)',
                ha='center', va='center', fontsize=11, style='italic')
        
        # Step 1: ML Models
        step1_x = 1.5
        step1_y = 5.0
        step_width = 2.5
        step_height = 1.5
        
        ax.add_patch(FancyBboxPatch((step1_x, step1_y), step_width, step_height,
                                    boxstyle="round,pad=0.1", facecolor=COLORS['ml'],
                                    edgecolor=COLORS['text'], linewidth=2.0))
        ax.text(step1_x + step_width/2, step1_y + step_height/2 + 0.3, 'ML Models',
                ha='center', va='center', fontsize=11, weight='bold')
        ax.text(step1_x + step_width/2, step1_y + step_height/2 - 0.1, 'ANN',
                ha='center', va='center', fontsize=10)
        ax.text(step1_x + step_width/2, step1_y + step_height/2 - 0.4, 'XGBoost',
                ha='center', va='center', fontsize=10)
        
        # Step 2: SHAP Explainer
        step2_x = 5.5
        step2_y = 5.0
        
        ax.add_patch(FancyBboxPatch((step2_x, step2_y), step_width, step_height,
                                    boxstyle="round,pad=0.1", facecolor=COLORS['xai'],
                                    edgecolor=COLORS['text'], linewidth=2.0))
        ax.text(step2_x + step_width/2, step2_y + step_height/2 + 0.3, 'SHAP Explainer',
                ha='center', va='center', fontsize=11, weight='bold')
        ax.text(step2_x + step_width/2, step2_y + step_height/2 - 0.1, 'Post-hoc',
                ha='center', va='center', fontsize=9, style='italic')
        ax.text(step2_x + step_width/2, step2_y + step_height/2 - 0.4, 'Explanation',
                ha='center', va='center', fontsize=9, style='italic')
        
        # Step 3: Outputs
        step3_x = 9.5
        step3_y = 5.0
        
        ax.add_patch(FancyBboxPatch((step3_x, step3_y), step_width, step_height,
                                    boxstyle="round,pad=0.1", facecolor=COLORS['output'],
                                    edgecolor=COLORS['text'], linewidth=2.0))
        ax.text(step3_x + step_width/2, step3_y + step_height/2 + 0.3, 'Clinical\nInterpretability',
                ha='center', va='center', fontsize=11, weight='bold')
        
        # Arrows
        arrow_props = dict(arrowstyle='->', lw=3.0, color=COLORS['arrow'])
        ax.annotate('', xy=(step2_x, step2_y + step_height/2),
                    xytext=(step1_x + step_width, step1_y + step_height/2), arrowprops=arrow_props)
        ax.annotate('', xy=(step3_x, step3_y + step_height/2),
                    xytext=(step2_x + step_width, step2_y + step_height/2), arrowprops=arrow_props)
        
        # Output branches from SHAP
        branch_y = 3.0
        branch_width = 2.0
        branch_height = 0.8
        
        # Branch 1: Feature Importance
        branch1_x = 2.0
        ax.add_patch(FancyBboxPatch((branch1_x, branch_y), branch_width, branch_height,
                                    boxstyle="round,pad=0.1", facecolor='#F0F0F0',
                                    edgecolor=COLORS['text'], linewidth=1.5))
        ax.text(branch1_x + branch_width/2, branch_y + branch_height/2, 'Feature\nImportance',
                ha='center', va='center', fontsize=9, weight='bold')
        
        # Branch 2: Directionality
        branch2_x = 5.5
        ax.add_patch(FancyBboxPatch((branch2_x, branch_y), branch_width, branch_height,
                                    boxstyle="round,pad=0.1", facecolor='#F0F0F0',
                                    edgecolor=COLORS['text'], linewidth=1.5))
        ax.text(branch2_x + branch_width/2, branch_y + branch_height/2, 'Directionality\n(+/-)',
                ha='center', va='center', fontsize=9, weight='bold')
        
        # Branch 3: Clinical Interpretability
        branch3_x = 9.0
        ax.add_patch(FancyBboxPatch((branch3_x, branch_y), branch_width, branch_height,
                                    boxstyle="round,pad=0.1", facecolor='#F0F0F0',
                                    edgecolor=COLORS['text'], linewidth=1.5))
        ax.text(branch3_x + branch_width/2, branch_y + branch_height/2, 'Clinical\nInterpretability',
                ha='center', va='center', fontsize=9, weight='bold')
        
        # Arrows from SHAP to branches
        branch_arrow_props = dict(arrowstyle='->', lw=2.0, color=COLORS['arrow'], linestyle='--')
        ax.annotate('', xy=(branch1_x + branch_width/2, branch_y + branch_height),
                    xytext=(step2_x + step_width/2, step2_y), arrowprops=branch_arrow_props)
        ax.annotate('', xy=(branch2_x + branch_width/2, branch_y + branch_height),
                    xytext=(step2_x + step_width/2, step2_y), arrowprops=branch_arrow_props)
        ax.annotate('', xy=(branch3_x + branch_width/2, branch_y + branch_height),
                    xytext=(step2_x + step_width/2, step2_y), arrowprops=branch_arrow_props)
        
        # Clinical protection box
        protection_y = 0.5
        ax.add_patch(FancyBboxPatch((2.0, protection_y), 10.0, 1.0,
                                    boxstyle="round,pad=0.1", facecolor='#FFE4E1',
                                    edgecolor=COLORS['text'], linewidth=1.5))
        ax.text(7.0, protection_y + 0.5, 'Clinical Protection: Explicit Feature Contributions Enable Validation',
                ha='center', va='center', fontsize=11, weight='bold')
        ax.text(7.0, protection_y + 0.1, 'SHAP is a post-hoc explanation tool, not a predictive model',
                ha='center', va='center', fontsize=9, style='italic')
        
        # Save
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / 'figure_shap_integration.png', dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.savefig(output_dir / 'figure_shap_integration.svg', format='svg', bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved: figure_shap_integration.png, figure_shap_integration.svg")
    except Exception as e:
        logger.error(f"[FAIL] Publication diagrams failed: {e}")
        raise


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description='Generate journal-grade scientific figures for py_ntcpx v1.0',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Generates publication-ready figures:
- Figure 1: Pipeline Workflow (DAG style)
- Figure 2: Feature-Model Matrix
- Figure 2a: Feature Taxonomy (NEW)
- Figure 3: NTCP Modeling Spectrum
- Figure 3a: Model Spectrum with Individual Blocks (NEW)
- Figure 4: Explainable AI (SHAP)
- Figure 4a: SHAP Integration - Conceptual (NEW)

All figures are saved as PNG (high-DPI) and SVG (vector) formats.

Software: py_ntcpx_v1.0.0
        """
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='out2/code6_output',
        help='Output directory for figures (default: out2/code6_output)'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=1200,
        help='DPI for PNG output (default: 1200)'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Generating Journal-Grade Scientific Figures")
    logger.info("Software: py_ntcpx_v1.0.0")
    logger.info("=" * 60)
    
    # Generate all figures with error handling
    figures_to_generate = [
        ("Figure 1: Pipeline Workflow", lambda: generate_figure1_workflow(output_dir, args.dpi)),
        ("Figure 2: Feature-Model Matrix", lambda: generate_figure2_feature_model_matrix(output_dir, args.dpi)),
        ("Figure 2a: Feature Taxonomy", lambda: generate_figure2a_feature_taxonomy(output_dir, args.dpi)),
        ("Figure 3: NTCP Modeling Spectrum", lambda: generate_figure3_methodology_spectrum(output_dir, args.dpi)),
        ("Figure 3a: Model Spectrum", lambda: generate_figure3a_model_spectrum(output_dir, args.dpi)),
        ("Figure 4: Explainable AI (SHAP)", lambda: generate_figure4_xai_shap(output_dir, args.dpi)),
        ("Figure 4a: SHAP Integration", lambda: generate_figure4a_shap_integration(output_dir, args.dpi)),
    ]
    
    successful = []
    failed = []
    
    for figure_name, generate_func in figures_to_generate:
        try:
            generate_func()
            successful.append(figure_name)
        except Exception as e:
            failed.append((figure_name, str(e)))
            logger.warning(f"Failed to generate {figure_name}: {e}")
            # Continue with next figure instead of crashing
    
    logger.info("=" * 60)
    if successful:
        logger.info(f"Successfully generated {len(successful)} figure(s):")
        for name in successful:
            logger.info(f"  - {name}")
    
    if failed:
        logger.warning(f"Failed to generate {len(failed)} figure(s):")
        for name, error in failed:
            logger.warning(f"  - {name}: {error}")
    
    logger.info(f"Output directory: {output_dir.absolute()}")
    logger.info("=" * 60)
    logger.info("Generated files:")
    logger.info("  - figure_workflow.png, figure_workflow.svg")
    logger.info("  - figure_feature_model_matrix.png, figure_feature_model_matrix.svg")
    logger.info("  - figure_feature_taxonomy.png, figure_feature_taxonomy.svg")
    logger.info("  - figure_methodology.png, figure_methodology.svg")
    logger.info("  - figure_model_spectrum.png, figure_model_spectrum.svg")
    logger.info("  - figure_xai_shap.png, figure_xai_shap.svg")
    logger.info("  - figure_shap_integration.png, figure_shap_integration.svg")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
