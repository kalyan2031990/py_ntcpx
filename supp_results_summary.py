#!/usr/bin/env python3
"""
supp_results_summary.py - Publication Tables Generator
======================================================

Auto-generates all publication tables and appendix for journal submission:
- Table 1: Cohort, Treatment & DVH Characteristics (per organ)
- Table 2: NTCP Model Performance (Internal Validation)
- Table 3: Uncertainty-Aware & QA Metrics (Expanded)
- Table 4: Clinical Factors vs NTCP
- Appendix A1: Model & Equation Reference
- Appendix A2: Computational Reproducibility

Software: py_ntcpx_v1.0.0
"""

from __future__ import annotations

import argparse
import logging
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Windows-safe encoding configuration
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except (AttributeError, ValueError):
    pass

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# QUANTEC/RTOG recommended ranges (organ-specific)
QUANTEC_RANGES = {
    'Parotid': {
        'MeanDose': {'recommended': '<26 Gy', 'constraint': '<20 Gy'},
        'V20': {'recommended': '<50%', 'constraint': '<30%'},
        'V30': {'recommended': '<50%', 'constraint': '<30%'},
        'gEUD': {'recommended': '<26 Gy', 'constraint': '<20 Gy'}
    },
    'SpinalCord': {
        'MaxDose': {'recommended': '<50 Gy', 'constraint': '<45 Gy'},
        'D0.1cc': {'recommended': '<50 Gy', 'constraint': '<45 Gy'},
        'D2': {'recommended': '<50 Gy', 'constraint': '<45 Gy'}
    },
    'Larynx': {
        'MeanDose': {'recommended': '<44 Gy', 'constraint': '<40 Gy'},
        'V50': {'recommended': '<50%', 'constraint': '<30%'},
        'V60': {'recommended': '<30%', 'constraint': '<20%'}
    },
    'OralCavity': {
        'MeanDose': {'recommended': '<40 Gy', 'constraint': '<35 Gy'},
        'V40': {'recommended': '<50%', 'constraint': '<30%'},
        'V50': {'recommended': '<30%', 'constraint': '<20%'}
    }
}


class PublicationTableGenerator:
    """Generate publication-ready tables for journal submission"""
    
    def __init__(self, code1_output: Path, code2_output: Path, code3_output: Path,
                 code4_output: Path, code5_output: Path, clinical_file: Optional[Path] = None):
        """
        Initialize table generator
        
        Args:
            code1_output: Path to code1 output directory
            code2_output: Path to code2 output directory
            code3_output: Path to code3 output directory
            code4_output: Path to code4 output directory
            code5_output: Path to code5 output directory
            clinical_file: Optional clinical factors Excel file
        """
        self.code1_output = Path(code1_output)
        self.code2_output = Path(code2_output)
        self.code3_output = Path(code3_output)
        self.code4_output = Path(code4_output)
        self.code5_output = Path(code5_output)
        self.clinical_file = Path(clinical_file) if clinical_file else None
        
        # Load data
        self.ntcp_results = None
        self.performance_summary = None
        self.qa_summary = None
        self.clinical_factors = None
        self.dose_metrics = None
        self.processed_dvh = None
        
        self._load_data()
    
    def _load_data(self):
        """Load all required data files"""
        logger.info("Loading data files...")
        
        # Load NTCP results
        ntcp_file = self.code3_output / 'enhanced_ntcp_calculations.csv'
        if ntcp_file.exists():
            self.ntcp_results = pd.read_csv(ntcp_file)
            logger.info(f"Loaded NTCP results: {len(self.ntcp_results)} records")
        else:
            logger.warning(f"NTCP results not found: {ntcp_file}")
        
        # Load performance summary
        perf_file = self.code3_output / 'enhanced_summary_performance.csv'
        if perf_file.exists():
            self.performance_summary = pd.read_csv(perf_file)
            logger.info(f"Loaded performance summary")
        else:
            logger.warning(f"Performance summary not found: {perf_file}")
        
        # Load QA summary
        qa_file = self.code4_output / 'qa_summary_tables.xlsx'
        if qa_file.exists():
            try:
                self.qa_summary = pd.read_excel(qa_file, sheet_name='PerOrganSummary')
                logger.info(f"Loaded QA summary")
            except:
                logger.warning(f"Could not load QA summary: {qa_file}")
        else:
            logger.warning(f"QA summary not found: {qa_file}")
        
        # Load clinical factors
        if self.clinical_file and self.clinical_file.exists():
            try:
                self.clinical_factors = pd.read_excel(self.clinical_file)
                logger.info(f"Loaded clinical factors: {len(self.clinical_factors)} records")
            except:
                logger.warning(f"Could not load clinical factors: {self.clinical_file}")
        
        # Load processed DVH summary
        dvh_file = self.code1_output / 'processed_dvh.xlsx'
        if dvh_file.exists():
            try:
                self.processed_dvh = pd.read_excel(dvh_file)
                logger.info(f"Loaded processed DVH summary")
            except:
                logger.warning(f"Could not load processed DVH: {dvh_file}")
        
        # Extract dose metrics from NTCP results if available
        if self.ntcp_results is not None:
            dose_cols = [c for c in self.ntcp_results.columns 
                        if any(x in c for x in ['mean_dose', 'max_dose', 'gEUD', 'V', 'D'])]
            if dose_cols:
                # Identity-safe: use PrimaryPatientID
                self.dose_metrics = self.ntcp_results[['PrimaryPatientID', 'Organ'] + dose_cols].copy()
    
    def generate_table1_cohort_characteristics(self) -> pd.DataFrame:
        """
        Generate Table 1: Cohort, Treatment & DVH Characteristics (per organ)
        
        Includes:
        - Cohort demographics (n, age, gender, comorbidity)
        - Treatment technique distribution
        - Diagnosis distribution
        - Dosimetric & biological metrics (QUANTEC-aligned)
        - Benchmark columns (QUANTEC/RTOG ranges)
        """
        logger.info("Generating Table 1: Cohort Characteristics...")
        
        if self.ntcp_results is None:
            logger.error("NTCP results not available for Table 1")
            return pd.DataFrame()
        
        table1_rows = []
        
        for organ in sorted(self.ntcp_results['Organ'].unique()):
            organ_data = self.ntcp_results[self.ntcp_results['Organ'] == organ].copy()
            
            # Basic cohort stats
            n_patients = organ_data['PrimaryPatientID'].nunique()
            n_organs = len(organ_data)
            
            # Event rate
            if 'Observed_Toxicity' in organ_data.columns:
                events = int(organ_data['Observed_Toxicity'].sum())
                event_rate = (events / n_organs * 100) if n_organs > 0 else 0
            else:
                events = np.nan
                event_rate = np.nan
            
            # Demographics (if available in clinical file)
            age_mean = np.nan
            age_std = np.nan
            gender_male_pct = np.nan
            
            if self.clinical_factors is not None:
                clin_organ = self.clinical_factors[
                    self.clinical_factors['Organ'] == organ
                ] if 'Organ' in self.clinical_factors.columns else pd.DataFrame()
                
                if 'Age' in clin_organ.columns:
                    age_data = pd.to_numeric(clin_organ['Age'], errors='coerce').dropna()
                    if len(age_data) > 0:
                        age_mean = age_data.mean()
                        age_std = age_data.std()
                
                if 'Sex' in clin_organ.columns:
                    sex_data = clin_organ['Sex'].str.upper()
                    male_count = (sex_data == 'M').sum() + (sex_data == 'MALE').sum()
                    total = len(sex_data[sex_data.notna()])
                    if total > 0:
                        gender_male_pct = (male_count / total * 100)
            
            # Treatment technique (if available)
            technique_dist = {}
            if self.clinical_factors is not None and 'Treatment_Technique' in self.clinical_factors.columns:
                clin_organ = self.clinical_factors[
                    self.clinical_factors['Organ'] == organ
                ] if 'Organ' in self.clinical_factors.columns else pd.DataFrame()
                
                if len(clin_organ) > 0:
                    tech_counts = clin_organ['Treatment_Technique'].value_counts()
                    total_tech = tech_counts.sum()
                    for tech, count in tech_counts.items():
                        technique_dist[tech] = (count / total_tech * 100) if total_tech > 0 else 0
            
            # Diagnosis distribution (if available)
            diagnosis_dist = {}
            if self.clinical_factors is not None and 'Diagnosis' in self.clinical_factors.columns:
                clin_organ = self.clinical_factors[
                    self.clinical_factors['Organ'] == organ
                ] if 'Organ' in self.clinical_factors.columns else pd.DataFrame()
                
                if len(clin_organ) > 0:
                    diag_counts = clin_organ['Diagnosis'].value_counts()
                    total_diag = diag_counts.sum()
                    for diag, count in diag_counts.items():
                        diagnosis_dist[diag] = (count / total_diag * 100) if total_diag > 0 else 0
            
            # Dosimetric metrics
            dose_metrics = {}
            
            # Mean/Median/Modal dose
            if 'mean_dose' in organ_data.columns:
                dose_metrics['MeanDose(Gy)'] = organ_data['mean_dose'].mean()
                dose_metrics['MeanDose_Median(Gy)'] = organ_data['mean_dose'].median()
            
            if 'max_dose' in organ_data.columns:
                dose_metrics['MaxDose(Gy)'] = organ_data['max_dose'].mean()
            
            # gEUD
            if 'gEUD' in organ_data.columns:
                dose_metrics['gEUD(Gy)'] = organ_data['gEUD'].mean()
            
            # Expanded Vx
            for vx in [5, 10, 15, 20, 30, 40, 50, 60]:
                vx_col = f'V{vx}'
                if vx_col in organ_data.columns:
                    dose_metrics[f'V{vx}(%)'] = organ_data[vx_col].mean()
            
            # Expanded Dx
            for dx in [2, 10, 50]:
                dx_col = f'D{dx}'
                if dx_col in organ_data.columns:
                    dose_metrics[f'D{dx}(Gy)'] = organ_data[dx_col].mean()
            
            if 'mean_dose' in organ_data.columns:
                dose_metrics['Dmean(Gy)'] = organ_data['mean_dose'].mean()
            
            if 'max_dose' in organ_data.columns:
                dose_metrics['Dmax(Gy)'] = organ_data['max_dose'].mean()
            
            # QUANTEC benchmarks
            quantec_refs = QUANTEC_RANGES.get(organ, {})
            
            row = {
                'Organ': organ,
                'N_Patients': n_patients,
                'N_Organs_Evaluated': n_organs,
                'Event_Rate(%)': f"{event_rate:.1f}" if not np.isnan(event_rate) else "N/A",
                'Age_Mean_SD': f"{age_mean:.1f}+/-{age_std:.1f}" if not np.isnan(age_mean) else "N/A",
                'Gender_Male(%)': f"{gender_male_pct:.1f}" if not np.isnan(gender_male_pct) else "N/A",
            }
            
            # Treatment technique
            if technique_dist:
                row['Treatment_Technique(%)'] = '; '.join([f"{k}:{v:.1f}%" for k, v in technique_dist.items()])
            else:
                row['Treatment_Technique(%)'] = "N/A"
            
            # Diagnosis
            if diagnosis_dist:
                row['Diagnosis_Distribution(%)'] = '; '.join([f"{k}:{v:.1f}%" for k, v in diagnosis_dist.items()])
            else:
                row['Diagnosis_Distribution(%)'] = "N/A"
            
            # Add dose metrics
            row.update(dose_metrics)
            
            # Add QUANTEC benchmarks
            for metric, ranges in quantec_refs.items():
                if 'recommended' in ranges:
                    row[f'{metric}_QUANTEC_Recommended'] = ranges['recommended']
                if 'constraint' in ranges:
                    row[f'{metric}_QUANTEC_Constraint'] = ranges['constraint']
            
            table1_rows.append(row)
        
        table1_df = pd.DataFrame(table1_rows)
        return table1_df
    
    def generate_table2_ntcp_performance(self) -> pd.DataFrame:
        """
        Generate Table 2: NTCP Model Performance (Internal Validation)
        
        Includes:
        - Per-organ performance metrics
        - AUC, Brier score, calibration
        - Best model identification
        """
        logger.info("Generating Table 2: NTCP Model Performance...")
        
        if self.performance_summary is None:
            logger.error("Performance summary not available for Table 2")
            return pd.DataFrame()
        
        # Start with performance summary
        table2 = self.performance_summary.copy()
        
        # Add detailed metrics from NTCP results if available
        if self.ntcp_results is not None:
            detailed_metrics = []
            
            for organ in sorted(self.ntcp_results['Organ'].unique()):
                organ_data = self.ntcp_results[self.ntcp_results['Organ'] == organ].copy()
                
                if 'Observed_Toxicity' not in organ_data.columns:
                    continue
                
                # Get all NTCP model columns
                ntcp_cols = [c for c in organ_data.columns if c.startswith('NTCP_')]
                
                organ_metrics = {'Organ': organ}
                
                for ntcp_col in ntcp_cols:
                    model_name = ntcp_col.replace('NTCP_', '')
                    
                    # Calculate AUC
                    try:
                        from sklearn.metrics import roc_curve, auc
                        valid_data = organ_data[[ntcp_col, 'Observed_Toxicity']].dropna()
                        if len(valid_data) >= 5 and valid_data['Observed_Toxicity'].nunique() >= 2:
                            fpr, tpr, _ = roc_curve(valid_data['Observed_Toxicity'], valid_data[ntcp_col])
                            auc_score = auc(fpr, tpr)
                            organ_metrics[f'{model_name}_AUC'] = auc_score
                        else:
                            organ_metrics[f'{model_name}_AUC'] = np.nan
                    except:
                        organ_metrics[f'{model_name}_AUC'] = np.nan
                    
                    # Calculate Brier score
                    try:
                        from sklearn.metrics import brier_score_loss
                        valid_data = organ_data[[ntcp_col, 'Observed_Toxicity']].dropna()
                        if len(valid_data) >= 5:
                            brier = brier_score_loss(valid_data['Observed_Toxicity'], valid_data[ntcp_col])
                            organ_metrics[f'{model_name}_Brier'] = brier
                        else:
                            organ_metrics[f'{model_name}_Brier'] = np.nan
                    except:
                        organ_metrics[f'{model_name}_Brier'] = np.nan
                
                detailed_metrics.append(organ_metrics)
            
            if detailed_metrics:
                detailed_df = pd.DataFrame(detailed_metrics)
                # Merge with performance summary
                table2 = table2.merge(detailed_df, on='Organ', how='left', suffixes=('', '_detailed'))
        
        return table2
    
    def generate_table3_uncertainty_qa(self) -> pd.DataFrame:
        """
        Generate Table 3: Uncertainty-Aware & QA Metrics (Expanded)
        
        Includes:
        - Mean uNTCP
        - CI width
        - % DO_NOT_USE (CCS)
        - % QA flags raised
        - Dominant QA reason
        """
        logger.info("Generating Table 3: Uncertainty & QA Metrics...")
        
        if self.ntcp_results is None:
            logger.error("NTCP results not available for Table 3")
            return pd.DataFrame()
        
        table3_rows = []
        
        for organ in sorted(self.ntcp_results['Organ'].unique()):
            organ_data = self.ntcp_results[self.ntcp_results['Organ'] == organ].copy()
            
            n_total = len(organ_data)
            
            # uNTCP metrics
            uNTCP_mean = np.nan
            uNTCP_std = np.nan
            ci_width_mean = np.nan
            
            if 'uNTCP' in organ_data.columns:
                uNTCP_data = pd.to_numeric(organ_data['uNTCP'], errors='coerce').dropna()
                if len(uNTCP_data) > 0:
                    uNTCP_mean = uNTCP_data.mean()
            
            if 'uNTCP_STD' in organ_data.columns:
                std_data = pd.to_numeric(organ_data['uNTCP_STD'], errors='coerce').dropna()
                if len(std_data) > 0:
                    uNTCP_std = std_data.mean()
            
            # CI width
            if 'uNTCP_CI_L' in organ_data.columns and 'uNTCP_CI_U' in organ_data.columns:
                ci_l = pd.to_numeric(organ_data['uNTCP_CI_L'], errors='coerce')
                ci_u = pd.to_numeric(organ_data['uNTCP_CI_U'], errors='coerce')
                ci_width = ci_u - ci_l
                ci_width_mean = ci_width.mean()
            
            # CCS metrics
            do_not_use_pct = np.nan
            qa_flags_pct = np.nan
            dominant_qa_reason = "N/A"
            
            if 'CCS_Safety' in organ_data.columns:
                do_not_use = (organ_data['CCS_Safety'] == 'DO_NOT_USE').sum()
                do_not_use_pct = (do_not_use / n_total * 100) if n_total > 0 else 0
            
            if 'CCS_Warning' in organ_data.columns:
                warnings = organ_data['CCS_Warning'].notna().sum()
                qa_flags_pct = (warnings / n_total * 100) if n_total > 0 else 0
                
                # Dominant QA reason
                warning_counts = organ_data['CCS_Warning'].value_counts()
                if len(warning_counts) > 0:
                    dominant_qa_reason = warning_counts.index[0]
            
            row = {
                'Organ': organ,
                'N_Total': n_total,
                'Mean_uNTCP': f"{uNTCP_mean:.4f}" if not np.isnan(uNTCP_mean) else "N/A",
                'Mean_uNTCP_STD': f"{uNTCP_std:.4f}" if not np.isnan(uNTCP_std) else "N/A",
                'Mean_CI_Width': f"{ci_width_mean:.4f}" if not np.isnan(ci_width_mean) else "N/A",
                'DO_NOT_USE_Percent(%)': f"{do_not_use_pct:.1f}" if not np.isnan(do_not_use_pct) else "N/A",
                'QA_Flags_Raised(%)': f"{qa_flags_pct:.1f}" if not np.isnan(qa_flags_pct) else "N/A",
                'Dominant_QA_Reason': dominant_qa_reason
            }
            
            table3_rows.append(row)
        
        table3_df = pd.DataFrame(table3_rows)
        return table3_df
    
    def generate_table4_clinical_factors(self) -> pd.DataFrame:
        """
        Generate Table 4: Clinical Factors vs NTCP
        
        Includes:
        - Factor associations with NTCP
        - p-values
        - Statistical test used
        - Directionality (↑ risk / ↓ risk)
        """
        logger.info("Generating Table 4: Clinical Factors vs NTCP...")
        
        # Try to load from code5 output (pipeline: files in code5_output root; standalone: under clinical_factors_analysis/)
        code5_file = self.code5_output / 'continuous_factors_analysis.xlsx'
        if not code5_file.exists():
            code5_file = self.code5_output / 'clinical_factors_analysis' / 'continuous_factors_analysis.xlsx'
        
        if code5_file.exists():
            try:
                # Load correlations sheet
                corr_df = pd.read_excel(code5_file, sheet_name='Correlations')
                return corr_df
            except:
                pass
        
        # Fallback: generate from available data
        if self.ntcp_results is None or self.clinical_factors is None:
            logger.warning("Insufficient data for Table 4")
            return pd.DataFrame()
        
        # Merge data
        merged = self.ntcp_results.merge(
            self.clinical_factors,
            on=['PrimaryPatientID', 'Organ'],
            how='inner',
            suffixes=('_ntcp', '_clin')
        )
        
        if len(merged) == 0:
            logger.warning("No merged data available for Table 4")
            return pd.DataFrame()
        
        table4_rows = []
        
        # Analyze continuous factors
        continuous_factors = ['Age', 'Dose_per_Fraction', 'Total_Dose']
        
        for factor in continuous_factors:
            if factor not in merged.columns:
                continue
            
            # Get NTCP columns
            ntcp_cols = [c for c in merged.columns if c.startswith('NTCP_')]
            
            for ntcp_col in ntcp_cols:
                model_name = ntcp_col.replace('NTCP_', '')
                
                # Correlation
                valid_data = merged[[factor, ntcp_col]].dropna()
                if len(valid_data) >= 10:
                    try:
                        from scipy.stats import pearsonr
                        corr, p_val = pearsonr(valid_data[factor], valid_data[ntcp_col])
                        
                        direction = "↑" if corr > 0 else "↓"
                        
                        row = {
                            'Factor': factor,
                            'NTCP_Model': model_name,
                            'Correlation': f"{corr:.3f}",
                            'P_Value': f"{p_val:.4f}",
                            'Significant': "Yes" if p_val < 0.05 else "No",
                            'Direction': direction,
                            'Test': 'Pearson correlation'
                        }
                        table4_rows.append(row)
                    except:
                        pass
        
        table4_df = pd.DataFrame(table4_rows)
        return table4_df
    
    def generate_appendix_a1_model_reference(self) -> pd.DataFrame:
        """Generate Appendix A1: Model & Equation Reference"""
        logger.info("Generating Appendix A1: Model Reference...")
        
        models = [
            {
                'Model': 'LKB Log-Logistic',
                'Equation': 'NTCP = 1 / (1 + (TD50/gEUD)^(4*γ50))',
                'Parameters': 'TD50, γ50, α/β',
                'Source': 'Lyman 1985, Niemierko 1999'
            },
            {
                'Model': 'LKB Probit',
                'Equation': 'NTCP = Φ((D - TD50)/(m*TD50))',
                'Parameters': 'TD50, m',
                'Source': 'Lyman 1985'
            },
            {
                'Model': 'RS Poisson',
                'Equation': 'NTCP = 1 - exp(-exp(γ*(D/D50)))',
                'Parameters': 'D50, γ, s',
                'Source': 'Källman 1992'
            },
            {
                'Model': 'Probabilistic gEUD',
                'Equation': 'NTCP with parameter uncertainty via Monte Carlo',
                'Parameters': 'TD50, γ50, α/β (with uncertainty)',
                'Source': 'Brodin 2017'
            },
            {
                'Model': 'Monte Carlo NTCP',
                'Equation': 'NTCP with DVH + parameter uncertainty',
                'Parameters': 'All model parameters (with uncertainty)',
                'Source': 'Fenwick 2001'
            },
            {
                'Model': 'ANN',
                'Equation': 'Multi-layer perceptron (non-linear)',
                'Parameters': 'Learned weights',
                'Source': 'This work'
            },
            {
                'Model': 'XGBoost',
                'Equation': 'Gradient boosting ensemble',
                'Parameters': 'Learned trees',
                'Source': 'This work'
            },
            {
                'Model': 'Random Forest',
                'Equation': 'Random forest ensemble',
                'Parameters': 'Learned trees',
                'Source': 'This work'
            },
            {
                'Model': 'uNTCP',
                'Equation': 'NTCP +/- sigma via Delta method',
                'Parameters': 'Uncertainty propagation',
                'Source': 'This work'
            },
            {
                'Model': 'CCS',
                'Equation': 'CCS = exp(-1/2 D^2_Mahalanobis)',
                'Parameters': 'Cohort distribution (mu, SIGMA)',
                'Source': 'This work'
            }
        ]
        
        return pd.DataFrame(models)
    
    def generate_appendix_a2_reproducibility(self) -> pd.DataFrame:
        """Generate Appendix A2: Computational Reproducibility"""
        logger.info("Generating Appendix A2: Reproducibility...")
        
        # System info
        sys_info = {
            'OS': platform.system(),
            'OS_Version': platform.version(),
            'CPU': platform.processor(),
            'Python_Version': sys.version.split()[0],
            'Platform': platform.platform()
        }
        
        # Try to get library versions
        try:
            import pandas as pd
            sys_info['Pandas_Version'] = pd.__version__
        except:
            sys_info['Pandas_Version'] = 'N/A'
        
        try:
            import numpy as np
            sys_info['NumPy_Version'] = np.__version__
        except:
            sys_info['NumPy_Version'] = 'N/A'
        
        try:
            import sklearn
            sys_info['Scikit-learn_Version'] = sklearn.__version__
        except:
            sys_info['Scikit-learn_Version'] = 'N/A'
        
        try:
            import xgboost as xgb
            sys_info['XGBoost_Version'] = xgb.__version__
        except:
            sys_info['XGBoost_Version'] = 'N/A'
        
        # Runtime info (if available from logs)
        sys_info['Analysis_Date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        sys_info['Software'] = 'py_ntcpx v1.0'
        
        # QA flags summary
        qa_summary_text = "N/A"
        if self.qa_summary is not None:
            total_flags = len(self.qa_summary)
            qa_summary_text = f"{total_flags} organs analyzed"
        
        sys_info['QA_Flags_Summary'] = qa_summary_text
        
        return pd.DataFrame([sys_info])
    
    def generate_all_tables(self, output_dir: Path):
        """Generate all tables and save to Excel"""
        logger.info("Generating all publication tables...")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        excel_file = output_dir / 'publication_tables.xlsx'
        
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # Table 1
            table1 = self.generate_table1_cohort_characteristics()
            if not table1.empty:
                table1.to_excel(writer, sheet_name='Table1_Cohort_Characteristics', index=False)
                logger.info(f"Table 1: {len(table1)} rows")
            
            # Table 2
            table2 = self.generate_table2_ntcp_performance()
            if not table2.empty:
                table2.to_excel(writer, sheet_name='Table2_NTCP_Performance', index=False)
                logger.info(f"Table 2: {len(table2)} rows")
            
            # Table 3
            table3 = self.generate_table3_uncertainty_qa()
            if not table3.empty:
                table3.to_excel(writer, sheet_name='Table3_Uncertainty_QA', index=False)
                logger.info(f"Table 3: {len(table3)} rows")
            
            # Table 4
            table4 = self.generate_table4_clinical_factors()
            if not table4.empty:
                table4.to_excel(writer, sheet_name='Table4_Clinical_Factors', index=False)
                logger.info(f"Table 4: {len(table4)} rows")
            
            # Appendix A1
            app_a1 = self.generate_appendix_a1_model_reference()
            if not app_a1.empty:
                app_a1.to_excel(writer, sheet_name='Appendix_A1_Model_Reference', index=False)
                logger.info(f"Appendix A1: {len(app_a1)} models")
            
            # Appendix A2
            app_a2 = self.generate_appendix_a2_reproducibility()
            if not app_a2.empty:
                app_a2.to_excel(writer, sheet_name='Appendix_A2_Reproducibility', index=False)
                logger.info(f"Appendix A2: Generated")
        
        logger.info(f"All tables saved to: {excel_file}")


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description='Generate publication tables for journal submission',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Generates comprehensive publication tables:
- Table 1: Cohort, Treatment & DVH Characteristics
- Table 2: NTCP Model Performance
- Table 3: Uncertainty-Aware & QA Metrics
- Table 4: Clinical Factors vs NTCP
- Appendix A1: Model & Equation Reference
- Appendix A2: Computational Reproducibility

Software: py_ntcpx_v1.0.0
        """
    )
    
    parser.add_argument(
        '--code1_output',
        type=str,
        required=True,
        help='Path to code1 output directory'
    )
    
    parser.add_argument(
        '--code2_output',
        type=str,
        required=True,
        help='Path to code2 output directory'
    )
    
    parser.add_argument(
        '--code3_output',
        type=str,
        required=True,
        help='Path to code3 output directory'
    )
    
    parser.add_argument(
        '--code4_output',
        type=str,
        required=True,
        help='Path to code4 output directory'
    )
    
    parser.add_argument(
        '--code5_output',
        type=str,
        required=True,
        help='Path to code5 output directory'
    )
    
    parser.add_argument(
        '--clinical_file',
        type=str,
        default=None,
        help='Optional clinical factors Excel file'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='supp_results_summary_output',
        help='Output directory for publication tables (default: supp_results_summary_output)'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Publication Tables Generator - py_ntcpx v1.0")
    logger.info("=" * 60)
    
    generator = PublicationTableGenerator(
        code1_output=args.code1_output,
        code2_output=args.code2_output,
        code3_output=args.code3_output,
        code4_output=args.code4_output,
        code5_output=args.code5_output,
        clinical_file=args.clinical_file
    )
    
    generator.generate_all_tables(args.output_dir)
    
    logger.info("=" * 60)
    logger.info("Publication tables generation completed!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

