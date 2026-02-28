
#!/usr/bin/env python3
"""
NTCP Clinical Factors Analysis (Patched)
=======================================

This patched script adds robust column standardization and automatic detection/
creation of a numeric binary 'Observed_Toxicity' column to prevent KeyErrors
(e.g., when the column is named slightly differently or has merge suffixes).

Changes vs original:
- Standardize clinical column names (e.g., "Technique" -> "Treatment_Technique",
  "DosePerFraction(Gy)" -> "Dose_per_Fraction", "Total_Dose(Gy)" -> "Total_Dose",
  "Duration(wk)" -> "Total_Treatment_Duration", "Follow_up(months)" -> "Follow_up_Duration").
- After merging, auto-detect any variant of an observed-toxicity column and
  create a clean numeric 'Observed_Toxicity' (0/1).
- Guard all analyses to proceed only if 'Observed_Toxicity' exists; otherwise,
  skip toxicity‑dependent steps with a clear console notice.
- Make correlation matrix construction and factor scans resilient to missing columns.

Author: K. Mondal (North Bengal Medical College, Darjeeling, India.)
Version: 1.1-patched
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import scipy.stats as stats
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from scipy.stats import chi2_contingency, mannwhitneyu, kruskal
import warnings
import sys
warnings.filterwarnings('ignore')

# Windows-safe encoding configuration
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except (AttributeError, ValueError):
    pass

# Set publication-quality plotting parameters
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.8,
    'legend.frameon': False,
    'legend.fontsize': 10,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.minor.size': 3,
    'ytick.minor.size': 3,
    'lines.linewidth': 2.5,
    'lines.markersize': 6,
    'figure.dpi': 100,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'savefig.facecolor': 'white'
})

# Professional color scheme
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#F24236', 
    'tertiary': '#F6AE2D',
    'quaternary': '#8B4B9E',
    'quinary': '#2ECC71',
    'observed': '#C73E1D',
    'predicted': '#592E83',
    'correlation_pos': '#27AE60',
    'correlation_neg': '#E74C3C',
    'neutral': '#95A5A6'
}

# ----------------------- Helper Utilities (NEW) -----------------------

def _strip_and_lower(s: str) -> str:
    return s.strip().lower().replace('\xa0', ' ') if isinstance(s, str) else s

def _standardize_columns_inplace(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize common clinical column names to the internal expected set.
    Performs case-insensitive matching and renaming where appropriate.
    """
    if df is None or df.empty:
        return df

    # Build a mapping by scanning for known variants
    col_map = {}
    cols_lower = {c.lower(): c for c in df.columns}

    def has(name_variants):
        for v in name_variants:
            if v.lower() in cols_lower:
                return cols_lower[v.lower()]
        return None

    # Expected internal names -> variants we may see
    rename_plan = {
        'Treatment_Technique': ['Treatment_Technique', 'Technique', 'Tx_Technique'],
        'Dose_per_Fraction': ['Dose_per_Fraction', 'DosePerFraction(Gy)', 'DosePerFraction', 'Dose/Fraction(Gy)', 'Dose/Fraction', 'DosePerFx(Gy)'],
        'Total_Dose': ['Total_Dose', 'Total_Dose(Gy)', 'TotalDose(Gy)', 'TotalDose'],
        'Total_Treatment_Duration': ['Total_Treatment_Duration', 'Duration(wk)', 'Treatment_Duration(weeks)', 'Duration_weeks'],
        'Follow_up_Duration': ['Follow_up_Duration', 'Follow_up(months)', 'FollowUp(months)', 'Followup_Months'],
        'Age': ['Age'],
        'Sex': ['Sex', 'Gender'],
        'Diagnosis': ['Diagnosis', 'Dx']
    }

    for std, variants in rename_plan.items():
        found = has(variants)
        if found and found != std:
            col_map[found] = std

    if col_map:
        df.rename(columns=col_map, inplace=True)

    return df

def _coerce_observed_toxicity(series: pd.Series) -> pd.Series:
    """
    Convert various encodings of observed toxicity to a binary numeric series (0/1).
    Accepts numeric 0/1, booleans, and common string encodings like 'yes/no', 'true/false', etc.
    Non-matching values will be coerced to NaN, then filled with 0 by default.
    """
    if series is None:
        return None

    # If already numeric 0/1
    if pd.api.types.is_numeric_dtype(series):
        # Coerce to 0/1 explicitly (handle floats like 0.0/1.0)
        return series.astype(float).round().clip(lower=0, upper=1).astype(int)

    # Map common strings to {0,1}
    mapping = {
        'yes': 1, 'y': 1, 'true': 1, 't': 1, '1': 1, 'present': 1, 'toxic': 1,
        'grade>=2': 1, 'grade≥2': 1, 'g2+': 1, '>=g2': 1, 'event': 1, 'positive': 1,
        'no': 0, 'n': 0, 'false': 0, 'f': 0, '0': 0, 'absent': 0, 'non-toxic': 0, 'negative': 0
    }

    s = series.astype(str).str.strip().str.lower()
    mapped = s.map(mapping)
    # For anything unmapped, try to parse numbers, else NaN
    unmapped = mapped.isna()
    if unmapped.any():
        # Try to coerce to numeric
        numeric = pd.to_numeric(s[unmapped], errors='coerce')
        mapped.loc[unmapped & numeric.notna()] = numeric.loc[unmapped & numeric.notna()].round().clip(lower=0, upper=1)
    # Fill remaining NaN with 0 (conservative)
    mapped = mapped.fillna(0).astype(int)
    return mapped

def _ensure_observed_toxicity_column(df: pd.DataFrame, verbose_prefix="") -> pd.DataFrame:
    """
    Detect any column that semantically represents observed toxicity and create a unified
    numeric 'Observed_Toxicity' column. Handles merge suffixes (_x/_y) and spacing/underscore variants.
    """
    if df is None or df.empty:
        return df

    candidates = []
    for c in df.columns:
        lc = c.lower().replace(' ', '_')
        if 'observed' in lc and 'tox' in lc:
            candidates.append(c)
        elif lc in ('toxicity', 'observed_toxicity', 'observedtoxicity', 'toxicity_observed'):
            candidates.append(c)
        elif lc.endswith('_x') or lc.endswith('_y'):
            base = lc[:-2]
            if base in ('observed_toxicity', 'observedtoxicity', 'toxicity_observed', 'toxicity'):
                candidates.append(c)

    # Prefer exact 'Observed_Toxicity' if present
    preferred = [c for c in candidates if c == 'Observed_Toxicity']
    selected = preferred[0] if preferred else (candidates[0] if candidates else None)

    if selected is None:
        print(f"{verbose_prefix}Warning: No observed-toxicity column found after merge. "
              f"Looking for common alternatives failed. Skipping toxicity-based analyses.")
        return df

    df['Observed_Toxicity'] = _coerce_observed_toxicity(df[selected])
    return df

# ----------------------- Main Analyzer Class (patched) -----------------------

class ClinicalFactorsAnalyzer:
    """Analyze clinical factors effects on NTCP predictions and observed toxicity"""

    def __init__(self, input_file, enhanced_output_dir, output_dir=None):
        self.input_file = Path(input_file)
        self.enhanced_output_dir = Path(enhanced_output_dir)
        self.output_dir = Path(output_dir) if output_dir else (self.enhanced_output_dir / 'clinical_factors_analysis')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        self.clinical_data = None
        self.ntcp_results = None
        self.merged_data = None

    def load_and_merge_data(self):
        """Load clinical factors and NTCP results, then merge them"""

        print(" Loading and merging clinical data with NTCP results...")

        # Load clinical factors from input file
        try:
            self.clinical_data = pd.read_excel(self.input_file)
            # Standardize clinical columns (NEW)
            _standardize_columns_inplace(self.clinical_data)
            print(f" Loaded clinical data: {len(self.clinical_data)} patient-organ combinations")
            print(f"📋 Available clinical factors (standardized): {list(self.clinical_data.columns)}")
        except Exception as e:
            print(f"Error: Error loading clinical data: {e}")
            return False

        # Load NTCP results from enhanced analysis
        ntcp_file = self.enhanced_output_dir / 'enhanced_ntcp_calculations.csv'
        try:
            self.ntcp_results = pd.read_csv(ntcp_file)
            print(f" Loaded NTCP results: {len(self.ntcp_results)} patient-organ combinations")

            # Count unique patients - identity-safe: use PrimaryPatientID
            if 'PrimaryPatientID' in self.ntcp_results.columns:
                unique_patients = self.ntcp_results['PrimaryPatientID'].nunique()
                print(f" Unique patients (PrimaryPatientID): {unique_patients}")
            else:
                print("Warning: 'PrimaryPatientID' column missing in NTCP results; unique patient count unavailable.")

        except Exception as e:
            print(f"Error: Error loading NTCP results: {e}")
            return False

        # Merge datasets on PrimaryPatientID + Organ (identity-safe matching)
        try:
            # Identity-safe: use PrimaryPatientID for matching
            # Handle backward compatibility for files with only PatientID/DVH_ID
            
            # Check clinical data for PrimaryPatientID
            has_primary_clinical = 'PrimaryPatientID' in self.clinical_data.columns
            has_primary_ntcp = 'PrimaryPatientID' in self.ntcp_results.columns
            
            # Map legacy columns to PrimaryPatientID if needed
            if not has_primary_clinical:
                if 'DVH_ID' in self.clinical_data.columns:
                    self.clinical_data['PrimaryPatientID'] = self.clinical_data['DVH_ID']
                    has_primary_clinical = True
                elif 'patient_id' in self.clinical_data.columns:
                    # Code0 reconciled output (Clinical Contract v2)
                    self.clinical_data['PrimaryPatientID'] = self.clinical_data['patient_id']
                    has_primary_clinical = True
                elif 'PatientID' in self.clinical_data.columns:
                    # Check if PatientID looks like AnonPatientID (PT format)
                    sample_pid = self.clinical_data['PatientID'].dropna().iloc[0] if len(self.clinical_data['PatientID'].dropna()) > 0 else ""
                    if isinstance(sample_pid, str) and sample_pid.strip().upper().startswith('PT'):
                        # PatientID is AnonPatientID - map via Step1 registry
                        if 'Organ' in self.clinical_data.columns:
                            try:
                                from contract_validator import ContractValidator
                                from pathlib import Path
                                contracts_dir = Path(self.input_file).parent.parent / "contracts"
                                validator = ContractValidator(contracts_dir)
                                registry_df = validator.load_step1_registry()
                                if registry_df is not None and 'AnonPatientID' in registry_df.columns:
                                    mapping_df = registry_df[['PrimaryPatientID', 'AnonPatientID', 'Organ']].drop_duplicates()
                                    self.clinical_data = pd.merge(
                                        self.clinical_data,
                                        mapping_df,
                                        left_on=['PatientID', 'Organ'],
                                        right_on=['AnonPatientID', 'Organ'],
                                        how='left',
                                        suffixes=('', '_mapped')
                                    )
                                    if 'PrimaryPatientID_mapped' in self.clinical_data.columns:
                                        self.clinical_data['PrimaryPatientID'] = self.clinical_data['PrimaryPatientID_mapped']
                                        self.clinical_data = self.clinical_data.drop(columns=['PrimaryPatientID_mapped'], errors='ignore')
                                        has_primary_clinical = self.clinical_data['PrimaryPatientID'].notna().any()
                            except Exception as e:
                                print(f"Warning: Could not map PatientID to PrimaryPatientID via registry: {e}")
                        if not has_primary_clinical:
                            raise ValueError("Clinical data contains PatientID (PT format) but could not map to PrimaryPatientID via Step1_DVHRegistry. Ensure Step1 registry exists.")
                    else:
                        # PatientID is PrimaryPatientID format
                        self.clinical_data['PrimaryPatientID'] = self.clinical_data['PatientID']
                        has_primary_clinical = True
                else:
                    raise ValueError("Clinical data must contain PrimaryPatientID, DVH_ID, patient_id, or PatientID")
            
            # NTCP results should already have PrimaryPatientID (from code3)
            if not has_primary_ntcp:
                raise ValueError("NTCP results must contain PrimaryPatientID. Run code3_ntcp_analysis_ml.py first.")
            
            # Verify PrimaryPatientID exists in both datasets
            if not has_primary_clinical or not has_primary_ntcp:
                print(f"Error: PrimaryPatientID not found in both datasets.")
                print(f"Clinical data has PrimaryPatientID: {has_primary_clinical}")
                print(f"NTCP results has PrimaryPatientID: {has_primary_ntcp}")
                print(f"Clinical data columns: {list(self.clinical_data.columns)}")
                print(f"NTCP results columns: {list(self.ntcp_results.columns)}")
                return False
            
            # Verify Organ column exists
            if 'Organ' not in self.clinical_data.columns or 'Organ' not in self.ntcp_results.columns:
                print(f"Error: Organ column missing in one or both datasets.")
                return False
            
            # Identity-safe merge: use PrimaryPatientID + Organ
            merge_on = ['PrimaryPatientID', 'Organ']
            print(f"Using PrimaryPatientID + Organ for merging (identity-safe key)")
            
            # Perform merge on PrimaryPatientID + Organ
            self.merged_data = pd.merge(
                self.clinical_data, 
                self.ntcp_results, 
                on=merge_on,
                how='inner',
                suffixes=('_clin', '_ntcp')
            )
            
            # Sanity check: warn if more than 10% of records dropped
            original_count = max(len(self.clinical_data), len(self.ntcp_results))
            merged_count = len(self.merged_data)
            drop_rate = 1.0 - (merged_count / original_count) if original_count > 0 else 0
            if drop_rate > 0.10:
                print(f"\n[WARNING] {drop_rate:.1%} of records were dropped during merge ({merged_count}/{original_count}).")
                print("This may indicate a mismatch between PrimaryPatientID in datasets.")

            # Ensure Observed_Toxicity exists and is numeric (NEW)
            self.merged_data = _ensure_observed_toxicity_column(self.merged_data, verbose_prefix=" ")
            
            # PART 2 FIX: Explicitly define observed_event from xerostomia_grade2plus
            if 'xerostomia_grade2plus' in self.merged_data.columns:
                self.merged_data['observed_event'] = self.merged_data['xerostomia_grade2plus'].astype(int)
                # Also update Observed_Toxicity if it exists
                if 'Observed_Toxicity' in self.merged_data.columns:
                    self.merged_data['Observed_Toxicity'] = self.merged_data['observed_event']
                else:
                    self.merged_data['Observed_Toxicity'] = self.merged_data['observed_event']
                print("  [FIX] Using xerostomia_grade2plus as observed_event for correlation analysis")
            elif 'Observed_Toxicity' in self.merged_data.columns:
                # Fallback to Observed_Toxicity if xerostomia_grade2plus not found
                self.merged_data['observed_event'] = self.merged_data['Observed_Toxicity'].astype(int)
                print("  [INFO] Using Observed_Toxicity as observed_event (xerostomia_grade2plus not found)")
            else:
                print("  [WARNING] Neither xerostomia_grade2plus nor Observed_Toxicity found - correlation may fail")
            
            # Enforce identity contract after merging
            try:
                from contract_validator import enforce_identity_contract
                enforce_identity_contract(self.merged_data)
                print(" [VALIDATED] Identity contract enforced: PrimaryPatientID+Organ unique and non-null")
            except Exception as e:
                raise ValueError(f"Identity governance failed after merge: {e}")
            
            print(f" Successfully merged data: {len(self.merged_data)} records")
            if 'PrimaryPatientID' in self.merged_data.columns:
                print(f" Unique patients in merged data (PrimaryPatientID): {self.merged_data['PrimaryPatientID'].nunique()}")

            # Display organs distribution
            if 'Organ' in self.merged_data.columns:
                organ_counts = self.merged_data['Organ'].value_counts()
                print(f"📋 Organ distribution:")
                for organ, count in organ_counts.items():
                    print(f"  {organ}: {count} cases")

            return True

        except Exception as e:
            print(f"Error: Error merging data: {e}")
            return False

    def analyze_categorical_factors(self):
        """Analyze categorical factors (Diagnosis, Treatment Techniques, Sex)"""

        print("\n Analyzing Categorical Clinical Factors...")

        if 'Observed_Toxicity' not in self.merged_data.columns:
            print("Warning: Skipping categorical analysis: 'Observed_Toxicity' not available.")
            return {}

        categorical_factors = []

        # Identify categorical columns using standardized names first
        for col in ['Diagnosis', 'Treatment_Technique', 'Sex']:
            if col in self.merged_data.columns:
                categorical_factors.append(col)
        # Fallbacks if standard names absent
        for alt in ['Gender', 'Technique']:
            if alt in self.merged_data.columns and alt not in categorical_factors:
                categorical_factors.append(alt)

        if not categorical_factors:
            print("Warning: No categorical factors found in the data")
            return {}

        print(f" Analyzing categorical factors: {categorical_factors}")

        results = {}

        for factor in categorical_factors:
            print(f"\n📈 Analyzing {factor}...")

            factor_results = {
                'factor_name': factor,
                'categories': {},
                'statistical_tests': {},
                'ntcp_model_effects': {}
            }

            # Get unique categories
            categories = self.merged_data[factor].dropna().unique()
            print(f"  Categories: {list(categories)}")

            # Analyze each category
            for category in categories:
                category_data = self.merged_data[self.merged_data[factor] == category]

                category_stats = {
                    'n_cases': len(category_data),
                    'n_patients': category_data['PrimaryPatientID'].nunique() if 'PrimaryPatientID' in category_data.columns else len(category_data),
                    'observed_toxicity_rate': category_data['Observed_Toxicity'].mean() if 'Observed_Toxicity' in category_data.columns else np.nan,
                    'organs': category_data['Organ'].value_counts().to_dict() if 'Organ' in category_data.columns else {}
                }

                factor_results['categories'][category] = category_stats
                tox_rate = category_stats['observed_toxicity_rate']
                tox_txt = f"{tox_rate:.3f}" if pd.notna(tox_rate) else "NA"
                print(f"    {category}: {category_stats['n_cases']} cases, {category_stats['n_patients']} patients, toxicity rate: {tox_txt}")

            # Statistical tests for observed toxicity
            if len(categories) >= 2:
                try:
                    # Chi-square test for observed toxicity vs factor
                    contingency_table = pd.crosstab(
                        self.merged_data[factor].fillna('Missing'), 
                        self.merged_data['Observed_Toxicity']
                    )

                    if contingency_table.shape[1] == 2:  # Ensure binary
                        chi2, p_value, dof, expected = chi2_contingency(contingency_table)

                        factor_results['statistical_tests']['chi_square'] = {
                            'chi2': chi2,
                            'p_value': p_value,
                            'degrees_of_freedom': dof,
                            'significant': p_value < 0.05
                        }

                        print(f"    Chi-square test: χ² = {chi2:.3f}, p = {p_value:.4f}")
                    else:
                        print("    Warning: Chi-square skipped: Observed_Toxicity is not binary after coercion.")

                except Exception as e:
                    print(f"    Warning: Chi-square test failed: {e}")

            # Analyze NTCP model predictions by factor
            ntcp_cols = [col for col in self.merged_data.columns if col.startswith('NTCP_')]

            for ntcp_col in ntcp_cols:
                if ntcp_col in self.merged_data.columns:
                    model_name = ntcp_col.replace('NTCP_', '')

                    # Calculate mean NTCP by category
                    category_ntcp_means = {}
                    for category in categories:
                        category_data = self.merged_data[self.merged_data[factor] == category]
                        mean_ntcp = category_data[ntcp_col].mean()
                        category_ntcp_means[category] = mean_ntcp

                    factor_results['ntcp_model_effects'][model_name] = category_ntcp_means

            results[factor] = factor_results

        # Save categorical analysis results
        self._save_categorical_results(results)
        self._plot_categorical_analysis(results)

        return results

    def analyze_continuous_factors(self):
        """Analyze continuous factors (Age, Dose per Fraction, Total Dose, etc.)"""

        print("\n Analyzing Continuous Clinical Factors...")

        if 'Observed_Toxicity' not in self.merged_data.columns:
            print("Warning: Skipping continuous analysis: 'Observed_Toxicity' not available.")
            return {}

        # Identify continuous factors
        continuous_factors = []
        potential_continuous = [
            'Age', 'Dose_per_Fraction', 'Total_Dose', 'Total_Treatment_Duration', 
            'Follow_up_Duration', 'age', 'dose_per_fraction', 'total_dose'
        ]

        for col in potential_continuous:
            if col in self.merged_data.columns and pd.api.types.is_numeric_dtype(self.merged_data[col]):
                continuous_factors.append(col)

        if not continuous_factors:
            print("Warning: No continuous factors found in the data")
            return {}

        print(f" Analyzing continuous factors: {continuous_factors}")

        results = {}

        for factor in continuous_factors:
            print(f"\n📈 Analyzing {factor}...")

            factor_data = self.merged_data[factor].dropna()

            factor_results = {
                'factor_name': factor,
                'descriptive_stats': {
                    'count': len(factor_data),
                    'mean': float(factor_data.mean()) if len(factor_data) else np.nan,
                    'std': float(factor_data.std()) if len(factor_data) else np.nan,
                    'min': float(factor_data.min()) if len(factor_data) else np.nan,
                    'max': float(factor_data.max()) if len(factor_data) else np.nan,
                    'median': float(factor_data.median()) if len(factor_data) else np.nan,
                    'q25': float(factor_data.quantile(0.25)) if len(factor_data) else np.nan,
                    'q75': float(factor_data.quantile(0.75)) if len(factor_data) else np.nan
                },
                'correlations': {},
                'group_comparisons': {}
            }

            print(f"  Descriptive stats: mean={factor_results['descriptive_stats']['mean']:.2f}, "
                  f"std={factor_results['descriptive_stats']['std']:.2f}, range=[{factor_results['descriptive_stats']['min']:.2f}, {factor_results['descriptive_stats']['max']:.2f}]")

            # Correlation with observed toxicity - use observed_event explicitly
            event_col = 'observed_event' if 'observed_event' in self.merged_data.columns else 'Observed_Toxicity'
            if event_col not in self.merged_data.columns:
                print(f"  Warning: {event_col} not found, skipping correlation")
                continue
                
            valid_data = self.merged_data[[factor, event_col]].dropna()
            
            # PART 2 FIX: Validate observed_event is binary and not NaN
            if valid_data[event_col].isna().any():
                print(f"  Warning: {event_col} contains NaN values, removing them")
                valid_data = valid_data[valid_data[event_col].notna()]
            
            unique_vals = valid_data[event_col].dropna().unique()
            if not all(v in [0, 1] for v in unique_vals):
                print(f"  Warning: {event_col} contains non-binary values: {unique_vals}, skipping correlation")
                continue

            if len(valid_data) > 10:
                # Point-biserial correlation (continuous vs binary)
                try:
                    correlation_coef, correlation_p = stats.pointbiserialr(
                        valid_data[event_col].astype(float), 
                        valid_data[factor].astype(float)
                    )
                    factor_results['correlations']['observed_toxicity'] = {
                        'correlation': float(correlation_coef),
                        'p_value': float(correlation_p),
                        'significant': bool(correlation_p < 0.05)
                    }
                    print(f"  Correlation with observed toxicity: r = {correlation_coef:.3f}, p = {correlation_p:.4f}")
                except Exception as e:
                    print(f"  Warning: Correlation computation failed for {factor}: {e}")

            # Correlations with NTCP model predictions
            ntcp_cols = [col for col in self.merged_data.columns if col.startswith('NTCP_')]

            for ntcp_col in ntcp_cols:
                if ntcp_col in self.merged_data.columns:
                    model_name = ntcp_col.replace('NTCP_', '')

                    valid_data = self.merged_data[[factor, ntcp_col]].dropna()

                    if len(valid_data) > 10:
                        try:
                            corr_coef, corr_p = stats.pearsonr(valid_data[factor].astype(float), valid_data[ntcp_col].astype(float))
                            factor_results['correlations'][model_name] = {
                                'correlation': float(corr_coef),
                                'p_value': float(corr_p),
                                'significant': bool(corr_p < 0.05)
                            }
                        except Exception as e:
                            pass

            # Group comparisons (toxicity vs no toxicity) - use observed_event
            event_col = 'observed_event' if 'observed_event' in self.merged_data.columns else 'Observed_Toxicity'
            if event_col in self.merged_data.columns:
                toxicity_group = self.merged_data[self.merged_data[event_col] == 1][factor].dropna()
                no_toxicity_group = self.merged_data[self.merged_data[event_col] == 0][factor].dropna()
            else:
                toxicity_group = pd.Series(dtype=float)
                no_toxicity_group = pd.Series(dtype=float)

            if len(toxicity_group) > 0 and len(no_toxicity_group) > 0:
                try:
                    # Mann-Whitney U test (non-parametric)
                    statistic, p_value = mannwhitneyu(toxicity_group, no_toxicity_group, alternative='two-sided')

                    effect_size = np.nan
                    std_all = self.merged_data[factor].std()
                    if pd.notna(std_all) and std_all != 0:
                        effect_size = (toxicity_group.mean() - no_toxicity_group.mean()) / std_all

                    factor_results['group_comparisons']['toxicity_vs_no_toxicity'] = {
                        'toxicity_group_mean': float(toxicity_group.mean()),
                        'no_toxicity_group_mean': float(no_toxicity_group.mean()),
                        'mann_whitney_u': float(statistic),
                        'p_value': float(p_value),
                        'significant': bool(p_value < 0.05),
                        'effect_size': float(effect_size) if pd.notna(effect_size) else np.nan
                    }

                    print(f"  Group comparison (toxicity vs no toxicity):")
                    print(f"    Toxicity group mean: {toxicity_group.mean():.2f}")
                    print(f"    No toxicity group mean: {no_toxicity_group.mean():.2f}")
                    print(f"    Mann-Whitney U test: p = {p_value:.4f}")
                except Exception as e:
                    print(f"  Warning: Mann-Whitney U failed for {factor}: {e}")

            results[factor] = factor_results

        # Save continuous analysis results
        self._save_continuous_results(results)
        self._plot_continuous_analysis(results)

        return results

    def analyze_organ_specific_effects(self):
        """Analyze how clinical factors affect different organs"""

        print("\n Analyzing Organ-Specific Effects...")

        if 'Organ' not in self.merged_data.columns:
            print("Warning: Skipping organ-specific analysis: 'Organ' column not available.")
            return {}
        if 'Observed_Toxicity' not in self.merged_data.columns:
            print("Warning: Skipping organ-specific analysis: 'Observed_Toxicity' not available.")
            return {}

        organs = self.merged_data['Organ'].unique()
        results = {}

        for organ in organs:
            print(f"\n Analyzing {organ}...")

            organ_data = self.merged_data[self.merged_data['Organ'] == organ].copy()

            if len(organ_data) < 10:
                print(f"  Warning: Insufficient data for {organ} ({len(organ_data)} cases)")
                continue

            organ_results = {
                'organ_name': organ,
                'sample_size': int(len(organ_data)),
                'toxicity_rate': float(organ_data['Observed_Toxicity'].mean()),
                'factor_effects': {}
            }

            print(f"  Sample size: {len(organ_data)}, Toxicity rate: {organ_results['toxicity_rate']:.3f}")

            # Analyze each clinical factor for this organ
            clinical_factors = []
            for col in organ_data.columns:
                if col in ['Diagnosis', 'Treatment_Technique', 'Sex', 'Age', 'Dose_per_Fraction', 
                          'Total_Dose', 'Total_Treatment_Duration', 'Follow_up_Duration']:
                    clinical_factors.append(col)

            for factor in clinical_factors:
                if factor not in organ_data.columns:
                    continue

                factor_data = organ_data[factor].dropna()

                if len(factor_data) < 5:
                    continue

                factor_effect = {
                    'factor_name': factor,
                    'data_type': 'categorical' if organ_data[factor].dtype == 'object' else 'continuous'
                }

                if factor_effect['data_type'] == 'categorical':
                    # Categorical factor analysis
                    categories = factor_data.unique()

                    if len(categories) >= 2:
                        category_effects = {}

                        for category in categories:
                            category_subset = organ_data[organ_data[factor] == category]
                            category_effects[category] = {
                                'n_cases': int(len(category_subset)),
                                'toxicity_rate': float(category_subset['Observed_Toxicity'].mean())
                            }

                        factor_effect['category_effects'] = category_effects

                else:
                    # Continuous factor analysis
                    valid_data = organ_data[[factor, 'Observed_Toxicity']].dropna()

                    if len(valid_data) > 5:
                        try:
                            correlation_coef, correlation_p = stats.pointbiserialr(
                                valid_data['Observed_Toxicity'].astype(float), 
                                valid_data[factor].astype(float)
                            )
                            factor_effect['correlation_with_toxicity'] = {
                                'correlation': float(correlation_coef),
                                'p_value': float(correlation_p),
                                'significant': bool(correlation_p < 0.05)
                            }
                        except Exception as e:
                            pass

                organ_results['factor_effects'][factor] = factor_effect

            results[organ] = organ_results

        # Save organ-specific results
        self._save_organ_specific_results(results)
        self._plot_organ_specific_analysis(results)

        return results

    def create_correlation_matrix(self):
        """Create correlation matrix for all factors and NTCP predictions"""

        print("\n Creating Comprehensive Correlation Matrix...")

        # Select numerical columns for correlation
        numerical_cols = []

        # Clinical factors
        for col in ['Age', 'Dose_per_Fraction', 'Total_Dose', 'Total_Treatment_Duration', 
                   'Follow_up_Duration', 'age', 'dose_per_fraction', 'total_dose']:
            if col in self.merged_data.columns and pd.api.types.is_numeric_dtype(self.merged_data[col]):
                numerical_cols.append(col)

        # Observed toxicity
        if 'Observed_Toxicity' in self.merged_data.columns:
            numerical_cols.append('Observed_Toxicity')

        # NTCP predictions
        ntcp_cols = [col for col in self.merged_data.columns if col.startswith('NTCP_')]
        numerical_cols.extend([c for c in ntcp_cols if pd.api.types.is_numeric_dtype(self.merged_data[c])])

        # Dose metrics (optional; include only if present and numeric)
        dose_cols = ['gEUD', 'mean_dose', 'max_dose', 'total_volume']
        for col in dose_cols:
            if col in self.merged_data.columns and pd.api.types.is_numeric_dtype(self.merged_data[col]):
                numerical_cols.append(col)

        if not numerical_cols:
            print("Warning: No numerical columns available for correlation matrix.")
            return pd.DataFrame()

        # Create correlation matrix
        correlation_data = self.merged_data[numerical_cols].copy()
        correlation_matrix = correlation_data.corr()

        # Create comprehensive correlation plot
        fig, ax = plt.subplots(figsize=(14, 12))

        # Create heatmap
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax,
                   fmt='.3f', annot_kws={'size': 8})

        ax.set_title('Correlation Matrix: Clinical Factors, Dose Metrics, and NTCP Predictions', 
                    fontsize=14, fontweight='bold', pad=20)

        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        # Save correlation matrix
        correlation_file = self.output_dir / 'correlation_matrix.png'
        plt.savefig(correlation_file, dpi=600, bbox_inches='tight')
        print(f" Correlation matrix saved: {correlation_file}")
        plt.close()

        # Save correlation matrix as CSV
        correlation_matrix.to_csv(self.output_dir / 'correlation_matrix.csv')

        return correlation_matrix

    # ----------------------- Save/Plot helpers (unchanged) -----------------------
    def _save_categorical_results(self, results):
        """Save categorical analysis results to Excel"""

        excel_file = self.output_dir / 'categorical_factors_analysis.xlsx'

        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:

            # Summary sheet
            summary_data = []
            for factor, factor_results in results.items():
                for category, stats_d in factor_results['categories'].items():
                    summary_data.append({
                        'Factor': factor,
                        'Category': category,
                        'N_Cases': stats_d['n_cases'],
                        'N_Patients': stats_d['n_patients'],
                        'Toxicity_Rate': stats_d['observed_toxicity_rate']
                    })

            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)

            # Statistical tests sheet
            stats_data = []
            for factor, factor_results in results.items():
                if 'chi_square' in factor_results['statistical_tests']:
                    chi_sq = factor_results['statistical_tests']['chi_square']
                    stats_data.append({
                        'Factor': factor,
                        'Test': 'Chi-square',
                        'Statistic': chi_sq['chi2'],
                        'P_Value': chi_sq['p_value'],
                        'Significant': chi_sq['significant']
                    })

            if stats_data:
                stats_df = pd.DataFrame(stats_data)
                stats_df.to_excel(writer, sheet_name='Statistical_Tests', index=False)

        print(f" Categorical analysis saved: {excel_file}")

    def _save_continuous_results(self, results):
        """Save continuous analysis results to Excel"""

        excel_file = self.output_dir / 'continuous_factors_analysis.xlsx'

        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:

            # Descriptive statistics
            desc_data = []
            for factor, factor_results in results.items():
                stats_d = factor_results['descriptive_stats']
                desc_data.append({
                    'Factor': factor,
                    'Count': stats_d['count'],
                    'Mean': stats_d['mean'],
                    'Std': stats_d['std'],
                    'Min': stats_d['min'],
                    'Max': stats_d['max'],
                    'Median': stats_d['median'],
                    'Q25': stats_d['q25'],
                    'Q75': stats_d['q75']
                })

            desc_df = pd.DataFrame(desc_data)
            desc_df.to_excel(writer, sheet_name='Descriptive_Stats', index=False)

            # Correlations
            corr_data = []
            for factor, factor_results in results.items():
                for target, corr_stats in factor_results['correlations'].items():
                    corr_data.append({
                        'Factor': factor,
                        'Target': target,
                        'Correlation': corr_stats['correlation'],
                        'P_Value': corr_stats['p_value'],
                        'Significant': corr_stats['significant']
                    })

            if corr_data:
                corr_df = pd.DataFrame(corr_data)
                corr_df.to_excel(writer, sheet_name='Correlations', index=False)

            # Group comparisons
            group_data = []
            for factor, factor_results in results.items():
                if 'toxicity_vs_no_toxicity' in factor_results['group_comparisons']:
                    comp = factor_results['group_comparisons']['toxicity_vs_no_toxicity']
                    group_data.append({
                        'Factor': factor,
                        'Toxicity_Group_Mean': comp['toxicity_group_mean'],
                        'No_Toxicity_Group_Mean': comp['no_toxicity_group_mean'],
                        'Mann_Whitney_U': comp['mann_whitney_u'],
                        'P_Value': comp['p_value'],
                        'Significant': comp['significant'],
                        'Effect_Size': comp['effect_size']
                    })

            if group_data:
                group_df = pd.DataFrame(group_data)
                group_df.to_excel(writer, sheet_name='Group_Comparisons', index=False)

        print(f" Continuous analysis saved: {excel_file}")

    def _save_organ_specific_results(self, results):
        """Save organ-specific analysis results"""

        excel_file = self.output_dir / 'organ_specific_analysis.xlsx'

        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:

            # Summary by organ
            summary_data = []
            for organ, organ_results in results.items():
                summary_data.append({
                    'Organ': organ,
                    'Sample_Size': organ_results['sample_size'],
                    'Toxicity_Rate': organ_results['toxicity_rate']
                })

            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Organ_Summary', index=False)

            # Factor effects by organ
            effect_data = []
            for organ, organ_results in results.items():
                for factor, factor_effect in organ_results['factor_effects'].items():

                    if factor_effect['data_type'] == 'continuous':
                        if 'correlation_with_toxicity' in factor_effect:
                            corr = factor_effect['correlation_with_toxicity']
                            effect_data.append({
                                'Organ': organ,
                                'Factor': factor,
                                'Data_Type': 'Continuous',
                                'Correlation': corr['correlation'],
                                'P_Value': corr['p_value'],
                                'Significant': corr['significant']
                            })

                    elif factor_effect['data_type'] == 'categorical':
                        if 'category_effects' in factor_effect:
                            for category, cat_effect in factor_effect['category_effects'].items():
                                effect_data.append({
                                    'Organ': organ,
                                    'Factor': factor,
                                    'Data_Type': 'Categorical',
                                    'Category': category,
                                    'N_Cases': cat_effect['n_cases'],
                                    'Toxicity_Rate': cat_effect['toxicity_rate']
                                })

            if effect_data:
                effect_df = pd.DataFrame(effect_data)
                effect_df.to_excel(writer, sheet_name='Factor_Effects', index=False)

        print(f" Organ-specific analysis saved: {excel_file}")

    def _plot_categorical_analysis(self, results):
        """Create plots for categorical factors analysis"""

        if not results:
            return

        for factor, factor_results in results.items():
            # Create subplot for this factor
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Categorical Factor Analysis: {factor}', fontsize=16, fontweight='bold')

            # Plot 1: Toxicity rates by category
            categories = list(factor_results['categories'].keys())
            toxicity_rates = [factor_results['categories'][cat]['observed_toxicity_rate'] 
                            for cat in categories]

            bars = ax1.bar(categories, toxicity_rates, color=COLORS['primary'], alpha=0.7)
            ax1.set_title('Observed Toxicity Rate by Category')
            ax1.set_ylabel('Toxicity Rate')
            ax1.set_xlabel(factor)

            # Add value labels
            for bar, rate in zip(bars, toxicity_rates):
                if pd.notna(rate):
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{rate:.3f}', ha='center', va='bottom')

            ax1.tick_params(axis='x', rotation=45)

            # Plot 2: Sample sizes by category
            sample_sizes = [factor_results['categories'][cat]['n_cases'] for cat in categories]

            bars = ax2.bar(categories, sample_sizes, color=COLORS['secondary'], alpha=0.7)
            ax2.set_title('Sample Size by Category')
            ax2.set_ylabel('Number of Cases')
            ax2.set_xlabel(factor)

            # Add value labels
            for bar, size in zip(bars, sample_sizes):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{size}', ha='center', va='bottom')

            ax2.tick_params(axis='x', rotation=45)

            # Plot 3: NTCP predictions by category (if available)
            ntcp_models = list(factor_results['ntcp_model_effects'].keys())

            if ntcp_models:
                # Take first NTCP model for visualization
                model = ntcp_models[0]
                ntcp_means = [factor_results['ntcp_model_effects'][model].get(cat, np.nan) 
                            for cat in categories]

                bars = ax3.bar(categories, ntcp_means, color=COLORS['tertiary'], alpha=0.7)
                ax3.set_title(f'Mean {model} NTCP by Category')
                ax3.set_ylabel('Mean NTCP')
                ax3.set_xlabel(factor)

                # Add value labels
                for bar, mean_ntcp in zip(bars, ntcp_means):
                    if pd.notna(mean_ntcp):
                        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                                f'{mean_ntcp:.3f}', ha='center', va='bottom')

                ax3.tick_params(axis='x', rotation=45)

            # Plot 4: Statistical significance
            if 'chi_square' in factor_results['statistical_tests']:
                chi_sq = factor_results['statistical_tests']['chi_square']

                # Create text summary
                significance_text = f"""
Statistical Test Results:
Chi-square test: χ² = {chi_sq['chi2']:.3f}
p-value = {chi_sq['p_value']:.4f}
Significant: {chi_sq['significant']}
Degrees of freedom: {chi_sq['degrees_of_freedom']}
                """

                ax4.text(0.1, 0.5, significance_text, transform=ax4.transAxes, 
                        fontsize=12, verticalalignment='center',
                        bbox=dict(boxstyle='round,pad=0.5', 
                                facecolor='lightblue' if chi_sq['significant'] else 'lightgray',
                                alpha=0.8))
                ax4.set_title('Statistical Significance')
                ax4.axis('off')
            else:
                ax4.axis('off')

            plt.tight_layout()

            # Save plot
            plot_file = self.output_dir / f'categorical_analysis_{factor}.png'
            plt.savefig(plot_file, dpi=600, bbox_inches='tight')
            print(f" Categorical plot saved: {plot_file}")
            plt.close()

    def _plot_continuous_analysis(self, results):
        """Create plots for continuous factors analysis"""

        if not results:
            return

        for factor, factor_results in results.items():
            # Create subplot for this factor
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Continuous Factor Analysis: {factor}', fontsize=16, fontweight='bold')

            # Plot 1: Distribution of factor
            factor_data = self.merged_data[factor].dropna()

            ax1.hist(factor_data, bins=20, color=COLORS['primary'], alpha=0.7, edgecolor='black')
            ax1.set_title(f'Distribution of {factor}')
            ax1.set_xlabel(factor)
            ax1.set_ylabel('Frequency')
            ax1.grid(True, alpha=0.3)

            # Add statistics text
            stats_d = factor_results['descriptive_stats']
            if pd.notna(stats_d['mean']):
                stats_text = f"Mean: {stats_d['mean']:.2f}\nStd: {stats_d['std']:.2f}\nMedian: {stats_d['median']:.2f}"
                ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # Plot 2: Factor by toxicity group
            if 'Observed_Toxicity' in self.merged_data.columns:
                toxicity_data = self.merged_data[self.merged_data['Observed_Toxicity'] == 1][factor].dropna()
                no_toxicity_data = self.merged_data[self.merged_data['Observed_Toxicity'] == 0][factor].dropna()

                ax2.boxplot([no_toxicity_data, toxicity_data], 
                           labels=['No Toxicity', 'Toxicity'],
                           patch_artist=True,
                           boxprops=dict(facecolor=COLORS['secondary'], alpha=0.7),
                           medianprops=dict(color='black', linewidth=2))

                ax2.set_title(f'{factor} by Toxicity Status')
                ax2.set_ylabel(factor)
                ax2.grid(True, alpha=0.3)

                # Add group comparison results
                if 'toxicity_vs_no_toxicity' in factor_results['group_comparisons']:
                    comp = factor_results['group_comparisons']['toxicity_vs_no_toxicity']
                    comp_text = f"p = {comp['p_value']:.4f}\nEffect size: {comp['effect_size']:.3f}" if pd.notna(comp['p_value']) else "p = NA"
                    ax2.text(0.02, 0.98, comp_text, transform=ax2.transAxes, 
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                ax2.axis('off')

            # Plot 3: Scatter plot with observed toxicity
            if 'Observed_Toxicity' in self.merged_data.columns:
                valid_data = self.merged_data[[factor, 'Observed_Toxicity']].dropna()

                # Add jitter to toxicity for better visualization
                jittered_toxicity = valid_data['Observed_Toxicity'].astype(float) + np.random.normal(0, 0.02, len(valid_data))

                ax3.scatter(valid_data[factor], jittered_toxicity, 
                           alpha=0.6, color=COLORS['tertiary'], s=30)
                ax3.set_xlabel(factor)
                ax3.set_ylabel('Observed Toxicity (jittered)')
                ax3.set_title(f'{factor} vs Observed Toxicity')
                ax3.grid(True, alpha=0.3)

                # Add correlation information
                if 'observed_toxicity' in factor_results['correlations']:
                    corr = factor_results['correlations']['observed_toxicity']
                    corr_text = f"r = {corr['correlation']:.3f}\np = {corr['p_value']:.4f}"
                    ax3.text(0.02, 0.98, corr_text, transform=ax3.transAxes, 
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                ax3.axis('off')

            # Plot 4: Correlations with NTCP models
            ntcp_correlations = {k: v for k, v in factor_results['correlations'].items() 
                               if k != 'observed_toxicity'}

            if ntcp_correlations:
                models = list(ntcp_correlations.keys())
                correlations = [ntcp_correlations[model]['correlation'] for model in models]
                p_values = [ntcp_correlations[model]['p_value'] for model in models]

                # Create bar plot
                colors = [COLORS['correlation_pos'] if corr > 0 else COLORS['correlation_neg'] 
                         for corr in correlations]
                bars = ax4.bar(range(len(models)), correlations, color=colors, alpha=0.7)

                ax4.set_title(f'Correlations with NTCP Models')
                ax4.set_ylabel('Correlation Coefficient')
                ax4.set_xlabel('NTCP Models')
                ax4.set_xticks(range(len(models)))
                ax4.set_xticklabels(models, rotation=45, ha='right')
                ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax4.grid(True, alpha=0.3)

                # Add significance indicators
                for i, (bar, p_val) in enumerate(zip(bars, p_values)):
                    if pd.notna(p_val) and p_val < 0.05:
                        ax4.text(bar.get_x() + bar.get_width()/2, 
                                bar.get_height() + 0.01 if bar.get_height() > 0 else bar.get_height() - 0.03,
                                '*', ha='center', va='bottom' if bar.get_height() > 0 else 'top', 
                                fontsize=16, fontweight='bold')
            else:
                ax4.axis('off')

            plt.tight_layout()

            # Save plot
            plot_file = self.output_dir / f'continuous_analysis_{factor}.png'
            plt.savefig(plot_file, dpi=600, bbox_inches='tight')
            print(f" Continuous plot saved: {plot_file}")
            plt.close()

    def _plot_organ_specific_analysis(self, results):
        """Create plots for organ-specific analysis"""

        if not results:
            return

        # Create overview plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Organ-Specific Clinical Factors Analysis', fontsize=16, fontweight='bold')

        organs = list(results.keys())
        if not organs:
            plt.close(fig)
            return

        # Plot 1: Sample sizes by organ
        sample_sizes = [results[organ]['sample_size'] for organ in organs]

        bars = axes[0, 0].bar(organs, sample_sizes, color=COLORS['primary'], alpha=0.7)
        axes[0, 0].set_title('Sample Size by Organ')
        axes[0, 0].set_ylabel('Number of Cases')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Add value labels
        for bar, size in zip(bars, sample_sizes):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           f'{size}', ha='center', va='bottom')

        # Plot 2: Toxicity rates by organ
        toxicity_rates = [results[organ]['toxicity_rate'] for organ in organs]

        bars = axes[0, 1].bar(organs, toxicity_rates, color=COLORS['secondary'], alpha=0.7)
        axes[0, 1].set_title('Toxicity Rate by Organ')
        axes[0, 1].set_ylabel('Toxicity Rate')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # Add value labels
        for bar, rate in zip(bars, toxicity_rates):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{rate:.3f}', ha='center', va='bottom')

        # Plot 3: Factor significance heatmap (if enough data)
        # Create a matrix of significant correlations by organ
        all_factors = set()
        for organ_result in results.values():
            all_factors.update(organ_result['factor_effects'].keys())

        all_factors = list(all_factors)

        if len(all_factors) > 0 and len(organs) > 0:
            significance_matrix = np.zeros((len(organs), len(all_factors)))

            for i, organ in enumerate(organs):
                for j, factor in enumerate(all_factors):
                    if factor in results[organ]['factor_effects']:
                        factor_effect = results[organ]['factor_effects'][factor]
                        if factor_effect['data_type'] == 'continuous':
                            if 'correlation_with_toxicity' in factor_effect:
                                corr_data = factor_effect['correlation_with_toxicity']
                                if corr_data['significant']:
                                    significance_matrix[i, j] = corr_data['correlation']

            # Create heatmap
            im = axes[1, 0].imshow(significance_matrix, cmap='RdBu_r', aspect='auto', 
                                  vmin=-1, vmax=1)
            axes[1, 0].set_title('Significant Factor Correlations by Organ')
            axes[1, 0].set_yticks(range(len(organs)))
            axes[1, 0].set_yticklabels(organs)
            axes[1, 0].set_xticks(range(len(all_factors)))
            axes[1, 0].set_xticklabels(all_factors, rotation=45, ha='right')

            # Add colorbar
            plt.colorbar(im, ax=axes[1, 0], label='Correlation Coefficient')

        # Plot 4: Summary statistics table
        axes[1, 1].axis('off')

        # Create summary table data
        table_data = []
        for organ in organs:
            table_data.append([
                organ,
                results[organ]['sample_size'],
                f"{results[organ]['toxicity_rate']:.3f}",
                len(results[organ]['factor_effects'])
            ])

        table = axes[1, 1].table(cellText=table_data,
                                colLabels=['Organ', 'Sample Size', 'Toxicity Rate', 'Factors Analyzed'],
                                cellLoc='center',
                                loc='center')

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        # Style the table
        for i in range(len(table_data) + 1):
            for j in range(4):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor(COLORS['primary'])
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')

        axes[1, 1].set_title('Summary by Organ')

        plt.tight_layout()

        # Save plot
        plot_file = self.output_dir / 'organ_specific_overview.png'
        plt.savefig(plot_file, dpi=600, bbox_inches='tight')
        print(f" Organ-specific overview saved: {plot_file}")
        plt.close()

    def create_comprehensive_summary_report(self, categorical_results, continuous_results, 
                                          organ_results, correlation_matrix):
        """Create comprehensive summary report"""

        print("\n📋 Creating Comprehensive Summary Report...")

        report_lines = []
        report_lines.append("NTCP CLINICAL FACTORS ANALYSIS REPORT")
        report_lines.append("=" * 50)
        report_lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Total patient-organ combinations: {len(self.merged_data)}")
        if 'PrimaryPatientID' in self.merged_data.columns:
            report_lines.append(f"Unique patients: {self.merged_data['PrimaryPatientID'].nunique()}")
        report_lines.append("")

        # Dataset overview
        if 'Organ' in self.merged_data.columns:
            report_lines.append("DATASET OVERVIEW")
            report_lines.append("-" * 20)

            organ_distribution = self.merged_data['Organ'].value_counts()
            for organ, count in organ_distribution.items():
                toxicity_rate = self.merged_data[self.merged_data['Organ'] == organ]['Observed_Toxicity'].mean() if 'Observed_Toxicity' in self.merged_data.columns else np.nan
                tox_txt = f"{toxicity_rate:.3f}" if pd.notna(toxicity_rate) else "NA"
                report_lines.append(f"{organ}: {count} cases, toxicity rate: {tox_txt}")

            overall_toxicity_rate = self.merged_data['Observed_Toxicity'].mean() if 'Observed_Toxicity' in self.merged_data.columns else np.nan
            report_lines.append(f"Overall toxicity rate: {overall_toxicity_rate:.3f}" if pd.notna(overall_toxicity_rate) else "Overall toxicity rate: NA")
            report_lines.append("")

        # Categorical factors summary
        if categorical_results:
            report_lines.append("CATEGORICAL FACTORS ANALYSIS")
            report_lines.append("-" * 30)

            for factor, results in categorical_results.items():
                report_lines.append(f"\n{factor.upper()}:")

                # Category summary
                for category, stats_d in results['categories'].items():
                    tox = stats_d['observed_toxicity_rate']
                    tox_txt = f"{tox:.3f}" if pd.notna(tox) else "NA"
                    report_lines.append(f"  {category}: {stats_d['n_cases']} cases, toxicity rate: {tox_txt}")

                # Statistical significance
                if 'chi_square' in results['statistical_tests']:
                    chi_sq = results['statistical_tests']['chi_square']
                    significance = "SIGNIFICANT" if chi_sq['significant'] else "NOT SIGNIFICANT"
                    report_lines.append(f"  Chi-square test: p = {chi_sq['p_value']:.4f} ({significance})")

        # Continuous factors summary
        if continuous_results:
            report_lines.append("\n\nCONTINUOUS FACTORS ANALYSIS")
            report_lines.append("-" * 32)

            for factor, results in continuous_results.items():
                report_lines.append(f"\n{factor.upper()}:")

                # Descriptive statistics
                stats_d = results['descriptive_stats']
                report_lines.append(f"  Range: {stats_d['min']:.2f} - {stats_d['max']:.2f}")
                report_lines.append(f"  Mean +/- SD: {stats_d['mean']:.2f} +/- {stats_d['std']:.2f}")

                # Correlation with observed toxicity
                if 'observed_toxicity' in results['correlations']:
                    corr = results['correlations']['observed_toxicity']
                    significance = "SIGNIFICANT" if corr['significant'] else "NOT SIGNIFICANT"
                    report_lines.append(f"  Correlation with toxicity: r = {corr['correlation']:.3f}, "
                                      f"p = {corr['p_value']:.4f} ({significance})")

                # Group comparison
                if 'toxicity_vs_no_toxicity' in results['group_comparisons']:
                    comp = results['group_comparisons']['toxicity_vs_no_toxicity']
                    significance = "SIGNIFICANT" if comp['significant'] else "NOT SIGNIFICANT"
                    report_lines.append(f"  Group difference: toxicity group mean = {comp['toxicity_group_mean']:.2f}, "
                                      f"no toxicity group mean = {comp['no_toxicity_group_mean']:.2f}")
                    report_lines.append(f"  Mann-Whitney U test: p = {comp['p_value']:.4f} ({significance})")
                    report_lines.append(f"  Effect size: {comp['effect_size']:.3f}")

        # Organ-specific findings
        if organ_results:
            report_lines.append("\n\nORGAN-SPECIFIC FINDINGS")
            report_lines.append("-" * 25)

            for organ, results in organ_results.items():
                report_lines.append(f"\n{organ.upper()}:")
                report_lines.append(f"  Sample size: {results['sample_size']}")
                report_lines.append(f"  Toxicity rate: {results['toxicity_rate']:.3f}")

                # Significant factor effects
                significant_factors = []
                for factor, factor_effect in results['factor_effects'].items():
                    if factor_effect['data_type'] == 'continuous':
                        if 'correlation_with_toxicity' in factor_effect:
                            if factor_effect['correlation_with_toxicity']['significant']:
                                corr = factor_effect['correlation_with_toxicity']['correlation']
                                significant_factors.append(f"{factor} (r={corr:.3f})")

                if significant_factors:
                    report_lines.append(f"  Significant factors: {', '.join(significant_factors)}")
                else:
                    report_lines.append("  No significant factor correlations found")

        # Key correlations from correlation matrix
        report_lines.append("\n\nKEY CORRELATIONS")
        report_lines.append("-" * 18)

        # Find strongest correlations with observed toxicity
        if isinstance(correlation_matrix, pd.DataFrame) and not correlation_matrix.empty and \
           'Observed_Toxicity' in correlation_matrix.columns:
            toxicity_correlations = correlation_matrix['Observed_Toxicity'].abs().sort_values(ascending=False)

            report_lines.append("Strongest correlations with observed toxicity:")
            for factor, corr in toxicity_correlations.head(10).items():
                if factor != 'Observed_Toxicity' and not np.isnan(corr):
                    actual_corr = correlation_matrix.loc[factor, 'Observed_Toxicity']
                    report_lines.append(f"  {factor}: r = {actual_corr:.3f}")

        # Clinical recommendations
        report_lines.append("\n\nCLINICAL RECOMMENDATIONS")
        report_lines.append("-" * 26)

        # Generate recommendations based on findings
        recommendations = self._generate_clinical_recommendations(
            categorical_results, continuous_results, organ_results, correlation_matrix
        )

        for recommendation in recommendations:
            report_lines.append(f"• {recommendation}")

        # Save report
        report_file = self.output_dir / 'clinical_factors_analysis_report.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        print(f" Comprehensive report saved: {report_file}")

        return report_lines

    def _generate_clinical_recommendations(self, categorical_results, continuous_results, 
                                         organ_results, correlation_matrix):
        """Generate clinical recommendations based on analysis results"""

        recommendations = []

        # Check for significant categorical factors
        for factor, results in categorical_results.items():
            if 'chi_square' in results['statistical_tests']:
                if results['statistical_tests']['chi_square']['significant']:
                    recommendations.append(f"{factor} shows significant association with toxicity outcomes and should be considered in treatment planning")

        # Check for significant continuous factors
        for factor, results in continuous_results.items():
            if 'observed_toxicity' in results['correlations']:
                corr_data = results['correlations']['observed_toxicity']
                if corr_data['significant']:
                    direction = "higher" if corr_data['correlation'] > 0 else "lower"
                    recommendations.append(f"{factor} is significantly correlated with toxicity - {direction} values associated with increased toxicity risk")

        # Organ-specific recommendations
        high_risk_organs = []
        for organ, results in organ_results.items():
            if results['toxicity_rate'] > 0.3:  # 30% threshold
                high_risk_organs.append(organ)

        if high_risk_organs:
            recommendations.append(f"Organs with highest toxicity risk ({', '.join(high_risk_organs)}) require enhanced monitoring and potential dose constraints")

        # Check sample size adequacy
        small_sample_organs = []
        for organ, results in organ_results.items():
            if results['sample_size'] < 20:
                small_sample_organs.append(organ)

        if small_sample_organs:
            recommendations.append(f"Larger datasets needed for {', '.join(small_sample_organs)} to improve statistical power")

        # NTCP model recommendations
        if isinstance(correlation_matrix, pd.DataFrame) and not correlation_matrix.empty and \
           'Observed_Toxicity' in correlation_matrix.columns:
            ntcp_correlations = {}
            for col in correlation_matrix.columns:
                if col.startswith('NTCP_'):
                    corr_val = correlation_matrix.loc['Observed_Toxicity', col]
                    if not np.isnan(corr_val):
                        ntcp_correlations[col] = abs(corr_val)

            if ntcp_correlations:
                best_model = max(ntcp_correlations.items(), key=lambda x: x[1])
                recommendations.append(f"Best performing NTCP model: {best_model[0]} (|r| = {best_model[1]:.3f})")

        # General recommendations
        recommendations.append("Consider multivariable modeling combining significant clinical factors with dose metrics")
        recommendations.append("Validate findings in external cohorts before clinical implementation")
        recommendations.append("Regular model recalibration recommended as more data becomes available")

        return recommendations

    def run_complete_analysis(self):
        """Run the complete clinical factors analysis pipeline"""

        print(" Starting Comprehensive NTCP Clinical Factors Analysis")
        print("=" * 60)

        # Step 1: Load and merge data
        if not self.load_and_merge_data():
            print("Error: Failed to load and merge data. Stopping analysis.")
            return False

        # Step 2: Analyze categorical factors
        print("\n" + "="*60)
        categorical_results = self.analyze_categorical_factors()

        # Step 3: Analyze continuous factors
        print("\n" + "="*60)
        continuous_results = self.analyze_continuous_factors()

        # Step 4: Analyze organ-specific effects
        print("\n" + "="*60)
        organ_results = self.analyze_organ_specific_effects()

        # Step 5: Create correlation matrix
        print("\n" + "="*60)
        correlation_matrix = self.create_correlation_matrix()

        # Step 6: Create comprehensive summary report
        print("\n" + "="*60)
        self.create_comprehensive_summary_report(
            categorical_results, continuous_results, organ_results, correlation_matrix
        )

        print("\n🎉 Clinical Factors Analysis Completed Successfully!")
        print("=" * 60)
        print(f"📁 All outputs saved to: {self.output_dir.absolute()}")
        print("\nGenerated files:")
        print("   categorical_factors_analysis.xlsx")
        print("   continuous_factors_analysis.xlsx") 
        print("   organ_specific_analysis.xlsx")
        print("  📈 correlation_matrix.png & correlation_matrix.csv")
        print("  📋 clinical_factors_analysis_report.txt")
        print("   Individual factor analysis plots")
        print("   organ_specific_overview.png")

        return True

def main():
    """Main execution function"""

    import argparse

    parser = argparse.ArgumentParser(description='NTCP Clinical Factors Analysis')
    parser.add_argument('--input_file', default='ntcp_analysis_input.xlsx',
                       help='Clinical factors input file (default: ntcp_analysis_input.xlsx)')
    parser.add_argument('--enhanced_output_dir', default='enhanced_ntcp_analysis',
                       help='Enhanced NTCP analysis output directory (default: enhanced_ntcp_analysis)')
    parser.add_argument('--output_dir', default=None,
                       help='Output directory for factors analysis (default: <enhanced_output_dir>/clinical_factors_analysis)')

    args = parser.parse_args()

    # Validate input files
    input_file = Path(args.input_file)
    enhanced_dir = Path(args.enhanced_output_dir)

    if not input_file.exists():
        print(f"Error: Error: Input file '{input_file}' not found")
        return

    if not enhanced_dir.exists():
        print(f"Error: Error: Enhanced output directory '{enhanced_dir}' not found")
        print("Please run enhanced_ntcp_analysis_ml.py first to generate NTCP results")
        return

    # Check for required NTCP results file
    ntcp_results_file = enhanced_dir / 'enhanced_ntcp_calculations.csv'
    if not ntcp_results_file.exists():
        print(f"Error: Error: NTCP results file '{ntcp_results_file}' not found")
        print("Please run enhanced_ntcp_analysis_ml.py first to generate NTCP results")
        return

    print(" Input validation passed")
    print(f" Clinical factors file: {input_file}")
    print(f" NTCP results directory: {enhanced_dir}")

    try:
        # Initialize analyzer (optional output_dir for single canonical location when run from pipeline)
        analyzer = ClinicalFactorsAnalyzer(input_file, enhanced_dir, output_dir=args.output_dir)

        # Run complete analysis
        success = analyzer.run_complete_analysis()

        if success:
            print("\n💡 Next Steps:")
            print("  1. Review clinical_factors_analysis_report.txt for key findings")
            print("  2. Examine correlation_matrix.png for factor relationships")
            print("  3. Check individual factor analysis plots for detailed insights")
            print("  4. Consider multivariable modeling for significant factors")
            print("  5. Validate findings in external cohorts")

    except Exception as e:
        print(f"\nError: Error during analysis: {e}")
        import traceback
        print("\nFull error traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
