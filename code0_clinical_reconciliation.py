#!/usr/bin/env python3
"""
code0_clinical_reconciliation.py - Clinical Data Reconciliation & Diagnostic Executive
======================================================================================

Runs BEFORE Step 3 to ensure clinical data matches DVH registry.
Provides interactive reconciliation and diagnostic capabilities.

Responsibilities:
1. Read Step1_DVHRegistry.xlsx and clinical_patient_data.xlsx
2. Detect mismatches (missing patients, extra patients, organ mismatches, duplicates)
3. Generate clinical_template.xlsx with valid rows only
4. Prompt user for auto-mapping acceptance
5. Self-diagnostic executive: parse logs, tracebacks, contracts to detect issues

Software: py_ntcpx v1.0 - Clinical Governance Upgrade
"""

from __future__ import annotations

import argparse
import sys
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
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

# Import contract validator
try:
    from contract_validator import ContractValidator, normalize_dvh_id
except ImportError:
    safe_print("ERROR: contract_validator module not found")
    sys.exit(1)


class ClinicalReconciler:
    """Clinical data reconciliation engine"""
    
    def __init__(self, contracts_dir: Path, output_dir: Path):
        self.contracts_dir = Path(contracts_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.validator = ContractValidator(self.contracts_dir)
        
    def load_registry(self) -> Optional[pd.DataFrame]:
        """Load Step1_DVHRegistry"""
        registry_path = self.validator.get_contract_path("Step1_DVHRegistry")
        if not registry_path.exists():
            safe_print(f"ERROR: Registry not found: {registry_path}")
            safe_print("Run Step 1 (code1_dvh_preprocess.py) first to generate registry.")
            return None
        
        try:
            with pd.ExcelFile(registry_path) as xl:
                registry_df = pd.read_excel(xl, sheet_name='DVHRegistry')
            
            # Validate required columns
            required_cols = ['PrimaryPatientID', 'Organ']
            missing_cols = [col for col in required_cols if col not in registry_df.columns]
            if missing_cols:
                safe_print(f"ERROR: Registry missing required columns: {missing_cols}")
                return None
            
            return registry_df
        except Exception as e:
            safe_print(f"ERROR: Failed to load registry: {e}")
            return None
    
    def load_clinical_data(self, clinical_path: Path) -> Optional[pd.DataFrame]:
        """Load clinical patient data"""
        if not clinical_path.exists():
            safe_print(f"ERROR: Clinical data file not found: {clinical_path}")
            return None
        
        try:
            with pd.ExcelFile(clinical_path) as xl:
                # Try to find the right sheet
                sheet_name = None
                for name in xl.sheet_names:
                    df_test = pd.read_excel(xl, sheet_name=name, nrows=5)
                    if 'PrimaryPatientID' in df_test.columns or 'PatientID' in df_test.columns:
                        sheet_name = name
                        break
                
                if sheet_name is None:
                    sheet_name = xl.sheet_names[0]
                
                clinical_df = pd.read_excel(xl, sheet_name=sheet_name)
            
            return clinical_df
        except Exception as e:
            safe_print(f"ERROR: Failed to load clinical data: {e}")
            return None
    
    def detect_mismatches(self, registry_df: pd.DataFrame, clinical_df: pd.DataFrame) -> Dict:
        """Detect all mismatches between registry and clinical data"""
        
        mismatches = {
            'missing_dvhs': [],  # Clinical rows without DVH
            'missing_clinical': [],  # DVHs without clinical data
            'organ_mismatches': [],  # Same patient, different organs
            'duplicates': [],  # Duplicate (PrimaryPatientID, Organ) in clinical
            'missing_primary_id': [],  # Clinical rows missing PrimaryPatientID
            'missing_organ': [],  # Clinical rows missing Organ
            'missing_toxicity': []  # Clinical rows missing toxicity column
        }
        
        # Normalize PrimaryPatientID for matching
        registry_df['PrimaryPatientID_norm'] = registry_df['PrimaryPatientID'].apply(normalize_dvh_id)
        
        # Check if clinical data has PrimaryPatientID
        has_primary = 'PrimaryPatientID' in clinical_df.columns
        has_anon = 'AnonPatientID' in clinical_df.columns
        has_patient_id = 'PatientID' in clinical_df.columns
        
        # Handle backward compatibility: map AnonPatientID to PrimaryPatientID if needed
        if not has_primary and has_patient_id:
            # Check if PatientID looks like AnonPatientID (PT format)
            sample_pid = clinical_df['PatientID'].dropna().iloc[0] if len(clinical_df['PatientID'].dropna()) > 0 else ""
            if isinstance(sample_pid, str) and sample_pid.strip().upper().startswith('PT'):
                # PatientID is AnonPatientID - need to map via registry
                if 'AnonPatientID' in registry_df.columns and 'Organ' in clinical_df.columns:
                    mapping_df = registry_df[['PrimaryPatientID', 'AnonPatientID', 'Organ']].drop_duplicates()
                    clinical_df = pd.merge(
                        clinical_df,
                        mapping_df,
                        left_on=['PatientID', 'Organ'],
                        right_on=['AnonPatientID', 'Organ'],
                        how='left',
                        suffixes=('', '_mapped')
                    )
                    if 'PrimaryPatientID_mapped' in clinical_df.columns:
                        clinical_df['PrimaryPatientID'] = clinical_df['PrimaryPatientID_mapped']
                        clinical_df = clinical_df.drop(columns=['PrimaryPatientID_mapped'], errors='ignore')
                        has_primary = clinical_df['PrimaryPatientID'].notna().any()
        
        # Check for missing PrimaryPatientID
        if not has_primary or 'PrimaryPatientID' not in clinical_df.columns:
            mismatches['missing_primary_id'] = list(range(len(clinical_df)))
            return mismatches, clinical_df
        
        # Check for missing Organ
        if 'Organ' not in clinical_df.columns:
            mismatches['missing_organ'] = list(range(len(clinical_df)))
            return mismatches, clinical_df
        
        # Check for missing toxicity column
        toxicity_cols = ['Toxicity', 'Observed_Toxicity', 'Endpoint', 'Event']
        has_toxicity = any(col in clinical_df.columns for col in toxicity_cols)
        if not has_toxicity:
            mismatches['missing_toxicity'] = ["Required: One of " + ", ".join(toxicity_cols)]
            return mismatches, clinical_df
        
        # Normalize PrimaryPatientID in clinical data
        clinical_df['PrimaryPatientID_norm'] = clinical_df['PrimaryPatientID'].apply(normalize_dvh_id)
        
        # Build key sets for comparison (PrimaryPatientID + Organ)
        registry_keys = set(zip(
            registry_df['PrimaryPatientID_norm'].dropna(),
            registry_df['Organ'].dropna()
        ))
        
        clinical_keys = set(zip(
            clinical_df['PrimaryPatientID_norm'].dropna(),
            clinical_df['Organ'].dropna()
        ))
        
        # Find mismatches
        missing_dvh_keys = clinical_keys - registry_keys
        missing_clinical_keys = registry_keys - clinical_keys
        
        # Convert keys back to readable format
        for key in missing_dvh_keys:
            primary_id_norm, organ = key
            # Find original PrimaryPatientID from registry
            primary_id = registry_df[
                (registry_df['PrimaryPatientID_norm'] == primary_id_norm) &
                (registry_df['Organ'] == organ)
            ]['PrimaryPatientID'].iloc[0] if len(registry_df[
                (registry_df['PrimaryPatientID_norm'] == primary_id_norm) &
                (registry_df['Organ'] == organ)
            ]) > 0 else primary_id_norm
            mismatches['missing_dvhs'].append({'PrimaryPatientID': primary_id, 'Organ': organ})
        
        for key in missing_clinical_keys:
            primary_id_norm, organ = key
            primary_id = registry_df[
                (registry_df['PrimaryPatientID_norm'] == primary_id_norm) &
                (registry_df['Organ'] == organ)
            ]['PrimaryPatientID'].iloc[0] if len(registry_df[
                (registry_df['PrimaryPatientID_norm'] == primary_id_norm) &
                (registry_df['Organ'] == organ)
            ]) > 0 else primary_id_norm
            mismatches['missing_clinical'].append({'PrimaryPatientID': primary_id, 'Organ': organ})
        
        # Check for duplicates in clinical data
        if len(clinical_df) > 0:
            duplicate_mask = clinical_df[['PrimaryPatientID_norm', 'Organ']].duplicated(keep=False)
            if duplicate_mask.any():
                duplicates = clinical_df[duplicate_mask][['PrimaryPatientID', 'Organ']].drop_duplicates()
                mismatches['duplicates'] = duplicates.to_dict('records')
        
        # Check for organ mismatches (same PrimaryPatientID, different organs in registry vs clinical)
        if len(clinical_df) > 0:
            for _, row in clinical_df.iterrows():
                primary_id_norm = row.get('PrimaryPatientID_norm')
                clinical_organ = row.get('Organ')
                
                if pd.isna(primary_id_norm) or pd.isna(clinical_organ):
                    continue
                
                # Find all organs for this patient in registry
                registry_organs = set(registry_df[
                    registry_df['PrimaryPatientID_norm'] == primary_id_norm
                ]['Organ'].unique())
                
                if registry_organs and clinical_organ not in registry_organs:
                    mismatches['organ_mismatches'].append({
                        'PrimaryPatientID': row.get('PrimaryPatientID', primary_id_norm),
                        'Clinical_Organ': clinical_organ,
                        'Registry_Organs': list(registry_organs)
                    })
        
        return mismatches, clinical_df
    
    def generate_reconciled_template(self, registry_df: pd.DataFrame, clinical_df: pd.DataFrame,
                                    mismatches: Dict) -> pd.DataFrame:
        """Generate reconciled clinical template using registry as source of truth"""
        
        # Start with registry (source of truth for DVH availability)
        template_df = registry_df[['PrimaryPatientID', 'Organ']].copy()
        
        # Add AnonPatientID if available in registry
        if 'AnonPatientID' in registry_df.columns:
            template_df = template_df.merge(
                registry_df[['PrimaryPatientID', 'Organ', 'AnonPatientID']].drop_duplicates(),
                on=['PrimaryPatientID', 'Organ'],
                how='left'
            )
        else:
            template_df['AnonPatientID'] = None
        
        # Merge clinical data where keys match
        if len(clinical_df) > 0 and 'PrimaryPatientID' in clinical_df.columns and 'Organ' in clinical_df.columns:
            # Normalize for merge
            clinical_df_norm = clinical_df.copy()
            clinical_df_norm['PrimaryPatientID_norm'] = clinical_df_norm['PrimaryPatientID'].apply(normalize_dvh_id)
            registry_df_norm = registry_df.copy()
            registry_df_norm['PrimaryPatientID_norm'] = registry_df_norm['PrimaryPatientID'].apply(normalize_dvh_id)
            
            # Merge on normalized keys
            template_df_norm = template_df.copy()
            template_df_norm['PrimaryPatientID_norm'] = template_df_norm['PrimaryPatientID'].apply(normalize_dvh_id)
            
            # Merge clinical data
            clinical_cols_to_merge = [col for col in clinical_df.columns 
                                     if col not in ['PrimaryPatientID_norm', 'PatientID']]
            
            template_df_norm = template_df_norm.merge(
                clinical_df_norm[['PrimaryPatientID_norm', 'Organ'] + clinical_cols_to_merge],
                on=['PrimaryPatientID_norm', 'Organ'],
                how='left',
                suffixes=('', '_clinical')
            )
            
            # Clean up duplicate columns
            for col in template_df_norm.columns:
                if col.endswith('_clinical') and col.replace('_clinical', '') in template_df_norm.columns:
                    # Keep non-null values from clinical version
                    base_col = col.replace('_clinical', '')
                    mask = template_df_norm[base_col].isna() & template_df_norm[col].notna()
                    template_df_norm[base_col] = template_df_norm[base_col].fillna(template_df_norm[col])
                    template_df_norm = template_df_norm.drop(columns=[col])
            
            template_df = template_df_norm.drop(columns=['PrimaryPatientID_norm'], errors='ignore')
        else:
            # No valid clinical data - create empty template
            template_df['Toxicity'] = None
            template_df['Observed_Toxicity'] = None
        
        # Ensure required columns exist
        required_cols = ['Toxicity', 'Observed_Toxicity']
        for col in required_cols:
            if col not in template_df.columns:
                template_df[col] = None
        
        # Remove duplicates (shouldn't happen, but safety check)
        template_df = template_df.drop_duplicates(subset=['PrimaryPatientID', 'Organ'])
        
        # Sort by PrimaryPatientID and Organ
        template_df = template_df.sort_values(['PrimaryPatientID', 'Organ']).reset_index(drop=True)
        
        return template_df
    
    def print_mismatch_summary(self, mismatches: Dict):
        """Print human-readable mismatch summary"""
        safe_print("\n" + "="*60)
        safe_print("CLINICAL RECONCILIATION SUMMARY")
        safe_print("="*60)
        
        missing_dvhs = len(mismatches.get('missing_dvhs', []))
        missing_clinical = len(mismatches.get('missing_clinical', []))
        organ_mismatches = len(mismatches.get('organ_mismatches', []))
        duplicates = len(mismatches.get('duplicates', []))
        missing_primary_id = len(mismatches.get('missing_primary_id', []))
        missing_organ = len(mismatches.get('missing_organ', []))
        missing_toxicity = len(mismatches.get('missing_toxicity', []))
        
        safe_print(f"\nMissing DVHs: {missing_dvhs}")
        safe_print(f"  (Clinical rows without matching DVH in registry)")
        
        safe_print(f"\nMissing clinical data: {missing_clinical}")
        safe_print(f"  (DVHs in registry without matching clinical data)")
        
        safe_print(f"\nOrgan mismatches: {organ_mismatches}")
        safe_print(f"  (Same PrimaryPatientID, different organs)")
        
        safe_print(f"\nDuplicate (PrimaryPatientID, Organ): {duplicates}")
        
        if missing_primary_id > 0:
            safe_print(f"\nMissing PrimaryPatientID: {missing_primary_id} rows")
        
        if missing_organ > 0:
            safe_print(f"\nMissing Organ: {missing_organ} rows")
        
        if missing_toxicity > 0:
            safe_print(f"\nMissing toxicity column: {missing_toxicity}")
        
        safe_print("\n" + "="*60)
    
    def reconcile(self, clinical_path: Path, auto_accept: bool = False) -> Tuple[bool, Optional[Path]]:
        """
        Perform clinical reconciliation
        
        Returns:
            (success, output_path) tuple
        """
        safe_print("\n[CODE0] Clinical Data Reconciliation")
        safe_print("="*60)
        
        # Load registry
        registry_df = self.load_registry()
        if registry_df is None:
            return False, None
        
        safe_print(f"Registry loaded: {len(registry_df)} DVH entries")
        safe_print(f"  Organs: {', '.join(sorted(registry_df['Organ'].unique()))}")
        safe_print(f"  Unique PrimaryPatientID: {registry_df['PrimaryPatientID'].nunique()}")
        
        # Load clinical data
        clinical_df = self.load_clinical_data(clinical_path)
        if clinical_df is None:
            return False, None
        
        safe_print(f"\nClinical data loaded: {len(clinical_df)} rows")
        
        # Detect mismatches
        mismatches, clinical_df_processed = self.detect_mismatches(registry_df, clinical_df)
        
        # Print summary
        self.print_mismatch_summary(mismatches)
        
        # Check if reconciliation needed
        total_issues = (len(mismatches.get('missing_dvhs', [])) +
                       len(mismatches.get('organ_mismatches', [])) +
                       len(mismatches.get('duplicates', [])) +
                       len(mismatches.get('missing_primary_id', [])) +
                       len(mismatches.get('missing_organ', [])) +
                       len(mismatches.get('missing_toxicity', [])))
        
        if total_issues == 0:
            safe_print("\n[OK] No mismatches detected. Clinical data matches registry.")
            return True, clinical_path
        
        # Generate reconciled template
        safe_print("\n[INFO] Generating reconciled clinical template...")
        reconciled_df = self.generate_reconciled_template(registry_df, clinical_df_processed, mismatches)
        
        # Save template
        template_path = self.output_dir / "clinical_template.xlsx"
        with pd.ExcelWriter(template_path, engine='openpyxl') as writer:
            reconciled_df.to_excel(writer, index=False, sheet_name='ClinicalData')
        
        safe_print(f"Reconciled template saved to: {template_path}")
        
        # Prompt user (unless auto_accept)
        if not auto_accept:
            safe_print("\nAccept auto-mapping from registry and continue? [Y/N]: ", end='')
            try:
                response = input().strip().upper()
                if response != 'Y':
                    safe_print("[ABORT] Pipeline aborted by user.")
                    return False, None
            except (EOFError, KeyboardInterrupt):
                safe_print("\n[ABORT] Pipeline aborted.")
                return False, None
        
        # Write corrected clinical file
        output_path = self.output_dir / "clinical_patient_data_reconciled.xlsx"
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            reconciled_df.to_excel(writer, index=False, sheet_name='ClinicalData')
        
        safe_print(f"\n[OK] Reconciled clinical data saved to: {output_path}")
        safe_print(f"  Rows: {len(reconciled_df)}")
        safe_print(f"  Organs: {', '.join(sorted(reconciled_df['Organ'].unique()))}")
        
        return True, output_path


class DiagnosticExecutive:
    """Self-diagnostic executive for pipeline issues"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.diagnostics = []
        
    def parse_log_file(self, log_path: Path) -> List[Dict]:
        """Parse pipeline log for error patterns"""
        issues = []
        
        if not log_path.exists():
            return issues
        
        try:
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Pattern: Step-X failed
            step_failures = re.findall(r'(Step\s*[\d\.]+\s*[:\-]?\s*.+?failed|ERROR.*Step)', content, re.I)
            for failure in step_failures:
                issues.append({
                    'type': 'step_failure',
                    'pattern': failure,
                    'severity': 'high'
                })
            
            # Pattern: Identity mismatch
            identity_issues = re.findall(r'(identity|PrimaryPatientID|mismatch|duplicate.*PrimaryPatientID)', content, re.I)
            if identity_issues:
                issues.append({
                    'type': 'identity_mismatch',
                    'pattern': 'Identity matching issue detected',
                    'severity': 'critical'
                })
            
            # Pattern: Missing contract
            contract_issues = re.findall(r'(contract.*not found|missing.*contract|Step\d+.*Registry)', content, re.I)
            if contract_issues:
                issues.append({
                    'type': 'missing_contract',
                    'pattern': 'Missing contract file',
                    'severity': 'high'
                })
            
            # Pattern: SHAP warnings
            shap_warnings = re.findall(r'(SHAP|feature.*drift|warning.*feature)', content, re.I)
            if shap_warnings:
                issues.append({
                    'type': 'feature_drift',
                    'pattern': 'Potential feature drift (SHAP warnings)',
                    'severity': 'medium'
                })
            
            # Pattern: Sample loss
            sample_loss = re.findall(r'(\d+%?\s*.*excluded|sample.*loss|\d+\s*DVH.*excluded)', content, re.I)
            if sample_loss:
                issues.append({
                    'type': 'sample_loss',
                    'pattern': 'Samples excluded from analysis',
                    'severity': 'medium'
                })
            
            # Pattern: Organ mapping issues
            organ_issues = re.findall(r'(organ.*mismatch|missing.*organ|Organ.*mapping)', content, re.I)
            if organ_issues:
                issues.append({
                    'type': 'organ_mapping',
                    'pattern': 'Organ mapping issue',
                    'severity': 'high'
                })
            
        except Exception as e:
            issues.append({
                'type': 'parse_error',
                'pattern': f'Failed to parse log: {e}',
                'severity': 'low'
            })
        
        return issues
    
    def parse_traceback(self, traceback_text: str) -> List[Dict]:
        """Parse traceback for error patterns"""
        issues = []
        
        # Pattern: FileNotFoundError
        if 'FileNotFoundError' in traceback_text:
            file_pattern = re.search(r'FileNotFoundError:?\s*(.+?)(?:\n|$)', traceback_text)
            if file_pattern:
                issues.append({
                    'type': 'file_not_found',
                    'pattern': file_pattern.group(1),
                    'severity': 'high'
                })
        
        # Pattern: KeyError (missing column)
        if 'KeyError' in traceback_text:
            key_pattern = re.search(r"KeyError:\s*['\"](.+?)['\"]", traceback_text)
            if key_pattern:
                issues.append({
                    'type': 'missing_column',
                    'pattern': f"Missing column: {key_pattern.group(1)}",
                    'severity': 'high'
                })
        
        # Pattern: ValueError (data validation)
        if 'ValueError' in traceback_text:
            value_pattern = re.search(r'ValueError:\s*(.+?)(?:\n|$)', traceback_text)
            if value_pattern:
                issues.append({
                    'type': 'data_validation',
                    'pattern': value_pattern.group(1),
                    'severity': 'medium'
                })
        
        return issues
    
    def check_contracts(self, contracts_dir: Path) -> List[Dict]:
        """Check contract files for issues"""
        issues = []
        contracts_dir = Path(contracts_dir)
        
        required_contracts = [
            'Step1_DVHRegistry',
            'Step3_NTCPDataset'  # May not exist if Step3 not run
        ]
        
        for contract_name in required_contracts:
            contract_path = contracts_dir / f"{contract_name}.xlsx"
            if not contract_path.exists():
                if contract_name == 'Step1_DVHRegistry':
                    issues.append({
                        'type': 'missing_contract',
                        'pattern': f'Required contract missing: {contract_name}',
                        'severity': 'critical',
                        'fix': 'Run Step 1 (code1_dvh_preprocess.py) first'
                    })
        
        return issues
    
    def generate_diagnostic_report(self, log_path: Optional[Path] = None,
                                  traceback_text: Optional[str] = None,
                                  contracts_dir: Optional[Path] = None) -> Path:
        """Generate comprehensive diagnostic report"""
        
        all_issues = []
        
        # Parse log file
        if log_path:
            log_issues = self.parse_log_file(log_path)
            all_issues.extend(log_issues)
        
        # Parse traceback
        if traceback_text:
            traceback_issues = self.parse_traceback(traceback_text)
            all_issues.extend(traceback_issues)
        
        # Check contracts
        if contracts_dir:
            contract_issues = self.check_contracts(contracts_dir)
            all_issues.extend(contract_issues)
        
        # Generate report
        report_lines = []
        report_lines.append("="*60)
        report_lines.append("PIPELINE DIAGNOSTIC REPORT")
        report_lines.append("="*60)
        report_lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        if not all_issues:
            report_lines.append("[OK] No issues detected")
        else:
            # Group by type
            issues_by_type = defaultdict(list)
            for issue in all_issues:
                issues_by_type[issue['type']].append(issue)
            
            for issue_type, type_issues in issues_by_type.items():
                report_lines.append(f"\n{issue_type.upper().replace('_', ' ')}")
                report_lines.append("-" * 40)
                
                for issue in type_issues:
                    report_lines.append(f"Problem: {issue.get('pattern', 'Unknown issue')}")
                    report_lines.append(f"Severity: {issue.get('severity', 'unknown')}")
                    
                    if 'fix' in issue:
                        report_lines.append(f"Fix: {issue['fix']}")
                    else:
                        # Generate generic fix based on type
                        fix = self._suggest_fix(issue_type)
                        if fix:
                            report_lines.append(f"Fix: {fix}")
                    
                    report_lines.append("")
        
        # Save report
        report_path = self.output_dir / "diagnostic_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        return report_path
    
    def _suggest_fix(self, issue_type: str) -> str:
        """Suggest fix based on issue type"""
        fixes = {
            'identity_mismatch': 'Run clinical reconciliation (code0_clinical_reconciliation.py) and accept auto-mapping',
            'missing_contract': 'Run previous pipeline steps to generate required contracts',
            'organ_mapping': 'Run clinical reconciliation to fix organ mismatches',
            'sample_loss': 'Check DVH files and registry for missing or invalid entries',
            'feature_drift': 'Review feature engineering and data preprocessing steps',
            'file_not_found': 'Verify input file paths and ensure all required files exist',
            'missing_column': 'Check clinical data format and ensure all required columns are present',
            'data_validation': 'Review data validation errors and fix data format issues'
        }
        return fixes.get(issue_type, 'Review pipeline logs and fix identified issues')


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description='Clinical Data Reconciliation & Diagnostic Executive (runs before Step 3)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--clinical_data',
        type=str,
        required=True,
        help='Path to clinical_patient_data.xlsx'
    )
    
    parser.add_argument(
        '--contracts_dir',
        type=str,
        default='out2/contracts',
        help='Directory containing contract files (default: out2/contracts)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='out2/code0_output',
        help='Output directory (default: out2/code0_output)'
    )
    
    parser.add_argument(
        '--auto_accept',
        action='store_true',
        help='Automatically accept auto-mapping without prompting'
    )
    
    parser.add_argument(
        '--diagnostic',
        action='store_true',
        help='Run diagnostic executive only (no reconciliation)'
    )
    
    parser.add_argument(
        '--log_file',
        type=str,
        default=None,
        help='Path to pipeline log file for diagnostics'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    contracts_dir = Path(args.contracts_dir)
    
    # Run diagnostic executive
    if args.diagnostic:
        diag = DiagnosticExecutive(output_dir)
        
        log_path = Path(args.log_file) if args.log_file else None
        report_path = diag.generate_diagnostic_report(
            log_path=log_path,
            contracts_dir=contracts_dir
        )
        
        safe_print(f"\n[OK] Diagnostic report generated: {report_path}")
        return 0
    
    # Run reconciliation
    reconciler = ClinicalReconciler(contracts_dir, output_dir)
    clinical_path = Path(args.clinical_data)
    
    success, output_path = reconciler.reconcile(clinical_path, auto_accept=args.auto_accept)
    
    if success:
        safe_print("\n[OK] Clinical reconciliation completed successfully")
        if output_path:
            safe_print(f"Use reconciled file for Step 3: {output_path}")
        return 0
    else:
        safe_print("\n[FAIL] Clinical reconciliation failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

