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

Software: py_ntcpx_v1.0.0 - Clinical Governance Upgrade
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
        """Load clinical patient data (Clinical Contract v2)"""
        if not clinical_path.exists():
            safe_print(f"ERROR: Clinical data file not found: {clinical_path}")
            return None
        
        try:
            with pd.ExcelFile(clinical_path) as xl:
                # Try to find the right sheet (check for patient_id, PrimaryPatientID, or PatientID)
                sheet_name = None
                for name in xl.sheet_names:
                    df_test = pd.read_excel(xl, sheet_name=name, nrows=5)
                    if 'patient_id' in df_test.columns or 'PrimaryPatientID' in df_test.columns or 'PatientID' in df_test.columns:
                        sheet_name = name
                        break
                
                if sheet_name is None:
                    sheet_name = xl.sheet_names[0]
                
                clinical_df = pd.read_excel(xl, sheet_name=sheet_name)
            
            # Handle backward compatibility: map PrimaryPatientID/PatientID to patient_id if needed
            if 'patient_id' not in clinical_df.columns:
                if 'PrimaryPatientID' in clinical_df.columns:
                    clinical_df['patient_id'] = clinical_df['PrimaryPatientID']
                elif 'PatientID' in clinical_df.columns:
                    clinical_df['patient_id'] = clinical_df['PatientID']
            
            return clinical_df
        except Exception as e:
            safe_print(f"ERROR: Failed to load clinical data: {e}")
            return None
    
    def validate_clinical_contract_v2(self, clinical_df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate Clinical Contract v2
        
        Mandatory columns:
        - patient_id: must exist, no missing values
        - xerostomia_grade2plus: must exist, no missing values, must be 0 or 1
        - followup_months: must exist, no missing values, must be > 0
        
        Returns:
            (is_valid, list_of_errors)
        """
        errors = []
        
        # Check mandatory columns exist
        mandatory_cols = ['patient_id', 'xerostomia_grade2plus', 'followup_months']
        missing_cols = [col for col in mandatory_cols if col not in clinical_df.columns]
        if missing_cols:
            errors.append(f"Missing mandatory columns: {', '.join(missing_cols)}")
            return False, errors
        
        # Check patient_id: no missing values
        if clinical_df['patient_id'].isna().any():
            n_missing = clinical_df['patient_id'].isna().sum()
            errors.append(f"patient_id has {n_missing} missing value(s)")
        
        # Check xerostomia_grade2plus: no missing values, must be 0 or 1
        if clinical_df['xerostomia_grade2plus'].isna().any():
            n_missing = clinical_df['xerostomia_grade2plus'].isna().sum()
            errors.append(f"xerostomia_grade2plus has {n_missing} missing value(s)")
        else:
            invalid_values = clinical_df[~clinical_df['xerostomia_grade2plus'].isin([0, 1])]
            if len(invalid_values) > 0:
                errors.append(f"xerostomia_grade2plus must be 0 or 1, found invalid values in {len(invalid_values)} row(s)")
        
        # Check followup_months: no missing values, must be > 0
        if clinical_df['followup_months'].isna().any():
            n_missing = clinical_df['followup_months'].isna().sum()
            errors.append(f"followup_months has {n_missing} missing value(s)")
        else:
            invalid_values = clinical_df[clinical_df['followup_months'] <= 0]
            if len(invalid_values) > 0:
                errors.append(f"followup_months must be > 0, found invalid values in {len(invalid_values)} row(s)")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def detect_mismatches(self, registry_df: pd.DataFrame, clinical_df: pd.DataFrame) -> Dict:
        """Detect all mismatches between registry and clinical data (Clinical Contract v2)"""
        
        mismatches = {
            'missing_dvhs': [],  # Clinical rows without DVH
            'missing_clinical': [],  # DVHs without clinical data
            'duplicates': [],  # Duplicate patient_id in clinical
            'identity_mismatch': False,  # patient_id mismatch with registry
            'contract_validation_errors': []  # Clinical Contract v2 validation errors
        }
        
        # Step 1: Validate Clinical Contract v2
        is_valid, contract_errors = self.validate_clinical_contract_v2(clinical_df)
        if not is_valid:
            mismatches['contract_validation_errors'] = contract_errors
            return mismatches, clinical_df
        
        # Step 2: Check identity integrity (patient_id must match PrimaryPatientID in registry)
        # Normalize patient_id for matching
        clinical_df['patient_id_norm'] = clinical_df['patient_id'].apply(normalize_dvh_id)
        registry_df['PrimaryPatientID_norm'] = registry_df['PrimaryPatientID'].apply(normalize_dvh_id)
        
        # Get unique patient IDs from both
        clinical_patient_ids = set(clinical_df['patient_id_norm'].dropna().unique())
        registry_patient_ids = set(registry_df['PrimaryPatientID_norm'].dropna().unique())
        
        # Check for identity mismatch
        missing_in_registry = clinical_patient_ids - registry_patient_ids
        missing_in_clinical = registry_patient_ids - clinical_patient_ids
        
        if missing_in_registry or missing_in_clinical:
            mismatches['identity_mismatch'] = True
            if missing_in_registry:
                # Find original patient_id values
                for pid_norm in missing_in_registry:
                    orig_pid = clinical_df[clinical_df['patient_id_norm'] == pid_norm]['patient_id'].iloc[0] if len(clinical_df[clinical_df['patient_id_norm'] == pid_norm]) > 0 else pid_norm
                    mismatches['missing_dvhs'].append({'patient_id': orig_pid})
            
            if missing_in_clinical:
                # Find original PrimaryPatientID values
                for pid_norm in missing_in_clinical:
                    orig_pid = registry_df[registry_df['PrimaryPatientID_norm'] == pid_norm]['PrimaryPatientID'].iloc[0] if len(registry_df[registry_df['PrimaryPatientID_norm'] == pid_norm]) > 0 else pid_norm
                    mismatches['missing_clinical'].append({'PrimaryPatientID': orig_pid})
        
        # Check for duplicates in clinical data (patient_id)
        if len(clinical_df) > 0:
            duplicate_mask = clinical_df['patient_id_norm'].duplicated(keep=False)
            if duplicate_mask.any():
                duplicates = clinical_df[duplicate_mask]['patient_id'].drop_duplicates().tolist()
                mismatches['duplicates'] = duplicates
        
        return mismatches, clinical_df
    
    def generate_dynamic_template(self, registry_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate dynamic template with patient_id from registry and optional columns blank
        
        Returns:
            DataFrame with columns: patient_id, age, sex, baseline_xerostomia, 
            tobacco_exposure, chemotherapy, hpv_status, followup_months, xerostomia_grade2plus
        """
        # Get unique PrimaryPatientIDs from registry
        unique_patients = registry_df['PrimaryPatientID'].drop_duplicates().sort_values()
        
        # Create template DataFrame
        template_df = pd.DataFrame({
            'patient_id': unique_patients.values,
            'age': None,
            'sex': None,
            'baseline_xerostomia': None,
            'tobacco_exposure': None,
            'chemotherapy': None,
            'hpv_status': None,
            'followup_months': None,
            'xerostomia_grade2plus': None
        })
        
        return template_df
    
    def generate_reconciled_output(self, clinical_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate reconciled output with mandatory columns + optional columns that exist
        
        Returns:
            DataFrame with patient_id, xerostomia_grade2plus, followup_months + optional columns
        """
        # Start with mandatory columns
        reconciled_df = clinical_df[['patient_id', 'xerostomia_grade2plus', 'followup_months']].copy()
        
        # Add optional columns if they exist in input
        optional_cols = ['age', 'sex', 'baseline_xerostomia', 'tobacco_exposure', 
                        'chemotherapy', 'hpv_status', 'Organ']
        
        for col in optional_cols:
            if col in clinical_df.columns:
                reconciled_df[col] = clinical_df[col]
        
        # Sort by patient_id
        reconciled_df = reconciled_df.sort_values('patient_id').reset_index(drop=True)
        
        return reconciled_df
    
    def print_mismatch_summary(self, mismatches: Dict):
        """Print human-readable mismatch summary (Clinical Contract v2)"""
        safe_print("\n" + "="*60)
        safe_print("CLINICAL RECONCILIATION SUMMARY (Contract v2)")
        safe_print("="*60)
        
        # Contract validation errors
        contract_errors = mismatches.get('contract_validation_errors', [])
        if contract_errors:
            safe_print("\n[CRITICAL] Clinical Contract v2 Validation Failed:")
            for error in contract_errors:
                safe_print(f"  - {error}")
        
        # Identity mismatch
        if mismatches.get('identity_mismatch', False):
            missing_dvhs = len(mismatches.get('missing_dvhs', []))
            missing_clinical = len(mismatches.get('missing_clinical', []))
            
            safe_print(f"\n[CRITICAL] Identity Mismatch Detected:")
            safe_print(f"  Missing in registry (no DVH): {missing_dvhs} patient(s)")
            safe_print(f"  Missing in clinical data: {missing_clinical} patient(s)")
        
        # Duplicates
        duplicates = mismatches.get('duplicates', [])
        if duplicates:
            safe_print(f"\n[ERROR] Duplicate patient_id found: {len(duplicates)}")
            for dup in duplicates[:5]:  # Show first 5
                safe_print(f"  - {dup}")
            if len(duplicates) > 5:
                safe_print(f"  ... and {len(duplicates) - 5} more")
        
        safe_print("\n" + "="*60)
    
    def reconcile(self, clinical_path: Path, auto_accept: bool = False) -> Tuple[bool, Optional[Path]]:
        """
        Perform clinical reconciliation (Clinical Contract v2)
        
        Returns:
            (success, output_path) tuple
        """
        safe_print("\n[CODE0] Clinical Data Reconciliation (Contract v2)")
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
        
        # Check for contract validation errors (STOP if any)
        contract_errors = mismatches.get('contract_validation_errors', [])
        if contract_errors:
            safe_print("\n[STOP] Clinical Contract v2 validation failed. Pipeline stopped.")
            safe_print("\nGenerating dynamic template...")
            
            # Generate dynamic template
            template_df = self.generate_dynamic_template(registry_df)
            template_path = self.output_dir / "clinical_template_required.xlsx"
            
            with pd.ExcelWriter(template_path, engine='openpyxl') as writer:
                template_df.to_excel(writer, index=False, sheet_name='ClinicalData')
            
            safe_print(f"\nClinical data did not satisfy the contract.")
            safe_print(f"A valid template has been generated at {template_path}")
            safe_print(f"Please fill it and rerun.")
            
            return False, None
        
        # Check for identity mismatch (STOP if any)
        if mismatches.get('identity_mismatch', False):
            safe_print("\n[STOP] Identity mismatch detected. Pipeline stopped.")
            safe_print("  patient_id in clinical data must match PrimaryPatientID in DVH registry.")
            safe_print("  No missing, no extra, no duplicates allowed.")
            
            # Generate dynamic template
            template_df = self.generate_dynamic_template(registry_df)
            template_path = self.output_dir / "clinical_template_required.xlsx"
            
            with pd.ExcelWriter(template_path, engine='openpyxl') as writer:
                template_df.to_excel(writer, index=False, sheet_name='ClinicalData')
            
            safe_print(f"\nA valid template has been generated at {template_path}")
            safe_print(f"Please fill it and rerun.")
            
            return False, None
        
        # Check for duplicates (STOP if any)
        duplicates = mismatches.get('duplicates', [])
        if duplicates:
            safe_print("\n[STOP] Duplicate patient_id found. Pipeline stopped.")
            safe_print("  Each patient_id must be unique.")
            
            # Generate dynamic template
            template_df = self.generate_dynamic_template(registry_df)
            template_path = self.output_dir / "clinical_template_required.xlsx"
            
            with pd.ExcelWriter(template_path, engine='openpyxl') as writer:
                template_df.to_excel(writer, index=False, sheet_name='ClinicalData')
            
            safe_print(f"\nA valid template has been generated at {template_path}")
            safe_print(f"Please fill it and rerun.")
            
            return False, None
        
        # All validations passed - generate reconciled output
        safe_print("\n[OK] Clinical Contract v2 validation passed.")
        safe_print("[OK] Identity integrity validated.")
        
        # Generate reconciled output
        reconciled_df = self.generate_reconciled_output(clinical_df_processed)
        
        # Always write reconciled output
        output_path = self.output_dir / "clinical_reconciled.xlsx"
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            reconciled_df.to_excel(writer, index=False, sheet_name='ClinicalData')
        
        safe_print(f"\n[OK] Reconciled clinical data saved to: {output_path}")
        safe_print(f"  Rows: {len(reconciled_df)}")
        safe_print(f"  Columns: {', '.join(reconciled_df.columns.tolist())}")
        
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

