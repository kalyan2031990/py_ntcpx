#!/usr/bin/env python3
"""
Contract-Driven Staged Execution System for NTCP Pipeline
==========================================================

Enforces data integrity through canonical contract files.
Each step produces a contract that must be validated before proceeding.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple
import re
import unicodedata

# Windows-safe encoding configuration
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except (AttributeError, ValueError):
    pass


def normalize_dvh_id(x):
    """Normalize DVH ID for canonical matching (same as in code3)"""
    if x is None:
        return None
    x = str(x)
    x = unicodedata.normalize("NFKC", x)
    x = x.strip().lower()
    x = x.replace(".csv", "")
    x = x.replace(" ", "").replace("_", "")
    x = re.sub(r"[^0-9\-]", "", x)
    return x


def enforce_identity_contract(df, required_cols=("PrimaryPatientID", "Organ")):
    """
    Enforce identity governance - validate that identity fields are present and valid.
    
    Args:
        df: DataFrame to validate
        required_cols: Tuple of required identity columns (default: PrimaryPatientID, Organ)
    
    Raises:
        ValueError if identity contract is violated
    """
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Identity contract violated: missing {col}")
    
    if df[list(required_cols)].isnull().any().any():
        raise ValueError("Identity contract violated: null identity fields")
    
    if df[list(required_cols)].duplicated().any():
        raise ValueError("Identity contract violated: duplicate PrimaryPatientID+Organ")


class ContractValidator:
    """Validates pipeline contracts and ensures data integrity"""
    
    def __init__(self, contracts_dir: Path):
        self.contracts_dir = Path(contracts_dir)
        self.contracts_dir.mkdir(parents=True, exist_ok=True)
    
    def get_contract_path(self, step_name: str) -> Path:
        """Get path to contract file for a step"""
        return self.contracts_dir / f"{step_name}.xlsx"
    
    def create_step1_registry(self, dvh_dir: Path, output_registry: Path) -> pd.DataFrame:
        """
        Create authoritative Step1_DVHRegistry from all DVH files.
        
        This is the single source of truth for what DVH data exists.
        Uses PrimaryPatientID (real patient ID) for matching.
        """
        dvh_dir = Path(dvh_dir)
        registry_rows = []
        
        # Try to load from processed_dvh.xlsx first (has PrimaryPatientID and AnonPatientID)
        processed_excel = dvh_dir / "processed_dvh.xlsx"
        if processed_excel.exists():
            try:
                with pd.ExcelFile(processed_excel) as xl:
                    processed_df = pd.read_excel(xl, sheet_name='DVH_Data')
                
                # Check if PrimaryPatientID exists
                if 'PrimaryPatientID' in processed_df.columns:
                    for _, row in processed_df.iterrows():
                        primary_id = row['PrimaryPatientID']
                        anon_id = row.get('AnonPatientID', None)
                        organ = row['Organ']
                        dvh_filename = f"{primary_id}_{organ}.csv"
                        
                        dvh_id_norm = normalize_dvh_id(primary_id)
                        if dvh_id_norm is None or dvh_id_norm == "":
                            continue
                        
                        registry_rows.append({
                            'PrimaryPatientID': primary_id,
                            'AnonPatientID': anon_id,
                            'Organ': organ,
                            'DVH_filename': dvh_filename,
                            'DVH_ID_norm': dvh_id_norm
                        })
                else:
                    # Fallback: scan files directly
                    raise ValueError("PrimaryPatientID not in processed_dvh.xlsx")
            except Exception as e:
                print(f"Warning: Could not load from processed_dvh.xlsx: {e}. Scanning files directly.")
                processed_excel = None
        
        # Fallback: scan CSV files directly if Excel not available
        if not processed_excel or not processed_excel.exists() or len(registry_rows) == 0:
            dDVH_dir = dvh_dir / "dDVH_csv"
            if not dDVH_dir.exists():
                dDVH_dir = dvh_dir  # Fallback to main directory
            
            for dvh_file in sorted(dDVH_dir.glob("*.csv")):
                # Parse filename: {PrimaryPatientID}_{Organ}.csv
                stem = dvh_file.stem
                if '_' not in stem:
                    continue
                
                parts = stem.split('_')
                primary_id = parts[0]
                organ = '_'.join(parts[1:])  # Handle organs with underscores
                
                # Normalize PrimaryPatientID
                dvh_id_norm = normalize_dvh_id(primary_id)
                
                if dvh_id_norm is None or dvh_id_norm == "":
                    continue
                
                registry_rows.append({
                    'PrimaryPatientID': primary_id,
                    'AnonPatientID': None,  # Will be mapped later if needed
                    'Organ': organ,
                    'DVH_filename': dvh_file.name,
                    'DVH_ID_norm': dvh_id_norm
                })
        
        registry_df = pd.DataFrame(registry_rows)
        
        # Save registry
        with pd.ExcelWriter(output_registry, engine='openpyxl') as writer:
            registry_df.to_excel(writer, index=False, sheet_name='DVHRegistry')
        
        return registry_df
    
    def normalize_dvh_id(self, x):
        """Normalize DVH ID for canonical matching (exposed method)"""
        return normalize_dvh_id(x)
    
    def load_step1_registry(self) -> Optional[pd.DataFrame]:
        """Load Step1_DVHRegistry contract"""
        registry_path = self.get_contract_path("Step1_DVHRegistry")
        if not registry_path.exists():
            return None
        
        try:
            with pd.ExcelFile(registry_path) as xl:
                df = pd.read_excel(xl, sheet_name='DVHRegistry')
            return df
        except Exception as e:
            print(f"[ERROR] Failed to load Step1_DVHRegistry: {e}")
            return None
    
    def validate_clinical_data(self, clinical_path: Path, registry_df: pd.DataFrame) -> Tuple[bool, Optional[Path], str]:
        """
        Validate clinical Excel against Step1_DVHRegistry.
        Uses PrimaryPatientID for matching (identity-safe).
        Handles backward compatibility for files with only AnonPatientID (PT IDs).
        
        Returns:
            (is_valid, template_path, error_message)
        """
        if not clinical_path.exists():
            return False, None, f"Clinical data file not found: {clinical_path}"
        
        try:
            with pd.ExcelFile(clinical_path) as xl:
                clinical_df = pd.read_excel(xl, sheet_name=xl.sheet_names[0])
        except Exception as e:
            return False, None, f"Failed to read clinical data: {e}"
        
        # Check for PrimaryPatientID (mandatory)
        has_primary = 'PrimaryPatientID' in clinical_df.columns
        has_anon = 'AnonPatientID' in clinical_df.columns
        
        # If only PatientID exists (and looks like AnonPatientID), map via registry
        if not has_primary and 'PatientID' in clinical_df.columns:
            # Check if PatientID looks like AnonPatientID (PT format)
            sample_pid = clinical_df['PatientID'].dropna().iloc[0] if len(clinical_df['PatientID'].dropna()) > 0 else ""
            if isinstance(sample_pid, str) and sample_pid.strip().upper().startswith('PT'):
                # PatientID is AnonPatientID - map via registry
                if 'Organ' in clinical_df.columns and 'AnonPatientID' in registry_df.columns:
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
                        unmapped = clinical_df['PrimaryPatientID'].isna().sum()
                        if unmapped > 0:
                            error_msg = f"{unmapped} clinical rows could not be mapped to PrimaryPatientID from registry (PatientID + Organ)"
                            template_path = self.generate_clinical_template(registry_df, [])
                            return False, template_path, error_msg
                else:
                    error_msg = "Cannot map PatientID (PT format) to PrimaryPatientID: registry missing AnonPatientID or Organ column"
                    template_path = self.generate_clinical_template(registry_df, [])
                    return False, template_path, error_msg
        
        # Check required columns
        if 'PrimaryPatientID' not in clinical_df.columns:
            error_msg = "Missing required column: PrimaryPatientID"
            template_path = self.generate_clinical_template(registry_df, [])
            return False, template_path, error_msg
        
        if 'Organ' not in clinical_df.columns:
            error_msg = "Missing required column: Organ"
            template_path = self.generate_clinical_template(registry_df, [])
            return False, template_path, error_msg
        
        # Check for Toxicity/Endpoint column
        has_toxicity = any(col.lower() in ['toxicity', 'observed_toxicity', 'endpoint', 'event'] 
                          for col in clinical_df.columns)
        if not has_toxicity:
            error_msg = "Missing toxicity/endpoint column (required: Toxicity, Observed_Toxicity, Endpoint, or Event)"
            template_path = self.generate_clinical_template(registry_df, ['Toxicity'])
            return False, template_path, error_msg
        
        # Normalize PrimaryPatientID for matching
        clinical_df['PrimaryPatientID_norm'] = clinical_df['PrimaryPatientID'].apply(normalize_dvh_id)
        registry_df['PrimaryPatientID_norm'] = registry_df['PrimaryPatientID'].apply(normalize_dvh_id)
        
        # Match on (PrimaryPatientID, Organ) - this is the identity-safe key
        registry_keys = set(zip(registry_df['PrimaryPatientID_norm'].dropna(), registry_df['Organ'].dropna()))
        clinical_keys = set(zip(clinical_df['PrimaryPatientID_norm'].dropna(), clinical_df['Organ'].dropna()))
        
        # Check for clinical rows without matching DVH
        missing_dvhs = clinical_keys - registry_keys
        if missing_dvhs:
            error_msg = f"{len(missing_dvhs)} clinical rows have no matching DVH in registry (PrimaryPatientID + Organ)"
            template_path = self.generate_clinical_template(registry_df, [])
            return False, template_path, error_msg
        
        # Check for DVHs without clinical data (warning, not error)
        missing_clinical = registry_keys - clinical_keys
        if missing_clinical:
            print(f"[WARNING] {len(missing_clinical)} DVHs in registry have no matching clinical data")
        
        return True, None, ""
    
    def generate_clinical_template(self, registry_df: pd.DataFrame, additional_cols: List[str]) -> Path:
        """Generate clinical data template from registry using PrimaryPatientID"""
        template_path = self.contracts_dir / "clinical_template.xlsx"
        
        # Create template with registry data - use PrimaryPatientID as key
        if 'PrimaryPatientID' not in registry_df.columns:
            raise ValueError("Registry missing PrimaryPatientID column - cannot generate template")
        
        template_df = registry_df[['PrimaryPatientID', 'Organ']].copy()
        
        # Add AnonPatientID if available (optional, for display)
        if 'AnonPatientID' in registry_df.columns:
            template_df['AnonPatientID'] = registry_df['AnonPatientID']
        else:
            template_df['AnonPatientID'] = None
        
        # Add required columns
        for col in ['Toxicity', 'Observed_Toxicity', 'dose_per_fraction', 'FollowupMonths', 'Age', 'Sex']:
            if col not in template_df.columns:
                template_df[col] = None
        
        # Add any additional required columns
        for col in additional_cols:
            if col not in template_df.columns:
                template_df[col] = None
        
        # Remove duplicates (same PrimaryPatientID, Organ combination)
        template_df = template_df.drop_duplicates(subset=['PrimaryPatientID', 'Organ'])
        
        # Sort by PrimaryPatientID and Organ
        template_df = template_df.sort_values(['PrimaryPatientID', 'Organ']).reset_index(drop=True)
        
        with pd.ExcelWriter(template_path, engine='openpyxl') as writer:
            template_df.to_excel(writer, index=False, sheet_name='ClinicalData')
        
        return template_path
    
    def validate_contract_exists(self, step_name: str) -> bool:
        """Check if required contract file exists"""
        contract_path = self.get_contract_path(step_name)
        exists = contract_path.exists()
        if not exists:
            print(f"[ERROR] Required contract not found: {step_name}.xlsx")
        return exists
    
    def validate_step3_dataset(self, step3_path: Path, registry_df: pd.DataFrame) -> Tuple[bool, str]:
        """Validate Step3_NTCPDataset against Step1_DVHRegistry using PrimaryPatientID"""
        if not step3_path.exists():
            return False, "Step3_NTCPDataset.xlsx not found"
        
        try:
            with pd.ExcelFile(step3_path) as xl:
                # Load from first sheet (usually 'Complete Results')
                step3_df = pd.read_excel(xl, sheet_name=xl.sheet_names[0])
        except Exception as e:
            return False, f"Failed to read Step3 dataset: {e}"
        
        # Check required columns - use PrimaryPatientID (identity-safe, mandatory)
        if 'PrimaryPatientID' not in step3_df.columns:
            return False, "Step3 dataset missing PrimaryPatientID column (required for identity-safe matching)"
        
        if 'Organ' not in step3_df.columns:
            return False, "Step3 dataset missing Organ column"
        
        # Normalize PrimaryPatientID for comparison
        step3_df['PrimaryPatientID_norm'] = step3_df['PrimaryPatientID'].apply(normalize_dvh_id)
        registry_df['PrimaryPatientID_norm'] = registry_df['PrimaryPatientID'].apply(normalize_dvh_id)
        
        # Match on (PrimaryPatientID, Organ) - identity-safe key
        registry_keys = set(zip(registry_df['PrimaryPatientID_norm'].dropna(), registry_df['Organ'].dropna()))
        step3_keys = set(zip(step3_df['PrimaryPatientID_norm'].dropna(), step3_df['Organ'].dropna()))
        
        # Check all registry entries are in Step3
        missing_in_step3 = registry_keys - step3_keys
        if missing_in_step3:
            return False, f"{len(missing_in_step3)} DVHs in registry missing from Step3 dataset (PrimaryPatientID + Organ)"
        
        return True, ""
    
    def log_match_statistics(self, registry_df: pd.DataFrame, clinical_df: Optional[pd.DataFrame] = None, 
                            step3_df: Optional[pd.DataFrame] = None):
        """Print comprehensive match statistics using PrimaryPatientID (identity-safe)"""
        print("\n" + "="*60)
        print("CONTRACT VALIDATION STATISTICS (Identity-Safe)")
        print("="*60)
        
        # Registry statistics - use PrimaryPatientID
        print(f"\n[Step1 Registry]")
        print(f"  Total DVHs: {len(registry_df)}")
        organs = registry_df['Organ'].unique() if 'Organ' in registry_df.columns else []
        print(f"  Organs: {', '.join(sorted(organs))}")
        if 'PrimaryPatientID' in registry_df.columns:
            unique_primary_ids = registry_df['PrimaryPatientID'].nunique()
            print(f"  Unique PrimaryPatientID: {unique_primary_ids}")
        if 'AnonPatientID' in registry_df.columns:
            unique_anon_ids = registry_df['AnonPatientID'].nunique()
            print(f"  Unique AnonPatientID: {unique_anon_ids}")
        
        if clinical_df is not None:
            # Clinical data statistics - use PrimaryPatientID for matching
            print(f"\n[Clinical Data]")
            print(f"  Total rows: {len(clinical_df)}")
            
            # Match rate using PrimaryPatientID + Organ (identity-safe key)
            if 'PrimaryPatientID' in registry_df.columns and 'PrimaryPatientID' in clinical_df.columns:
                # Normalize PrimaryPatientID for matching
                registry_df['PrimaryPatientID_norm'] = registry_df['PrimaryPatientID'].apply(normalize_dvh_id)
                clinical_df['PrimaryPatientID_norm'] = clinical_df['PrimaryPatientID'].apply(normalize_dvh_id)
                
                # Match on (PrimaryPatientID, Organ) key
                registry_keys = set(zip(registry_df['PrimaryPatientID_norm'].dropna(), registry_df['Organ'].dropna()))
                clinical_keys = set(zip(clinical_df['PrimaryPatientID_norm'].dropna(), clinical_df['Organ'].dropna()))
                matched = len(registry_keys & clinical_keys)
                match_rate = (matched / len(registry_keys) * 100) if len(registry_keys) > 0 else 0
                print(f"  Match rate (PrimaryPatientID + Organ): {matched}/{len(registry_keys)} ({match_rate:.1f}%)")
        
        if step3_df is not None:
            # Step3 statistics - use PrimaryPatientID for matching
            print(f"\n[Step3 Dataset]")
            print(f"  Total rows: {len(step3_df)}")
            
            if 'PrimaryPatientID' in registry_df.columns and 'PrimaryPatientID' in step3_df.columns:
                # Normalize PrimaryPatientID for matching
                registry_df['PrimaryPatientID_norm'] = registry_df['PrimaryPatientID'].apply(normalize_dvh_id)
                step3_df['PrimaryPatientID_norm'] = step3_df['PrimaryPatientID'].apply(normalize_dvh_id)
                
                # Match on (PrimaryPatientID, Organ) key
                registry_keys = set(zip(registry_df['PrimaryPatientID_norm'].dropna(), registry_df['Organ'].dropna()))
                step3_keys = set(zip(step3_df['PrimaryPatientID_norm'].dropna(), step3_df['Organ'].dropna()))
                matched = len(registry_keys & step3_keys)
                match_rate = (matched / len(registry_keys) * 100) if len(registry_keys) > 0 else 0
                print(f"  Match rate (PrimaryPatientID + Organ): {matched}/{len(registry_keys)} ({match_rate:.1f}%)")
        
        print("="*60 + "\n")

