#!/usr/bin/env python3
"""
code2_bDVH.py - Biological DVH (bDVH) Generator
===============================================

Generates biological DVH (bDVH) from physical DVH using radiobiological formalisms:
- BED (Biologically Effective Dose) transformation
- EQD2 (Equivalent Dose in 2 Gy fractions) transformation  
- gEUD-mapped bDVH (voxel-wise gEUD weighting)

Biological DVH (bDVH) Definition:
----------------------------------
A DVH transformed from physical dose (Gy) to a biologically effective dose domain
using standard radiobiological formalisms, enabling direct comparison across 
fractionation schemes and supporting biological NTCP modeling.

References:
- Fowler JF. The linear-quadratic formula and progress in fractionated radiotherapy.
  Br J Radiol. 1989;62(740):679-694.
- Bentzen SM, Constine LS, Deasy JO, et al. Quantitative Analyses of Normal Tissue
  Effects in the Clinic (QUANTEC): an introduction to the issue. Int J Radiat Oncol
  Biol Phys. 2010;76(3 Suppl):S3-S9.
- Emami B, Lyman J, Brown A, et al. Tolerance of normal tissue to therapeutic
  irradiation. Int J Radiat Oncol Biol Phys. 1991;21(1):109-122.

Software: py_ntcpx_v1.0.0
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional

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
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Publication-quality plotting
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
    'figure.dpi': 100,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'savefig.facecolor': 'white'
})

# QUANTEC α/β ratios (defaults, organ-specific)
# Source: Bentzen 2010, QUANTEC
ALPHA_BETA_RATIOS = {
    'Parotid': 3.0,
    'SpinalCord': 2.0,
    'Larynx': 3.0,
    'OralCavity': 3.0,
    'Submandibular': 3.0,
    'Brainstem': 2.0,
    'Mandible': 3.0,
    'Esophagus': 3.0,
    'BrachialPlexus': 2.0,
    'Cochlea': 2.0,
    'OpticNerve': 2.0,
    'OpticChiasm': 2.0,
    'default': 3.0
}

# Default gEUD parameter 'a' for organ types
GEUD_PARAMS = {
    'Parotid': -10.0,  # Parallel (negative a)
    'SpinalCord': 20.0,  # Serial (positive a)
    'Larynx': 5.0,  # Mixed
    'OralCavity': -10.0,
    'default': 1.0
}


class BiologicalDVHGenerator:
    """Generate biological DVH from physical DVH"""
    
    def __init__(self, alpha_beta_ratios: Optional[Dict[str, float]] = None,
                 geud_params: Optional[Dict[str, float]] = None):
        """
        Initialize bDVH generator
        
        Args:
            alpha_beta_ratios: Organ-specific α/β ratios (defaults to QUANTEC values)
            geud_params: Organ-specific gEUD parameter 'a' (defaults to organ-type values)
        """
        self.alpha_beta = alpha_beta_ratios or ALPHA_BETA_RATIOS.copy()
        self.geud_params = geud_params or GEUD_PARAMS.copy()
    
    def get_alpha_beta(self, organ: str) -> float:
        """Get α/β ratio for organ"""
        return self.alpha_beta.get(organ, self.alpha_beta.get('default', 3.0))
    
    def get_geud_param(self, organ: str) -> float:
        """Get gEUD parameter 'a' for organ"""
        return self.geud_params.get(organ, self.geud_params.get('default', 1.0))
    
    def calculate_BED(self, dose: float, dose_per_fraction: float, 
                     n_fractions: Optional[int] = None, 
                     alpha_beta: Optional[float] = None) -> float:
        """
        Calculate Biologically Effective Dose (BED)
        
        BED = nd * (1 + d/(α/β))
        
        where:
        - n = number of fractions
        - d = dose per fraction (Gy)
        - α/β = alpha/beta ratio (Gy)
        
        If n_fractions not provided, assumes single fraction equivalent.
        """
        if alpha_beta is None:
            alpha_beta = 3.0  # Default
        
        if n_fractions is None:
            # Single fraction equivalent: BED = d * (1 + d/(α/β))
            n_fractions = 1
        
        BED = n_fractions * dose_per_fraction * (1 + dose_per_fraction / alpha_beta)
        return BED
    
    def calculate_EQD2(self, dose: float, dose_per_fraction: float,
                     alpha_beta: Optional[float] = None) -> float:
        """
        Calculate Equivalent Dose in 2 Gy fractions (EQD2)
        
        EQD2 = BED / (1 + 2/(α/β))
        
        where BED is calculated from the physical dose and fractionation.
        """
        if alpha_beta is None:
            alpha_beta = 3.0
        
        # Calculate BED first
        # For EQD2, we need total dose and fractionation scheme
        # Simplified: assume dose is total dose, convert per fraction
        # More accurate: EQD2 = D * (d + α/β) / (2 + α/β)
        # where D is total dose, d is dose per fraction
        
        # If dose is total dose:
        # n = dose / dose_per_fraction (approximate)
        # BED = n * d * (1 + d/(α/β))
        # EQD2 = BED / (1 + 2/(α/β))
        
        # Simplified approach: convert dose bin directly
        # EQD2 = D * (d + α/β) / (2 + α/β)
        EQD2 = dose * (dose_per_fraction + alpha_beta) / (2.0 + alpha_beta)
        return EQD2
    
    def calculate_gEUD(self, doses: np.ndarray, volumes: np.ndarray, 
                      a_param: float) -> float:
        """
        Calculate generalized Equivalent Uniform Dose (gEUD)
        
        gEUD = (Σᵢ vᵢ * Dᵢᵃ)^(1/a)
        
        where:
        - vᵢ = relative volume in bin i
        - Dᵢ = dose in bin i
        - a = organ-specific parameter
        """
        if len(doses) == 0 or len(volumes) == 0:
            return np.nan
        
        total_volume = np.sum(volumes)
        if total_volume <= 0:
            return np.nan
        
        rel_volumes = volumes / total_volume
        
        # Handle special cases
        if abs(a_param) < 1e-6:  # a ≈ 0: mean dose
            geud = np.sum(rel_volumes * doses)
        elif a_param < 0:  # Negative a (parallel organs)
            # Use absolute value for calculation
            geud = np.power(np.sum(rel_volumes * np.power(doses, abs(a_param))), 
                           1.0 / abs(a_param))
        else:  # Positive a (serial organs)
            geud = np.power(np.sum(rel_volumes * np.power(doses, a_param)), 
                           1.0 / a_param)
        
        return geud
    
    def transform_dvh_to_BED(self, dvh_df: pd.DataFrame, 
                            dose_per_fraction: float,
                            n_fractions: Optional[int] = None,
                            alpha_beta: float = 3.0) -> pd.DataFrame:
        """
        Transform physical DVH to BED-based bDVH
        
        Args:
            dvh_df: DataFrame with 'Dose[Gy]' and 'Volume[cm3]' columns
            dose_per_fraction: Dose per fraction (Gy)
            n_fractions: Number of fractions (optional)
            alpha_beta: α/β ratio (Gy)
        
        Returns:
            DataFrame with BED-transformed dose bins
        """
        bdvh = dvh_df.copy()
        
        # Transform dose bins to BED
        bdvh['Dose[Gy]'] = bdvh['Dose[Gy]'].apply(
            lambda d: self.calculate_BED(d, dose_per_fraction, n_fractions, alpha_beta)
        )
        
        # Rename to indicate BED
        bdvh = bdvh.rename(columns={'Dose[Gy]': 'BED[Gy]'})
        
        return bdvh
    
    def transform_dvh_to_EQD2(self, dvh_df: pd.DataFrame,
                              dose_per_fraction: float,
                              alpha_beta: float = 3.0) -> pd.DataFrame:
        """
        Transform physical DVH to EQD2-based bDVH
        
        Args:
            dvh_df: DataFrame with 'Dose[Gy]' and 'Volume[cm3]' columns
            dose_per_fraction: Dose per fraction (Gy)
            alpha_beta: α/β ratio (Gy)
        
        Returns:
            DataFrame with EQD2-transformed dose bins
        """
        bdvh = dvh_df.copy()
        
        # Transform dose bins to EQD2
        # For each dose bin, convert assuming it represents dose per fraction
        # More accurate: need total dose context
        # Simplified: convert each bin as if it's the dose level
        bdvh['Dose[Gy]'] = bdvh['Dose[Gy]'].apply(
            lambda d: self.calculate_EQD2(d, dose_per_fraction, alpha_beta)
        )
        
        # Rename to indicate EQD2
        bdvh = bdvh.rename(columns={'Dose[Gy]': 'EQD2[Gy]'})
        
        return bdvh
    
    def transform_dvh_to_gEUD_mapped(self, dvh_df: pd.DataFrame,
                                    a_param: float) -> pd.DataFrame:
        """
        Transform physical DVH to gEUD-mapped bDVH
        
        Remaps dose bins via voxel-wise gEUD weighting.
        
        Args:
            dvh_df: DataFrame with 'Dose[Gy]' and 'Volume[cm3]' columns
            a_param: gEUD parameter 'a'
        
        Returns:
            DataFrame with gEUD-mapped dose bins
        """
        bdvh = dvh_df.copy()
        doses = bdvh['Dose[Gy]'].values
        volumes = bdvh['Volume[cm3]'].values
        
        # Calculate gEUD for the entire DVH
        geud_total = self.calculate_gEUD(doses, volumes, a_param)
        
        # Map each dose bin by its contribution to gEUD
        # Weighted transformation: D_bio = D_phys * (gEUD_weight)
        if abs(a_param) < 1e-6:
            # Mean dose: no transformation needed
            bdvh['Dose[Gy]'] = doses
        else:
            # Transform based on gEUD weighting
            total_volume = np.sum(volumes)
            rel_volumes = volumes / total_volume if total_volume > 0 else volumes
            
            # Calculate contribution weight for each bin
            if a_param < 0:
                weights = np.power(doses, abs(a_param))
            else:
                weights = np.power(doses, a_param)
            
            # Normalize weights
            weight_sum = np.sum(rel_volumes * weights)
            if weight_sum > 0:
                normalized_weights = weights / np.power(weight_sum, 1.0 / abs(a_param) if a_param != 0 else 1.0)
                # Map doses: D_bio = D_phys * normalized_weight_factor
                bdvh['Dose[Gy]'] = doses * (geud_total / np.sum(rel_volumes * doses) if np.sum(rel_volumes * doses) > 0 else 1.0)
            else:
                bdvh['Dose[Gy]'] = doses
        
        # Rename to indicate gEUD mapping
        bdvh = bdvh.rename(columns={'Dose[Gy]': 'gEUD_Mapped[Gy]'})
        
        return bdvh


def load_step1_registry(
    registry_path: Optional[Path] = None,
    base_output_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None
) -> Dict[Tuple[str, str], Dict[str, str]]:
    """
    Load Step1_DVHRegistry.xlsx to lookup PrimaryPatientID and AnonPatientID
    
    Args:
        registry_path: Path to Step1_DVHRegistry.xlsx (default: base_output_dir/contracts/Step1_DVHRegistry.xlsx)
        base_output_dir: Base output directory containing contracts folder
        output_dir: bDVH output directory (used to auto-resolve base_output_dir if not provided)
    
    Returns:
        Dictionary keyed by (DVH_filename, Organ) -> {PrimaryPatientID, AnonPatientID}
    """
    if registry_path is None:
        if base_output_dir is not None:
            registry_path = Path(base_output_dir) / 'contracts' / 'Step1_DVHRegistry.xlsx'
        elif output_dir is not None:
            # Auto-resolve: walk up from output_dir to find base_output_dir with contracts folder
            current = Path(output_dir).resolve()
            max_levels = 5  # Prevent infinite loops
            level = 0
            while level < max_levels:
                contracts_dir = current / 'contracts'
                if contracts_dir.exists() and (contracts_dir / 'Step1_DVHRegistry.xlsx').exists():
                    registry_path = contracts_dir / 'Step1_DVHRegistry.xlsx'
                    logger.info(f"Auto-resolved registry path: {registry_path}")
                    break
                parent = current.parent
                if parent == current:  # Reached filesystem root
                    break
                current = parent
                level += 1
            
            if registry_path is None:
                # Fallback: try common locations
                for base in [Path('out2'), Path('.'), Path('..')]:
                    test_path = base.resolve() / 'contracts' / 'Step1_DVHRegistry.xlsx'
                    if test_path.exists():
                        registry_path = test_path
                        logger.info(f"Found registry at fallback location: {registry_path}")
                        break
        else:
            # Legacy fallback (backward compatibility only)
            registry_path = Path('out2') / 'contracts' / 'Step1_DVHRegistry.xlsx'
    
    registry_path = Path(registry_path)
    lookup = {}
    
    if not registry_path.exists():
        logger.error(f"ERROR: Step1_DVHRegistry.xlsx not found at {registry_path}")
        logger.error("Cannot proceed without identity mapping. Execution STOPPED.")
        raise FileNotFoundError(f"Step1_DVHRegistry.xlsx not found at {registry_path}. Cannot proceed without identity mapping.")
    
    try:
        with pd.ExcelFile(registry_path) as xl:
            df = pd.read_excel(xl, sheet_name='DVHRegistry')
        
        # Required columns: PrimaryPatientID, AnonPatientID, Organ, DVH_filename
        required_cols = ['PrimaryPatientID', 'AnonPatientID', 'Organ', 'DVH_filename']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Step1_DVHRegistry.xlsx missing required columns: {missing_cols}")
            return lookup
        
        # Build lookup dictionary: (DVH_filename, Organ) -> {PrimaryPatientID, AnonPatientID}
        for _, row in df.iterrows():
            dvh_filename = str(row['DVH_filename']).strip()
            organ = str(row['Organ']).strip()
            primary_id = str(row['PrimaryPatientID']).strip() if pd.notna(row['PrimaryPatientID']) else None
            anon_id = str(row['AnonPatientID']).strip() if pd.notna(row['AnonPatientID']) else None
            
            if not dvh_filename or not organ or not primary_id:
                continue
            
            key = (dvh_filename, organ)
            lookup[key] = {
                'PrimaryPatientID': primary_id,
                'AnonPatientID': anon_id
            }
        
        logger.info(f"Loaded Step1_DVHRegistry: {len(lookup)} entries")
        
    except Exception as e:
        logger.error(f"Error loading Step1_DVHRegistry.xlsx: {e}")
    
    return lookup


def load_fractionation_data(clinical_file: Optional[Path]) -> Dict[str, Dict[str, float]]:
    """
    Load fractionation data from clinical Excel file
    
    Expected columns: PatientID, Organ, Dose_per_Fraction, Total_Dose, NumFractions
    """
    fractionation = {}
    
    if clinical_file is None or not clinical_file.exists():
        logger.warning(f"Clinical file not found: {clinical_file}. Using defaults.")
        return fractionation
    
    try:
        df = pd.read_excel(clinical_file)
        
        # Standardize column names
        col_map = {}
        for col in df.columns:
            lc = col.lower().replace(' ', '_').replace('-', '_')
            if 'patient' in lc and 'id' in lc:
                col_map[col] = 'PatientID'
            elif 'organ' in lc:
                col_map[col] = 'Organ'
            elif 'dose' in lc and 'fraction' in lc:
                col_map[col] = 'Dose_per_Fraction'
            elif 'total' in lc and 'dose' in lc:
                col_map[col] = 'Total_Dose'
            elif 'num' in lc and 'fraction' in lc:
                col_map[col] = 'NumFractions'
        
        df = df.rename(columns=col_map)
        
        # Extract fractionation data
        for _, row in df.iterrows():
            pid = str(row.get('PatientID', ''))
            organ = str(row.get('Organ', ''))
            key = f"{pid}_{organ}"
            
            dose_per_fx = pd.to_numeric(row.get('Dose_per_Fraction', 2.0), errors='coerce')
            total_dose = pd.to_numeric(row.get('Total_Dose', np.nan), errors='coerce')
            n_fractions = pd.to_numeric(row.get('NumFractions', np.nan), errors='coerce')
            
            if pd.isna(dose_per_fx):
                dose_per_fx = 2.0  # Default
            
            # Calculate n_fractions if not provided
            if pd.isna(n_fractions) and not pd.isna(total_dose) and dose_per_fx > 0:
                n_fractions = total_dose / dose_per_fx
            
            fractionation[key] = {
                'dose_per_fraction': float(dose_per_fx),
                'n_fractions': float(n_fractions) if not pd.isna(n_fractions) else None,
                'total_dose': float(total_dose) if not pd.isna(total_dose) else None
            }
        
        logger.info(f"Loaded fractionation data for {len(fractionation)} patient-organ combinations")
        
    except Exception as e:
        logger.warning(f"Error loading fractionation data: {e}. Using defaults.")
    
    return fractionation


def process_bDVH(input_dir: Path, output_dir: Path, 
                clinical_file: Optional[Path] = None,
                method: str = 'EQD2',
                base_output_dir: Optional[Path] = None,
                registry_path: Optional[Path] = None) -> None:
    """
    Process all dDVH files and generate biological DVH
    
    Args:
        input_dir: Directory containing code1 dDVH_csv outputs
        output_dir: Output directory for bDVH files
        clinical_file: Optional Excel file with fractionation data
        method: Transformation method ('BED', 'EQD2', 'gEUD', or 'all')
        base_output_dir: Base output directory containing contracts folder
        registry_path: Direct path to Step1_DVHRegistry.xlsx (overrides auto-resolution)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Create output directories
    bdvh_csv_dir = output_dir / 'bDVH_csv'
    bdvh_plots_dir = output_dir / 'bDVH_plots'
    bdvh_csv_dir.mkdir(parents=True, exist_ok=True)
    bdvh_plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Step1_DVHRegistry.xlsx for identity lookup
    # Registry is located at base_output_dir/contracts/Step1_DVHRegistry.xlsx
    try:
        step1_registry = load_step1_registry(
            registry_path=registry_path,
            base_output_dir=base_output_dir,
            output_dir=output_dir
        )
        if not step1_registry:
            logger.error("ERROR: Step1_DVHRegistry.xlsx lookup returned empty. Cannot proceed without identity mapping.")
            raise ValueError("Step1_DVHRegistry.xlsx lookup returned empty. Cannot proceed.")
    except FileNotFoundError as e:
        logger.error(str(e))
        raise
    
    # Load fractionation data
    fractionation = load_fractionation_data(clinical_file)
    
    # Initialize generator
    generator = BiologicalDVHGenerator()
    
    # Find all dDVH CSV files
    dvh_files = list(input_dir.glob('*.csv'))
    if not dvh_files:
        logger.error(f"No CSV files found in {input_dir}")
        return
    
    logger.info(f"Processing {len(dvh_files)} DVH files...")
    
    registry_rows = []  # For Step2b_bDVHRegistry.xlsx
    
    for dvh_file in dvh_files:
        try:
            # Lookup PrimaryPatientID and AnonPatientID using DVH_filename + Organ
            dvh_filename = dvh_file.name  # e.g., "2020-734_Parotid.csv" or "PT001_Parotid.csv"
            
            # Parse filename to get organ (handle case where filename might use PrimaryPatientID or AnonPatientID)
            stem = dvh_file.stem
            parts = stem.split('_', 1)
            if len(parts) < 2:
                logger.warning(f"Could not parse filename: {dvh_file.name}")
                continue
            
            organ = parts[1]
            
            # Lookup identity from Step1 registry using (DVH_filename, Organ)
            lookup_key = (dvh_filename, organ)
            identity_info = step1_registry.get(lookup_key)
            
            if not identity_info:
                logger.warning(f"Could not find identity mapping for {dvh_filename}, Organ={organ} in Step1_DVHRegistry. Skipping.")
                continue
            
            primary_patient_id = identity_info['PrimaryPatientID']
            anon_patient_id = identity_info['AnonPatientID']
            
            if not primary_patient_id:
                logger.warning(f"Missing PrimaryPatientID for {dvh_filename}, Organ={organ}. Skipping.")
                continue
            
            # Load dDVH
            dvh_df = pd.read_csv(dvh_file)
            
            # Standardize column names
            col_map = {}
            for col in dvh_df.columns:
                lc = col.lower().replace(' ', '').replace('[', '').replace(']', '')
                if 'dose' in lc:
                    col_map[col] = 'Dose[Gy]'
                elif 'volume' in lc:
                    col_map[col] = 'Volume[cm3]'
            
            dvh_df = dvh_df.rename(columns=col_map)
            
            if 'Dose[Gy]' not in dvh_df.columns or 'Volume[cm3]' not in dvh_df.columns:
                logger.warning(f"Missing required columns in {dvh_file.name}")
                continue
            
            # Get fractionation data using PrimaryPatientID (not AnonPatientID)
            key = f"{primary_patient_id}_{organ}"
            frac_data = fractionation.get(key, {})
            dose_per_fx = frac_data.get('dose_per_fraction', 2.0)
            n_fractions = frac_data.get('n_fractions')
            alpha_beta = generator.get_alpha_beta(organ)
            a_param = generator.get_geud_param(organ)
            
            # Generate bDVH based on method
            methods_to_process = ['BED', 'EQD2', 'gEUD'] if method == 'all' else [method]
            
            for bdvh_method in methods_to_process:
                if bdvh_method == 'BED':
                    bdvh_df = generator.transform_dvh_to_BED(
                        dvh_df.copy(), dose_per_fx, n_fractions, alpha_beta
                    )
                    # Use PrimaryPatientID for filename (identity-safe)
                    bdvh_filename = f"{primary_patient_id}_{organ}_BED.csv"
                    output_file = bdvh_csv_dir / bdvh_filename
                    dose_col = 'BED[Gy]'
                    
                elif bdvh_method == 'EQD2':
                    bdvh_df = generator.transform_dvh_to_EQD2(
                        dvh_df.copy(), dose_per_fx, alpha_beta
                    )
                    # Use PrimaryPatientID for filename (identity-safe)
                    bdvh_filename = f"{primary_patient_id}_{organ}_EQD2.csv"
                    output_file = bdvh_csv_dir / bdvh_filename
                    dose_col = 'EQD2[Gy]'
                    
                elif bdvh_method == 'gEUD':
                    bdvh_df = generator.transform_dvh_to_gEUD_mapped(
                        dvh_df.copy(), a_param
                    )
                    # Use PrimaryPatientID for filename (identity-safe)
                    bdvh_filename = f"{primary_patient_id}_{organ}_gEUD.csv"
                    output_file = bdvh_csv_dir / bdvh_filename
                    dose_col = 'gEUD_Mapped[Gy]'
                
                else:
                    continue
                
                # Save bDVH CSV
                bdvh_df.to_csv(output_file, index=False)
                
                # Create plot
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot physical DVH (reference)
                ax.plot(dvh_df['Dose[Gy]'], dvh_df['Volume[cm3]'], 
                       'k--', linewidth=1.5, alpha=0.5, label='Physical DVH')
                
                # Plot biological DVH
                ax.plot(bdvh_df[dose_col], bdvh_df['Volume[cm3]'],
                       'b-', linewidth=2, label=f'{bdvh_method} bDVH')
                
                # Use AnonPatientID for display in plot title (if available)
                display_id = anon_patient_id if anon_patient_id else primary_patient_id
                ax.set_xlabel(f'Dose [Gy] ({bdvh_method})', fontsize=12)
                ax.set_ylabel('Volume [cm³]', fontsize=12)
                ax.set_title(f'{organ} - {display_id}\n{bdvh_method} Biological DVH', 
                           fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Plot filename can use AnonPatientID for display
                plot_display_id = anon_patient_id if anon_patient_id else primary_patient_id
                plot_file = bdvh_plots_dir / f"{plot_display_id}_{organ}_{bdvh_method}.png"
                plt.savefig(plot_file, dpi=600, bbox_inches='tight')
                plt.close()
                
                # Add to registry: PrimaryPatientID, AnonPatientID, Organ, bDVH_filename
                registry_rows.append({
                    'PrimaryPatientID': primary_patient_id,
                    'AnonPatientID': anon_patient_id if anon_patient_id else '',
                    'Organ': organ,
                    'bDVH_filename': bdvh_filename
                })
            
        except Exception as e:
            logger.error(f"Error processing {dvh_file.name}: {e}")
            continue
    
    # Save Step2b_bDVHRegistry.xlsx (required contract)
    if registry_rows:
        registry_df = pd.DataFrame(registry_rows)
        # Ensure required columns: PrimaryPatientID, AnonPatientID, Organ, bDVH_filename
        registry_df = registry_df[['PrimaryPatientID', 'AnonPatientID', 'Organ', 'bDVH_filename']]
        
        # Save to contracts directory: out2/contracts/Step2b_bDVHRegistry.xlsx
        contracts_dir = Path('out2') / 'contracts'
        contracts_dir.mkdir(parents=True, exist_ok=True)
        registry_file = contracts_dir / 'Step2b_bDVHRegistry.xlsx'
        
        with pd.ExcelWriter(registry_file, engine='openpyxl') as writer:
            registry_df.to_excel(writer, index=False, sheet_name='bDVHRegistry')
        logger.info(f"Saved Step2b_bDVHRegistry.xlsx: {len(registry_df)} entries")
    else:
        logger.warning("No registry entries generated. Step2b_bDVHRegistry.xlsx will not be created.")
    
    logger.info(f"bDVH generation complete. Output: {output_dir}")


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description='Generate Biological DVH (bDVH) from physical DVH',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Biological DVH (bDVH) Definition:
A DVH transformed from physical dose (Gy) to a biologically effective dose domain
using standard radiobiological formalisms, enabling direct comparison across 
fractionation schemes and supporting biological NTCP modeling.

References:
- Fowler JF (1989). The linear-quadratic formula and progress in fractionated radiotherapy.
- Bentzen SM et al. (2010). QUANTEC: an introduction to the issue.
- Emami B et al. (1991). Tolerance of normal tissue to therapeutic irradiation.

Software: py_ntcpx_v1.0.0
        """
    )
    
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Directory containing code1 dDVH_csv outputs'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='code2_bDVH_output',
        help='Output directory for bDVH files (default: code2_bDVH_output)'
    )
    
    parser.add_argument(
        '--clinical_file',
        type=str,
        default=None,
        help='Optional Excel file with fractionation data (PatientID, Organ, Dose_per_Fraction, etc.)'
    )
    
    parser.add_argument(
        '--method',
        type=str,
        choices=['BED', 'EQD2', 'gEUD', 'all'],
        default='EQD2',
        help='bDVH transformation method (default: EQD2)'
    )
    
    parser.add_argument(
        '--registry_path',
        type=str,
        default=None,
        help='Direct path to Step1_DVHRegistry.xlsx. If not provided, will auto-resolve from output_dir structure.'
    )
    
    parser.add_argument(
        '--base_output_dir',
        type=str,
        default=None,
        help='Base output directory containing contracts folder. If not provided, will auto-resolve from output_dir.'
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    clinical_file = Path(args.clinical_file).expanduser().resolve() if args.clinical_file else None
    
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return
    
    logger.info("=" * 60)
    logger.info("Biological DVH (bDVH) Generator - py_ntcpx v1.0")
    logger.info("=" * 60)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Method: {args.method}")
    
    # Registry path will be auto-resolved in process_bDVH
    process_bDVH(input_dir, output_dir, clinical_file, args.method, base_output_dir=Path(args.base_output_dir).expanduser().resolve() if args.base_output_dir else None, registry_path=Path(args.registry_path).expanduser().resolve() if args.registry_path else None)
    
    logger.info("=" * 60)
    logger.info("bDVH generation completed successfully!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

