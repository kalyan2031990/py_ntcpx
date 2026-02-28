#!/usr/bin/env python3
"""
Tier 1: Legacy-A (QUANTEC LKB / RS, fixed)
==============================================
Wraps existing QUANTEC LKB and RS models from code3_ntcp_analysis_ml.py
This module provides a clean interface to the fixed literature parameters.
"""

import sys
from pathlib import Path

# Import existing NTCP calculator
sys.path.insert(0, str(Path(__file__).parent.parent))
from code3_ntcp_analysis_ml import NTCPCalculator

import numpy as np
import pandas as pd


class LegacyFixedNTCP:
    """Tier 1: QUANTEC literature-based NTCP models (fixed parameters)"""
    
    def __init__(self):
        """Initialize with existing NTCP calculator"""
        self.calculator = NTCPCalculator()
    
    def calculate_ntcp_lkb_loglogit(self, geud, organ, dose_per_fraction=2.0):
        """
        Calculate LKB Log-Logistic NTCP using QUANTEC parameters
        
        Args:
            geud: Generalized Equivalent Uniform Dose (Gy)
            organ: Organ name (Parotid, Larynx, SpinalCord)
            dose_per_fraction: Dose per fraction (default 2.0 Gy)
        
        Returns:
            dict with NTCP value and parameters used
        """
        if organ not in self.calculator.literature_params:
            return {'NTCP': np.nan, 'error': f'No parameters for {organ}'}
        
        params = self.calculator.literature_params[organ]['LKB_LogLogit']
        geud_eqd2 = self.calculator.convert_to_eqd2(
            geud, params['alpha_beta'], dose_per_fraction
        )
        ntcp = self.calculator.ntcp_lkb_loglogit(
            geud_eqd2, params['TD50'], params['gamma50']
        )
        
        return {
            'NTCP': ntcp,
            'gEUD_physical': geud,
            'gEUD_EQD2': geud_eqd2,
            'parameters': params,
            'model': 'LKB_LogLogit_QUANTEC'
        }
    
    def calculate_ntcp_lkb_probit(self, dose_metrics, organ, dose_per_fraction=2.0):
        """
        Calculate LKB Probit NTCP using QUANTEC parameters
        
        Args:
            dose_metrics: dict with v_effective, max_dose
            organ: Organ name
            dose_per_fraction: Dose per fraction (default 2.0 Gy)
        
        Returns:
            dict with NTCP value and parameters used
        """
        if organ not in self.calculator.literature_params:
            return {'NTCP': np.nan, 'error': f'No parameters for {organ}'}
        
        params = self.calculator.literature_params[organ]['LKB_Probit']
        dose_metrics_copy = dose_metrics.copy()
        
        ntcp = self.calculator.ntcp_lkb_probit(
            dose_metrics_copy, params['TD50'], params['m'], params['n']
        )
        
        return {
            'NTCP': ntcp,
            'parameters': params,
            'model': 'LKB_Probit_QUANTEC'
        }
    
    def calculate_ntcp_rs_poisson(self, dvh, organ, dose_per_fraction=2.0):
        """
        Calculate RS Poisson NTCP using QUANTEC parameters
        
        Args:
            dvh: DataFrame with dose_gy and volume_cm3 columns
            organ: Organ name
            dose_per_fraction: Dose per fraction (default 2.0 Gy)
        
        Returns:
            dict with NTCP value and parameters used
        """
        if organ not in self.calculator.literature_params:
            return {'NTCP': np.nan, 'error': f'No parameters for {organ}'}
        
        params = self.calculator.literature_params[organ]['RS_Poisson']
        dvh_eqd2 = dvh.copy()
        dvh_eqd2['dose_gy'] = dvh_eqd2['dose_gy'].apply(
            lambda d: self.calculator.convert_to_eqd2(
                d, params['alpha_beta'], dose_per_fraction
            )
        )
        
        ntcp = self.calculator.ntcp_rs_poisson(
            dvh_eqd2, params['D50'], params['gamma'], params['s']
        )
        
        return {
            'NTCP': ntcp,
            'parameters': params,
            'model': 'RS_Poisson_QUANTEC'
        }
    
    def calculate_all_legacy_fixed(self, dvh, dose_metrics, organ, dose_per_fraction=2.0):
        """
        Calculate all Tier 1 (Legacy-A) NTCP models
        
        Returns:
            dict with all model results
        """
        results = {}
        
        # LKB Log-Logistic
        geud = dose_metrics.get('gEUD', np.nan)
        if not np.isnan(geud):
            results['NTCP_LKB_LogLogit_QUANTEC'] = self.calculate_ntcp_lkb_loglogit(
                geud, organ, dose_per_fraction
            )['NTCP']
        
        # LKB Probit
        results['NTCP_LKB_Probit_QUANTEC'] = self.calculate_ntcp_lkb_probit(
            dose_metrics, organ, dose_per_fraction
        )['NTCP']
        
        # RS Poisson
        results['NTCP_RS_Poisson_QUANTEC'] = self.calculate_ntcp_rs_poisson(
            dvh, organ, dose_per_fraction
        )['NTCP']
        
        return results

