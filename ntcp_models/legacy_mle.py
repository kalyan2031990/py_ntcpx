#!/usr/bin/env python3
"""
Tier 2: Legacy-B (MLE-refitted LKB / RS)
=========================================
Maximum likelihood estimation (Moiseenko-style) for LKB and RS models.
Fits TD50, m, n for LKB and D50, gamma, s for RS to cohort data.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import json
import warnings
warnings.filterwarnings('ignore')

# Import existing NTCP calculator for model functions
sys.path.insert(0, str(Path(__file__).parent.parent))
from code3_ntcp_analysis_ml import NTCPCalculator


class LegacyMLENTCP:
    """Tier 2: MLE-refitted LKB and RS models"""
    
    def __init__(self):
        """Initialize with existing NTCP calculator"""
        self.calculator = NTCPCalculator()
        self.fitted_params = {}  # Store fitted parameters per organ
    
    def log_likelihood_lkb(self, params, geud_values, outcomes, dose_metrics_list, model_type='probit'):
        """
        Calculate log-likelihood for LKB model
        
        Args:
            params: [TD50, m, n] for probit or [TD50, gamma50] for loglogit
            geud_values: array of gEUD values
            outcomes: array of binary outcomes (0/1)
            dose_metrics_list: list of dose_metrics dicts for probit model
            model_type: 'probit' or 'loglogit'
        
        Returns:
            Negative log-likelihood (for minimization)
        """
        if model_type == 'probit':
            TD50, m, n = params
            if TD50 <= 0 or m <= 0 or n < 0:
                return 1e10
            
            ntcp_values = []
            for i, (geud, dose_metrics) in enumerate(zip(geud_values, dose_metrics_list)):
                dose_metrics_copy = dose_metrics.copy()
                dose_metrics_copy['v_effective'] = dose_metrics.get('v_effective', 1.0)
                ntcp = self.calculator.ntcp_lkb_probit(dose_metrics_copy, TD50, m, n)
                ntcp_values.append(ntcp)
            
            ntcp_array = np.array(ntcp_values)
        else:  # loglogit
            TD50, gamma50 = params
            if TD50 <= 0 or gamma50 <= 0:
                return 1e10
            
            ntcp_array = np.array([
                self.calculator.ntcp_lkb_loglogit(geud, TD50, gamma50)
                for geud in geud_values
            ])
        
        # Clip to avoid log(0)
        ntcp_array = np.clip(ntcp_array, 1e-10, 1 - 1e-10)
        
        # Log-likelihood: sum(r_i * ln(p_i) + (1-r_i) * ln(1-p_i))
        log_likelihood = np.sum(
            outcomes * np.log(ntcp_array) + (1 - outcomes) * np.log(1 - ntcp_array)
        )
        
        return -log_likelihood  # Negative for minimization
    
    def log_likelihood_rs(self, params, dvh_list, outcomes):
        """
        Calculate log-likelihood for RS Poisson model
        
        Args:
            params: [D50, gamma, s]
            dvh_list: list of DVH DataFrames
            outcomes: array of binary outcomes (0/1)
        
        Returns:
            Negative log-likelihood (for minimization)
        """
        D50, gamma, s = params
        if D50 <= 0 or gamma <= 0 or s <= 0:
            return 1e10
        
        ntcp_values = []
        for dvh in dvh_list:
            ntcp = self.calculator.ntcp_rs_poisson(dvh, D50, gamma, s)
            ntcp_values.append(ntcp)
        
        ntcp_array = np.array(ntcp_values)
        ntcp_array = np.clip(ntcp_array, 1e-10, 1 - 1e-10)
        
        log_likelihood = np.sum(
            outcomes * np.log(ntcp_array) + (1 - outcomes) * np.log(1 - ntcp_array)
        )
        
        return -log_likelihood
    
    def fit_lkb_probit_mle(self, geud_values, dose_metrics_list, outcomes, initial_params=None):
        """
        Fit LKB Probit model using MLE
        
        Args:
            geud_values: array of gEUD values
            dose_metrics_list: list of dose_metrics dicts
            outcomes: array of binary outcomes
            initial_params: [TD50, m, n] initial guess
        
        Returns:
            dict with fitted parameters and convergence info
        """
        if initial_params is None:
            # Use median gEUD as initial TD50 guess
            initial_params = [np.median(geud_values), 0.2, 0.5]
        
        bounds = [(1.0, 200.0), (0.01, 1.0), (0.01, 2.0)]  # TD50, m, n
        
        result = minimize(
            self.log_likelihood_lkb,
            initial_params,
            args=(geud_values, outcomes, dose_metrics_list, 'probit'),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000}
        )
        
        if result.success:
            return {
                'TD50': float(result.x[0]),
                'm': float(result.x[1]),
                'n': float(result.x[2]),
                'log_likelihood': float(-result.fun),
                'converged': True
            }
        else:
            return {
                'TD50': float(result.x[0]),
                'm': float(result.x[1]),
                'n': float(result.x[2]),
                'log_likelihood': float(-result.fun),
                'converged': False,
                'message': result.message
            }
    
    def fit_lkb_loglogit_mle(self, geud_values, outcomes, initial_params=None):
        """
        Fit LKB Log-Logistic model using MLE
        
        Args:
            geud_values: array of gEUD values
            outcomes: array of binary outcomes
            initial_params: [TD50, gamma50] initial guess
        
        Returns:
            dict with fitted parameters
        """
        if initial_params is None:
            initial_params = [np.median(geud_values), 1.0]
        
        bounds = [(1.0, 200.0), (0.1, 10.0)]  # TD50, gamma50
        
        result = minimize(
            self.log_likelihood_lkb,
            initial_params,
            args=(geud_values, outcomes, [], 'loglogit'),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000}
        )
        
        if result.success:
            return {
                'TD50': float(result.x[0]),
                'gamma50': float(result.x[1]),
                'log_likelihood': float(-result.fun),
                'converged': True
            }
        else:
            return {
                'TD50': float(result.x[0]),
                'gamma50': float(result.x[1]),
                'log_likelihood': float(-result.fun),
                'converged': False,
                'message': result.message
            }
    
    def fit_rs_poisson_mle(self, dvh_list, outcomes, initial_params=None):
        """
        Fit RS Poisson model using MLE
        
        Args:
            dvh_list: list of DVH DataFrames
            outcomes: array of binary outcomes
            initial_params: [D50, gamma, s] initial guess
        
        Returns:
            dict with fitted parameters
        """
        if initial_params is None:
            # Estimate initial D50 from mean doses
            mean_doses = [dvh['dose_gy'].mean() for dvh in dvh_list]
            initial_params = [np.median(mean_doses), 1.0, 0.1]
        
        bounds = [(1.0, 200.0), (0.1, 10.0), (0.001, 10.0)]  # D50, gamma, s
        
        result = minimize(
            self.log_likelihood_rs,
            initial_params,
            args=(dvh_list, outcomes),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000}
        )
        
        if result.success:
            return {
                'D50': float(result.x[0]),
                'gamma': float(result.x[1]),
                's': float(result.x[2]),
                'log_likelihood': float(-result.fun),
                'converged': True
            }
        else:
            return {
                'D50': float(result.x[0]),
                'gamma': float(result.x[1]),
                's': float(result.x[2]),
                'log_likelihood': float(-result.fun),
                'converged': False,
                'message': result.message
            }
    
    def fit_models_for_organ(self, organ_data, organ):
        """
        Fit MLE models for a specific organ
        
        Args:
            organ_data: DataFrame with columns: gEUD, v_effective, max_dose, Observed_Toxicity, etc.
            organ: Organ name
        
        Returns:
            dict with fitted parameters for all models
        """
        # Prepare data
        geud_values = organ_data['gEUD'].values
        outcomes = organ_data['Observed_Toxicity'].values.astype(int)
        
        # Prepare dose_metrics for probit model
        dose_metrics_list = []
        for _, row in organ_data.iterrows():
            dose_metrics_list.append({
                'gEUD': row['gEUD'],
                'v_effective': row.get('v_effective', 1.0),
                'max_dose': row.get('max_dose', row['gEUD'])
            })
        
        # Fit models
        fitted_params = {}
        
        # LKB Probit MLE
        try:
            probit_params = self.fit_lkb_probit_mle(geud_values, dose_metrics_list, outcomes)
            fitted_params['LKB_Probit_MLE'] = probit_params
        except Exception as e:
            print(f"  Warning: LKB Probit MLE fitting failed for {organ}: {e}")
            fitted_params['LKB_Probit_MLE'] = None
        
        # LKB LogLogit MLE
        try:
            loglogit_params = self.fit_lkb_loglogit_mle(geud_values, outcomes)
            fitted_params['LKB_LogLogit_MLE'] = loglogit_params
        except Exception as e:
            print(f"  Warning: LKB LogLogit MLE fitting failed for {organ}: {e}")
            fitted_params['LKB_LogLogit_MLE'] = None
        
        # RS Poisson MLE (requires DVH data - skip if not available)
        # Note: This would require loading DVH files, which is done in the main script
        
        self.fitted_params[organ] = fitted_params
        return fitted_params
    
    def calculate_ntcp_lkb_mle(self, dose_metrics, organ, model_type='probit'):
        """
        Calculate NTCP using MLE-fitted LKB parameters
        
        Args:
            dose_metrics: dict with gEUD, v_effective, max_dose
            organ: Organ name
            model_type: 'probit' or 'loglogit'
        
        Returns:
            NTCP value
        """
        if organ not in self.fitted_params:
            return np.nan
        
        model_key = f'LKB_{model_type.capitalize()}_MLE'
        if model_key not in self.fitted_params[organ] or self.fitted_params[organ][model_key] is None:
            return np.nan
        
        params = self.fitted_params[organ][model_key]
        
        if model_type == 'probit':
            return self.calculator.ntcp_lkb_probit(
                dose_metrics, params['TD50'], params['m'], params['n']
            )
        else:  # loglogit
            geud = dose_metrics.get('gEUD', np.nan)
            if np.isnan(geud):
                return np.nan
            return self.calculator.ntcp_lkb_loglogit(
                geud, params['TD50'], params['gamma50']
            )
    
    def calculate_ntcp_rs_mle(self, dvh, organ):
        """
        Calculate NTCP using MLE-fitted RS parameters
        
        Args:
            dvh: DataFrame with dose_gy and volume_cm3
            organ: Organ name
        
        Returns:
            NTCP value
        """
        if organ not in self.fitted_params:
            return np.nan
        
        if 'RS_Poisson_MLE' not in self.fitted_params[organ] or self.fitted_params[organ]['RS_Poisson_MLE'] is None:
            return np.nan
        
        params = self.fitted_params[organ]['RS_Poisson_MLE']
        return self.calculator.ntcp_rs_poisson(
            dvh, params['D50'], params['gamma'], params['s']
        )
    
    def save_parameters(self, output_dir):
        """Save fitted parameters to JSON file"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        params_file = output_path / 'model_parameters_mle.json'
        
        # Convert to JSON-serializable format
        json_params = {}
        for organ, models in self.fitted_params.items():
            json_params[organ] = {}
            for model_name, params in models.items():
                if params is not None:
                    json_params[organ][model_name] = {
                        k: v for k, v in params.items()
                        if k != 'message'  # Skip non-serializable fields
                    }
                else:
                    json_params[organ][model_name] = None
        
        with open(params_file, 'w') as f:
            json.dump(json_params, f, indent=2)
        
        print(f"  Saved MLE parameters to: {params_file}")
        return params_file

