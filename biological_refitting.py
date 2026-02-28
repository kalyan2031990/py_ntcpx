#!/usr/bin/env python3
"""
Biological Dose–Response Refitting for Classical NTCP Models
=============================================================

This module provides biologically interpretable dose–response refitting for
classical NTCP models (LKB log-logit, LKB probit, RS Poisson) using mean dose (Gy)
as the dose metric. This is SEPARATE from probability calibration.

Refitted parameters are for interpretability ONLY and do not overwrite prediction pipelines.

Models:
1. LKB Log-Logit: NTCP(D) = 1 / (1 + (TD50 / D) ** gamma50)
2. LKB Probit: NTCP(D) = Phi((D - TD50) / (m * TD50))
3. RS Poisson: NTCP(D) = 1 - exp(-(D / D50) ** gamma)

All models use mean dose (Gy) as x-axis, NOT gEUD.

Author: K. Mondal (North Bengal Medical College, Darjeeling, India.)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.special import expit  # logistic function
import warnings
warnings.filterwarnings('ignore')

# Dose–response refitting uses binomial likelihood with bootstrap resampling
# to estimate biologically interpretable NTCP parameters. These estimates
# are independent of probability calibration and predictive evaluation.


class BiologicalDoseResponseRefitter:
    """Biological dose–response refitting for classical NTCP models"""
    
    def __init__(self, n_bootstrap=1000, config_path=None):
        """
        Initialize refitter
        
        Args:
            n_bootstrap: Number of bootstrap iterations (default: 1000)
            config_path: Path to organ-specific bounds config file (optional)
        """
        self.n_bootstrap = n_bootstrap
        self.fitted_params = {}  # Store fitted parameters per organ
        self.config_path = config_path
        self.organ_configs = self._load_config() if config_path else {}
    
    def _load_config(self):
        """Load organ-specific bounds from config file"""
        if self.config_path is None:
            return {}
        try:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                return config
            else:
                print(f"  Warning: Config file not found: {config_file}, using defaults")
                return {}
        except Exception as e:
            print(f"  Warning: Failed to load config: {e}, using defaults")
            return {}
    
    def logistic_p(self, d, TD50, k):
        """
        Logistic probability function that enforces p(TD50) = 0.5
        
        p(d) = expit((d - TD50) / k)
        
        Args:
            d: Dose (Gy)
            TD50: Dose for 50% complication probability (Gy)
            k: Logistic scale parameter (slope)
        
        Returns:
            NTCP probability
        """
        return expit((d - TD50) / k)
    
    def neg_log_likelihood_binomial(self, params, doses, events, totals):
        """
        Binomial negative log-likelihood for logistic model
        
        Args:
            params: [TD50, k]
            doses: Array of mean doses (Gy)
            events: Array of event counts (binary outcomes)
            totals: Array of total counts (for per-patient binary outcomes, totals=1)
        
        Returns:
            Negative log-likelihood
        """
        TD50, k = params
        if TD50 <= 0 or k <= 0:
            return 1e10
        
        p = self.logistic_p(doses, TD50, k)
        # Protect against 0/1
        p = np.clip(p, 1e-6, 1 - 1e-6)
        # Binomial negative log likelihood
        nll = -np.sum(events * np.log(p) + (totals - events) * np.log(1 - p))
        return nll
    
    def fit_biological_logistic(self, doses, events, totals, bounds, x0=None):
        """
        Fit logistic model using binomial NLL
        
        Args:
            doses: Array of mean doses (Gy)
            events: Array of event counts (binary outcomes, 0/1)
            totals: Array of total counts (for per-patient binary outcomes, totals=1)
            bounds: [(TD50_min, TD50_max), (k_min, k_max)]
            x0: Initial guess [TD50, k] (optional)
        
        Returns:
            dict with fitted parameters
        """
        if x0 is None:
            x0 = [np.median(doses), (bounds[1][0] + bounds[1][1]) / 2.0]
        
        res = minimize(
            self.neg_log_likelihood_binomial,
            x0,
            args=(doses, events, totals),
            bounds=bounds,
            method='L-BFGS-B',
            options={'maxiter': 10000}
        )
        
        if not res.success:
            raise RuntimeError(f"Biological refit optimization failed: {res.message}")
        
        TD50_hat, k_hat = res.x
        return {
            'TD50': float(TD50_hat),
            'k': float(k_hat),
            'nll': float(res.fun),
            'converged': True
        }
    
    def bootstrap_ci_logistic(self, doses, events, totals, bounds, n_iter=None, seed=42):
        """
        Bootstrap confidence intervals for logistic model
        
        Args:
            doses: Array of mean doses (Gy)
            events: Array of event counts (binary outcomes)
            totals: Array of total counts
            bounds: [(TD50_min, TD50_max), (k_min, k_max)]
            n_iter: Number of bootstrap iterations (default: self.n_bootstrap)
            seed: Random seed for reproducibility
        
        Returns:
            dict with point estimate and 95% CI, or None if failed
        """
        if n_iter is None:
            n_iter = self.n_bootstrap
        
        rng = np.random.RandomState(seed)
        params = []
        N = len(doses)
        
        # Fit point estimate first
        try:
            point_estimate = self.fit_biological_logistic(doses, events, totals, bounds)
        except Exception as e:
            print(f"      Warning: Point estimate failed: {e}")
            return None
        
        # Bootstrap resampling
        for i in range(n_iter):
            # Nonparametric bootstrap: resample indices
            idx = rng.choice(N, size=N, replace=True)
            doses_bs = doses[idx]
            events_bs = events[idx]
            totals_bs = totals[idx]
            try:
                fit = self.fit_biological_logistic(doses_bs, events_bs, totals_bs, bounds)
                params.append((fit['TD50'], fit['k']))
            except Exception:
                continue
        
        params = np.array(params)
        if params.size == 0:
            return {
                'point_estimate': point_estimate,
                'ci95': None,
                'n_successful': 0,
                'unstable': True
            }
        
        ci = {
            'TD50': {
                'ci95_lower': float(np.percentile(params[:, 0], 2.5)),
                'ci95_upper': float(np.percentile(params[:, 0], 97.5))
            },
            'k': {
                'ci95_lower': float(np.percentile(params[:, 1], 2.5)),
                'ci95_upper': float(np.percentile(params[:, 1], 97.5))
            }
        }
        
        return {
            'point_estimate': point_estimate,
            'ci95': ci,
            'n_successful': len(params),
            'unstable': len(params) < 0.7 * n_iter
        }
    
    def ntcp_lkb_loglogit_biological(self, mean_dose, TD50, gamma50):
        """
        LKB log-logit model using mean dose (biological interpretation)
        
        NTCP(D) = 1 / (1 + (TD50 / D) ** gamma50)
        
        Args:
            mean_dose: Mean dose (Gy)
            TD50: Dose for 50% complication probability (Gy)
            gamma50: Steepness parameter
        
        Returns:
            NTCP probability
        """
        if np.any(mean_dose <= 0) or TD50 <= 0 or gamma50 <= 0:
            return np.zeros_like(mean_dose) if np.ndim(mean_dose) > 0 else 0.0
        
        mean_dose = np.asarray(mean_dose)
        ratio = TD50 / np.maximum(mean_dose, 1e-6)
        ntcp = 1.0 / (1.0 + np.power(ratio, gamma50))
        ntcp = np.clip(ntcp, 1e-6, 1.0 - 1e-6)
        return ntcp[0] if ntcp.size == 1 else ntcp
    
    def ntcp_lkb_probit_biological(self, mean_dose, TD50, m):
        """
        LKB probit model using mean dose (biological interpretation)
        
        NTCP(D) = Phi((D - TD50) / (m * TD50))
        
        Args:
            mean_dose: Mean dose (Gy)
            TD50: Dose for 50% complication probability (Gy)
            m: Steepness parameter
        
        Returns:
            NTCP probability
        """
        if np.any(mean_dose <= 0) or TD50 <= 0 or m <= 0:
            return np.zeros_like(mean_dose) if np.ndim(mean_dose) > 0 else 0.0
        
        mean_dose = np.asarray(mean_dose)
        t = (mean_dose - TD50) / (m * TD50)
        ntcp = norm.cdf(t)
        ntcp = np.clip(ntcp, 1e-6, 1.0 - 1e-6)
        return ntcp[0] if ntcp.size == 1 else ntcp
    
    def ntcp_rs_poisson_biological(self, mean_dose, D50, gamma):
        """
        RS Poisson model using mean dose (biological interpretation)
        
        NTCP(D) = 1 - exp(-(D / D50) ** gamma)
        
        Args:
            mean_dose: Mean dose (Gy)
            D50: Dose for 50% complication probability (Gy)
            gamma: Steepness parameter
        
        Returns:
            NTCP probability
        """
        if np.any(mean_dose <= 0) or D50 <= 0 or gamma <= 0:
            return np.zeros_like(mean_dose) if np.ndim(mean_dose) > 0 else 0.0
        
        mean_dose = np.asarray(mean_dose)
        dose_ratio = mean_dose / D50
        exponent = np.power(dose_ratio, gamma)
        ntcp = 1.0 - np.exp(-exponent)
        ntcp = np.clip(ntcp, 1e-6, 1.0 - 1e-6)
        return ntcp[0] if ntcp.size == 1 else ntcp
    
    def negative_log_likelihood_lkb_loglogit(self, params, mean_doses, outcomes):
        """
        Negative log-likelihood for LKB log-logit model
        
        Args:
            params: [TD50, gamma50]
            mean_doses: Array of mean doses (Gy)
            outcomes: Array of binary outcomes (0/1)
        
        Returns:
            Negative log-likelihood
        """
        TD50, gamma50 = params
        if TD50 <= 0 or gamma50 <= 0:
            return 1e10
        
        ntcp = self.ntcp_lkb_loglogit_biological(mean_doses, TD50, gamma50)
        ntcp = np.clip(ntcp, 1e-6, 1.0 - 1e-6)
        
        log_likelihood = np.sum(
            outcomes * np.log(ntcp) + (1 - outcomes) * np.log(1 - ntcp)
        )
        return -log_likelihood
    
    def negative_log_likelihood_lkb_probit(self, params, mean_doses, outcomes):
        """
        Negative log-likelihood for LKB probit model
        
        Args:
            params: [TD50, m]
            mean_doses: Array of mean doses (Gy)
            outcomes: Array of binary outcomes (0/1)
        
        Returns:
            Negative log-likelihood
        """
        TD50, m = params
        if TD50 <= 0 or m <= 0:
            return 1e10
        
        ntcp = self.ntcp_lkb_probit_biological(mean_doses, TD50, m)
        ntcp = np.clip(ntcp, 1e-6, 1.0 - 1e-6)
        
        log_likelihood = np.sum(
            outcomes * np.log(ntcp) + (1 - outcomes) * np.log(1 - ntcp)
        )
        return -log_likelihood
    
    def negative_log_likelihood_rs_poisson(self, params, mean_doses, outcomes):
        """
        Negative log-likelihood for RS Poisson model
        
        Args:
            params: [D50, gamma]
            mean_doses: Array of mean doses (Gy)
            outcomes: Array of binary outcomes (0/1)
        
        Returns:
            Negative log-likelihood
        """
        D50, gamma = params
        if D50 <= 0 or gamma <= 0:
            return 1e10
        
        ntcp = self.ntcp_rs_poisson_biological(mean_doses, D50, gamma)
        ntcp = np.clip(ntcp, 1e-6, 1.0 - 1e-6)
        
        log_likelihood = np.sum(
            outcomes * np.log(ntcp) + (1 - outcomes) * np.log(1 - ntcp)
        )
        return -log_likelihood
    
    def fit_lkb_loglogit(self, mean_doses, outcomes, bounds=None):
        """
        Fit LKB log-logit model using mean dose
        
        Args:
            mean_doses: Array of mean doses (Gy)
            outcomes: Array of binary outcomes (0/1)
            bounds: Optional bounds [(TD50_min, TD50_max), (gamma50_min, gamma50_max)]
        
        Returns:
            dict with fitted parameters and convergence info
        """
        # Initialize TD50 using median mean dose
        initial_TD50 = np.median(mean_doses)
        initial_gamma50 = 1.0
        
        if bounds is None:
            bounds = [(20.0, 45.0), (0.1, 5.0)]  # TD50, gamma50 (literature-bounded)
        
        result = minimize(
            self.negative_log_likelihood_lkb_loglogit,
            [initial_TD50, initial_gamma50],
            args=(mean_doses, outcomes),
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
                'log_likelihood': float(-result.fun) if hasattr(result, 'fun') else None,
                'converged': False,
                'message': str(result.message) if hasattr(result, 'message') else 'Optimization failed'
            }
    
    def fit_lkb_probit(self, mean_doses, outcomes, bounds=None):
        """
        Fit LKB probit model using mean dose
        
        Args:
            mean_doses: Array of mean doses (Gy)
            outcomes: Array of binary outcomes (0/1)
            bounds: Optional bounds [(TD50_min, TD50_max), (m_min, m_max)]
        
        Returns:
            dict with fitted parameters and convergence info
        """
        # Initialize TD50 using median mean dose
        initial_TD50 = np.median(mean_doses)
        initial_m = 0.2
        
        if bounds is None:
            bounds = [(20.0, 45.0), (0.05, 0.8)]  # TD50, m (literature-bounded)
        
        result = minimize(
            self.negative_log_likelihood_lkb_probit,
            [initial_TD50, initial_m],
            args=(mean_doses, outcomes),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000}
        )
        
        if result.success:
            return {
                'TD50': float(result.x[0]),
                'm': float(result.x[1]),
                'log_likelihood': float(-result.fun),
                'converged': True
            }
        else:
            return {
                'TD50': float(result.x[0]),
                'm': float(result.x[1]),
                'log_likelihood': float(-result.fun) if hasattr(result, 'fun') else None,
                'converged': False,
                'message': str(result.message) if hasattr(result, 'message') else 'Optimization failed'
            }
    
    def fit_rs_poisson(self, mean_doses, outcomes, bounds=None):
        """
        Fit RS Poisson model using mean dose
        
        Args:
            mean_doses: Array of mean doses (Gy)
            outcomes: Array of binary outcomes (0/1)
            bounds: Optional bounds [(D50_min, D50_max), (gamma_min, gamma_max)]
        
        Returns:
            dict with fitted parameters and convergence info
        """
        # Initialize D50 using median mean dose
        initial_D50 = np.median(mean_doses)
        initial_gamma = 1.0
        
        if bounds is None:
            bounds = [(20.0, 45.0), (0.1, 5.0)]  # D50, gamma (literature-bounded)
        
        result = minimize(
            self.negative_log_likelihood_rs_poisson,
            [initial_D50, initial_gamma],
            args=(mean_doses, outcomes),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000}
        )
        
        if result.success:
            return {
                'D50': float(result.x[0]),
                'gamma': float(result.x[1]),
                'log_likelihood': float(-result.fun),
                'converged': True
            }
        else:
            return {
                'D50': float(result.x[0]),
                'gamma': float(result.x[1]),
                'log_likelihood': float(-result.fun) if hasattr(result, 'fun') else None,
                'converged': False,
                'message': str(result.message) if hasattr(result, 'message') else 'Optimization failed'
            }
    
    def bootstrap_confidence_intervals(self, mean_doses, outcomes, fit_func, param_names, n_bootstrap=None, bounds=None):
        """
        Calculate bootstrap confidence intervals for fitted parameters
        
        Args:
            mean_doses: Array of mean doses (Gy)
            outcomes: Array of binary outcomes (0/1)
            fit_func: Function to fit model (returns dict with parameters)
            param_names: List of parameter names to extract
            n_bootstrap: Number of bootstrap iterations (default: self.n_bootstrap)
            bounds: Optional bounds to pass to fit_func
        
        Returns:
            dict with point estimates, bootstrap samples, and 95% CI
        """
        if n_bootstrap is None:
            n_bootstrap = self.n_bootstrap
        
        n_patients = len(mean_doses)
        if n_patients < 10:
            return {
                'point_estimate': None,
                'bootstrap_samples': [],
                'ci95_lower': None,
                'ci95_upper': None,
                'n_successful': 0,
                'unstable': True
            }
        
        # Fit point estimate
        if bounds is not None:
            point_estimate_result = fit_func(mean_doses, outcomes, bounds)
        else:
            point_estimate_result = fit_func(mean_doses, outcomes)
        if not point_estimate_result or not point_estimate_result.get('converged', False):
            return {
                'point_estimate': point_estimate_result,
                'bootstrap_samples': [],
                'ci95_lower': None,
                'ci95_upper': None,
                'n_successful': 0,
                'unstable': True
            }
        
        point_estimate = {name: point_estimate_result.get(name) for name in param_names}
        
        # Bootstrap resampling
        bootstrap_samples = []
        for i in range(n_bootstrap):
            # Resample patients with replacement
            bootstrap_indices = np.random.choice(n_patients, size=n_patients, replace=True)
            bootstrap_mean_doses = mean_doses[bootstrap_indices]
            bootstrap_outcomes = outcomes[bootstrap_indices]
            
            try:
                if bounds is not None:
                    bootstrap_result = fit_func(bootstrap_mean_doses, bootstrap_outcomes, bounds)
                else:
                    bootstrap_result = fit_func(bootstrap_mean_doses, bootstrap_outcomes)
                if bootstrap_result and bootstrap_result.get('converged', False):
                    bootstrap_params = {name: bootstrap_result.get(name) for name in param_names}
                    bootstrap_samples.append(bootstrap_params)
            except Exception as e:
                # Skip failed optimizations, log warning if needed
                continue
        
        n_successful = len(bootstrap_samples)
        
        if n_successful < 0.7 * n_bootstrap:
            # Flag CI as unstable if <70% converge
            unstable = True
        else:
            unstable = False
        
        if n_successful == 0:
            return {
                'point_estimate': point_estimate,
                'bootstrap_samples': [],
                'ci95_lower': None,
                'ci95_upper': None,
                'n_successful': 0,
                'unstable': True
            }
        
        # Calculate 95% percentile CI (2.5%, 97.5%)
        ci_results = {}
        for param_name in param_names:
            param_values = [sample[param_name] for sample in bootstrap_samples]
            ci_lower = np.percentile(param_values, 2.5)
            ci_upper = np.percentile(param_values, 97.5)
            ci_results[param_name] = {
                'ci95_lower': float(ci_lower),
                'ci95_upper': float(ci_upper)
            }
        
        return {
            'point_estimate': point_estimate,
            'bootstrap_samples': bootstrap_samples,
            'ci95': ci_results,
            'n_successful': n_successful,
            'unstable': unstable
        }
    
    def refit_organ_models(self, organ_data, organ):
        """
        Refit biological dose-response model using logistic form with p(TD50)=0.5
        
        Args:
            organ_data: DataFrame with columns: mean_dose, Observed_Toxicity (or event column)
            organ: Organ name
        
        Returns:
            dict with fitted parameters and bootstrap CI
        """
        # Safety check: ensure we're using binary events, not calibrated probabilities
        if 'refit_source' in organ_data.columns and organ_data['refit_source'].iloc[0] == 'probabilities':
            raise ValueError(f"ERROR: Cannot use calibrated probabilities for biological refit of {organ}. "
                           f"Refit must use binary events (0/1) from clinical outcomes.")
        
        # Check for mean_dose column
        if 'mean_dose' not in organ_data.columns:
            print(f"  Warning: mean_dose not available for {organ}, skipping biological refitting")
            return None
        
        # Check for event column (Observed_Toxicity or event)
        event_col = None
        if 'Observed_Toxicity' in organ_data.columns:
            event_col = 'Observed_Toxicity'
        elif 'event' in organ_data.columns:
            event_col = 'event'
        else:
            print(f"  Warning: No event column (Observed_Toxicity or event) found for {organ}, skipping biological refitting")
            return None
        
        # Safety assert: event column must be integer/binary
        if not organ_data[event_col].dtype in [np.int64, np.int32, int] and not all(organ_data[event_col].dropna().isin([0, 1])):
            raise ValueError(f"ERROR: Event column '{event_col}' for {organ} must be integer binary (0/1), "
                           f"not probabilities. Found dtype: {organ_data[event_col].dtype}")
        
        # Filter valid data
        valid_mask = (
            organ_data['mean_dose'].notna() &
            (organ_data['mean_dose'] > 0) &
            organ_data[event_col].notna() &
            organ_data[event_col].isin([0, 1])
        )
        
        organ_data_valid = organ_data[valid_mask].copy()
        
        if len(organ_data_valid) < 10:
            print(f"  Warning: Insufficient valid data for {organ} (n={len(organ_data_valid)}), skipping biological refitting")
            return None
        
        # Check event counts per bin (safety check)
        dose_bins = np.linspace(organ_data_valid['mean_dose'].min(), organ_data_valid['mean_dose'].max(), 6)
        bin_counts = []
        for i in range(len(dose_bins) - 1):
            mask = (organ_data_valid['mean_dose'] >= dose_bins[i]) & (organ_data_valid['mean_dose'] < dose_bins[i+1])
            bin_counts.append(mask.sum())
        
        if min(bin_counts) < 3:
            print(f"  Warning: Some dose bins have <3 events for {organ}, refit may be unstable")
        
        # Extract data
        doses = organ_data_valid['mean_dose'].values
        events = organ_data_valid[event_col].values.astype(int)
        totals = np.ones_like(events)  # For per-patient binary outcomes, totals=1
        
        print(f"    Refitting biological model for {organ} (n={len(organ_data_valid)})...")
        
        # Get organ-specific bounds from config, or use defaults
        if organ in self.organ_configs and 'refit' in self.organ_configs[organ]:
            config = self.organ_configs[organ]['refit']
            TD50_bounds = config.get('TD50_bounds', [15.0, 60.0])
            slope_bounds = config.get('slope_bounds', [0.05, 5.0])
            n_bootstrap = config.get('bootstrap_iterations', self.n_bootstrap)
        else:
            # Default bounds
            TD50_bounds = [15.0, 60.0]
            slope_bounds = [0.05, 5.0]
            n_bootstrap = self.n_bootstrap
        
        bounds = [(TD50_bounds[0], TD50_bounds[1]), (slope_bounds[0], slope_bounds[1])]
        
        # Fit logistic model with bootstrap CI
        try:
            print(f"      Fitting logistic model (p(TD50)=0.5)...")
            logistic_result = self.bootstrap_ci_logistic(
                doses, events, totals, bounds, n_iter=n_bootstrap, seed=42
            )
            
            if logistic_result is None:
                print(f"      Warning: Logistic fitting failed for {organ}")
                self.fitted_params[organ] = None
                return None
            
            # Store logistic fit result (Tier A: Biological dose-response)
            results = {
                'Logistic': logistic_result
            }
            
            # Tier B: Fit classical models independently with their own likelihoods
            # Get bounds from config or use defaults
            if organ in self.organ_configs and 'refit' in self.organ_configs[organ]:
                config = self.organ_configs[organ]['refit']
                td50_bounds_config = config.get('TD50_bounds', [20.0, 45.0])
                slope_bounds_config = config.get('slope_bounds', [0.05, 5.0])
            else:
                td50_bounds_config = [20.0, 45.0]  # Literature-bounded default
                slope_bounds_config = [0.05, 5.0]
            
            # Fit LKB Log-Logit independently
            try:
                print(f"      Fitting LKB Log-Logit (independent)...")
                lkb_loglogit_bounds = [(td50_bounds_config[0], td50_bounds_config[1]), (0.1, 5.0)]  # TD50, gamma50
                lkb_loglogit_ci = self.bootstrap_confidence_intervals(
                    doses, events,
                    self.fit_lkb_loglogit,
                    ['TD50', 'gamma50'],
                    n_bootstrap=max(500, n_bootstrap),  # At least 500 iterations
                    bounds=lkb_loglogit_bounds
                )
                results['LKB_LogLogit'] = lkb_loglogit_ci
            except Exception as e:
                print(f"      Warning: LKB Log-Logit fitting failed: {e}")
                results['LKB_LogLogit'] = None
            
            # Fit LKB Probit independently
            try:
                print(f"      Fitting LKB Probit (independent)...")
                lkb_probit_bounds = [(td50_bounds_config[0], td50_bounds_config[1]), (0.05, 0.8)]  # TD50, m
                lkb_probit_ci = self.bootstrap_confidence_intervals(
                    doses, events,
                    self.fit_lkb_probit,
                    ['TD50', 'm'],
                    n_bootstrap=max(500, n_bootstrap),
                    bounds=lkb_probit_bounds
                )
                results['LKB_Probit'] = lkb_probit_ci
            except Exception as e:
                print(f"      Warning: LKB Probit fitting failed: {e}")
                results['LKB_Probit'] = None
            
            # Fit RS Poisson independently
            try:
                print(f"      Fitting RS Poisson (independent)...")
                rs_poisson_bounds = [(td50_bounds_config[0], td50_bounds_config[1]), (0.1, 5.0)]  # D50, gamma
                rs_poisson_ci = self.bootstrap_confidence_intervals(
                    doses, events,
                    self.fit_rs_poisson,
                    ['D50', 'gamma'],
                    n_bootstrap=max(500, n_bootstrap),
                    bounds=rs_poisson_bounds
                )
                results['RS_Poisson'] = rs_poisson_ci
            except Exception as e:
                print(f"      Warning: RS Poisson fitting failed: {e}")
                results['RS_Poisson'] = None
            
        except Exception as e:
            print(f"      Warning: Biological refit failed for {organ}: {e}")
            import traceback
            traceback.print_exc()
            results = None
        
        self.fitted_params[organ] = results
        return results
    
    def save_parameters(self, output_dir):
        """
        Save fitted biological and classical parameters to JSON files
        
        Args:
            output_dir: Output directory path
        
        Returns:
            Tuple of (biological_params_file, classical_params_file)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save biological (logistic) parameters
        bio_params_file = output_path / 'local_biological_parameters.json'
        bio_json_params = {}
        
        # Save classical parameters separately
        classical_params_file = output_path / 'local_classical_parameters.json'
        classical_json_params = {}
        
        for organ, models in self.fitted_params.items():
            if models is None:
                continue
                
            # Biological (logistic) parameters
            if 'Logistic' in models and models['Logistic'] is not None:
                result = models['Logistic']
                if result.get('point_estimate') is not None:
                    bio_json_params[organ] = {
                        'Logistic': {
                            'point_estimate': result['point_estimate'],
                            'ci95': result.get('ci95', {}),
                            'n_bootstrap_successful': result.get('n_successful', 0),
                            'unstable': result.get('unstable', False)
                        }
                    }
            
            # Classical model parameters
            classical_json_params[organ] = {}
            for model_name in ['LKB_LogLogit', 'LKB_Probit', 'RS_Poisson']:
                if model_name in models and models[model_name] is not None:
                    result = models[model_name]
                    if result.get('point_estimate') is not None:
                        classical_json_params[organ][model_name] = {
                            'point_estimate': result['point_estimate'],
                            'ci95': result.get('ci95', {}),
                            'n_bootstrap_successful': result.get('n_successful', 0),
                            'unstable': result.get('unstable', False)
                        }
                    else:
                        classical_json_params[organ][model_name] = None
                else:
                    classical_json_params[organ][model_name] = None
        
        # Save biological parameters
        with open(bio_params_file, 'w') as f:
            json.dump(bio_json_params, f, indent=2)
        print(f"  Saved biological parameters to: {bio_params_file}")
        
        # Save classical parameters
        with open(classical_params_file, 'w') as f:
            json.dump(classical_json_params, f, indent=2)
        print(f"  Saved classical parameters to: {classical_params_file}")
        
        return bio_params_file, classical_params_file
    
    def plot_dose_response_organ(self, organ_data, organ, output_dir):
        """
        Plot unified publication-quality biological dose–response curves for a specific organ
        
        Args:
            organ_data: DataFrame with mean_dose and Observed_Toxicity
            organ: Organ name
            output_dir: Output directory for saving plots
        """
        if organ not in self.fitted_params:
            return
        
        organ_params = self.fitted_params[organ]
        
        # Check if we have any valid models
        has_valid_model = any(
            result is not None and result.get('point_estimate') is not None
            for result in organ_params.values()
        )
        
        if not has_valid_model:
            print(f"  Warning: No valid biological models for {organ}, skipping plot")
            return
        
        # Filter valid data - check for event column
        event_col = 'Observed_Toxicity' if 'Observed_Toxicity' in organ_data.columns else 'event'
        if event_col not in organ_data.columns:
            return
        
        valid_mask = (
            organ_data['mean_dose'].notna() &
            (organ_data['mean_dose'] > 0) &
            organ_data[event_col].notna()
        )
        organ_data_valid = organ_data[valid_mask].copy()
        
        if len(organ_data_valid) == 0:
            return
        
        # Create figure with publication-quality settings
        fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
        ax.set_facecolor('white')
        
        # Get observed data points (NO jitter, NO binning, NO error bars)
        mean_doses_obs = organ_data_valid['mean_dose'].values
        toxicity_obs = organ_data_valid[event_col].values.astype(float)
        
        # Simple scatter: small circles, semi-transparent, neutral gray
        ax.scatter(mean_doses_obs, toxicity_obs, 
                  color='gray', alpha=0.5, s=20, zorder=5, edgecolors='none')
        
        # Generate dose range for curves
        dose_min = max(0.1, mean_doses_obs.min() * 0.5)
        dose_max = mean_doses_obs.max() * 1.2
        dose_range = np.linspace(dose_min, dose_max, 200)
        
        # Plot curves in order: Biological Logistic first, then classical models
        legend_entries = []
        
        # 1) Biological Logistic (reference curve) - Solid thick dark blue, NO CI bands
        if (organ_params.get('Logistic') is not None and
            organ_params['Logistic'].get('point_estimate') is not None):
            params = organ_params['Logistic']['point_estimate']
            ntcp_curve = self.logistic_p(dose_range, params['TD50'], params['k'])
            ax.plot(dose_range, ntcp_curve, color='#003366', linestyle='-', 
                   linewidth=3.0, zorder=4, label='Biological logistic')
            
            td50_val = params['TD50']
            legend_entries.append(f"Biological logistic: TD50 = {td50_val:.1f} Gy")
        
        # 2) LKB Log-Logit - Dashed steel blue
        if (organ_params.get('LKB_LogLogit') is not None and
            organ_params['LKB_LogLogit'].get('point_estimate') is not None):
            params = organ_params['LKB_LogLogit']['point_estimate']
            ntcp_curve = self.ntcp_lkb_loglogit_biological(dose_range, params['TD50'], params['gamma50'])
            ax.plot(dose_range, ntcp_curve, color='#4682B4', linestyle='--', 
                   linewidth=2.0, zorder=3, label='LKB (Log-logit)')
            
            td50_val = params['TD50']
            gamma50_val = params['gamma50']
            legend_entries.append(f"LKB (Log-logit): TD50 = {td50_val:.1f} Gy, γ50 = {gamma50_val:.2f}")
        
        # 3) LKB Probit - Dashed red
        if (organ_params.get('LKB_Probit') is not None and
            organ_params['LKB_Probit'].get('point_estimate') is not None):
            params = organ_params['LKB_Probit']['point_estimate']
            ntcp_curve = self.ntcp_lkb_probit_biological(dose_range, params['TD50'], params['m'])
            ax.plot(dose_range, ntcp_curve, color='#DC143C', linestyle='--', 
                   linewidth=2.0, zorder=3, label='LKB (Probit)')
            
            td50_val = params['TD50']
            m_val = params['m']
            legend_entries.append(f"LKB (Probit): TD50 = {td50_val:.1f} Gy, m = {m_val:.2f}")
        
        # 4) RS Poisson - Dash-dot goldenrod
        if (organ_params.get('RS_Poisson') is not None and
            organ_params['RS_Poisson'].get('point_estimate') is not None):
            params = organ_params['RS_Poisson']['point_estimate']
            ntcp_curve = self.ntcp_rs_poisson_biological(dose_range, params['D50'], params['gamma'])
            ax.plot(dose_range, ntcp_curve, color='#DAA520', linestyle='-.', 
                   linewidth=2.0, zorder=3, label='RS Poisson')
            
            d50_val = params['D50']
            gamma_val = params['gamma']
            legend_entries.append(f"RS (Poisson): D50 = {d50_val:.1f} Gy, γ = {gamma_val:.2f}")
        
        # Format axes - publication quality
        ax.set_xlabel('Mean dose (Gy)', fontsize=14, fontweight='bold')
        ax.set_ylabel('NTCP', fontsize=14, fontweight='bold')
        ax.set_title(f'Biological Dose–Response: {organ}', fontsize=16, fontweight='bold', pad=15)
        
        # Set limits
        ax.set_xlim([dose_min, dose_max])
        ax.set_ylim([-0.05, 1.05])
        
        # Thin gridlines (light gray)
        ax.grid(True, alpha=0.3, color='lightgray', linewidth=0.5, linestyle='-')
        ax.set_axisbelow(True)
        
        # PART 3: Validation warnings (console only, not in plot)
        td50_values = []
        if organ_params.get('LKB_LogLogit') and organ_params['LKB_LogLogit'] and organ_params['LKB_LogLogit'].get('point_estimate'):
            td50_values.append(organ_params['LKB_LogLogit']['point_estimate'].get('TD50'))
        if organ_params.get('LKB_Probit') and organ_params['LKB_Probit'] and organ_params['LKB_Probit'].get('point_estimate'):
            td50_values.append(organ_params['LKB_Probit']['point_estimate'].get('TD50'))
        if organ_params.get('RS_Poisson') and organ_params['RS_Poisson'] and organ_params['RS_Poisson'].get('point_estimate'):
            td50_values.append(organ_params['RS_Poisson']['point_estimate'].get('D50'))
        
        if len(td50_values) >= 2:
            td50_array = np.array([v for v in td50_values if v is not None])
            if len(td50_array) >= 2:
                if np.allclose(td50_array, td50_array[0], atol=1e-3):
                    print(f"  [WARNING] All classical TD50/D50 values are numerically identical (±1e-3): {td50_array}")
        
        if event_col in organ_data_valid.columns:
            nan_count = organ_data_valid[event_col].isna().sum()
            if nan_count > 0:
                print(f"  [WARNING] observed_event contains {nan_count} NaN values")
            unique_vals = organ_data_valid[event_col].dropna().unique()
            if not all(v in [0, 1] for v in unique_vals):
                print(f"  [WARNING] observed_event contains non-binary values: {unique_vals}")
        
        # Legend inside figure, bottom-right - parameters only (NO CI)
        if legend_entries:
            ax.legend(legend_entries, loc='lower right', fontsize=10, frameon=True, 
                     fancybox=True, shadow=False, framealpha=0.9, facecolor='white', 
                     edgecolor='gray', borderpad=0.8)
        
        plt.tight_layout()
        
        # Save plots to plots/ subdirectory
        output_path = Path(output_dir)
        plots_dir = output_path / 'plots'
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as PNG at ≥600 DPI (using 600 DPI for publication)
        plot_name = f'dose_response_biological_{organ}.png'
        plot_path_png = plots_dir / plot_name
        fig.savefig(plot_path_png, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
        
        # Also save SVG for vector editing
        plot_name_svg = f'dose_response_biological_{organ}.svg'
        plot_path_svg = plots_dir / plot_name_svg
        fig.savefig(plot_path_svg, format='svg', bbox_inches='tight', facecolor='white', edgecolor='none')
        
        plt.close(fig)
        
        print(f"    Saved dose–response plot: {plot_path_png}")


def refit_all_organs(results_df, output_dir, n_bootstrap=1000, config_path=None):
    """
    Refit biological dose–response models for all organs
    
    Args:
        results_df: DataFrame with columns: Organ, mean_dose, Observed_Toxicity (or event)
        output_dir: Output directory for saving parameters and plots
        n_bootstrap: Number of bootstrap iterations (default: 1000)
        config_path: Path to organ-specific bounds config file (optional)
    
    Returns:
        BiologicalDoseResponseRefitter instance
    """
    print("\n" + "="*60)
    print("Biological Dose–Response Refitting")
    print("="*60)
    print("# Dose–response refitting uses binomial NLL on logistic form (p(TD50)=0.5)")
    print("# with bootstrap resampling to estimate biologically interpretable NTCP parameters.")
    print("# These estimates are independent of probability calibration and predictive evaluation.")
    print("="*60)
    
    # Safety check: ensure we're not using calibrated probabilities
    if any(col.startswith('Calibrated_') or 'calibrated' in col.lower() for col in results_df.columns):
        prob_cols = [col for col in results_df.columns if 'calibrated' in col.lower()]
        print(f"  Warning: Found calibrated probability columns: {prob_cols}")
        print("  These will NOT be used for biological refit (only binary events)")
    
    refitter = BiologicalDoseResponseRefitter(n_bootstrap=n_bootstrap, config_path=config_path)
    
    # Process each organ separately
    for organ in sorted(results_df['Organ'].unique()):
        organ_data = results_df[results_df['Organ'] == organ].copy()
        
        if len(organ_data) < 10:
            print(f"\n  Skipping {organ}: insufficient data (n={len(organ_data)})")
            continue
        
        print(f"\n  Processing {organ} (n={len(organ_data)})...")
        
        # Refit models
        refitter.refit_organ_models(organ_data, organ)
        
        # Plot dose–response curves
        refitter.plot_dose_response_organ(organ_data, organ, output_dir)
    
    # Save parameters
    refitter.save_parameters(output_dir)
    
    return refitter
