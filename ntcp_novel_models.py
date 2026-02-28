#!/usr/bin/env python3
"""
Novel NTCP Models for Head & Neck Cancer
=========================================
Probabilistic gEUD and Monte Carlo NTCP models with uncertainty quantification

All models implement the Universal NTCP Model API:
    calculate_ntcp(dvh, *, n_samples=0, return_distribution=False, random_state=None) -> dict
"""

import numpy as np
import pandas as pd
from scipy import stats
import sys

# Windows-safe encoding configuration
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except (AttributeError, ValueError):
    pass


class ProbabilisticgEUDModel:
    """Probabilistic gEUD NTCP Model (Niemierko 1999, Brodin 2017)"""
    
    def __init__(self, organ):
        # Parameter distributions for H&N OARs
        # Based on published literature for HNSCC
        self.param_distributions = {
            'Parotid': {
                'n': (0.7, 0.15),      # Parallel organ
                'TD50': (28.4, 5.0),   # Gy
                'm': (0.4, 0.1),
                'endpoint': 'Xerostomia Grade ≥2'
            },
            'Larynx': {
                'n': (0.2, 0.05),      # Mixed serial-parallel
                'TD50': (43.5, 8.0),   # Gy
                'm': (0.35, 0.08),
                'endpoint': 'Dysphagia Grade ≥2'
            },
            'SpinalCord': {
                'n': (0.05, 0.01),     # Serial organ
                'TD50': (66.5, 5.0),   # Gy
                'm': (0.175, 0.04),
                'endpoint': 'Myelopathy'
            },
            'OralCavity': {
                'n': (0.6, 0.12),      # Parallel organ
                'TD50': (39.8, 6.0),   # Gy
                'm': (0.38, 0.09),
                'endpoint': 'Mucositis Grade ≥3'
            },
            'Submandibular': {
                'n': (0.7, 0.14),      # Parallel organ
                'TD50': (39.0, 7.0),   # Gy
                'm': (0.46, 0.11),
                'endpoint': 'Xerostomia Grade ≥2'
            },
            'PharyngealConstrictor': {
                'n': (0.25, 0.06),     # Serial-like
                'TD50': (55.0, 9.0),   # Gy
                'm': (0.30, 0.07),
                'endpoint': 'Dysphagia Grade ≥2'
            },
            'Mandible': {
                'n': (0.1, 0.02),      # Serial organ
                'TD50': (65.0, 8.0),   # Gy
                'm': (0.15, 0.04),
                'endpoint': 'Osteoradionecrosis'
            },
            'Esophagus': {
                'n': (0.28, 0.06),     # Serial-parallel
                'TD50': (68.0, 10.0),  # Gy
                'm': (0.23, 0.06),
                'endpoint': 'Esophagitis Grade ≥3'
            },
            'BrachialPlexus': {
                'n': (0.08, 0.02),     # Serial organ
                'TD50': (60.4, 7.0),   # Gy
                'm': (0.18, 0.05),
                'endpoint': 'Brachial Plexopathy'
            },
            'Cochlea': {
                'n': (0.15, 0.03),     # Serial organ
                'TD50': (45.0, 6.0),   # Gy
                'm': (0.16, 0.04),
                'endpoint': 'Hearing Loss'
            },
            'OpticNerve': {
                'n': (0.05, 0.01),     # Serial organ
                'TD50': (65.0, 5.0),   # Gy
                'm': (0.14, 0.03),
                'endpoint': 'Optic Neuropathy'
            },
            'Chiasm': {
                'n': (0.05, 0.01),     # Serial organ
                'TD50': (65.0, 5.0),   # Gy
                'm': (0.14, 0.03),
                'endpoint': 'Optic Neuropathy'
            },
            'Brainstem': {
                'n': (0.05, 0.01),     # Serial organ
                'TD50': (65.0, 5.0),   # Gy
                'm': (0.16, 0.04),
                'endpoint': 'Brainstem Necrosis'
            }
        }
        self.organ = organ
    
    def calculate_ntcp(self, dvh, *, n_samples=1000, return_distribution=False, random_state=None):
        """
        Universal NTCP Model API - Calculate NTCP with uncertainty quantification
        
        Args:
            dvh: DataFrame with 'dose'/'dose_gy' and 'volume'/'volume_cm3' columns
            n_samples: Number of Monte Carlo samples (default: 1000)
            return_distribution: If True, return full distribution array
            random_state: Random seed for reproducibility
        
        Returns:
            dict with keys:
                'mean': float - Mean NTCP
                'std': float - Standard deviation
                'ci95': tuple - 95% confidence interval (low, high)
                'distribution': np.ndarray | None - Full distribution if return_distribution=True
                'model_name': str - Model identifier
                'assumptions': str - Model assumptions
        """
        params = self.param_distributions.get(self.organ)
        if not params:
            ci95_nan = (np.nan, np.nan)
            return {
                'mean': np.nan,
                'std': np.nan,
                'ci95': ci95_nan,
                'ci_lower': ci95_nan[0],
                'ci_upper': ci95_nan[1],
                'distribution': None,
                'model_name': f'ProbabilisticgEUD_{self.organ}',
                'assumptions': 'Organ not in parameter database'
            }
        
        # Set random state for reproducibility
        if random_state is not None:
            np.random.seed(random_state)
        
        # Sample parameters from distributions
        n_samples_arr = stats.norm(params['n'][0], params['n'][1]).rvs(n_samples)
        n_samples_arr = np.clip(n_samples_arr, 0.01, 1.0)  # Keep physically meaningful
        
        td50_samples = stats.norm(params['TD50'][0], params['TD50'][1]).rvs(n_samples)
        td50_samples = np.clip(td50_samples, 10, 100)  # Gy range
        
        m_samples = stats.norm(params['m'][0], params['m'][1]).rvs(n_samples)
        m_samples = np.clip(m_samples, 0.01, 1.0)
        
        ntcp_samples = []
        for n, td50, m in zip(n_samples_arr, td50_samples, m_samples):
            # Calculate gEUD
            geud = self._calculate_geud(dvh, n)
            if geud == 0:
                ntcp_samples.append(0.0)
                continue
            
            # Calculate NTCP using LKB log-logistic
            # NTCP = 1 / (1 + (TD50/gEUD)^(4*gamma50))
            # where gamma50 = log(10) / (m * log(e))
            gamma50 = np.log(10) / (m * np.log(np.e))
            ratio = td50 / geud if geud > 0 else 1e10
            exponent = 4.0 * gamma50
            ntcp = 1.0 / (1.0 + np.power(ratio, exponent))
            ntcp = np.clip(ntcp, 0, 1)
            ntcp_samples.append(ntcp)
        
        ntcp_samples = np.array(ntcp_samples)
        
        ci95 = (float(np.percentile(ntcp_samples, 2.5)), float(np.percentile(ntcp_samples, 97.5)))
        
        return {
            'mean': float(np.mean(ntcp_samples)),
            'std': float(np.std(ntcp_samples)),
            'ci95': ci95,
            'ci_lower': ci95[0],
            'ci_upper': ci95[1],
            'distribution': ntcp_samples.copy() if return_distribution else None,
            'model_name': f'ProbabilisticgEUD_{self.organ}',
            'assumptions': f'Parameter uncertainty: n~N({params["n"][0]:.2f}, {params["n"][1]:.2f}), TD50~N({params["TD50"][0]:.1f}, {params["TD50"][1]:.1f}) Gy, m~N({params["m"][0]:.2f}, {params["m"][1]:.2f}); Endpoint: {params["endpoint"]}'
        }
    
    def calculate_ntcp_distribution(self, dvh, n_samples=1000):
        """Legacy method - wrapper for calculate_ntcp() for backward compatibility"""
        result = self.calculate_ntcp(dvh, n_samples=n_samples, return_distribution=True)
        return {
            'mean': result['mean'],
            'std': result['std'],
            'ci_lower': result['ci95'][0],
            'ci_upper': result['ci95'][1],
            'median': np.median(result['distribution']) if result['distribution'] is not None else np.nan,
            'endpoint': self.param_distributions.get(self.organ, {}).get('endpoint', 'Unknown')
        }
    
    def _calculate_geud(self, dvh, n):
        """Calculate generalized EUD"""
        if isinstance(dvh, pd.DataFrame):
            doses = dvh['dose'].values if 'dose' in dvh.columns else dvh['dose_gy'].values
            volumes = dvh['volume'].values if 'volume' in dvh.columns else dvh['volume_cm3'].values
        else:
            doses = dvh['dose'].values
            volumes = dvh['volume'].values
        
        total_volume = volumes.sum()
        
        if total_volume == 0:
            return 0
        
        a = 1/n if n != 0 else 1
        geud = np.power(np.sum(volumes * np.power(doses, a)) / total_volume, n)
        return geud


class MonteCarloNTCPModel:
    """Monte Carlo NTCP Model with DVH uncertainty (Fenwick 2001)"""
    
    def __init__(self, organ):
        self.organ = organ
        # H&N specific uncertainty parameters
        self.systematic_error = 0.03  # 3% systematic dose uncertainty
        self.random_error = 0.02      # 2% random dose uncertainty
    
    def calculate_ntcp(self, dvh, params=None, *, n_samples=1000, return_distribution=False, random_state=None):
        """
        Universal NTCP Model API - Calculate NTCP with DVH and parameter uncertainty
        
        Args:
            dvh: DataFrame with 'dose'/'dose_gy' and 'volume'/'volume_cm3' columns
            params: Optional dict with 'n', 'TD50', 'm' keys. If None, uses literature defaults
            n_samples: Number of Monte Carlo iterations (default: 1000)
            return_distribution: If True, return full distribution array
            random_state: Random seed for reproducibility
        
        Returns:
            dict with keys:
                'mean': float - Mean NTCP
                'std': float - Standard deviation
                'ci95': tuple - 95% confidence interval (low, high)
                'distribution': np.ndarray | None - Full distribution if return_distribution=True
                'model_name': str - Model identifier
                'assumptions': str - Model assumptions
        """
        # Use provided params or get from literature
        if params is None:
            # Default literature parameters for common H&N organs
            default_params = {
                'Parotid': {'n': 0.45, 'TD50': 28.4, 'm': 0.18},
                'Larynx': {'n': 1.0, 'TD50': 44.0, 'm': 0.20},
                'SpinalCord': {'n': 0.03, 'TD50': 66.5, 'm': 0.10}
            }
            params = default_params.get(self.organ, {'n': 0.5, 'TD50': 50.0, 'm': 0.2})
        
        # Set random state for reproducibility
        if random_state is not None:
            np.random.seed(random_state)
        
        ntcp_samples = []
        
        for _ in range(n_samples):
            # Perturb DVH (systematic + random errors)
            perturbed_dvh = self._perturb_dvh(dvh, random_state=random_state)
            
            # Perturb parameters (10% relative uncertainty)
            n_perturbed = np.random.normal(params['n'], abs(params['n'] * 0.1))
            n_perturbed = np.clip(n_perturbed, 0.01, 1.0)
            
            td50_perturbed = np.random.normal(params['TD50'], abs(params['TD50'] * 0.1))
            td50_perturbed = np.clip(td50_perturbed, 10, 100)
            
            m_perturbed = np.random.normal(params['m'], abs(params['m'] * 0.1))
            m_perturbed = np.clip(m_perturbed, 0.01, 1.0)
            
            # Calculate NTCP
            geud = self._calculate_geud(perturbed_dvh, n_perturbed)
            if geud == 0:
                ntcp_samples.append(0.0)
                continue
            
            # LKB log-logistic model
            gamma50 = np.log(10) / (m_perturbed * np.log(np.e))
            ratio = td50_perturbed / geud if geud > 0 else 1e10
            exponent = 4.0 * gamma50
            ntcp = 1.0 / (1.0 + np.power(ratio, exponent))
            ntcp = np.clip(ntcp, 0, 1)
            ntcp_samples.append(ntcp)
        
        ntcp_samples = np.array(ntcp_samples)
        
        ci95 = (float(np.percentile(ntcp_samples, 2.5)), float(np.percentile(ntcp_samples, 97.5)))
        
        return {
            'mean': float(np.mean(ntcp_samples)),
            'std': float(np.std(ntcp_samples)),
            'ci95': ci95,
            'ci_lower': ci95[0],
            'ci_upper': ci95[1],
            'distribution': ntcp_samples.copy() if return_distribution else None,
            'model_name': f'MonteCarloNTCP_{self.organ}',
            'assumptions': f'DVH uncertainty: {self.systematic_error*100:.1f}% systematic, {self.random_error*100:.1f}% random; Parameter uncertainty: 10% relative; Organ: {self.organ}'
        }
    
    def calculate_ntcp_with_uncertainty(self, dvh, params, n_iterations=1000):
        """Legacy method - wrapper for calculate_ntcp() for backward compatibility"""
        result = self.calculate_ntcp(dvh, params=params, n_samples=n_iterations, return_distribution=True)
        return {
            'mean': result['mean'],
            'std': result['std'],
            'ci_lower': result['ci95'][0],
            'ci_upper': result['ci95'][1],
            'median': np.median(result['distribution']) if result['distribution'] is not None else np.nan,
            'samples': result['distribution'].tolist() if result['distribution'] is not None else [],
            'dvh_contribution': self._estimate_dvh_uncertainty_contribution(result['distribution']) if result['distribution'] is not None else 0.0,
            'param_contribution': self._estimate_param_uncertainty_contribution(result['distribution']) if result['distribution'] is not None else 0.0
        }
    
    def _perturb_dvh(self, dvh, random_state=None):
        """Add uncertainty to DVH"""
        if isinstance(dvh, pd.DataFrame):
            perturbed = dvh.copy()
        else:
            perturbed = pd.DataFrame(dvh)
        
        # Systematic error (affects all bins)
        sys_factor = 1 + np.random.normal(0, self.systematic_error)
        
        # Random error (per bin)
        n_bins = len(perturbed)
        rand_factors = 1 + np.random.normal(0, self.random_error, n_bins)
        
        dose_col = 'dose' if 'dose' in perturbed.columns else 'dose_gy'
        if dose_col not in perturbed.columns:
            dose_col = perturbed.columns[0]  # Fallback to first column
        
        perturbed[dose_col] = perturbed[dose_col] * sys_factor * rand_factors
        perturbed[dose_col] = np.clip(perturbed[dose_col], 0, None)
        
        return perturbed
    
    def _calculate_geud(self, dvh, n):
        """Calculate generalized EUD"""
        if isinstance(dvh, pd.DataFrame):
            doses = dvh['dose'].values if 'dose' in dvh.columns else dvh['dose_gy'].values
            volumes = dvh['volume'].values if 'volume' in dvh.columns else dvh['volume_cm3'].values
        else:
            doses = dvh['dose'].values
            volumes = dvh['volume'].values
        
        total_volume = volumes.sum()
        
        if total_volume == 0:
            return 0
        
        a = 1/n if n != 0 else 1
        geud = np.power(np.sum(volumes * np.power(doses, a)) / total_volume, n)
        return geud
    
    def _estimate_dvh_uncertainty_contribution(self, samples):
        """Estimate contribution of DVH uncertainty to total uncertainty"""
        if samples is None or len(samples) == 0:
            return 0.0
        return float(np.std(samples) * 0.6)  # Approximate: DVH uncertainty contributes ~60%
    
    def _estimate_param_uncertainty_contribution(self, samples):
        """Estimate contribution of parameter uncertainty to total uncertainty"""
        if samples is None or len(samples) == 0:
            return 0.0
        return float(np.std(samples) * 0.4)  # Approximate: Parameter uncertainty contributes ~40%

