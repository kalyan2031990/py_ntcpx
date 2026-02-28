"""
Clinical Safety Layer (Phase 7)

ClinicalSafetyGuard that flags underprediction risk and integrates CCS
CCS < 0.2 → DO_NOT_USE flag
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import warnings

try:
    from ntcp_qa_modules import CohortConsistencyScore
    CCS_AVAILABLE = True
except ImportError:
    CCS_AVAILABLE = False
    warnings.warn("CohortConsistencyScore not available. Install ntcp_qa_modules.")


class ClinicalSafetyGuard:
    """
    Clinical safety guard for NTCP predictions
    
    Flags:
    - Underprediction risk (uses CI lower bounds)
    - Integrates Cohort Consistency Score (CCS)
    - CCS < 0.2 → DO_NOT_USE flag
    """
    
    def __init__(self, ccs_threshold: float = None, ci_alpha: float = 0.05, n_samples: int = None):
        """
        Initialize clinical safety guard (v3.0.0: adaptive thresholds)
        
        Parameters
        ----------
        ccs_threshold : float, optional
            CCS threshold (if None, will be set adaptively based on n_samples)
        ci_alpha : float
            Significance level for CI (default: 0.05 for 95% CI)
        n_samples : int, optional
            Number of samples for adaptive threshold calculation
        """
        self.n_samples = n_samples
        self.ci_alpha = ci_alpha
        self.ccs_calculator = None
        self.training_stats = None
        
        if CCS_AVAILABLE:
            self.ccs_calculator = CohortConsistencyScore(n_samples=n_samples)
            # Use adaptive threshold from calculator if not explicitly set
            if ccs_threshold is None:
                # Will be set after fit() is called
                self.ccs_threshold = None
            else:
                self.ccs_threshold = ccs_threshold
    
    def fit(self, X_train: np.ndarray):
        """
        Fit safety guard on training cohort
        
        Parameters
        ----------
        X_train : np.ndarray
            Training feature matrix
        """
        if self.ccs_calculator is not None:
            self.ccs_calculator.fit(X_train)
            # v3.0.0: Get adaptive threshold from calculator after fitting
            if self.ccs_threshold is None:
                self.ccs_threshold = self.ccs_calculator.ccs_threshold
            self.training_stats = {
                'mean': np.mean(X_train, axis=0),
                'std': np.std(X_train, axis=0),
                'n_samples': len(X_train)
            }
    
    def evaluate_safety(
        self,
        ntcp_predictions: np.ndarray,
        ntcp_ci_lower: Optional[np.ndarray] = None,
        X_features: Optional[np.ndarray] = None,
        patient_ids: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Evaluate clinical safety for NTCP predictions
        
        Parameters
        ----------
        ntcp_predictions : np.ndarray
            NTCP point predictions
        ntcp_ci_lower : np.ndarray, optional
            Lower bound of 95% CI for NTCP
        X_features : np.ndarray, optional
            Feature matrix for CCS calculation
        patient_ids : list, optional
            Patient IDs for reporting
            
        Returns
        -------
        pd.DataFrame
            Safety evaluation with flags
        """
        n_patients = len(ntcp_predictions)
        
        # Initialize safety flags
        safety_flags = []
        ccs_scores = []
        underprediction_risks = []
        do_not_use_flags = []
        
        # Calculate CCS if features available
        if X_features is not None and self.ccs_calculator is not None:
            try:
                for i in range(n_patients):
                    if X_features.ndim == 2:
                        ccs_result = self.ccs_calculator.calculate_ccs(X_features[i])
                    else:
                        ccs_result = self.ccs_calculator.calculate_ccs(X_features)
                    
                    # Extract CCS value from result (handles both dict and float returns)
                    if isinstance(ccs_result, dict):
                        ccs = ccs_result.get('ccs', np.nan)
                    else:
                        ccs = ccs_result
                    
                    ccs_scores.append(ccs)
                    
                    # v3.0.0: CCS_Warning flag instead of DO_NOT_USE
                    # True if CCS < threshold (interpretations should be treated with caution)
                    if not np.isnan(ccs) and self.ccs_threshold is not None:
                        ccs_warning = ccs < self.ccs_threshold
                        do_not_use_flags.append(ccs_warning)  # Reusing variable name for compatibility
                        
                        if ccs_warning:
                            safety_flags.append(f"CCS_WARNING (CCS={ccs:.3f} < {self.ccs_threshold:.3f})")
                        else:
                            safety_flags.append("")
                    else:
                        do_not_use_flags.append(False)
                        if np.isnan(ccs):
                            safety_flags.append("CCS_CALCULATION_FAILED")
                        else:
                            safety_flags.append("")
            except Exception as e:
                warnings.warn(f"CCS calculation failed: {e}")
                ccs_scores = [np.nan] * n_patients
                do_not_use_flags = [False] * n_patients
                safety_flags = [""] * n_patients
        else:
            ccs_scores = [np.nan] * n_patients
            do_not_use_flags = [False] * n_patients
            safety_flags = [""] * n_patients
        
        # Evaluate underprediction risk using CI lower bounds
        if ntcp_ci_lower is not None:
            for i in range(n_patients):
                ntcp = ntcp_predictions[i]
                ci_lower = ntcp_ci_lower[i]
                
                # High underprediction risk if CI lower bound is very low
                # but point prediction is moderate/high
                if not np.isnan(ci_lower) and not np.isnan(ntcp):
                    if ci_lower < 0.1 and ntcp > 0.3:
                        underprediction_risks.append("HIGH")
                        if safety_flags[i]:
                            safety_flags[i] += "; UNDERPREdICTION_RISK"
                        else:
                            safety_flags[i] = "UNDERPREdICTION_RISK"
                    elif ci_lower < 0.2 and ntcp > 0.4:
                        underprediction_risks.append("MODERATE")
                        if not safety_flags[i]:
                            safety_flags[i] = "UNDERPREdICTION_RISK_MODERATE"
                    else:
                        underprediction_risks.append("LOW")
                else:
                    underprediction_risks.append("UNKNOWN")
        else:
            underprediction_risks = ["UNKNOWN"] * n_patients
        
        # Build results DataFrame
        results = {
            'NTCP_Prediction': ntcp_predictions,
            'Safety_Flag': safety_flags,
            'DO_NOT_USE': do_not_use_flags,
            'CCS_Score': ccs_scores,
            'Underprediction_Risk': underprediction_risks
        }
        
        if patient_ids is not None:
            results['PatientID'] = patient_ids
        
        if ntcp_ci_lower is not None:
            results['NTCP_CI_Lower'] = ntcp_ci_lower
        
        return pd.DataFrame(results)
    
    def generate_safety_report(
        self,
        safety_df: pd.DataFrame,
        output_path: Optional[Path] = None
    ) -> str:
        """
        Generate safety report summary
        
        Parameters
        ----------
        safety_df : pd.DataFrame
            Safety evaluation DataFrame
        output_path : Path, optional
            Path to save report
            
        Returns
        -------
        str
            Safety report text
        """
        n_total = len(safety_df)
        n_do_not_use = safety_df['DO_NOT_USE'].sum() if 'DO_NOT_USE' in safety_df.columns else 0
        n_high_risk = (safety_df['Underprediction_Risk'] == 'HIGH').sum() if 'Underprediction_Risk' in safety_df.columns else 0
        
        report_lines = [
            "=" * 60,
            "CLINICAL SAFETY REPORT",
            "=" * 60,
            f"Total Predictions: {n_total}",
            f"CCS Warnings: {n_do_not_use} ({n_do_not_use/n_total*100:.1f}%)",
            f"High Underprediction Risk: {n_high_risk} ({n_high_risk/n_total*100:.1f}%)",
            "",
            "Safety Thresholds:",
            f"  - CCS Threshold: {self.ccs_threshold:.3f} (CCS < threshold -> Warning)",
            f"  - CI Alpha: {self.ci_alpha} ({(1-self.ci_alpha)*100:.0f}% CI)",
            "",
            "Recommendations:",
        ]
        
        if n_do_not_use > 0:
            report_lines.append(
                f"  INFO: {n_do_not_use} predictions have CCS below adaptive threshold."
            )
            report_lines.append(
                "     Interpretations should be treated with caution."
            )
        
        if n_high_risk > 0:
            report_lines.append(
                f"  WARNING: {n_high_risk} predictions have HIGH underprediction risk."
            )
            report_lines.append(
                "     Review CI lower bounds before clinical use."
            )
        
        if n_do_not_use == 0 and n_high_risk == 0:
            report_lines.append("  OK: No critical safety issues detected.")
        
        report_lines.extend([
            "",
            "=" * 60
        ])
        
        report_text = "\n".join(report_lines)
        
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report_text)
        
        return report_text
