"""
Domain-knowledge guided feature selection for NTCP modeling

Reference: QUANTEC guidelines, Parotid: Deasy et al. IJROBP 2010
"""

import numpy as np
import pandas as pd
from typing import List, Optional
from scipy.stats import mannwhitneyu


class RadiobiologyGuidedFeatureSelector:
    """
    Domain-knowledge guided feature selection for NTCP modeling
    
    Reference: QUANTEC guidelines, Parotid: Deasy et al. IJROBP 2010
    """
    
    # Literature-supported features for parotid xerostomia
    PAROTID_ESSENTIAL = ['Dmean', 'V30', 'V45']  # Primary predictors
    PAROTID_EXPLORATORY = ['D50', 'V15', 'V20', 'gEUD']  # Secondary
    
    # Clinical features with evidence
    CLINICAL_VALIDATED = ['Age', 'Baseline_Salivary_Function']
    CLINICAL_EXPLORATORY = ['Chemotherapy', 'T_Stage', 'Diabetes']
    
    def select_features(self, 
                       X: pd.DataFrame, 
                       y: np.ndarray,
                       organ: str = 'Parotid',
                       max_features: Optional[int] = None) -> List[str]:
        """
        Select features using domain knowledge + statistical filtering
        
        Strategy:
        1. Start with literature-validated features
        2. Add statistical filtering (univariate p < 0.1)
        3. Cap at max_features based on EPV rule
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : np.ndarray
            Binary outcomes
        organ : str
            Organ name (default: 'Parotid')
        max_features : int, optional
            Maximum number of features. If None, calculated from EPV rule.
            
        Returns
        -------
        list of str
            Selected feature names
        """
        n_events = np.sum(y)
        
        # EPV-based maximum features
        if max_features is None:
            max_features = max(int(n_events / 10), 3)  # Min 3 features
        
        selected = []
        
        # 1. Essential domain features (always include if available)
        if organ.lower() == 'parotid':
            for feat in self.PAROTID_ESSENTIAL:
                if feat in X.columns and feat not in selected:
                    selected.append(feat)
        elif organ.lower() in ['larynx', 'oralcavity']:
            # For other organs, use general dose metrics
            for feat in ['Dmean', 'V30', 'V45']:
                if feat in X.columns and feat not in selected:
                    selected.append(feat)
        
        # 2. Statistical filtering on remaining features
        candidate_features = [c for c in X.columns if c not in selected]
        
        if len(candidate_features) == 0:
            return selected
        
        p_values = {}
        for feat in candidate_features:
            try:
                # Handle missing values
                X_pos = X.loc[y == 1, feat].dropna()
                X_neg = X.loc[y == 0, feat].dropna()
                
                if len(X_pos) < 2 or len(X_neg) < 2:
                    p_values[feat] = 1.0
                    continue
                
                # Mann-Whitney U test (non-parametric)
                stat, p = mannwhitneyu(X_pos, X_neg, alternative='two-sided')
                p_values[feat] = p
            except Exception:
                p_values[feat] = 1.0
        
        # Sort by p-value and add until max_features
        sorted_features = sorted(p_values.items(), key=lambda x: x[1])
        for feat, p in sorted_features:
            if len(selected) >= max_features:
                break
            if p < 0.1:  # Univariate significance threshold
                selected.append(feat)
        
        # If we still don't have enough features, add the best remaining ones
        if len(selected) < 3:
            for feat, p in sorted_features:
                if feat not in selected:
                    selected.append(feat)
                    if len(selected) >= 3:
                        break
        
        return selected
