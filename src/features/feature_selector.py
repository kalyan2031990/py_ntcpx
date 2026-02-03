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
    # FIXED: Use actual column names from prepare_features()
    PAROTID_ESSENTIAL = ['mean_dose', 'V30', 'V45']  # Primary predictors (was 'Dmean', now 'mean_dose')
    PAROTID_EXPLORATORY = ['D50', 'V15', 'V20', 'gEUD']  # Secondary
    
    # Clinical features with evidence
    CLINICAL_VALIDATED = ['Age', 'Baseline_Salivary_Function']
    CLINICAL_EXPLORATORY = ['Chemotherapy', 'T_Stage', 'Diabetes']
    
    # Feature name mappings for robustness (handles variations)
    FEATURE_NAME_MAPPINGS = {
        'Dmean': ['mean_dose', 'Dmean', 'MeanDose', 'mean_dose_Gy', 'D_mean'],
        'Dmax': ['max_dose', 'Dmax', 'MaxDose', 'max_dose_Gy', 'D_max'],
        'gEUD': ['gEUD', 'geud', 'gEUD_a2', 'GEUD', 'gEUD_a1'],
        'V5': ['V5', 'v5', 'V5_%', 'v5_percent'],
        'V30': ['V30', 'v30', 'V30_%', 'v30_percent'],
        'V45': ['V45', 'v45', 'V45_%', 'v45_percent'],
        'V15': ['V15', 'v15', 'V15_%', 'v15_percent'],
        'V20': ['V20', 'v20', 'V20_%', 'v20_percent'],
        'D50': ['D50', 'd50', 'D50_Gy', 'd50_gy'],
    }
    
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
        # DEBUG: Feature selector input
        print(f"\n[DEBUG] RadiobiologyGuidedFeatureSelector.select_features()")
        print(f"  - Input X_df shape: {X.shape}")
        print(f"  - All available features: {list(X.columns)}")
        print(f"  - Number of events (y.sum()): {int(np.sum(y))}")
        print(f"  - Total samples: {len(y)}")
        print(f"  - Organ: {organ}")
        
        n_events = np.sum(y)
        n_samples = len(y)
        
        # EPV-based maximum features - ADJUSTED for small datasets
        if max_features is None:
            if n_samples < 100:  # Small dataset heuristic
                # For small datasets, allow more features relative to events
                # Less restrictive: n_events/5 instead of n_events/10, minimum 5 features
                max_features = max(int(n_events / 5), 5)
                print(f"  - Small dataset detected (n_samples={n_samples} < 100)")
                print(f"  - Adjusted EPV rule: max_features = max(int({n_events} / 5), 5) = {max_features}")
            else:
                # Original rule for larger datasets
                max_features = max(int(n_events / 10), 3)
                print(f"  - Standard EPV rule: max_features = max(int({n_events} / 10), 3) = {max_features}")
        else:
            print(f"  - Using provided max_features: {max_features}")
        
        print(f"  - n_events: {n_events}")
        print(f"  - n_samples: {n_samples}")
        
        selected = []
        
        # Helper function to find feature name variations
        def _find_feature_variation(feature_name, available_features):
            """Find feature name variations in available features."""
            if feature_name in available_features:
                return feature_name
            
            # Check mappings
            for canonical_name, variations in self.FEATURE_NAME_MAPPINGS.items():
                if feature_name == canonical_name:
                    for variation in variations:
                        if variation in available_features:
                            return variation
            
            # Try case-insensitive match
            feature_lower = feature_name.lower()
            for avail_feature in available_features:
                if avail_feature.lower() == feature_lower:
                    return avail_feature
            
            return None  # Not found
        
        # 1. Essential domain features (always include if available)
        if organ.lower() == 'parotid':
            essential_features = self.PAROTID_ESSENTIAL
            exploratory_features = self.PAROTID_EXPLORATORY
        elif organ.lower() in ['larynx', 'oralcavity']:
            # For other organs, use general dose metrics (with name mapping)
            essential_features = ['mean_dose', 'V30', 'V45']  # Updated to use actual names
            exploratory_features = []
        else:
            essential_features = ['mean_dose', 'V30', 'V45']  # Default
            exploratory_features = []
        
        print(f"  - Essential features for {organ}: {essential_features}")
        print(f"  - Exploratory features for {organ}: {exploratory_features}")
        
        # Add essential features with name mapping
        for feat in essential_features:
            # Try direct match first
            found_feat = _find_feature_variation(feat, X.columns)
            found = found_feat is not None
            print(f"    - Looking for '{feat}': {'FOUND' if found else 'NOT FOUND'}", end="")
            if found and found_feat != feat:
                print(f" (as '{found_feat}')")
            else:
                print()
            
            if found_feat and found_feat not in selected:
                selected.append(found_feat)
                print(f"      -> Added '{found_feat}' to selected")
        
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
        stat_sig_features = [f for f, p in sorted_features if p < 0.1]
        print(f"  - Features with p<0.1: {stat_sig_features[:10]}... (showing first 10)")
        
        for feat, p in sorted_features:
            if len(selected) >= max_features:
                break
            if p < 0.1:  # Univariate significance threshold
                selected.append(feat)
        
        # If we still don't have enough features, add the best remaining ones
        if len(selected) < 3:
            print(f"  - WARNING: Only {len(selected)} features selected, adding best remaining...")
            for feat, p in sorted_features:
                if feat not in selected:
                    selected.append(feat)
                    if len(selected) >= 3:
                        break
        
        print(f"  - FINAL selected features: {selected}")
        print(f"  - Number selected: {len(selected)}")
        
        return selected
