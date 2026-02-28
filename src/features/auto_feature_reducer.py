"""
Auto Feature Reduction (Phase 4.1)

Automatically reduces features when EPV < 5 to meet minimum EPV requirement
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from src.features.feature_selector import RadiobiologyGuidedFeatureSelector


class AutoFeatureReducer:
    """
    Automatically reduce features to meet EPV >= 5 requirement
    
    Strategy:
    1. Start with domain-essential features
    2. Add features by statistical significance until EPV >= 5
    3. If still EPV < 5, use only essential features
    """
    
    def __init__(self, min_epv: float = 5.0):
        """
        Initialize auto feature reducer
        
        Parameters
        ----------
        min_epv : float
            Minimum EPV required (default: 5.0)
        """
        self.min_epv = min_epv
        self.selector = RadiobiologyGuidedFeatureSelector()
    
    def reduce_features(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        organ: str = 'Parotid',
        n_events: Optional[int] = None
    ) -> Tuple[pd.DataFrame, List[str], float]:
        """
        Reduce features to meet EPV >= 5 requirement
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : np.ndarray
            Binary outcomes
        organ : str
            Organ name
        n_events : int, optional
            Number of events (if None, calculated from y)
            
        Returns
        -------
        X_reduced : pd.DataFrame
            Reduced feature matrix
        selected_features : list
            List of selected feature names
        final_epv : float
            Final EPV after reduction
        """
        if n_events is None:
            n_events = int(np.sum(y))
        
        n_features = len(X.columns)
        current_epv = n_events / n_features if n_features > 0 else 0
        
        # If EPV already >= min_epv, return all features
        if current_epv >= self.min_epv:
            return X, list(X.columns), current_epv
        
        # Calculate target number of features
        max_features = max(int(n_events / self.min_epv), 1)
        
        # Use feature selector to get essential features first
        selected_features = self.selector.select_features(
            X, y, organ=organ, max_features=max_features
        )
        
        # If we still don't have enough EPV, use only essential features
        if len(selected_features) > max_features:
            # Prioritize essential features
            if organ.lower() == 'parotid':
                essential = ['Dmean', 'V30', 'V45']
                # Keep essential + top statistical features
                essential_present = [f for f in essential if f in X.columns]
                remaining = [f for f in selected_features if f not in essential_present]
                selected_features = essential_present + remaining[:max(0, max_features - len(essential_present))]
            else:
                # For other organs, just take first max_features
                selected_features = selected_features[:max_features]
        
        # Extract reduced features
        X_reduced = X[selected_features].copy()
        final_epv = n_events / len(selected_features) if len(selected_features) > 0 else 0
        
        return X_reduced, selected_features, final_epv
