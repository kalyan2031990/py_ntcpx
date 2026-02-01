"""
Nested cross-validation for unbiased performance estimation
"""

import numpy as np
from typing import Dict, Optional, Callable
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
import warnings


class NestedCrossValidation:
    """
    Nested cross-validation for unbiased performance estimation
    
    Outer loop: Performance estimation (default: 5-fold)
    Inner loop: Hyperparameter tuning (default: 3-fold)
    """
    
    def __init__(self, 
                 outer_folds: int = 5,
                 inner_folds: int = 3,
                 random_seed: int = 42,
                 scoring: str = 'roc_auc'):
        """
        Initialize nested CV
        
        Parameters
        ----------
        outer_folds : int
            Number of outer CV folds
        inner_folds : int
            Number of inner CV folds for hyperparameter tuning
        random_seed : int
            Random seed for reproducibility
        scoring : str
            Scoring metric (default: 'roc_auc')
        """
        self.outer_folds = outer_folds
        self.inner_folds = inner_folds
        self.random_seed = random_seed
        self.scoring = scoring
    
    def fit_and_evaluate(self,
                        model,
                        X: np.ndarray,
                        y: np.ndarray,
                        param_grid: Optional[Dict] = None) -> Dict:
        """
        Perform nested CV and return performance metrics
        
        Parameters
        ----------
        model : sklearn-like model
            Model with fit() and predict_proba() methods
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Binary outcomes
        param_grid : dict, optional
            Hyperparameter grid for inner CV tuning
            
        Returns
        -------
        dict with keys:
            - 'nested_cv_auc_mean': Mean AUC across outer folds
            - 'nested_cv_auc_std': Standard deviation of AUC
            - 'nested_cv_auc_scores': List of AUC scores for each fold
            - 'best_params': List of best parameters for each fold
        """
        # Ensure binary classification
        if len(np.unique(y)) < 2:
            raise ValueError("Both classes must be present for classification")
        
        outer_cv = StratifiedKFold(
            n_splits=self.outer_folds, 
            shuffle=True, 
            random_state=self.random_seed
        )
        inner_cv = StratifiedKFold(
            n_splits=self.inner_folds, 
            shuffle=True, 
            random_state=self.random_seed
        )
        
        outer_scores = []
        best_params_list = []
        
        for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Check if both classes present in train and test
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                warnings.warn(f"Fold {fold}: Only one class in train or test, skipping")
                continue
            
            # Inner CV for hyperparameter selection
            if param_grid:
                gs = GridSearchCV(
                    clone(model),
                    param_grid,
                    cv=inner_cv,
                    scoring=self.scoring,
                    n_jobs=-1
                )
                gs.fit(X_train, y_train)
                best_model = gs.best_estimator_
                best_params_list.append(gs.best_params_)
            else:
                best_model = clone(model)
                best_model.fit(X_train, y_train)
                best_params_list.append({})
            
            # Test on outer fold
            try:
                y_pred = best_model.predict_proba(X_test)[:, 1]
                score = roc_auc_score(y_test, y_pred)
                outer_scores.append(score)
            except Exception as e:
                warnings.warn(f"Fold {fold}: Error computing AUC: {e}")
                continue
        
        if len(outer_scores) == 0:
            raise ValueError("Could not compute any AUC scores")
        
        return {
            'nested_cv_auc_mean': np.mean(outer_scores),
            'nested_cv_auc_std': np.std(outer_scores),
            'nested_cv_auc_scores': outer_scores,
            'best_params': best_params_list
        }
