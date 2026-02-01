"""
ML models with aggressive overfitting prevention

Designed for small sample sizes with proper EPV (Events Per Variable) validation
"""

import numpy as np
import warnings
from typing import Dict, List, Optional, Tuple
from sklearn.neural_network import MLPClassifier
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available. Install with: pip install xgboost")


class OverfitResistantMLModels:
    """
    ML models with aggressive overfitting prevention
    
    Validates Events Per Variable (EPV) rule:
    - EPV >= 10 for reliable logistic regression
    - EPV >= 20 for reliable ML
    """
    
    # CONSERVATIVE ANN ARCHITECTURE
    ANN_CONFIG = {
        'hidden_layer_sizes': (16, 8),  # Reduced from (64, 32, 16)
        'activation': 'relu',
        'solver': 'adam',
        'alpha': 0.01,  # Strong L2 regularization (increased from 0.001)
        'learning_rate': 'adaptive',
        'learning_rate_init': 0.001,
        'max_iter': 500,  # Reduced from 1000
        'early_stopping': True,
        'validation_fraction': 0.2,
        'n_iter_no_change': 20,  # Reduced from 50
        'random_state': 42
    }
    
    # CONSERVATIVE XGBOOST CONFIG
    XGBOOST_CONFIG = {
        'n_estimators': 50,  # Reduced from 200
        'max_depth': 2,  # Reduced from 4
        'learning_rate': 0.05,
        'subsample': 0.7,  # Reduced from 0.8
        'colsample_bytree': 0.7,
        'reg_alpha': 0.5,  # Increased L1
        'reg_lambda': 2.0,  # Increased L2
        'min_child_weight': 3,  # Minimum samples per leaf
        'gamma': 0.1,  # Minimum loss reduction for split
        'random_state': 42,
        'eval_metric': 'logloss'
    }
    
    def __init__(self, n_features: int, n_samples: int, n_events: int, random_seed: int = 42):
        """
        Initialize with sample size awareness
        
        Validates Events Per Variable (EPV) rule:
        - EPV >= 10 for reliable logistic regression
        - EPV >= 20 for reliable ML
        
        Parameters
        ----------
        n_features : int
            Number of features
        n_samples : int
            Total number of samples
        n_events : int
            Number of positive events (toxicity cases)
        random_seed : int
            Random seed for reproducibility
        """
        self.n_features = n_features
        self.n_samples = n_samples
        self.n_events = n_events
        self.random_seed = random_seed
        self.epv = n_events / max(n_features, 1)
        
        # Update random seed in configs
        self.ANN_CONFIG['random_state'] = random_seed
        self.XGBOOST_CONFIG['random_state'] = random_seed
        
        # Validate sample size adequacy
        # Note: EPV < 5 should trigger auto-feature reduction before model creation
        # This check is a safety net - feature reduction should happen earlier
        if self.epv < 5:
            # Try to auto-reduce features if possible
            # This requires feature reduction to happen before model creation
            raise ValueError(
                f"CRITICAL: EPV too low ({self.epv:.1f}). "
                f"Minimum EPV = 5 required. Cannot train model safely. "
                f"Use AutoFeatureReducer to reduce features before creating model."
            )
        elif self.epv < 10:
            warnings.warn(
                f"LOW EPV WARNING: {self.epv:.1f} events per variable. "
                f"Recommended EPV >= 10. Consider feature reduction. "
                f"Model complexity will be automatically reduced."
            )
        
        # Auto-adjust complexity based on sample size
        self._adjust_model_complexity()
    
    def _adjust_model_complexity(self):
        """Dynamically reduce model complexity for small samples"""
        if self.n_samples < 50:
            # Very small sample: minimal model
            self.ANN_CONFIG['hidden_layer_sizes'] = (8,)
            self.XGBOOST_CONFIG['max_depth'] = 1
            self.XGBOOST_CONFIG['n_estimators'] = 30
        elif self.n_samples < 100:
            # Small sample: conservative model
            self.ANN_CONFIG['hidden_layer_sizes'] = (16,)
            self.XGBOOST_CONFIG['max_depth'] = 2
    
    def create_ann_model(self) -> Pipeline:
        """Create ANN model with conservative configuration"""
        ann = MLPClassifier(**self.ANN_CONFIG)
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('ann', ann)
        ])
        return pipeline
    
    def create_xgboost_model(self) -> Optional[xgb.XGBClassifier]:
        """Create XGBoost model with conservative configuration"""
        if not XGBOOST_AVAILABLE:
            return None
        
        return xgb.XGBClassifier(**self.XGBOOST_CONFIG)
    
    def train_with_nested_cv(self, X: np.ndarray, y: np.ndarray, 
                            model_type: str = 'ann',
                            param_grid: Optional[Dict] = None) -> Dict:
        """
        NESTED CROSS-VALIDATION for unbiased performance estimation
        
        Outer loop: Performance estimation (5-fold)
        Inner loop: Hyperparameter tuning (3-fold)
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Binary outcomes
        model_type : str
            'ann' or 'xgboost'
        param_grid : dict, optional
            Hyperparameter grid for tuning
            
        Returns
        -------
        dict with keys:
            - 'nested_cv_auc_mean': Mean AUC across outer folds
            - 'nested_cv_auc_std': Standard deviation of AUC
            - 'nested_cv_auc_scores': List of AUC scores
            - 'best_params': List of best parameters for each fold
            - 'epv': Events per variable
            - 'sample_size_adequate': Whether EPV >= 10
        """
        from sklearn.metrics import roc_auc_score
        
        # Ensure binary classification
        if len(np.unique(y)) < 2:
            raise ValueError("Both classes must be present for classification")
        
        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_seed)
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_seed)
        
        outer_scores = []
        best_params_list = []
        
        for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Create model
            if model_type.lower() == 'ann':
                model = self.create_ann_model()
            elif model_type.lower() == 'xgboost':
                model = self.create_xgboost_model()
                if model is None:
                    raise ValueError("XGBoost not available")
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Inner CV for hyperparameter selection
            if param_grid:
                gs = GridSearchCV(
                    model,
                    param_grid,
                    cv=inner_cv,
                    scoring='roc_auc',
                    n_jobs=-1
                )
                gs.fit(X_train, y_train)
                best_model = gs.best_estimator_
                best_params_list.append(gs.best_params_)
            else:
                best_model = model
                best_model.fit(X_train, y_train)
            
            # Test on outer fold
            y_pred = best_model.predict_proba(X_test)[:, 1]
            
            # Calculate AUC
            try:
                score = roc_auc_score(y_test, y_pred)
                outer_scores.append(score)
            except ValueError:
                # If only one class in test fold, skip
                warnings.warn(f"Fold {fold}: Only one class in test set, skipping")
                continue
        
        if len(outer_scores) == 0:
            raise ValueError("Could not compute any AUC scores")
        
        return {
            'nested_cv_auc_mean': np.mean(outer_scores),
            'nested_cv_auc_std': np.std(outer_scores),
            'nested_cv_auc_scores': outer_scores,
            'best_params': best_params_list,
            'epv': self.epv,
            'sample_size_adequate': self.epv >= 10
        }
