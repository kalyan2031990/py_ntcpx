#!/usr/bin/env python3
"""
Tier 3: Modern Classical NTCP (de Vette / CITOR style)
=======================================================
Multivariable logistic regression with DVH and clinical features.
Uses L2-regularized logistic regression, bootstrap variable stability,
and calibration curves.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss
try:
    from sklearn.calibration import calibration_curve
except ImportError:
    from sklearn.metrics import calibration_curve
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from ntcp_models import check_epv, EPVError

# Windows-safe encoding
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except (AttributeError, ValueError):
    pass


class ModernLogisticNTCP:
    """Tier 3: Modern multivariable logistic regression NTCP"""
    
    def __init__(self, random_state=42, include_age: bool = True):
        """
        Initialize modern logistic NTCP model
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.include_age = include_age
        self.models = {}  # Store models per organ
        self.scalers = {}  # Store scalers per organ
        self.feature_names = {}  # Store feature names per organ
        self.bootstrap_stability = {}  # Store bootstrap variable stability
        self.calibration_curves = {}  # Store calibration curves
        self.epv_ = {}  # EPV per organ
        self.epv_reduced_ = {}  # Whether auto feature reduction was applied
    
    def prepare_features(self, organ_data, clinical_data=None):
        """
        Prepare feature matrix from DVH metrics and optional clinical data
        
        Args:
            organ_data: DataFrame with DVH metrics
            clinical_data: Optional DataFrame with clinical factors
        
        Returns:
            X: Feature matrix
            y: Outcome vector
            feature_names: List of feature names
        """
        # DVH features (always available)
        dvh_features = [
            'mean_dose', 'gEUD', 'V30', 'V50', 'D20'
        ]
        
        # Additional DVH features if available
        optional_dvh = [
            'V5', 'V10', 'V15', 'V20', 'V25', 'V35', 'V40', 'V45',
            'D1', 'D2', 'D5', 'D10', 'D30', 'D50', 'D70', 'D90', 'D95',
            'max_dose', 'total_volume'
        ]
        
        feature_cols = []
        for feat in dvh_features:
            if feat in organ_data.columns:
                feature_cols.append(feat)
        
        for feat in optional_dvh:
            if feat in organ_data.columns and feat not in feature_cols:
                feature_cols.append(feat)
        
        # Clinical features (optional, EPV-gated)
        clinical_features = []
        if clinical_data is not None:
            # Standardized candidate set
            clinical_candidates = ['age', 'sex_binary', 'tobacco_exposure', 'chemotherapy',
                                   'hpv_status', 'baseline_xerostomia']
            clinical_priority = ['age', 'baseline_xerostomia', 'tobacco_exposure']

            # Derive outcome vector early for EPV-based budget
            y = organ_data['Observed_Toxicity'].values.astype(int)
            n_events = int(y.sum())
            n_dvh_features = len(feature_cols)
            budget_for_clinical = max(0, int(n_events / 10) - n_dvh_features)

            # Prepare binary age flag if requested
            if self.include_age and 'age' in clinical_data.columns and 'age_over_50' not in clinical_data.columns:
                try:
                    clinical_data = clinical_data.copy()
                    clinical_data['age_over_50'] = (clinical_data['age'] >= 50).astype(int)
                except Exception:
                    pass

            # Priority-based inclusion within EPV budget
            if budget_for_clinical > 0:
                for feat in clinical_priority:
                    if feat in clinical_data.columns and budget_for_clinical > 0:
                        clinical_features.append(feat)
                        budget_for_clinical -= 1

            # Optionally include additional non-priority clinical candidates if budget remains
            if budget_for_clinical > 0:
                for feat in clinical_candidates:
                    if feat in clinical_data.columns and feat not in clinical_features and budget_for_clinical > 0:
                        clinical_features.append(feat)
                        budget_for_clinical -= 1

            # Also allow age_over_50 as a binary flag if created
            if self.include_age and 'age_over_50' in clinical_data.columns and 'age_over_50' not in clinical_features:
                # Prefer to include age_over_50 if budget still allows
                clinical_features.append('age_over_50')

        # Combine features
        all_features = feature_cols + clinical_features

        # Determine merge key for linking clinical data
        merge_key = None
        if clinical_data is not None and clinical_features:
            if 'PrimaryPatientID' in organ_data.columns and 'PrimaryPatientID' in clinical_data.columns:
                merge_key = 'PrimaryPatientID'
            elif 'PatientID' in organ_data.columns and 'PatientID' in clinical_data.columns:
                merge_key = 'PatientID'
        
        # Start feature matrix, optionally including ID column for merging
        if merge_key is not None:
            cols_for_X = [merge_key] + feature_cols
            X = organ_data[cols_for_X].copy()
        else:
            X = organ_data[feature_cols].copy()
        
        # Merge clinical data if available and merge key resolved
        if clinical_features and clinical_data is not None and merge_key is not None:
            X = pd.merge(
                X,
                clinical_data[[merge_key] + clinical_features],
                on=merge_key,
                how='left'
            )
            # Fill missing clinical data with median/mode
            for feat in clinical_features:
                if X[feat].dtype in [np.float64, np.int64]:
                    X[feat].fillna(X[feat].median(), inplace=True)
                else:
                    X[feat].fillna(X[feat].mode()[0] if len(X[feat].mode()) > 0 else 0, inplace=True)
        
        # Get outcomes
        y = organ_data['Observed_Toxicity'].values.astype(int)
        
        # Drop ID column from features if it was used for merging
        if merge_key is not None and merge_key in X.columns:
            X = X.drop(columns=[merge_key])
        
        # Handle missing values
        X = X.fillna(X.median() if len(X) > 0 else 0)
        
        return X.values, y, all_features
    
    def bootstrap_variable_stability(self, X, y, feature_names, n_bootstrap=1000):
        """
        Calculate variable stability using bootstrap
        
        Args:
            X: Feature matrix
            y: Outcome vector
            feature_names: List of feature names
            n_bootstrap: Number of bootstrap iterations
        
        Returns:
            dict with stability metrics for each feature
        """
        n_samples = len(X)
        n_features = X.shape[1]
        
        # Store coefficients across bootstrap iterations
        bootstrap_coefs = np.zeros((n_bootstrap, n_features))
        
        for i in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            
            # Fit model
            try:
                model = LogisticRegression(
                    C=1.0,  # L2 regularization
                    penalty='l2',
                    max_iter=1000,
                    random_state=self.random_state + i,
                    solver='lbfgs'
                )
                model.fit(X_boot, y_boot)
                bootstrap_coefs[i] = model.coef_[0]
            except:
                bootstrap_coefs[i] = np.nan
        
        # Calculate stability metrics
        stability = {}
        for j, feat_name in enumerate(feature_names):
            coefs = bootstrap_coefs[:, j]
            coefs = coefs[~np.isnan(coefs)]
            
            if len(coefs) > 0:
                stability[feat_name] = {
                    'mean_coef': float(np.mean(coefs)),
                    'std_coef': float(np.std(coefs)),
                    'stability_ratio': float(np.abs(np.mean(coefs)) / (np.std(coefs) + 1e-10)),
                    'non_zero_fraction': float(np.sum(np.abs(coefs) > 1e-6) / len(coefs))
                }
            else:
                stability[feat_name] = {
                    'mean_coef': 0.0,
                    'std_coef': 0.0,
                    'stability_ratio': 0.0,
                    'non_zero_fraction': 0.0
                }
        
        return stability
    
    def train_model(self, organ_data, organ, clinical_data=None):
        """
        Train multivariable logistic regression model for an organ
        
        Args:
            organ_data: DataFrame with DVH metrics and outcomes
            organ: Organ name
            clinical_data: Optional DataFrame with clinical factors
        
        Returns:
            dict with training results
        """
        print(f"  Training modern logistic NTCP for {organ}...")
        
        # Prepare features
        X, y, feature_names = self.prepare_features(organ_data, clinical_data)
        
        if len(X) < 20:
            print(f"  Warning: Insufficient data for {organ} (n={len(X)}). Need at least 20 samples.")
            return None
        
        # Check for sufficient positive/negative cases
        if np.sum(y) < 3 or np.sum(1 - y) < 3:
            print(f"  Warning: Insufficient class balance for {organ}.")
            return None
        
        # Train-test split (70/30)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state, stratify=y
        )

        # EPV-aware feature check
        n_events = int(y_train.sum())
        n_features = X_train.shape[1]
        try:
            epv = check_epv(
                n_events=n_events,
                n_features=n_features,
                min_epv=10.0,
                model_name="Tier3 Logistic",
            )
            self.epv_[organ] = epv
            self.epv_reduced_[organ] = False
        except EPVError as e:
            print(f"  WARNING: {e}")
            print(f"  Auto-reducing features to max_features = {int(n_events / 10) if n_events > 0 else 1}")
            # Auto-reduce: select top features by univariate correlation
            max_feat = max(1, int(n_events / 10)) if n_events > 0 else 1
            corrs = []
            for j in range(n_features):
                try:
                    corr = np.corrcoef(X_train[:, j], y_train)[0, 1]
                    corrs.append(abs(corr))
                except Exception:
                    corrs.append(0.0)
            corrs = np.array(corrs)
            top_idx = np.argsort(corrs)[::-1][:max_feat]
            X_train = X_train[:, top_idx]
            X_test = X_test[:, top_idx]
            feature_names = [feature_names[i] for i in top_idx]

            # Re-check EPV after reduction (hard gate)
            epv = check_epv(
                n_events=n_events,
                n_features=len(feature_names),
                min_epv=10.0,
                model_name="Tier3 Logistic (reduced)",
            )
            self.epv_[organ] = epv
            self.epv_reduced_[organ] = True

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train L2-regularized logistic regression
        model = LogisticRegression(
            C=1.0,  # L2 regularization strength
            penalty='l2',
            max_iter=1000,
            random_state=self.random_state,
            solver='lbfgs'
        )
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_train_pred = model.predict_proba(X_train_scaled)[:, 1]
        y_test_pred = model.predict_proba(X_test_scaled)[:, 1]
        
        # Metrics
        train_auc = roc_auc_score(y_train, y_train_pred)
        test_auc = roc_auc_score(y_test, y_test_pred)
        train_brier = brier_score_loss(y_train, y_train_pred)
        test_brier = brier_score_loss(y_test, y_test_pred)
        
        # Calibration curve
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_test, y_test_pred, n_bins=10
            )
            calibration_data = {
                'fraction_of_positives': fraction_of_positives.tolist(),
                'mean_predicted_value': mean_predicted_value.tolist()
            }
        except:
            calibration_data = None
        
        # Bootstrap variable stability
        print(f"    Running bootstrap variable stability (1000 iterations)...")
        stability = self.bootstrap_variable_stability(X_train_scaled, y_train, feature_names, n_bootstrap=1000)
        
        # Store model and scaler
        self.models[organ] = model
        self.scalers[organ] = scaler
        self.feature_names[organ] = feature_names
        self.bootstrap_stability[organ] = stability
        self.calibration_curves[organ] = calibration_data
        
        print(f"    Train AUC: {train_auc:.3f}, Test AUC: {test_auc:.3f}")
        print(f"    Train Brier: {train_brier:.3f}, Test Brier: {test_brier:.3f}")
        print(f"    Overfitting gap (AUC): {train_auc - test_auc:.3f}")
        
        return {
            'train_auc': train_auc,
            'test_auc': test_auc,
            'train_brier': train_brier,
            'test_brier': test_brier,
            'overfitting_gap_auc': train_auc - test_auc,
            'overfitting_gap_brier': train_brier - test_brier,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'n_features': len(feature_names),
            'feature_names': feature_names,
            'coefficients': {feat: float(coef) for feat, coef in zip(feature_names, model.coef_[0])}
        }
    
    def predict_ntcp(self, organ_data, organ, clinical_data=None):
        """
        Predict NTCP for organ data
        
        Args:
            organ_data: DataFrame with DVH metrics
            organ: Organ name
            clinical_data: Optional DataFrame with clinical factors
        
        Returns:
            array of NTCP predictions
        """
        if organ not in self.models:
            return np.full(len(organ_data), np.nan)
        
        # Prepare features
        X, _, _ = self.prepare_features(organ_data, clinical_data)
        
        # Scale
        scaler = self.scalers[organ]
        X_scaled = scaler.transform(X)
        
        # Predict
        model = self.models[organ]
        predictions = model.predict_proba(X_scaled)[:, 1]
        
        return predictions
    
    def get_feature_importance(self, organ):
        """
        Get feature importance (absolute coefficients) for an organ
        
        Args:
            organ: Organ name
        
        Returns:
            dict with feature names and importance scores
        """
        if organ not in self.models:
            return {}
        
        model = self.models[organ]
        feature_names = self.feature_names[organ]
        
        importance = {
            feat: float(np.abs(coef))
            for feat, coef in zip(feature_names, model.coef_[0])
        }
        
        # Sort by importance
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        
        return importance

    def predict_ntcp_cv(self, organ_data: pd.DataFrame, organ: str,
                        clinical_data=None, cv_strategy: str = 'auto') -> dict:
        """
        Generate cross-validated NTCP predictions (honest, not apparent).

        For n < 100: use Leave-One-Out CV (LOO)
        For n >= 100: use 5-fold stratified CV

        Returns:
            dict with keys:
                'predictions_cv'      : np.ndarray, CV predictions for all patients
                'predictions_apparent': np.ndarray, apparent predictions (full model)
                'cv_auc'              : float
                'cv_auc_std'          : float (fold-level SD; NaN for LOO)
                'loo_auc'             : float (only for LOO)
                'apparent_auc'        : float
                'overfitting_gap'     : float (apparent_auc - cv_auc)
                'cv_strategy'         : str ('LOO' or '5-fold')
                'epv'                 : float
                'n_features'          : int
                'feature_names'       : list
                'fold_aucs'           : list or None
        """
        from sklearn.model_selection import LeaveOneOut, StratifiedKFold
        from sklearn.base import clone

        X, y, feature_names = self.prepare_features(organ_data, clinical_data)
        n = len(X)
        n_events = int(y.sum())

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model_template = LogisticRegression(
            C=1.0,
            penalty='l2',
            max_iter=1000,
            random_state=self.random_state,
            solver='lbfgs'
        )

        # Choose CV strategy
        if cv_strategy == 'auto':
            cv_strategy = 'LOO' if n < 100 else '5-fold'

        cv_preds = np.full(n, np.nan)

        if cv_strategy == 'LOO':
            loo = LeaveOneOut()
            for train_idx, test_idx in loo.split(X_scaled):
                m = clone(model_template)
                m.fit(X_scaled[train_idx], y[train_idx])
                cv_preds[test_idx] = m.predict_proba(X_scaled[test_idx])[:, 1]
            cv_auc = roc_auc_score(y, cv_preds)
            cv_auc_std = np.nan
            loo_auc = cv_auc
            fold_aucs = None
        else:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            fold_aucs = []
            for train_idx, test_idx in skf.split(X_scaled, y):
                m = clone(model_template)
                m.fit(X_scaled[train_idx], y[train_idx])
                fold_pred = m.predict_proba(X_scaled[test_idx])[:, 1]
                cv_preds[test_idx] = fold_pred
                try:
                    fold_auc = roc_auc_score(y[test_idx], fold_pred)
                    fold_aucs.append(fold_auc)
                except Exception:
                    continue
            try:
                cv_auc = roc_auc_score(y, cv_preds)
            except Exception:
                cv_auc = np.nan
            cv_auc_std = np.std(fold_aucs) if fold_aucs else np.nan
            loo_auc = np.nan

        # Apparent AUC (full model)
        full_model = clone(model_template)
        full_model.fit(X_scaled, y)
        apparent_preds = full_model.predict_proba(X_scaled)[:, 1]
        try:
            apparent_auc = roc_auc_score(y, apparent_preds)
        except Exception:
            apparent_auc = np.nan

        epv = n_events / len(feature_names) if len(feature_names) > 0 else np.nan

        return {
            'predictions_cv': cv_preds,
            'predictions_apparent': apparent_preds,
            'cv_auc': cv_auc,
            'cv_auc_std': cv_auc_std,
            'loo_auc': loo_auc,
            'apparent_auc': apparent_auc,
            'overfitting_gap': apparent_auc - cv_auc if (cv_auc is not None and not np.isnan(cv_auc) and not np.isnan(apparent_auc)) else np.nan,
            'cv_strategy': cv_strategy,
            'epv': epv,
            'n_features': len(feature_names),
            'feature_names': feature_names,
            'fold_aucs': fold_aucs,
        }

