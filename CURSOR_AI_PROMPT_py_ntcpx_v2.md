# COMPREHENSIVE CURSOR AI PROMPT: py_ntcpx Pipeline Transformation
## From Experimental to High-End Publishable Software

---

# EXECUTIVE SUMMARY

You are tasked with transforming the **py_ntcpx** Python pipeline from an experimental research tool into a **publication-ready, methodologically rigorous** software package suitable for peer-reviewed journal submission (target: IJROBP, Radiotherapy & Oncology, Physics in Medicine & Biology).

**Current State**: Functional pipeline with 6 sequential modules (code1-code5 + SHAP), tested on 54 head & neck cancer patients, producing NTCP predictions using traditional (LKB, RS) and ML (ANN, XGBoost) models.

**Target State**: Statistically rigorous, reproducible, well-documented software with proper validation methodology, no data leakage, publication-ready outputs, and comprehensive test coverage.

---

# PART 1: CRITICAL ISSUES REQUIRING IMMEDIATE CORRECTION

## 1.1 DATA LEAKAGE ARCHITECTURE (CRITICAL - Priority 1)

### Current Problem
The pipeline has potential data leakage in the following areas:

**Issue 1: Clinical Data Double-Dipping**
```
Step 0: Clinical Reconciliation → Creates reconciled data
Step 3: NTCP Analysis → May use ORIGINAL clinical data (bypassing reconciliation)
```

**Issue 2: Train-Test Split Timing**
- Features are extracted from ALL patients before train-test split
- Scaling/normalization may use entire dataset statistics
- Cross-validation folds may not be patient-stratified by institution

### Required Fix
```python
# FILE: src/validation/data_splitter.py

class PatientDataSplitter:
    """
    CRITICAL: Patient-level splitting with no data leakage
    """
    
    def __init__(self, random_seed: int = 42, test_size: float = 0.2):
        self.random_seed = random_seed
        self.test_size = test_size
        self._fitted = False
        
    def create_splits(self, 
                     patient_df: pd.DataFrame, 
                     outcome_col: str = 'Observed_Toxicity',
                     stratify_cols: List[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create patient-level train/test splits with stratification.
        
        IMPORTANT: 
        - Split by PATIENT ID, not by rows
        - Stratify by toxicity outcome AND institution if multi-site
        - NEVER allow same patient in both train and test
        """
        np.random.seed(self.random_seed)
        
        # Get unique patients
        unique_patients = patient_df['PatientID'].unique()
        
        # Create stratification key
        if stratify_cols:
            patient_strata = patient_df.groupby('PatientID')[stratify_cols].first()
            strata_key = patient_strata.apply(lambda x: '_'.join(x.astype(str)), axis=1)
        else:
            # Stratify by outcome only
            patient_outcomes = patient_df.groupby('PatientID')[outcome_col].max()
            strata_key = patient_outcomes
        
        # Stratified split
        from sklearn.model_selection import train_test_split
        train_patients, test_patients = train_test_split(
            unique_patients,
            test_size=self.test_size,
            random_state=self.random_seed,
            stratify=strata_key[unique_patients] if len(strata_key.unique()) > 1 else None
        )
        
        train_df = patient_df[patient_df['PatientID'].isin(train_patients)].copy()
        test_df = patient_df[patient_df['PatientID'].isin(test_patients)].copy()
        
        self._fitted = True
        self._train_patients = set(train_patients)
        self._test_patients = set(test_patients)
        
        return train_df, test_df
    
    def validate_no_leakage(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> bool:
        """Verify no patient overlap between train and test"""
        train_patients = set(train_df['PatientID'].unique())
        test_patients = set(test_df['PatientID'].unique())
        overlap = train_patients & test_patients
        
        if overlap:
            raise ValueError(f"DATA LEAKAGE DETECTED: {len(overlap)} patients in both sets: {overlap}")
        return True
```

### Implementation Checklist
- [ ] Modify `code3_ntcp_analysis_ml.py` to use `PatientDataSplitter`
- [ ] Ensure StandardScaler is fit ONLY on training data
- [ ] Verify feature extraction happens AFTER split
- [ ] Add leakage detection to QA reporter (code4)
- [ ] Document split methodology in output reports

---

## 1.2 OVERFITTING QUANTIFICATION (CRITICAL - Priority 2)

### Current Problem
Evidence of overfitting from log analysis:
- **Feature-to-Sample Ratio**: 24 features / 54 samples = 0.44 (should be <0.1)
- **AUC Train-Test Gaps**:
  - ANN: ~35.9% gap (Train ~0.85, Test 0.545)
  - XGBoost: ~52.6% gap (Train ~0.80, Test 0.379)
- **CV Instability**: XGBoost CV AUC SD = 0.162 (high variance)

### Required Fix
```python
# FILE: src/models/ml_models.py

class OverfitResistantMLModels:
    """
    ML models with aggressive overfitting prevention
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
        'min_child_weight': 3,  # NEW: Minimum samples per leaf
        'gamma': 0.1,  # NEW: Minimum loss reduction for split
        'random_state': 42,
        'eval_metric': 'logloss'
    }
    
    def __init__(self, n_features: int, n_samples: int, n_events: int):
        """
        Initialize with sample size awareness
        
        Validates Events Per Variable (EPV) rule:
        - EPV >= 10 for reliable logistic regression
        - EPV >= 20 for reliable ML
        """
        self.n_features = n_features
        self.n_samples = n_samples
        self.n_events = n_events
        self.epv = n_events / max(n_features, 1)
        
        # Validate sample size adequacy
        if self.epv < 10:
            warnings.warn(
                f"LOW EPV WARNING: {self.epv:.1f} events per variable. "
                f"Recommended EPV >= 10. Consider feature reduction."
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
    
    def train_with_nested_cv(self, X, y, param_grid=None) -> Dict:
        """
        NESTED CROSS-VALIDATION for unbiased performance estimation
        
        Outer loop: Performance estimation (5-fold)
        Inner loop: Hyperparameter tuning (3-fold)
        """
        from sklearn.model_selection import StratifiedKFold, GridSearchCV
        
        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        outer_scores = []
        best_params_list = []
        
        for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Inner CV for hyperparameter selection
            if param_grid:
                gs = GridSearchCV(
                    self._create_model(),
                    param_grid,
                    cv=inner_cv,
                    scoring='roc_auc',
                    n_jobs=-1
                )
                gs.fit(X_train, y_train)
                best_model = gs.best_estimator_
                best_params_list.append(gs.best_params_)
            else:
                best_model = self._create_model()
                best_model.fit(X_train, y_train)
            
            # Test on outer fold
            y_pred = best_model.predict_proba(X_test)[:, 1]
            from sklearn.metrics import roc_auc_score
            score = roc_auc_score(y_test, y_pred)
            outer_scores.append(score)
        
        return {
            'nested_cv_auc_mean': np.mean(outer_scores),
            'nested_cv_auc_std': np.std(outer_scores),
            'nested_cv_auc_scores': outer_scores,
            'best_params': best_params_list,
            'epv': self.epv,
            'sample_size_adequate': self.epv >= 10
        }
```

### Feature Selection Protocol
```python
# FILE: src/features/feature_selector.py

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
                       organ: str,
                       max_features: int = None) -> List[str]:
        """
        Select features using domain knowledge + statistical filtering
        
        Strategy:
        1. Start with literature-validated features
        2. Add statistical filtering (univariate p < 0.1)
        3. Cap at max_features based on EPV rule
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
        
        # 2. Statistical filtering on remaining features
        from scipy.stats import mannwhitneyu
        candidate_features = [c for c in X.columns if c not in selected]
        
        p_values = {}
        for feat in candidate_features:
            try:
                stat, p = mannwhitneyu(
                    X.loc[y == 1, feat].dropna(),
                    X.loc[y == 0, feat].dropna()
                )
                p_values[feat] = p
            except:
                p_values[feat] = 1.0
        
        # Sort by p-value and add until max_features
        sorted_features = sorted(p_values.items(), key=lambda x: x[1])
        for feat, p in sorted_features:
            if len(selected) >= max_features:
                break
            if p < 0.1:  # Univariate significance threshold
                selected.append(feat)
        
        return selected
```

### Implementation Checklist
- [ ] Replace current ML configs with conservative versions
- [ ] Implement nested CV for unbiased performance estimation
- [ ] Add EPV calculation and warnings to QA report
- [ ] Implement feature selection based on sample size
- [ ] Add train-test gap monitoring to output

---

## 1.3 STATISTICAL METHODOLOGY CORRECTIONS (CRITICAL - Priority 3)

### Current Problem: Monte Carlo NTCP Misimplementation
```python
# SUSPECTED FLAWED IMPLEMENTATION:
def monte_carlo_ntcp(predictions, stds):
    # WRONG: Adding std to probability is mathematically invalid!
    return predictions + 2 * stds
```

### Correct Implementation
```python
# FILE: src/models/uncertainty/monte_carlo_ntcp.py

class MonteCarloNTCPCorrect:
    """
    Correct Monte Carlo NTCP with proper uncertainty propagation
    
    References:
    - Taylor JR. An Introduction to Error Analysis. 2nd Ed.
    - Burman et al. IJROBP 1991 (LKB model uncertainty)
    """
    
    def __init__(self, n_samples: int = 10000, random_seed: int = 42):
        self.n_samples = n_samples
        np.random.seed(random_seed)
    
    def predict_with_parameter_uncertainty(
        self,
        model,
        geud_values: np.ndarray,
        param_mean: Dict[str, float],
        param_cov: np.ndarray
    ) -> Dict:
        """
        Monte Carlo NTCP prediction with parameter uncertainty
        
        Parameters
        ----------
        model : NTCPModel
            Fitted NTCP model (LKB, RS, etc.)
        geud_values : np.ndarray
            gEUD values for each patient
        param_mean : dict
            Mean parameter values {'TD50': float, 'm': float}
        param_cov : np.ndarray
            Parameter covariance matrix (2x2 for LKB)
            
        Returns
        -------
        dict with keys:
            - 'mean': Mean NTCP prediction
            - 'std': Standard deviation
            - 'ci_lower': 2.5th percentile
            - 'ci_upper': 97.5th percentile
            - 'samples': All MC samples (for diagnostics)
        """
        # Sample parameters from multivariate normal
        param_names = list(param_mean.keys())
        param_values = np.array([param_mean[k] for k in param_names])
        
        param_samples = np.random.multivariate_normal(
            param_values, param_cov, self.n_samples
        )
        
        # Ensure parameters stay in valid range
        # TD50 > 0, m > 0 (for LKB)
        param_samples = np.clip(param_samples, 0.01, None)
        
        # Calculate NTCP for each parameter sample
        n_patients = len(geud_values)
        ntcp_samples = np.zeros((self.n_samples, n_patients))
        
        for i, params in enumerate(param_samples):
            param_dict = dict(zip(param_names, params))
            ntcp_samples[i] = model.predict(geud_values, **param_dict)
        
        # Calculate statistics
        mean_pred = np.mean(ntcp_samples, axis=0)
        std_pred = np.std(ntcp_samples, axis=0)
        ci_lower = np.percentile(ntcp_samples, 2.5, axis=0)
        ci_upper = np.percentile(ntcp_samples, 97.5, axis=0)
        
        return {
            'mean': mean_pred,
            'std': std_pred,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'samples': ntcp_samples
        }
    
    def predict_with_data_uncertainty(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        n_bootstrap: int = 2000
    ) -> Dict:
        """
        Bootstrap NTCP uncertainty (data-driven)
        
        More robust than parameter uncertainty when:
        - Model may be misspecified
        - Parameter covariance is unknown
        """
        n_samples = len(y)
        bootstrap_preds = np.zeros((n_bootstrap, n_samples))
        
        for b in range(n_bootstrap):
            # Bootstrap resample
            idx = np.random.choice(n_samples, n_samples, replace=True)
            X_boot, y_boot = X[idx], y[idx]
            
            # Refit model
            model_boot = clone(model)
            model_boot.fit(X_boot, y_boot)
            
            # Predict on original data
            bootstrap_preds[b] = model_boot.predict_proba(X)[:, 1]
        
        return {
            'mean': np.mean(bootstrap_preds, axis=0),
            'std': np.std(bootstrap_preds, axis=0),
            'ci_lower': np.percentile(bootstrap_preds, 2.5, axis=0),
            'ci_upper': np.percentile(bootstrap_preds, 97.5, axis=0)
        }
```

### AUC with Proper Confidence Intervals
```python
# FILE: src/metrics/auc_calculator.py

def calculate_auc_with_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    method: str = 'bootstrap',
    n_bootstraps: int = 2000,
    alpha: float = 0.05
) -> Tuple[float, Tuple[float, float]]:
    """
    Calculate AUC with proper confidence intervals
    
    Parameters
    ----------
    y_true : array
        True binary labels
    y_pred : array
        Predicted probabilities
    method : str
        'bootstrap' (recommended) or 'delong'
    n_bootstraps : int
        Number of bootstrap iterations
    alpha : float
        Significance level (default 0.05 for 95% CI)
        
    Returns
    -------
    auc : float
        Point estimate of AUC
    ci : tuple
        (lower, upper) confidence bounds
    """
    from sklearn.metrics import roc_auc_score
    
    # Point estimate
    auc = roc_auc_score(y_true, y_pred)
    
    if method == 'bootstrap':
        bootstrapped_aucs = []
        n = len(y_true)
        
        for _ in range(n_bootstraps):
            # Resample with replacement
            idx = np.random.choice(n, n, replace=True)
            
            # Ensure both classes present
            if len(np.unique(y_true[idx])) < 2:
                continue
            
            try:
                boot_auc = roc_auc_score(y_true[idx], y_pred[idx])
                bootstrapped_aucs.append(boot_auc)
            except:
                continue
        
        # Percentile CI
        ci_lower = np.percentile(bootstrapped_aucs, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrapped_aucs, 100 * (1 - alpha / 2))
        
    elif method == 'delong':
        # DeLong's method for AUC variance
        # Reference: DeLong et al. Biometrics 1988
        ci_lower, ci_upper = _delong_ci(y_true, y_pred, alpha)
    
    return auc, (ci_lower, ci_upper)


def compare_aucs_delong(
    y_true: np.ndarray,
    y_pred1: np.ndarray,
    y_pred2: np.ndarray
) -> Tuple[float, float]:
    """
    DeLong test for comparing two AUCs
    
    Returns
    -------
    z_stat : float
        Z statistic
    p_value : float
        Two-sided p-value
    """
    from scipy import stats
    
    # Implement DeLong's method
    # Reference: Sun & Xu, Bioinformatics 2014
    
    n1 = np.sum(y_true == 1)
    n0 = np.sum(y_true == 0)
    
    # ... (full implementation)
    
    return z_stat, p_value
```

### Implementation Checklist
- [ ] Replace all Monte Carlo NTCP calculations with correct implementation
- [ ] Add bootstrap CI to all AUC calculations
- [ ] Implement DeLong test for model comparisons
- [ ] Add parameter covariance estimation to traditional models
- [ ] Include uncertainty estimates in all output reports

---

# PART 2: CODE ARCHITECTURE IMPROVEMENTS

## 2.1 CONFIGURATION MANAGEMENT

```yaml
# FILE: config/pipeline_config.yaml

pipeline:
  version: "2.0.0"
  random_seed: 42
  
validation:
  method: "nested_cv"  # Options: nested_cv, train_test_split, loocv
  outer_folds: 5
  inner_folds: 3
  test_size: 0.2
  stratify: true
  stratify_columns: ["Observed_Toxicity"]

models:
  traditional:
    lkb_log_logit:
      enabled: true
      TD50_bounds: [10, 70]
      m_bounds: [0.05, 0.5]
      optimization: "differential_evolution"
      
    lkb_probit:
      enabled: true
      TD50_bounds: [10, 70]
      gamma50_bounds: [0.5, 3.0]
      
    rs_poisson:
      enabled: true
      D50_bounds: [10, 70]
      gamma_bounds: [0.5, 3.0]
      
  machine_learning:
    ann:
      enabled: true
      hidden_layers: [16, 8]
      dropout_rate: 0.3
      l2_alpha: 0.01
      learning_rate: 0.001
      max_epochs: 500
      early_stopping_patience: 20
      
    xgboost:
      enabled: true
      max_depth: 2
      n_estimators: 50
      learning_rate: 0.05
      subsample: 0.7
      colsample_bytree: 0.7
      reg_alpha: 0.5
      reg_lambda: 2.0

features:
  selection_method: "domain_knowledge"  # Options: domain_knowledge, lasso, univariate, none
  max_features_epv_ratio: 10  # max_features = n_events / this value
  parotid_essential: ["Dmean", "V30", "V45"]
  
reporting:
  confidence_level: 0.95
  n_bootstraps: 2000
  figure_dpi: 600
  decimal_places: 3
  
quality_assurance:
  check_data_leakage: true
  check_overfitting: true
  auc_gap_threshold: 0.15
  cv_std_threshold: 0.15
```

## 2.2 PROJECT STRUCTURE

```
py_ntcpx_v2.0/
├── config/
│   ├── pipeline_config.yaml
│   ├── model_parameters.yaml
│   └── logging_config.yaml
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dvh_preprocessor.py
│   │   ├── clinical_reconciler.py
│   │   └── data_validator.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── traditional/
│   │   │   ├── lkb_model.py
│   │   │   ├── rs_model.py
│   │   │   └── geud_calculator.py
│   │   ├── machine_learning/
│   │   │   ├── ann_model.py
│   │   │   ├── xgboost_model.py
│   │   │   └── model_factory.py
│   │   └── uncertainty/
│   │       ├── monte_carlo_ntcp.py
│   │       └── bootstrap_estimator.py
│   │
│   ├── validation/
│   │   ├── __init__.py
│   │   ├── data_splitter.py
│   │   ├── nested_cv.py
│   │   └── calibration.py
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── feature_extractor.py
│   │   └── feature_selector.py
│   │
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── auc_calculator.py
│   │   ├── brier_score.py
│   │   └── calibration_metrics.py
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── publication_plots.py
│   │   ├── roc_plots.py
│   │   └── calibration_plots.py
│   │
│   ├── reporting/
│   │   ├── __init__.py
│   │   ├── statistical_reporter.py
│   │   ├── qa_reporter.py
│   │   └── latex_tables.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       └── config_loader.py
│
├── scripts/
│   ├── run_pipeline.py
│   ├── code1_dvh_preprocess.py
│   ├── code2_dvh_plot_and_summary.py
│   ├── code3_ntcp_analysis_ml.py
│   ├── code4_ntcp_output_QA_reporter.py
│   ├── code5_ntcp_factors_analysis.py
│   └── shap_analysis.py
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_data_splitter.py
│   ├── test_ntcp_models.py
│   ├── test_ml_models.py
│   ├── test_metrics.py
│   ├── test_integration.py
│   └── test_data/
│       ├── sample_dvh.csv
│       └── sample_clinical.xlsx
│
├── docs/
│   ├── user_guide.md
│   ├── api_reference.md
│   └── methodology.md
│
├── requirements.txt
├── setup.py
├── pyproject.toml
├── README.md
├── CITATION.cff
└── .github/
    └── workflows/
        └── ci.yml
```

---

# PART 3: TEST SUITE REQUIREMENTS

## 3.1 Unit Tests

```python
# FILE: tests/test_ntcp_models.py

import pytest
import numpy as np
from src.models.traditional.lkb_model import LKBModel

class TestLKBModel:
    """Test suite for LKB NTCP model"""
    
    @pytest.fixture
    def lkb_model(self):
        return LKBModel()
    
    def test_ntcp_bounds(self, lkb_model):
        """NTCP must be in [0, 1]"""
        doses = np.linspace(0, 100, 101)
        predictions = lkb_model.predict(doses, TD50=30, m=0.1)
        
        assert np.all(predictions >= 0), "NTCP < 0 detected"
        assert np.all(predictions <= 1), "NTCP > 1 detected"
    
    def test_monotonicity(self, lkb_model):
        """NTCP should increase with dose"""
        doses = np.linspace(10, 60, 51)
        predictions = lkb_model.predict(doses, TD50=30, m=0.1)
        
        assert np.all(np.diff(predictions) >= 0), "NTCP not monotonic"
    
    def test_td50_crossover(self, lkb_model):
        """NTCP(TD50) should be ~0.5"""
        TD50 = 30
        ntcp_at_td50 = lkb_model.predict(np.array([TD50]), TD50=TD50, m=0.1)
        
        assert np.abs(ntcp_at_td50 - 0.5) < 0.01, "NTCP(TD50) != 0.5"
    
    def test_known_values(self, lkb_model):
        """Test against known literature values"""
        # QUANTEC parotid: TD50=39.9, m=0.4
        # At 40 Gy, NTCP should be ~50%
        ntcp = lkb_model.predict(np.array([40]), TD50=39.9, m=0.4)
        assert 0.45 < ntcp < 0.55, f"NTCP at TD50 should be ~0.5, got {ntcp}"


class TestDataLeakage:
    """Test suite for data leakage prevention"""
    
    def test_train_test_no_overlap(self):
        """Verify no patient overlap between train and test"""
        from src.validation.data_splitter import PatientDataSplitter
        
        df = pd.DataFrame({
            'PatientID': [f'P{i}' for i in range(50)],
            'Observed_Toxicity': np.random.binomial(1, 0.3, 50),
            'Dmean': np.random.uniform(20, 50, 50)
        })
        
        splitter = PatientDataSplitter(random_seed=42)
        train_df, test_df = splitter.create_splits(df)
        
        train_patients = set(train_df['PatientID'])
        test_patients = set(test_df['PatientID'])
        
        assert len(train_patients & test_patients) == 0, "Data leakage detected!"
    
    def test_scaler_fit_on_train_only(self):
        """StandardScaler must be fit only on training data"""
        from sklearn.preprocessing import StandardScaler
        
        X_train = np.random.randn(80, 5)
        X_test = np.random.randn(20, 5)
        
        scaler = StandardScaler()
        scaler.fit(X_train)  # Fit only on train
        
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train mean should be ~0, test mean may differ
        assert np.abs(X_train_scaled.mean()) < 0.1
        # This is expected - test data is scaled with train statistics


class TestReproducibility:
    """Test suite for reproducibility"""
    
    def test_deterministic_results(self):
        """Same seed should give identical results"""
        from src.models.machine_learning.ann_model import ANNModel
        
        X = np.random.randn(100, 5)
        y = np.random.binomial(1, 0.3, 100)
        
        model1 = ANNModel(random_seed=42)
        model1.fit(X, y)
        pred1 = model1.predict_proba(X)
        
        model2 = ANNModel(random_seed=42)
        model2.fit(X, y)
        pred2 = model2.predict_proba(X)
        
        np.testing.assert_array_almost_equal(pred1, pred2, decimal=10)
```

## 3.2 Integration Tests

```python
# FILE: tests/test_integration.py

class TestEndToEndPipeline:
    """Integration tests for complete pipeline"""
    
    def test_full_pipeline_execution(self, tmp_path):
        """Test complete pipeline from DVH to reports"""
        from scripts.run_pipeline import NTCPPipeline
        
        # Setup test data
        test_dvh_dir = tmp_path / "dvh"
        test_clinical = tmp_path / "clinical.xlsx"
        output_dir = tmp_path / "output"
        
        # Create minimal test data
        self._create_test_data(test_dvh_dir, test_clinical)
        
        # Run pipeline
        pipeline = NTCPPipeline(
            dvh_dir=test_dvh_dir,
            clinical_file=test_clinical,
            output_dir=output_dir,
            config_file="config/pipeline_config.yaml"
        )
        
        results = pipeline.run()
        
        # Validate outputs
        assert (output_dir / "ntcp_results.xlsx").exists()
        assert (output_dir / "qa_report.docx").exists()
        assert 'auc' in results['metrics']
        assert 0 <= results['metrics']['auc'] <= 1
    
    def test_pipeline_reproducibility(self, tmp_path):
        """Same inputs + seed should produce identical outputs"""
        # Run twice with same seed
        results1 = self._run_pipeline(tmp_path, seed=42)
        results2 = self._run_pipeline(tmp_path, seed=42)
        
        np.testing.assert_almost_equal(
            results1['metrics']['auc'],
            results2['metrics']['auc'],
            decimal=10
        )
```

---

# PART 4: PUBLICATION-READY OUTPUTS

## 4.1 Statistical Reporting

```python
# FILE: src/reporting/statistical_reporter.py

class PublicationStatisticalReporter:
    """Generate publication-ready statistical tables"""
    
    def create_table2_model_comparison(self, results: Dict) -> pd.DataFrame:
        """
        Create Table 2: Model Performance Comparison
        
        Format follows IJROBP guidelines
        """
        table_data = []
        
        for model_name, metrics in results.items():
            auc_mean, auc_ci = metrics['auc'], metrics['auc_ci']
            brier_mean, brier_ci = metrics['brier'], metrics['brier_ci']
            
            row = {
                'Model': model_name,
                'AUC (95% CI)': f"{auc_mean:.3f} ({auc_ci[0]:.3f}-{auc_ci[1]:.3f})",
                'Brier Score (95% CI)': f"{brier_mean:.3f} ({brier_ci[0]:.3f}-{brier_ci[1]:.3f})",
                'Calibration Slope': f"{metrics['cal_slope']:.3f}",
                'CV AUC (mean ± SD)': f"{metrics['cv_auc_mean']:.3f} ± {metrics['cv_auc_std']:.3f}"
            }
            table_data.append(row)
        
        return pd.DataFrame(table_data)
    
    def create_latex_table(self, df: pd.DataFrame, caption: str) -> str:
        """Export table as LaTeX for direct manuscript inclusion"""
        latex = df.to_latex(
            index=False,
            escape=False,
            caption=caption,
            label="tab:model_comparison"
        )
        return latex
```

## 4.2 Publication Figures

```python
# FILE: src/visualization/publication_plots.py

class PublicationFigureGenerator:
    """Generate 600 DPI publication-ready figures"""
    
    # IJROBP color scheme
    COLORS = {
        'LKB_LogLogit': '#1f77b4',
        'LKB_Probit': '#ff7f0e', 
        'RS_Poisson': '#2ca02c',
        'ANN': '#d62728',
        'XGBoost': '#9467bd'
    }
    
    def __init__(self, output_dir: Path, dpi: int = 600):
        self.output_dir = output_dir
        self.dpi = dpi
        
        # Set publication defaults
        plt.rcParams.update({
            'font.family': 'Arial',
            'font.size': 10,
            'axes.linewidth': 1.0,
            'figure.dpi': dpi
        })
    
    def plot_roc_curves(self, results: Dict, filename: str = "fig2_roc_curves.png"):
        """
        Figure 2: ROC Curves for All Models
        
        Requirements:
        - 600 DPI
        - AUC values in legend
        - 95% CI bands
        - Diagonal reference line
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        for model_name, metrics in results.items():
            fpr = metrics['fpr']
            tpr = metrics['tpr']
            auc_val = metrics['auc']
            auc_ci = metrics['auc_ci']
            
            label = f"{model_name} (AUC = {auc_val:.3f}, 95% CI: {auc_ci[0]:.3f}-{auc_ci[1]:.3f})"
            
            ax.plot(fpr, tpr, 
                   color=self.COLORS.get(model_name, 'gray'),
                   linewidth=2,
                   label=label)
        
        # Reference line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Reference')
        
        ax.set_xlabel('1 - Specificity (False Positive Rate)')
        ax.set_ylabel('Sensitivity (True Positive Rate)')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.legend(loc='lower right', fontsize=9)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        fig.savefig(self.output_dir / filename, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return self.output_dir / filename
```

---

# PART 5: IMPLEMENTATION CHECKLIST

## Phase 1: Critical Fixes (Week 1-2)
- [ ] 1.1 Fix data leakage in train-test split
- [ ] 1.2 Implement patient-level splitting with stratification
- [ ] 1.3 Add leakage detection to QA reporter
- [ ] 1.4 Fix StandardScaler to fit only on training data
- [ ] 1.5 Correct Monte Carlo NTCP implementation
- [ ] 1.6 Add bootstrap CI to all AUC calculations

## Phase 2: Model Improvements (Week 3-4)
- [ ] 2.1 Implement nested cross-validation
- [ ] 2.2 Add conservative ML hyperparameters
- [ ] 2.3 Implement domain-guided feature selection
- [ ] 2.4 Add EPV warnings and auto-adjustment
- [ ] 2.5 Implement DeLong test for model comparison

## Phase 3: Code Quality (Week 5-6)
- [ ] 3.1 Create modular project structure
- [ ] 3.2 Implement configuration management (YAML)
- [ ] 3.3 Add comprehensive logging
- [ ] 3.4 Write unit tests (>80% coverage)
- [ ] 3.5 Write integration tests

## Phase 4: Documentation & Outputs (Week 7-8)
- [ ] 4.1 Generate publication-ready figures (600 DPI)
- [ ] 4.2 Create LaTeX tables for manuscript
- [ ] 4.3 Write comprehensive documentation
- [ ] 4.4 Create reproducibility README
- [ ] 4.5 Prepare GitHub release with Zenodo DOI

---

# PART 6: VALIDATION REQUIREMENTS

## Before Publication Submission

### Statistical Validation
- [ ] All AUCs have 95% bootstrap CIs
- [ ] Model comparisons use DeLong test with Bonferroni correction
- [ ] Brier scores and calibration slopes reported
- [ ] Cross-validation stability (SD < 0.15) confirmed

### Methodological Validation  
- [ ] No data leakage verified
- [ ] EPV >= 10 confirmed or feature reduction performed
- [ ] Train-test gap < 15% confirmed
- [ ] Nested CV for unbiased performance estimation

### Reproducibility
- [ ] Random seed documented
- [ ] All outputs reproducible with same seed
- [ ] Code version control (GitHub)
- [ ] Dependencies pinned in requirements.txt

### Documentation
- [ ] User guide complete
- [ ] API reference generated
- [ ] Methodology section ready for manuscript
- [ ] CITATION.cff for proper attribution

---

# CRITICAL REMINDERS FOR CURSOR AI

1. **ALWAYS preserve patient-level integrity** - Never split rows, split patients
2. **ALWAYS fit scalers on training data only** - Transform test with train statistics
3. **ALWAYS report confidence intervals** - No point estimates without uncertainty
4. **ALWAYS use nested CV for ML models** - Prevents overfitting bias
5. **ALWAYS check EPV** - Reduce features if EPV < 10
6. **ALWAYS save random seeds** - Reproducibility is non-negotiable
7. **ALWAYS validate outputs** - QA checks should be automated
8. **ALWAYS document methodology** - Peer reviewers will scrutinize

---

*This prompt was generated based on analysis of py_ntcpx v1.1.0 and comparison with the diagnostic document for pipeline transformation. Implementation should follow the priority order specified.*
