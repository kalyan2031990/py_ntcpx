# STEP 1: ML Code Comparison - v1.2.1 vs v2.0.0

## Section 1: Overall Statistics

### File Changes
- **File**: `code3_ntcp_analysis_ml.py`
- **Total lines changed**: 198
- **Lines added**: 174
- **Lines removed**: 24
- **Net change**: +150 lines

### Major Functions Modified
1. `train_and_evaluate_ml_models()` - Complete refactoring with v2.0 components
   - v1.2.1: Line 636
   - v2.0.0: Line 649
   - **Change**: Added patient-level splitting, feature selection, overfit-resistant models, AUC CI calculation

2. `train_ann_model()` - No changes to function signature, but v2.0 uses OverfitResistantMLModels wrapper
   - v1.2.1: Line 584
   - v2.0.0: Line 597

3. `train_xgboost_model()` - No changes to function signature, but v2.0 uses OverfitResistantMLModels wrapper
   - v1.2.1: Line 610
   - v2.0.0: Line 623

### New Imports in v2.0.0
```python
# Lines 9-20 in v2.0.0
from src.validation.data_splitter import PatientDataSplitter
from src.models.machine_learning.ml_models import OverfitResistantMLModels
from src.features.feature_selector import RadiobiologyGuidedFeatureSelector
from src.metrics.auc_calculator import calculate_auc_with_ci
from src.reporting.leakage_detector import DataLeakageDetector
```

---

## Section 2: Key Differences Found

### Difference 1: Data Splitting Strategy
**Category**: DATA_PROCESSING

**v1.2.1 (Lines 641-661)**:
```python
# Prepare features
X, y, feature_cols = self.prepare_features(organ_data)

if X is None:
    print(f"    Warning: Insufficient data for ML models")
    return {}

n_events = y.sum()
n_samples = len(y)

print(f"     Features: {len(feature_cols)}, Samples: {n_samples}, Events: {int(n_events)}")

if n_events < 5 or n_samples < 20:
    print(f"    Warning: Insufficient events/samples for reliable ML training")
    return {}

# Use stratified train-test split to prevent data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=self.random_state, 
    stratify=y if n_events >= 3 else None
)
```

**v2.0.0 (Lines 654-726)**:
```python
# Check if v2.0 components are available
use_v2_components = V2_COMPONENTS_AVAILABLE and 'PrimaryPatientID' in organ_data.columns

if use_v2_components:
    # V2.0: Patient-level splitting to prevent data leakage
    print(f"     Using v2.0 patient-level splitting...")
    
    # PATIENT-LEVEL SPLITTING (v2.0 - prevents data leakage)
    splitter = PatientDataSplitter(random_seed=self.random_state, test_size=0.2)
    train_df, test_df = splitter.create_splits(
        organ_data,
        patient_id_col='PrimaryPatientID',
        outcome_col='Observed_Toxicity'
    )
    
    # Check for leakage
    leakage_detector = DataLeakageDetector()
    leakage_check = leakage_detector.check_patient_overlap(
        train_df, test_df, 'PrimaryPatientID'
    )
    if not leakage_check:
        leakage_report = leakage_detector.generate_report()
        print(f"     WARNING: {leakage_report['errors']}")
    
    # Extract features AFTER split (prevents leakage)
    X_train_df, y_train_series, feature_cols = self.prepare_features(train_df)
    X_test_df, y_test_series, _ = self.prepare_features(test_df)
    
    # Convert to numpy arrays
    X_train = X_train_df.values
    X_test = X_test_df.values
    y_train = y_train_series.values
    y_test = y_test_series.values
    
    n_events = y_train.sum()
    n_samples = len(y_train)
    
    print(f"     Features: {len(feature_cols)}, Train Samples: {len(y_train)}, Events: {int(n_events)}")
    print(f"     Test Samples: {len(y_test)}, Test Events: {int(y_test.sum())}")

else:
    # FALLBACK: Row-level splitting (original method)
    print(f"     Using row-level splitting (v2.0 components not available)...")
    
    # Prepare features
    X, y, feature_cols = self.prepare_features(organ_data)
    
    if X is None:
        print(f"    Warning: Insufficient data for ML models")
        return {}
    
    n_events = y.sum()
    n_samples = len(y)
    
    print(f"     Features: {len(feature_cols)}, Samples: {n_samples}, Events: {int(n_events)}")
    
    # Use stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X.values if hasattr(X, 'values') else X,
        y.values if hasattr(y, 'values') else y,
        test_size=0.3, random_state=self.random_state, 
        stratify=y if n_events >= 3 else None
    )
```

**Key Changes**:
- v1.2.1: Row-level splitting (test_size=0.3)
- v2.0.0: Patient-level splitting (test_size=0.2) with leakage detection, fallback to row-level if components unavailable

---

### Difference 2: Feature Selection
**Category**: FEATURE_SELECTION

**v1.2.1**: 
- **No feature selection implemented**
- Uses all features from `prepare_features()`

**v2.0.0 (Lines 734-749)**:
```python
# V2.0: Feature selection before training (if available)
if use_v2_components and len(feature_cols) > 3:
    try:
        selector = RadiobiologyGuidedFeatureSelector()
        X_train_df = pd.DataFrame(X_train, columns=feature_cols)
        selected_features = selector.select_features(X_train_df, y_train, organ=organ)
        
        if len(selected_features) < len(feature_cols):
            # Use only selected features
            selected_indices = [i for i, f in enumerate(feature_cols) if f in selected_features]
            X_train = X_train[:, selected_indices]
            X_test = X_test[:, selected_indices]
            feature_cols = selected_features
            print(f"     Selected {len(selected_features)} features: {selected_features[:5]}...")
    except Exception as e:
        print(f"     Warning: Feature selection failed: {e}, using all features")
```

**Key Changes**:
- v1.2.1: No feature selection
- v2.0.0: Uses `RadiobiologyGuidedFeatureSelector` which:
  - Starts with literature-validated features (e.g., Dmean, V30, V45 for Parotid)
  - Adds features based on statistical filtering (univariate p < 0.1)
  - Caps features based on EPV rule (max_features = n_events / 10, minimum 3)

---

### Difference 3: Model Training - ANN
**Category**: MODEL_TRAINING

**v1.2.1 (Lines 665-667)**:
```python
# Train ANN
print(f"     Training ANN...")
ann_model = self.train_ann_model(X_train, y_train, organ)
```

**v2.0.0 (Lines 751-769)**:
```python
# Train ANN
print(f"     Training ANN...")

# V2.0: Use OverfitResistantMLModels if available
if use_v2_components:
    try:
        ml_model = OverfitResistantMLModels(
            n_features=X_train.shape[1],
            n_samples=len(X_train),
            n_events=int(np.sum(y_train)),
            random_seed=self.random_state
        )
        ann_model = ml_model.create_ann_model()
        print(f"       EPV: {ml_model.epv:.2f} events per variable")
    except Exception as e:
        print(f"     Warning: OverfitResistantMLModels failed: {e}, using basic model")
        ann_model = self.train_ann_model(X_train, y_train, organ)
else:
    ann_model = self.train_ann_model(X_train, y_train, organ)
```

**Key Changes**:
- v1.2.1: Direct call to `train_ann_model()`
- v2.0.0: Uses `OverfitResistantMLModels` wrapper which:
  - Calculates EPV (Events Per Variable)
  - Adjusts model complexity based on sample size
  - Uses more conservative hyperparameters
  - Falls back to basic model if wrapper fails

---

### Difference 4: Model Training - XGBoost
**Category**: MODEL_TRAINING

**v1.2.1 (Lines 700-702)**:
```python
# Train XGBoost
if XGBOOST_AVAILABLE:
    print(f"     Training XGBoost...")
    xgb_model = self.train_xgboost_model(X_train, y_train, organ)
```

**v2.0.0 (Lines 817-836)**:
```python
# Train XGBoost
if XGBOOST_AVAILABLE:
    print(f"     Training XGBoost...")
    
    # V2.0: Use OverfitResistantMLModels if available
    if use_v2_components:
        try:
            if 'ml_model' not in locals():
                ml_model = OverfitResistantMLModels(
                    n_features=X_train.shape[1],
                    n_samples=len(X_train),
                    n_events=int(np.sum(y_train)),
                    random_seed=self.random_state
                )
            xgb_model = ml_model.create_xgboost_model()
        except Exception as e:
            print(f"     Warning: OverfitResistantMLModels XGBoost failed: {e}, using basic model")
            xgb_model = self.train_xgboost_model(X_train, y_train, organ)
    else:
        xgb_model = self.train_xgboost_model(X_train, y_train, organ)
```

**Key Changes**:
- v1.2.1: Direct call to `train_xgboost_model()`
- v2.0.0: Uses `OverfitResistantMLModels` wrapper (reuses ml_model instance if already created for ANN)

---

### Difference 5: AUC Calculation
**Category**: MODEL_TRAINING

**v1.2.1 (Lines 675-676)**:
```python
fpr, tpr, _ = roc_curve(y_test, y_pred_ann)
auc_ann = auc(fpr, tpr)
```

**v2.0.0 (Lines 777-790)**:
```python
fpr, tpr, _ = roc_curve(y_test, y_pred_ann)

# V2.0: AUC with confidence intervals
if use_v2_components:
    try:
        auc_ann, auc_ci = calculate_auc_with_ci(y_test, y_pred_ann)
        print(f"       ANN - Test AUC: {auc_ann:.3f} (95% CI: {auc_ci[0]:.3f}-{auc_ci[1]:.3f})")
    except Exception as e:
        print(f"     Warning: AUC CI calculation failed: {e}")
        auc_ann = auc(fpr, tpr)
        auc_ci = (auc_ann, auc_ann)
else:
    auc_ann = auc(fpr, tpr)
    auc_ci = (auc_ann, auc_ann)
```

**Key Changes**:
- v1.2.1: Simple AUC calculation
- v2.0.0: AUC with 95% confidence intervals using `calculate_auc_with_ci()`

---

### Difference 6: Results Dictionary
**Category**: OTHER

**v1.2.1 (Lines 683-692)**:
```python
results['ANN'] = {
    'model': ann_model,
    'test_AUC': auc_ann,
    'test_Brier': brier_ann,
    'cv_AUC_mean': np.mean(cv_scores),
    'cv_AUC_std': np.std(cv_scores),
    'n_train': len(X_train),
    'n_test': len(X_test),
    'feature_names': feature_cols
}
```

**v2.0.0 (Lines 798-809)**:
```python
results['ANN'] = {
    'model': ann_model,
    'test_AUC': auc_ann,
    'test_AUC_CI': auc_ci if use_v2_components else None,
    'test_Brier': brier_ann,
    'cv_AUC_mean': np.mean(cv_scores),
    'cv_AUC_std': np.std(cv_scores),
    'n_train': len(X_train),
    'n_test': len(X_test),
    'feature_names': feature_cols,
    'epv': ml_model.epv if use_v2_components and 'ml_model' in locals() else None
}
```

**Key Changes**:
- v1.2.1: Basic metrics
- v2.0.0: Added `test_AUC_CI` and `epv` fields

---

## Section 3: Feature Selection Comparison

### v1.2.1: Feature Selection
**Status**: **NOT IMPLEMENTED**

- No feature selection code present
- All features from `prepare_features()` are used directly
- No filtering, ranking, or selection mechanism

**Code Location**: N/A

---

### v2.0.0: Feature Selection
**Status**: **IMPLEMENTED** (Conditional on v2.0 components)

**Implementation**: `RadiobiologyGuidedFeatureSelector`

**Code Location**: Lines 734-749 in `code3_ntcp_analysis_ml.py`

**Exact Code Snippet**:
```python
# V2.0: Feature selection before training (if available)
if use_v2_components and len(feature_cols) > 3:
    try:
        selector = RadiobiologyGuidedFeatureSelector()
        X_train_df = pd.DataFrame(X_train, columns=feature_cols)
        selected_features = selector.select_features(X_train_df, y_train, organ=organ)
        
        if len(selected_features) < len(feature_cols):
            # Use only selected features
            selected_indices = [i for i, f in enumerate(feature_cols) if f in selected_features]
            X_train = X_train[:, selected_indices]
            X_test = X_test[:, selected_indices]
            feature_cols = selected_features
            print(f"     Selected {len(selected_features)} features: {selected_features[:5]}...")
    except Exception as e:
        print(f"     Warning: Feature selection failed: {e}, using all features")
```

**Feature Selection Strategy** (from `src/features/feature_selector.py`):
1. **Domain Knowledge**: Starts with literature-validated features
   - Parotid: `['Dmean', 'V30', 'V45']` (essential), `['D50', 'V15', 'V20', 'gEUD']` (exploratory)
   - Clinical: `['Chemotherapy', 'T_Stage', 'Diabetes']` (exploratory)
2. **Statistical Filtering**: Adds features with univariate p < 0.1 (Mann-Whitney U test)
3. **EPV Rule**: Caps features at `max_features = max(int(n_events / 10), 3)`
4. **Organ-Specific**: Different essential features for different organs

---

## Section 4: Cross-Validation Comparison

### v1.2.1: Cross-Validation Implementation

**Code Location**: Lines 680-681 (ANN), Lines 714-715 (XGBoost)

**ANN Cross-Validation**:
```python
cv_scores = cross_val_score(ann_model, X_train, y_train, 
                           cv=min(5, len(X_train)//3), scoring='roc_auc')
```

**XGBoost Cross-Validation**:
```python
cv_scores = cross_val_score(xgb_model, X_train, y_train, 
                           cv=min(5, len(X_train)//3), scoring='roc_auc')
```

**Parameters**:
- **CV Strategy**: `cross_val_score` (default: StratifiedKFold if available)
- **Number of Folds**: `min(5, len(X_train)//3)` - adaptive based on training set size
- **Scoring Metric**: `'roc_auc'`
- **Data**: Uses `X_train, y_train` (training set only)

**Import Statement** (Line 31):
```python
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
```

---

### v2.0.0: Cross-Validation Implementation

**Code Location**: Lines 795-796 (ANN), Lines 861-862 (XGBoost)

**ANN Cross-Validation**:
```python
cv_scores = cross_val_score(ann_model, X_train, y_train, 
                           cv=min(5, len(X_train)//3), scoring='roc_auc')
```

**XGBoost Cross-Validation**:
```python
cv_scores = cross_val_score(xgb_model, X_train, y_train, 
                           cv=min(5, len(X_train)//3), scoring='roc_auc')
```

**Parameters**:
- **CV Strategy**: `cross_val_score` (default: StratifiedKFold if available)
- **Number of Folds**: `min(5, len(X_train)//3)` - adaptive based on training set size
- **Scoring Metric**: `'roc_auc'`
- **Data**: Uses `X_train, y_train` (training set only)

**Import Statement** (Line 31):
```python
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
```

**Key Finding**: **NO CHANGES** in cross-validation implementation between v1.2.1 and v2.0.0. The CV code is identical.

---

## Section 5: XGBoost Configuration

### v1.2.1: XGBoost Parameters

**Code Location**: Lines 618-628 in `code3_ntcp_analysis_ml_v1.2.1.py`

**Exact Code Snippet**:
```python
xgb_model = xgb.XGBClassifier(
    n_estimators=50,  # Small number of trees
    max_depth=3,      # Shallow trees
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,    # L1 regularization
    reg_lambda=1.0,   # L2 regularization
    random_state=self.random_state,
    eval_metric='logloss'
)
```

**Parameter Table**:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_estimators` | 50 | Number of boosting rounds |
| `max_depth` | 3 | Maximum tree depth |
| `learning_rate` | 0.1 | Step size shrinkage |
| `subsample` | 0.8 | Row subsampling ratio |
| `colsample_bytree` | 0.8 | Column subsampling ratio |
| `reg_alpha` | 0.1 | L1 regularization term |
| `reg_lambda` | 1.0 | L2 regularization term |
| `random_state` | `self.random_state` | Random seed |
| `eval_metric` | 'logloss' | Evaluation metric |

**Missing Parameters** (defaults used):
- `min_child_weight`: Default (1.0)
- `gamma`: Default (0.0)
- `scale_pos_weight`: Default (1.0)

---

### v2.0.0: XGBoost Parameters

**Two Configurations**:

#### Configuration A: Basic Model (Fallback)
**Code Location**: Lines 631-641 in `code3_ntcp_analysis_ml_v2.0.0.py`

**Exact Code Snippet** (same as v1.2.1):
```python
xgb_model = xgb.XGBClassifier(
    n_estimators=50,  # Small number of trees
    max_depth=3,      # Shallow trees
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,    # L1 regularization
    reg_lambda=1.0,   # L2 regularization
    random_state=self.random_state,
    eval_metric='logloss'
)
```

#### Configuration B: OverfitResistantMLModels (Primary)
**Code Location**: `src/models/machine_learning/ml_models.py` Lines 49-61

**Exact Code Snippet**:
```python
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
```

**Parameter Table - OverfitResistantMLModels**:

| Parameter | v1.2.1 | v2.0.0 (Basic) | v2.0.0 (OverfitResistant) | Change |
|-----------|--------|----------------|---------------------------|--------|
| `n_estimators` | 50 | 50 | 50 | No change |
| `max_depth` | 3 | 3 | **2** | **Reduced** |
| `learning_rate` | 0.1 | 0.1 | **0.05** | **Reduced** |
| `subsample` | 0.8 | 0.8 | **0.7** | **Reduced** |
| `colsample_bytree` | 0.8 | 0.8 | **0.7** | **Reduced** |
| `reg_alpha` | 0.1 | 0.1 | **0.5** | **Increased** |
| `reg_lambda` | 1.0 | 1.0 | **2.0** | **Increased** |
| `min_child_weight` | 1.0 (default) | 1.0 (default) | **3** | **Added** |
| `gamma` | 0.0 (default) | 0.0 (default) | **0.1** | **Added** |
| `random_state` | `self.random_state` | `self.random_state` | 42 | Changed |
| `eval_metric` | 'logloss' | 'logloss' | 'logloss' | No change |

**Dynamic Adjustments** (from `OverfitResistantMLModels._adjust_model_complexity()`):
- If `n_samples < 50`: `max_depth = 1`, `n_estimators = 30`
- If `n_samples < 100`: `max_depth = 2`

---

## Section 6: ANN Configuration

### v1.2.1: ANN Parameters

**Code Location**: Lines 590-600 in `code3_ntcp_analysis_ml_v1.2.1.py`

**Exact Code Snippet**:
```python
ann_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('ann', MLPClassifier(
        hidden_layer_sizes=(20, 10),  # Conservative architecture
        activation='relu',
        solver='lbfgs',  # Good for small datasets
        alpha=0.01,  # L2 regularization
        max_iter=1000,
        random_state=self.random_state,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=20
    ))
])
```

**Parameter Table**:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `hidden_layer_sizes` | (20, 10) | Two hidden layers with 20 and 10 neurons |
| `activation` | 'relu' | ReLU activation function |
| `solver` | 'lbfgs' | Limited-memory BFGS optimizer |
| `alpha` | 0.01 | L2 regularization strength |
| `max_iter` | 1000 | Maximum iterations |
| `random_state` | `self.random_state` | Random seed |
| `early_stopping` | True | Enable early stopping |
| `validation_fraction` | 0.2 | Fraction of data for validation |
| `n_iter_no_change` | 20 | Iterations without improvement before stopping |

**Missing Parameters** (defaults used):
- `learning_rate`: Default ('constant')
- `learning_rate_init`: Default (0.001)
- `batch_size`: Default ('auto')
- `beta_1`, `beta_2`: Default (for adam/sgd)

---

### v2.0.0: ANN Parameters

**Two Configurations**:

#### Configuration A: Basic Model (Fallback)
**Code Location**: Lines 603-613 in `code3_ntcp_analysis_ml_v2.0.0.py`

**Exact Code Snippet** (same as v1.2.1):
```python
ann_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('ann', MLPClassifier(
        hidden_layer_sizes=(20, 10),  # Conservative architecture
        activation='relu',
        solver='lbfgs',  # Good for small datasets
        alpha=0.01,  # L2 regularization
        max_iter=1000,
        random_state=self.random_state,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=20
    ))
])
```

#### Configuration B: OverfitResistantMLModels (Primary)
**Code Location**: `src/models/machine_learning/ml_models.py` Lines 34-46

**Exact Code Snippet**:
```python
ANN_CONFIG = {
    'hidden_layer_sizes': (16, 8),  # Reduced from (64, 32, 16)
    'activation': 'relu',
    'solver': 'adam',  # Changed from 'lbfgs'
    'alpha': 0.01,  # Strong L2 regularization (increased from 0.001)
    'learning_rate': 'adaptive',
    'learning_rate_init': 0.001,
    'max_iter': 500,  # Reduced from 1000
    'early_stopping': True,
    'validation_fraction': 0.2,
    'n_iter_no_change': 20,  # Reduced from 50
    'random_state': 42
}
```

**Parameter Table - OverfitResistantMLModels**:

| Parameter | v1.2.1 | v2.0.0 (Basic) | v2.0.0 (OverfitResistant) | Change |
|-----------|--------|----------------|---------------------------|--------|
| `hidden_layer_sizes` | (20, 10) | (20, 10) | **(16, 8)** | **Reduced** |
| `activation` | 'relu' | 'relu' | 'relu' | No change |
| `solver` | 'lbfgs' | 'lbfgs' | **'adam'** | **Changed** |
| `alpha` | 0.01 | 0.01 | 0.01 | No change |
| `learning_rate` | 'constant' (default) | 'constant' (default) | **'adaptive'** | **Added** |
| `learning_rate_init` | 0.001 (default) | 0.001 (default) | **0.001** | **Explicit** |
| `max_iter` | 1000 | 1000 | **500** | **Reduced** |
| `random_state` | `self.random_state` | `self.random_state` | 42 | Changed |
| `early_stopping` | True | True | True | No change |
| `validation_fraction` | 0.2 | 0.2 | 0.2 | No change |
| `n_iter_no_change` | 20 | 20 | 20 | No change |

**Dynamic Adjustments** (from `OverfitResistantMLModels._adjust_model_complexity()`):
- If `n_samples < 50`: `hidden_layer_sizes = (8,)` (single layer)
- If `n_samples < 100`: `hidden_layer_sizes = (16,)` (single layer)

**Pipeline Structure**: Both versions use `Pipeline([('scaler', StandardScaler()), ('ann', MLPClassifier(...))])`

---

## Summary of Key Findings

### Major Architectural Changes
1. **Data Splitting**: Row-level → Patient-level (prevents data leakage)
2. **Feature Selection**: None → RadiobiologyGuidedFeatureSelector (domain knowledge + statistical filtering)
3. **Model Training**: Direct → OverfitResistantMLModels wrapper (EPV-aware, complexity adjustment)
4. **AUC Calculation**: Simple → With confidence intervals
5. **Results Storage**: Basic metrics → Includes EPV and AUC CI

### Hyperparameter Changes (OverfitResistantMLModels vs Basic)
- **XGBoost**: More conservative (deeper regularization, lower learning rate, shallower trees)
- **ANN**: Smaller architecture, different solver (adam vs lbfgs), adaptive learning rate

### Unchanged Components
- Cross-validation implementation (identical)
- Basic model fallback parameters (identical)
- Core training logic structure

---

**Document Generated**: Comparison between v1.2.1 and v2.0.0 ML training code
**Files Analyzed**: 
- `code3_ntcp_analysis_ml_v1.2.1.py` (v1.2.1)
- `code3_ntcp_analysis_ml_v2.0.0.py` (v2.0.0)
- `src/models/machine_learning/ml_models.py` (v2.0.0 OverfitResistantMLModels)
- `src/features/feature_selector.py` (v2.0.0 RadiobiologyGuidedFeatureSelector)

