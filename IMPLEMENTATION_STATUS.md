# Implementation Status: py_ntcpx v2.0 Transformation

## Overview
This document tracks the implementation of critical fixes and improvements based on `CURSOR_AI_PROMPT_py_ntcpx_v2.md`.

## ✅ Completed Components

### 1. Modular Project Structure
- ✅ Created `src/` directory structure:
  - `src/validation/` - Data splitting and cross-validation
  - `src/models/machine_learning/` - ML models with overfitting prevention
  - `src/models/uncertainty/` - Monte Carlo NTCP uncertainty quantification
  - `src/features/` - Feature selection
  - `src/metrics/` - AUC calculation with confidence intervals
  - `src/reporting/` - QA and leakage detection
- ✅ Created `config/` directory with `pipeline_config.yaml`

### 2. Critical Fixes (Priority 1-3)

#### ✅ Priority 1: Data Leakage Prevention
- **`src/validation/data_splitter.py`**: `PatientDataSplitter` class
  - Patient-level splitting (not row-level)
  - Stratification by outcome and institution
  - Leakage detection and validation
  - Test suite: `tests/test_data_splitter.py`

#### ✅ Priority 2: Overfitting Prevention
- **`src/models/machine_learning/ml_models.py`**: `OverfitResistantMLModels` class
  - Conservative ANN and XGBoost configurations
  - EPV (Events Per Variable) validation
  - Automatic complexity adjustment for small samples
  - Nested cross-validation support

- **`src/features/feature_selector.py`**: `RadiobiologyGuidedFeatureSelector` class
  - Domain-knowledge guided feature selection
  - QUANTEC-based parotid features (Dmean, V30, V45)
  - Statistical filtering (univariate p < 0.1)
  - EPV-based feature capping

#### ✅ Priority 3: Statistical Methodology
- **`src/models/uncertainty/monte_carlo_ntcp.py`**: `MonteCarloNTCPCorrect` class
  - Correct Monte Carlo NTCP with parameter uncertainty
  - Bootstrap uncertainty estimation
  - Proper uncertainty propagation (not simple addition)

- **`src/metrics/auc_calculator.py`**: AUC with confidence intervals
  - Bootstrap method (default, recommended)
  - DeLong's method for comparison
  - `compare_aucs_delong()` for model comparison

- **`src/validation/nested_cv.py`**: `NestedCrossValidation` class
  - Outer loop: Performance estimation
  - Inner loop: Hyperparameter tuning
  - Unbiased performance metrics

### 3. QA and Reporting
- ✅ **`src/reporting/leakage_detector.py`**: `DataLeakageDetector` class
  - Patient overlap detection
  - Feature scaling validation
  - Feature extraction timing checks
  - Comprehensive leakage report generation

### 4. Test Suite
- ✅ **`tests/test_data_splitter.py`**: Unit tests for `PatientDataSplitter`
  - Patient-level splitting validation
  - Leakage detection tests
  - Stratification tests
  - Reproducibility tests
  - Uses synthetic data (follows test data policy)

### 5. Integration Examples
- ✅ **`src/integration_example.py`**: Example code showing how to integrate new components
  - Patient-level splitting integration
  - Feature selection integration
  - Conservative ML model integration

## 🔄 Integration Required

### Update `code3_ntcp_analysis_ml.py`
The existing `code3_ntcp_analysis_ml.py` needs to be updated to use the new components:

**Current Issue** (line ~658):
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=self.random_state, 
    stratify=y if n_events >= 3 else None
)
```

**Should become**:
```python
from src.validation.data_splitter import PatientDataSplitter

splitter = PatientDataSplitter(random_seed=self.random_state, test_size=0.2)
train_df, test_df = splitter.create_splits(
    patient_df,  # Must pass full DataFrame with PrimaryPatientID
    patient_id_col='PrimaryPatientID',
    outcome_col='Observed_Toxicity'
)

# Extract features AFTER split
X_train = train_df[feature_cols].values
X_test = test_df[feature_cols].values
y_train = train_df['Observed_Toxicity'].values
y_test = test_df['Observed_Toxicity'].values

# Check for leakage
from src.reporting.leakage_detector import DataLeakageDetector
leakage_detector = DataLeakageDetector()
leakage_detector.check_patient_overlap(train_df, test_df, 'PrimaryPatientID')
```

### Update ML Model Training
Replace current ANN/XGBoost configs with `OverfitResistantMLModels`:

**Current**: Uses `MLPClassifier` directly with potentially overfitting config
**Should use**: `OverfitResistantMLModels.create_ann_model()` or `.create_xgboost_model()`

### Update AUC Calculation
Replace current AUC calculation with confidence intervals:

**Current**: `auc(fpr, tpr)` (point estimate only)
**Should use**: `calculate_auc_with_ci(y_true, y_pred)` (returns AUC + 95% CI)

### Update QA Reporter (`code4_ntcp_output_QA_reporter.py`)
Add leakage detection to QA checks:

```python
from src.reporting.leakage_detector import DataLeakageDetector

leakage_detector = DataLeakageDetector()
leakage_detector.check_patient_overlap(train_df, test_df)
report = leakage_detector.generate_report()
# Include in QA report output
```

## 📋 Remaining Tasks

### High Priority
- [ ] Update `code3_ntcp_analysis_ml.py` to use `PatientDataSplitter`
- [ ] Replace ML model configs with `OverfitResistantMLModels`
- [ ] Add AUC confidence intervals to all model evaluations
- [ ] Integrate feature selection using `RadiobiologyGuidedFeatureSelector`
- [ ] Add leakage detection to `code4_ntcp_output_QA_reporter.py`

### Medium Priority
- [ ] Fix Monte Carlo NTCP implementation in existing code
- [ ] Update all reports to include confidence intervals
- [ ] Add nested CV to model evaluation pipeline
- [ ] Create comprehensive integration tests

### Low Priority
- [ ] Migrate to full YAML configuration
- [ ] Create publication-ready figure generators
- [ ] Generate LaTeX tables for manuscript
- [ ] Complete documentation

## 🧪 Testing

### Test Data Policy
- ✅ All tests use synthetic data from `tests/test_data/generate_synthetic_data.py`
- ✅ Real patient data is used ONLY for final validation runs
- ✅ Before running tests, execute:
  ```bash
  python tests/test_data/generate_synthetic_data.py
  ```

### Test Execution
```bash
# Run data splitter tests
python -m pytest tests/test_data_splitter.py -v

# Run all tests
python -m pytest tests/ -v
```

## 📝 Usage Examples

See `src/integration_example.py` for detailed examples of:
1. Patient-level data splitting
2. Domain-guided feature selection
3. Conservative ML model training
4. AUC calculation with confidence intervals
5. Leakage detection

## 🔍 Validation Checklist

Before publication submission, verify:
- [x] Patient-level splitting implemented
- [x] EPV validation and warnings
- [x] Conservative ML hyperparameters
- [x] Monte Carlo NTCP correct implementation
- [x] AUC with confidence intervals
- [ ] All tests pass with synthetic data
- [ ] No data leakage detected
- [ ] Train-test AUC gap < 15%
- [ ] CV stability (SD < 0.15)

## 📚 References

- QUANTEC guidelines: Deasy et al. IJROBP 2010
- DeLong et al. Biometrics 1988 (AUC variance)
- Taylor JR. An Introduction to Error Analysis. 2nd Ed.
- Burman et al. IJROBP 1991 (LKB model uncertainty)
