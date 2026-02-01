# Changelog - Version 2.0.0

## 🎉 Major Release: Publication-Ready py_ntcpx v2.0.0

This release transforms py_ntcpx from an experimental research tool into a publication-ready, methodologically rigorous software package.

---

## 🚀 Major Features

### Data Integrity & Leakage Prevention
- ✅ Patient-level data splitting with stratification
- ✅ LeakageAudit utility for automated leakage detection
- ✅ StandardScaler fit only on training data
- ✅ Split-before-transform enforcement

### Overfitting Prevention & Model Containment
- ✅ EPV (Events Per Variable) enforcement (refuses to train if EPV < 5)
- ✅ Auto feature reduction when EPV < 5
- ✅ Conservative ML architectures (ANN, XGBoost)
- ✅ Dynamic model complexity adjustment
- ✅ Domain-guided feature selection

### Statistical Rigor
- ✅ Correct Monte Carlo NTCP with parameter uncertainty
- ✅ Bootstrap confidence intervals for all metrics
- ✅ DeLong test for AUC comparison with Bonferroni correction
- ✅ Nested cross-validation for unbiased performance estimation
- ✅ Calibration correction (Platt scaling, isotonic regression)

### Clinical Safety Layer
- ✅ ClinicalSafetyGuard with underprediction risk detection
- ✅ Cohort Consistency Score (CCS) integration
- ✅ DO_NOT_USE flags for unsafe predictions
- ✅ Automated safety reports

### Model Documentation
- ✅ Auto-generated model cards
- ✅ EXPLORATORY labels for ML models
- ✅ Intended use, limitations, and failure modes documented

### Publication-Ready Outputs
- ✅ 600 DPI publication-ready figures
- ✅ LaTeX tables for manuscript
- ✅ Statistical reporting with confidence intervals
- ✅ Comprehensive documentation

### Reproducibility
- ✅ Global random seed management
- ✅ YAML configuration management
- ✅ Dependency locking
- ✅ Baseline capture for regression testing

---

## 📊 Test Coverage

- **49 unit and integration tests** - All passing ✅
- Test coverage includes:
  - Data splitting and leakage prevention
  - ML model training and validation
  - Feature selection
  - AUC calculation with CI
  - Clinical safety checks
  - DVH validation
  - NTCP mathematics
  - Calibration correction
  - Auto feature reduction

---

## 📁 New Components

### Core Modules
- `src/validation/` - Data splitting, nested CV, leakage audit, calibration
- `src/models/` - Traditional, ML, uncertainty, model cards
- `src/features/` - Feature selection, auto reduction
- `src/metrics/` - AUC with CI, DeLong test
- `src/reporting/` - Statistical reporter, leakage detector
- `src/visualization/` - Publication plots (600 DPI)
- `src/safety/` - Clinical safety guard
- `src/utils/` - Logging utilities

### Configuration
- `config/pipeline_config.yaml` - Centralized configuration
- `requirements.txt` - Pinned dependencies

### Tests
- `tests/` - Comprehensive test suite (49 tests)
- `tests/baseline/` - Baseline capture for regression
- `tests/regression/` - Regression tests

### Scripts
- `scripts/publication_checklist.py` - Publication readiness verification

---

## 🔧 Breaking Changes

### Data Splitting
- **BREAKING**: Train-test split now uses patient-level splitting (not row-level)
- All patients with multiple organs stay together in same split

### EPV Enforcement
- **BREAKING**: Models now refuse to train if EPV < 5 (was warning)
- Use `AutoFeatureReducer` to automatically reduce features

### StandardScaler
- **BREAKING**: StandardScaler now fits only on training data
- Test data transformed using training statistics

---

## 📝 Migration Guide

### For Existing Users

1. **Update data splitting**:
   ```python
   from src.validation.data_splitter import PatientDataSplitter
   splitter = PatientDataSplitter(test_size=0.3, random_seed=42)
   train_idx, test_idx = splitter.split(X, y, patient_ids)
   ```

2. **Check EPV before training**:
   ```python
   from src.features.auto_feature_reducer import AutoFeatureReducer
   reducer = AutoFeatureReducer(min_epv=5.0)
   X_reduced, features, epv = reducer.reduce_features(X, y, organ='Parotid')
   ```

3. **Use new ML models**:
   ```python
   from src.models.machine_learning.ml_models import OverfitResistantMLModels
   ml_model = OverfitResistantMLModels(n_features, n_samples, n_events, random_seed)
   ```

---

## 🐛 Bug Fixes

- Fixed data leakage in train-test split
- Fixed StandardScaler fitting on test data
- Fixed Monte Carlo NTCP parameter uncertainty propagation
- Fixed AUC calculation without confidence intervals

---

## 📚 Documentation

- Comprehensive README updates
- Reproducibility guide
- API documentation
- Methodology documentation

---

## 🙏 Acknowledgments

This release represents a complete methodological overhaul to ensure publication-ready quality and clinical safety.

---

**Release Date**: 2024
**Version**: 2.0.0
**Status**: Production Ready ✅
