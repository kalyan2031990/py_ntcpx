# Changelog - Version 2.1.0

## Extended Release: py_ntcpx v2.1.0 - Enhanced for Small Datasets

This release extends v2.0.0 with enhanced capabilities for small datasets (n < 100) while maintaining all the methodological rigor, statistical validation, and explainable AI features introduced in v2.0.0.

**Release Date:** 2026-02-03

---

## 📋 All Features from v2.0.0

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

### Outputs
- ✅ 600 DPI figures
- ✅ LaTeX tables for manuscript
- ✅ Statistical reporting with confidence intervals
- ✅ Comprehensive documentation

### Reproducibility
- ✅ Global random seed management
- ✅ YAML configuration management
- ✅ Dependency locking
- ✅ Baseline capture for regression testing

---

## 🆕 New Enhancements in v2.1.0 - Small Dataset Support

### 1. Dynamic CCS Threshold (`ntcp_qa_modules.py`)
- ✅ **Adaptive threshold calculation** based on dataset size
  - n < 30: Percentile-based threshold (95th percentile)
  - n 30-100: Relaxed threshold (0.5)
  - n 100-200: Moderate threshold (0.3)
  - n ≥ 200: Strict threshold (0.2)
- ✅ **Small dataset warnings** instead of DO_NOT_USE flags for n < 100
- ✅ **Backward compatible** with existing workflows

### 2. Clinical Factor Integration (`src/features/feature_selector.py`)
- ✅ **Automatic inclusion** of significant clinical factors (p < 0.05)
- ✅ **Statistical testing** for both continuous (Mann-Whitney U) and categorical (Chi-square) variables
- ✅ **Prioritized feature selection**: Essential features → Clinical factors → Statistical features
- ✅ **EPV-aware** feature capping maintains statistical rigor

### 3. Small Dataset Adaptations (`code3_ntcp_analysis_ml.py`)
- ✅ **Adaptive CV strategy**:
  - n < 30: Leave-One-Out CV
  - n 30-100: StratifiedKFold (3-5 folds)
  - n ≥ 100: Standard 5-fold CV
- ✅ **Model complexity reduction** for small datasets:
  - n < 50: Simplified models (ANN: 8 neurons, XGBoost: 20 trees)
  - n 50-100: Moderate complexity (ANN: 16 neurons, XGBoost: 30 trees)
  - n ≥ 100: Standard complexity
- ✅ **Conservative feature selection** with adjusted EPV rules

### 4. Robust SHAP Analysis (`shap_code7.py`)
- ✅ **Bootstrap SHAP** for stability assessment (automatic for n < 100)
- ✅ **Feature stability metrics** with ranking consistency analysis
- ✅ **Inconsistent feature flagging** between models
- ✅ **Stability reports** saved as Excel files

### 5. Enhanced Reporting (`code4_ntcp_output_QA_reporter.py`)
- ✅ **Small dataset advisory** with dataset-specific warnings
- ✅ **Clinical factor significance** reporting
- ✅ **Enhanced QA report** sections in DOCX output
- ✅ **Statistical considerations** clearly documented

### 6. Bug Fixes
- ✅ **CCS return type handling** in `clinical_safety_guard.py` - now correctly handles dict return from `calculate_ccs()`
- ✅ **Backward compatibility** maintained for float returns

---

## 📊 Test Coverage

### v2.0.0 Baseline
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

### v2.1.0 Additions
- **Total Tests**: 80
- **Passed**: 78 (100% of runnable tests)
- **Failed**: 0
- **Skipped**: 2 (baseline regression - expected)
- **Test Coverage**: All enhancements validated

---

## 📁 Components

### Core Modules (from v2.0.0)
- `src/validation/` - Data splitting, nested CV, leakage audit, calibration
- `src/models/` - Traditional, ML, uncertainty, model cards
- `src/features/` - Feature selection, auto reduction
- `src/metrics/` - AUC with CI, DeLong test
- `src/reporting/` - Statistical reporter, leakage detector
- `src/visualization/` - High-resolution plots (600 DPI)
- `src/safety/` - Clinical safety guard
- `src/utils/` - Logging utilities

### Configuration
- `config/pipeline_config.yaml` - Centralized configuration
- `requirements.txt` - Pinned dependencies

### Tests
- `tests/` - Comprehensive test suite (80 tests in v2.1.0)
- `tests/baseline/` - Baseline capture for regression
- `tests/regression/` - Regression tests

### Scripts
- `scripts/publication_checklist.py` - Quality verification

---

## 🔧 Breaking Changes

### From v2.0.0 (Still Apply)
- **BREAKING**: Train-test split now uses patient-level splitting (not row-level)
- **BREAKING**: Models now refuse to train if EPV < 5 (was warning)
- **BREAKING**: StandardScaler now fits only on training data

### v2.1.0
- **No breaking changes** - all enhancements are backward compatible

---

## 📝 Migration Guide

### For Existing v2.0.0 Users

**No breaking changes** - all enhancements are backward compatible:

1. **Dynamic CCS**: Automatically adapts based on dataset size. No code changes needed.
2. **Clinical Factors**: Pass `clinical_data` parameter to `RadiobiologyGuidedFeatureSelector` to enable. Optional.
3. **Small Dataset Adaptations**: Automatically applied when n < 100. No configuration needed.
4. **SHAP Bootstrap**: Automatically runs for datasets < 100 samples. No changes needed.

### New Parameters (Optional)

```python
# Feature Selector with Clinical Data
selector = RadiobiologyGuidedFeatureSelector(
    organ='Parotid',
    clinical_data=clinical_df,  # NEW: Optional clinical data
    outcome_column='xerostomia_grade2plus'
)

# CCS with Sample Size
ccs_calculator = CohortConsistencyScore(n_samples=len(X_train))  # NEW: Optional
```

### For Users Migrating from v1.x

See v2.0.0 migration guide for breaking changes. All v2.0.0 features are included in v2.1.0.

---

## 🐛 Bug Fixes

### From v2.0.0
- Fixed data leakage in train-test split
- Fixed StandardScaler fitting on test data
- Fixed Monte Carlo NTCP parameter uncertainty propagation
- Fixed AUC calculation without confidence intervals

### v2.1.0
- Fixed CCS return type handling in `clinical_safety_guard.py` - now correctly handles dict return from `calculate_ccs()`
- Maintained backward compatibility for float returns

---

## 📚 Documentation

- Comprehensive README updates
- Reproducibility guide
- API documentation
- Methodology documentation
- Small dataset handling guide (new in v2.1.0)

---

## 🎓 Scientific Improvements

### From v2.0.0
- Complete methodological overhaul to ensure rigorous quality and clinical safety
- Enhanced machine learning capabilities with overfitting prevention
- Statistical validation with proper confidence intervals
- Explainable AI features with SHAP integration

### v2.1.0 Additions
1. **Better handling of small datasets** without sacrificing rigor
2. **Automatic clinical factor integration** when statistically significant
3. **Stability assessment** for feature importance via bootstrap SHAP
4. **Clear warnings** about dataset limitations in reports
5. **Adaptive methodologies** that scale with dataset size

---

## 🔗 References

- Repository: https://github.com/kalyan2031990/py_ntcpx
- Documentation: See README.md and ARCHITECTURE_REPORT.md
- Test Report: test_report.md
- Previous Release: See CHANGELOG_v2.0.0.md

---

## 🙏 Acknowledgments

This release extends v2.0.0's methodological rigor with enhanced capabilities for real-world clinical datasets, which are often smaller than ideal. All changes maintain scientific rigor while providing appropriate warnings and adaptations.

---

**Release Date**: February 3, 2026  
**Version**: 2.1.0  
**Based on**: v2.0.0 (February 3, 2026)

**Next Steps**: Continue collecting data to reach recommended sample sizes (≥150 patients for parotid NTCP) for more reliable model performance.
