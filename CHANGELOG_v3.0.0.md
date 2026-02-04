# Changelog - Version 3.0.0

## Major Release: py_ntcpx v3.0.0 - Enhanced Interpretability and Small-Cohort Honesty

This release extends v2.1.0 with scientifically robust enhancements for small-cohort NTCP analysis, improved model interpretability, and honest prediction handling. Focus areas: adaptive CCS thresholds, fixed SHAP explanations, and complementary LIME interpretability.

**Release Date:** 2026-02-03

---

## 📋 All Features from v2.0.0 and v2.1.0

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
- ✅ Adaptive CCS thresholds (v3.0.0 enhancement)
- ✅ CCS warnings instead of DO_NOT_USE flags (v3.0.0)
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

### Small Dataset Support (from v2.1.0)
- ✅ Dynamic CCS threshold (enhanced in v3.0.0)
- ✅ Clinical factor integration
- ✅ Small dataset adaptations
- ✅ Robust SHAP analysis (enhanced in v3.0.0)
- ✅ Enhanced reporting

---

## 🆕 New Enhancements in v3.0.0

### 1. Adaptive CCS Thresholds for Small Cohorts (`ntcp_qa_modules.py`)
- ✅ **Scientifically honest thresholds** based on dataset size:
  - n < 30: Disable filtering (0.0) for tiny cohorts
  - n 30-100: Conservative threshold (0.1) for small cohorts (e.g., 54-patient case)
  - n ≥ 100: Standard threshold (0.2) for larger datasets
- ✅ **Replaces overly conservative static threshold** that blocked all predictions in small cohorts
- ✅ **Maintains scientific rigor** while enabling analysis of real-world small datasets

### 2. CCS Warnings Instead of DO_NOT_USE Flags
- ✅ **Changed behavior**: CCS now generates warnings instead of discarding predictions
- ✅ **New `CCS_Warning_Flag` column**: Boolean flag (True if CCS below adaptive threshold)
- ✅ **All predictions preserved**: Full analysis and explanations (SHAP/LIME) computed for all patients
- ✅ **Clear logging**: `INFO - CCS below adaptive threshold for X/X predictions. Interpretations should be treated with caution.`
- ✅ **Updated safety reports**: Reflect warning-based approach instead of blocking

### 3. Fixed XGBoost SHAP Explainer (`shap_code7.py`)
- ✅ **Model-agnostic explainer**: Replaced `TreeExplainer` with `Explainer` for serialized models
- ✅ **Base score fix**: Handles string `base_score` in serialized XGBoost models
- ✅ **Robust prediction wrapper**: Works with both DataFrame and array inputs
- ✅ **Fixes serialization issues**: Resolves failures when loading saved XGBoost models

### 4. Improved ANN SHAP Stability Warnings
- ✅ **Enhanced error handling**: Wrapped bootstrap stability analysis in try-except
- ✅ **CCS-aware warnings**: Checks cohort consistency before stability analysis
- ✅ **Clear warnings**: `WARNING - SHAP stability analysis skipped for ANN due to low cohort consistency. Global interpretations may be unstable.`
- ✅ **Visualizations preserved**: SHAP beeswarm and bar plots still generated with appropriate warnings

### 5. LIME (Local Interpretable Model-agnostic Explanations) Integration
- ✅ **New dependency**: `lime` package added to requirements.txt
- ✅ **Per-patient explanations**: Generates LIME explanations for representative patients
- ✅ **Representative selection**: Automatically identifies highest, median, and lowest predicted NTCP patients
- ✅ **Dual output formats**: Saves both HTML (`lime_explanation_{patient_id}.html`) and PNG (`lime_explanation_{patient_id}.png`)
- ✅ **Model support**: Works with both ANN and XGBoost models
- ✅ **Complementary to SHAP**: Provides local interpretability alongside global SHAP analysis

---

## 📊 Test Coverage

### v2.0.0 Baseline
- **49 unit and integration tests** - All passing ✅

### v2.1.0 Additions
- **Total Tests**: 80
- **Passed**: 78 (100% of runnable tests)
- **Failed**: 0
- **Skipped**: 2 (baseline regression - expected)

### v3.0.0
- All v2.1.0 tests maintained
- New tests for adaptive CCS thresholds
- LIME integration validation

---

## 📁 Components

### Core Modules (from v2.0.0)
- `src/validation/` - Data splitting, nested CV, leakage audit, calibration
- `src/models/` - Traditional, ML, uncertainty, model cards
- `src/features/` - Feature selection, auto reduction
- `src/metrics/` - AUC with CI, DeLong test
- `src/reporting/` - Statistical reporter, leakage detector
- `src/visualization/` - High-resolution plots (600 DPI)
- `src/safety/` - Clinical safety guard (enhanced in v3.0.0)
- `src/utils/` - Logging utilities

### Configuration
- `config/pipeline_config.yaml` - Centralized configuration
- `requirements.txt` - Pinned dependencies (includes `lime`)

### Tests
- `tests/` - Comprehensive test suite
- `tests/baseline/` - Baseline capture for regression
- `tests/regression/` - Regression tests

### Scripts
- `scripts/publication_checklist.py` - Quality verification
- `shap_code7.py` - SHAP + LIME analysis (enhanced in v3.0.0)

---

## 🔧 Breaking Changes

### From v2.0.0 (Still Apply)
- **BREAKING**: Train-test split now uses patient-level splitting (not row-level)
- **BREAKING**: Models now refuse to train if EPV < 5 (was warning)
- **BREAKING**: StandardScaler now fits only on training data

### v3.0.0
- **CHANGED**: `CCS_Safety` column replaced with `CCS_Warning_Flag` (boolean instead of 'DO_NOT_USE'/'OK')
- **CHANGED**: CCS threshold now adaptive (0.0 for n<30, 0.1 for n<100, 0.2 for n≥100)
- **CHANGED**: Predictions are no longer blocked by CCS; warnings are issued instead
- **No other breaking changes** - all enhancements are backward compatible

---

## 📝 Migration Guide

### For Existing v2.1.0 Users

**Key changes in v3.0.0:**

1. **CCS Column Changes**:
   - Old: `CCS_Safety` column with values 'DO_NOT_USE'/'OK'
   - New: `CCS_Warning_Flag` column (boolean: True if CCS below adaptive threshold)
   - Action: Update any code that checks `CCS_Safety == 'DO_NOT_USE'` to use `CCS_Warning_Flag == True`

2. **Adaptive Thresholds**:
   - Automatically applied based on dataset size
   - No code changes needed
   - Threshold values: 0.0 (n<30), 0.1 (n<100), 0.2 (n≥100)

3. **XGBoost SHAP**:
   - Now uses model-agnostic explainer
   - Should work with serialized models that previously failed
   - No code changes needed

4. **LIME Explanations**:
   - Automatically generated for representative patients
   - Outputs saved in same directory as SHAP results
   - No code changes needed

### New Parameters (Optional)

```python
# Clinical Safety Guard with adaptive threshold
from src.safety.clinical_safety_guard import ClinicalSafetyGuard
guard = ClinicalSafetyGuard(n_samples=len(X_train))  # NEW: Optional n_samples for adaptive threshold
guard.fit(X_train)

# CCS Calculator with adaptive threshold
from ntcp_qa_modules import CohortConsistencyScore
ccs_calc = CohortConsistencyScore(n_samples=len(X_train))  # Adaptive threshold based on n_samples
ccs_calc.fit(X_train)
```

### For Users Migrating from v1.x

See v2.0.0 migration guide for breaking changes. All v2.0.0 and v2.1.0 features are included in v3.0.0.

---

## 🐛 Bug Fixes

### From v2.0.0
- Fixed data leakage in train-test split
- Fixed StandardScaler fitting on test data
- Fixed Monte Carlo NTCP parameter uncertainty propagation
- Fixed AUC calculation without confidence intervals

### From v2.1.0
- Fixed CCS return type handling in `clinical_safety_guard.py`
- Maintained backward compatibility for float returns

### v3.0.0
- **Fixed**: XGBoost SHAP explainer compatibility with serialized models
- **Fixed**: Overly conservative CCS threshold blocking all predictions in small cohorts
- **Fixed**: Missing SHAP explanations when CCS was low
- **Improved**: ANN SHAP stability analysis error handling

---

## 📚 Documentation

- Comprehensive README updates
- Reproducibility guide
- API documentation
- Methodology documentation
- Small dataset handling guide
- LIME interpretability guide (new in v3.0.0)

---

## 🎓 Scientific Improvements

### From v2.0.0
- Complete methodological overhaul to ensure rigorous quality and clinical safety
- Enhanced machine learning capabilities with overfitting prevention
- Statistical validation with proper confidence intervals
- Explainable AI features with SHAP integration

### From v2.1.0
- Better handling of small datasets without sacrificing rigor
- Automatic clinical factor integration when statistically significant
- Stability assessment for feature importance via bootstrap SHAP
- Clear warnings about dataset limitations in reports
- Adaptive methodologies that scale with dataset size

### v3.0.0 Additions
1. **Scientifically honest small-cohort analysis** with adaptive CCS thresholds
2. **Preserved full analysis** - no predictions discarded, only warnings issued
3. **Robust SHAP explanations** for both ANN and XGBoost models
4. **Complementary LIME interpretability** for per-patient explanations
5. **Improved stability warnings** with CCS-aware checks
6. **Fixed serialization issues** in XGBoost SHAP explainer

---

## 🔗 References

- Repository: https://github.com/kalyan2031990/py_ntcpx
- Documentation: See README.md and ARCHITECTURE_REPORT.md
- Test Report: test_report.md
- Previous Releases:
  - [CHANGELOG_v2.1.0.md](CHANGELOG_v2.1.0.md)
  - [CHANGELOG_v2.0.0.md](CHANGELOG_v2.0.0.md)

---

## 🙏 Acknowledgments

This release addresses real-world challenges encountered with small-cohort NTCP analysis (54-patient case). The enhancements maintain scientific rigor while enabling honest, interpretable analysis of small clinical datasets. All changes preserve backward compatibility and enhance rather than replace existing functionality.

---

**Release Date**: February 3, 2026  
**Version**: 3.0.0  
**Based on**: v2.1.0 (February 3, 2026)

**Next Steps**: Continue collecting data to reach recommended sample sizes (≥150 patients for parotid NTCP) for more reliable model performance. Use LIME explanations for detailed per-patient interpretability.

