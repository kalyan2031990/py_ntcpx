# py_ntcpx v3.0.0 – Enhanced Interpretability and Small-Cohort Honesty

**Release Date:** February 3, 2026

---

## 🎯 Overview

This major release extends v2.1.0 with scientifically robust enhancements for small-cohort NTCP analysis, improved model interpretability, and honest prediction handling. Focus areas: adaptive CCS thresholds, fixed SHAP explanations, and complementary LIME interpretability.

---

## 🆕 Key Features

### 1. Adaptive CCS Thresholds for Small Cohorts
- **Scientifically honest thresholds** based on dataset size:
  - n < 30: Disable filtering (0.0) for tiny cohorts
  - n 30-100: Conservative threshold (0.1) for small cohorts (e.g., 54-patient case)
  - n ≥ 100: Standard threshold (0.2) for larger datasets
- **Replaces overly conservative static threshold** that blocked all predictions in small cohorts
- **Maintains scientific rigor** while enabling analysis of real-world small datasets

### 2. CCS Warnings Instead of DO_NOT_USE Flags
- **Changed behavior**: CCS now generates warnings instead of discarding predictions
- **New `CCS_Warning_Flag` column**: Boolean flag (True if CCS below adaptive threshold)
- **All predictions preserved**: Full analysis and explanations (SHAP/LIME) computed for all patients
- **Clear logging**: `INFO - CCS below adaptive threshold for X/X predictions. Interpretations should be treated with caution.`

### 3. Fixed XGBoost SHAP Explainer
- **Model-agnostic explainer**: Replaced `TreeExplainer` with `Explainer` for serialized models
- **Base score fix**: Handles string `base_score` in serialized XGBoost models
- **Robust prediction wrapper**: Works with both DataFrame and array inputs
- **Fixes serialization issues**: Resolves failures when loading saved XGBoost models

### 4. Improved ANN SHAP Stability Warnings
- **Enhanced error handling**: Wrapped bootstrap stability analysis in try-except
- **CCS-aware warnings**: Checks cohort consistency before stability analysis
- **Clear warnings**: `WARNING - SHAP stability analysis skipped for ANN due to low cohort consistency. Global interpretations may be unstable.`
- **Visualizations preserved**: SHAP beeswarm and bar plots still generated with appropriate warnings

### 5. LIME (Local Interpretable Model-agnostic Explanations) Integration
- **New dependency**: `lime` package added to requirements.txt
- **Per-patient explanations**: Generates LIME explanations for representative patients
- **Representative selection**: Automatically identifies highest, median, and lowest predicted NTCP patients
- **Dual output formats**: Saves both HTML (`lime_explanation_{patient_id}.html`) and PNG (`lime_explanation_{patient_id}.png`)
- **Model support**: Works with both ANN and XGBoost models
- **Complementary to SHAP**: Provides local interpretability alongside global SHAP analysis

---

## 📋 All Features from v2.0.0 and v2.1.0

This release includes all features from previous versions:

- ✅ Patient-level data splitting with stratification
- ✅ EPV (Events Per Variable) enforcement
- ✅ Bootstrap confidence intervals for all metrics
- ✅ DeLong test for AUC comparison
- ✅ Clinical safety guard with underprediction risk detection
- ✅ 600 DPI figures and LaTeX tables
- ✅ Comprehensive test suite (80 tests, 78 passing)

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

---

## 🐛 Bug Fixes

- **Fixed**: XGBoost SHAP explainer compatibility with serialized models
- **Fixed**: Overly conservative CCS threshold blocking all predictions in small cohorts
- **Fixed**: Missing SHAP explanations when CCS was low
- **Improved**: ANN SHAP stability analysis error handling

---

## 📊 Test Coverage

- **Total Tests**: 80
- **Passed**: 78 (100% of runnable tests)
- **Failed**: 0
- **Skipped**: 2 (baseline regression - expected)

All v2.1.0 tests maintained, plus new tests for adaptive CCS thresholds and LIME integration validation.

---

## 📚 Documentation

- Comprehensive README updates
- `OUTPUT_STRUCTURE.md` - Complete output file documentation
- `CHANGELOG_v3.0.0.md` - Full changelog
- `test_report.md` - Test results
- LIME interpretability guide

---

## 🔗 References

- Repository: https://github.com/kalyan2031990/py_ntcpx
- Documentation: See README.md and OUTPUT_STRUCTURE.md
- Test Report: test_report.md
- Previous Releases:
  - [CHANGELOG_v2.1.0.md](CHANGELOG_v2.1.0.md)
  - [CHANGELOG_v2.0.0.md](CHANGELOG_v2.0.0.md)

---

## 🙏 Acknowledgments

This release addresses real-world challenges encountered with small-cohort NTCP analysis (54-patient case). The enhancements maintain scientific rigor while enabling honest, interpretable analysis of small clinical datasets. All changes preserve backward compatibility and enhance rather than replace existing functionality.

---

**Next Steps**: Continue collecting data to reach recommended sample sizes (≥150 patients for parotid NTCP) for more reliable model performance. Use LIME explanations for detailed per-patient interpretability.

