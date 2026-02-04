# Changelog - Version 3.0.1

## Patch Release: py_ntcpx v3.0.1 - Bug Fixes and XGBoost SHAP

**Release Date:** 2026-02-04

---

## Fixes

### 1. ML CV-AUC Now Saved and Reported
- **Issue:** CV-AUC values were computed during ML training but never persisted or included in reports.
- **Fix:** Added `save_ml_validation_results()` to write CV_AUC_Mean and CV_AUC_Std to `ml_validation.xlsx`.
- **Fix:** `ntcp_results.xlsx` Summary by Organ and Performance Matrix now include ML_ANN_CV_AUC and ML_XGBoost_CV_AUC columns.
- **Fix:** Manuscript materials bundle reads `ml_validation.xlsx` and includes CV-AUC in ML_Performance sheet.

### 2. QUANTEC-RS (Poisson) No Longer Returns NaN
- **Issue:** RS Poisson model returned NaN for some DVH formats due to numerical instability and column name mismatches.
- **Fix:** Enhanced DVH column handling (Volume[%], Volume[cm3]).
- **Fix:** Improved numerical stability in `ntcp_rs_poisson()` (exponent clipping, s_safe, product_term bounds).
- **Fix:** gEUD-based fallback when DVH-based calculation fails.

### 3. gEUD Calculation Verified
- **Issue:** User requested verification of gEUD formula (e.g., 40.17 Gy for Parotid).
- **Fix:** Formula and QUANTEC parameters verified correct; docstring updated with references.

### 4. XGBoost SHAP/LIME Now Generated
- **Issue:** XGBoost SHAP failed with `AttributeError: module 'numpy' has no attribute 'trapz'` (numba/numpy compatibility).
- **Fix:** Switched XGBoost to `TreeExplainer` instead of model-agnostic Explainer; fallback to KernelExplainer if needed.
- **Result:** XGBoost SHAP beeswarm, bar, table, stability report and LIME explanations (HTML+PNG) now generated.

---

## Documentation Added

- `FIXES_v3.0.0_to_v3.0.1.md` - Summary of all fixes
- `INVESTIGATION_ISSUE1_ML_CV_AUC.md` - ML CV-AUC investigation
- `INVESTIGATION_ISSUE2_QUANTEC_RS_NAN.md` - QUANTEC-RS NaN investigation
- `INVESTIGATION_ISSUE3_gEUD_VERIFICATION.md` - gEUD verification
- `ARCHITECTURE_REPORT.md` - Complete architecture documentation
- `COMPLETE_RESULTS_REPORT.md` - Comprehensive results from pipeline run

---

## Files Modified

- `code3_ntcp_analysis_ml.py` - ML validation save, QUANTEC-RS fixes, gEUD docstring
- `shap_code7.py` - XGBoost TreeExplainer
- `OUTPUT_STRUCTURE.md` - ml_validation.xlsx, pipeline run date
- `ARCHITECTURE_REPORT.md` - v3.0.1 fixes
- `README.md` - v3.0.1 patches
- `REPRODUCIBILITY_README.md` - v3.0.1
- `requirements.txt` - Version pins, xlsxwriter, joblib
- `VERSION` - 3.0.1
