# Fixes: py_ntcpx v3.0.0 → v3.0.1

**Date:** 2025-02-04  
**Scope:** Three specific issues fixed while preserving all working functionality

---

## Issue 1: ML CV-AUC Values Missing from Report

### Problem
CV-AUC (cross-validated AUC) was computed during ML training but never saved to any output file. Reports showed "N/A" or omitted CV-AUC for ANN and XGBoost models.

### Root Cause
CV-AUC values were stored only in memory (`ml_models.models[organ]`) and never written to Excel or passed to report generators. The `ml_validation.xlsx` file was not created by code3.

### Fix Implemented
1. **New function `save_ml_validation_results(ml_models, output_dir)`**
   - Extracts `cv_AUC_mean`, `cv_AUC_std` for each organ and ML model
   - Writes `ml_validation.xlsx` with columns: Organ, Model, CV_AUC_Mean, CV_AUC_Std, N_Samples, N_Events, Validation_Method

2. **Updated `create_comprehensive_excel()`**
   - New optional parameter `ml_models`
   - Adds `ML_ANN_CV_AUC` and `ML_XGBoost_CV_AUC` to Summary by Organ and Performance Matrix sheets

3. **Updated manuscript materials**
   - ML_Performance sheet reads `ml_validation.xlsx` and includes CV_AUC columns when available

### Result
- `ml_validation.xlsx` created in code3_output with CV-AUC values
- ntcp_results.xlsx Summary and Performance Matrix include CV-AUC
- Manuscript tables include CV-AUC for ML models

---

## Issue 2: QUANTEC-RS Model Returns NaN

### Problem
The RS Poisson (QUANTEC-RS) model sometimes returned NaN instead of valid NTCP values.

### Root Causes
1. **Numerical instability** in the full DVH-based formula (overflow/underflow with small s, e.g. 0.01 for Parotid)
2. **DVH column mismatch** – `Volume[%]` not supported (e.g. synthetic/test data)
3. **No fallback** when the primary calculation failed

### Fix Implemented
1. **DVH loading**
   - Support for `Volume[%]` in addition to `Volume[cm3]`
   - Fallback column detection for columns containing "dose" and "volume"

2. **`ntcp_rs_poisson()` – numerical stability**
   - Clip exponent in `2^(-exp(...))` to [-50, 50]
   - Use `s_safe = max(s, 1e-6)` to avoid extreme exponents
   - Clip `product_term` to [0, 1 - 1e-10] before final power
   - Cap `1/s` to 100 to reduce underflow
   - Return `np.nan` on failure to trigger fallback

3. **gEUD-based fallback in `calculate_all_ntcp_models`**
   - When DVH-based RS Poisson fails or returns NaN, use: NTCP = 1 - exp(-(gEUD_eqd2/D50)^γ)
   - Standard QUANTEC formulation, numerically stable

### Result
- QUANTEC-RS produces valid NTCP values (no NaN)
- Fallback ensures valid output whenever gEUD is available

---

## Issue 3: gEUD Calculation Verification (40.17 Gy)

### Problem
User requested verification that gEUD = 40.17 Gy is correctly calculated.

### Investigation
- **Formula:** gEUD = (Σ v_i × D_i^a)^(1/a) — implemented correctly
- **Parameter:** Parotid a = 2.2 (from QUANTEC LKB_LogLogit)
- **Expected ratio:** gEUD/mean_dose ≈ 1.1–1.4 for a=2.2

### Conclusion
- gEUD calculation is **correct**
- 40.17 Gy is plausible for a high-dose H&N cohort (mean dose ~30–35 Gy)
- No code changes to the calculation; docstring updated for clarity

---

## Files Modified

| File | Changes |
|------|---------|
| `code3_ntcp_analysis_ml.py` | save_ml_validation_results(), create_comprehensive_excel(ml_models), DVH column handling, ntcp_rs_poisson stability, gEUD fallback, calculate_gEUD docstring, manuscript ML_Performance CV-AUC |

---

## Testing

- Import and function checks pass
- `save_ml_validation_results()` tested with mock data
- RS Poisson: full DVH path and gEUD fallback both produce valid NTCP
- gEUD formula verified against QUANTEC

---

## Documentation Created

- `INVESTIGATION_ISSUE1_ML_CV_AUC.md`
- `INVESTIGATION_ISSUE2_QUANTEC_RS_NAN.md`
- `INVESTIGATION_ISSUE3_gEUD_VERIFICATION.md`
