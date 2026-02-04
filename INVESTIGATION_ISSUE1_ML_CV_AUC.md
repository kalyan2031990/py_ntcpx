# Issue 1: ML CV-AUC Values Missing from Report – Investigation Report

**Date:** 2025-02-04  
**Status:** Root cause identified, fix implemented

---

## Summary

CV-AUC values are computed during ML training but **never saved to any output file**. They exist only in memory (`ml_models.models[organ]`) and are lost when the pipeline completes.

---

## Investigation Findings

### 1. CV-AUC is Computed Correctly

**Location:** `code3_ntcp_analysis_ml.py`

- **Small datasets (< 100 samples):** 5-fold cross-validation (lines 916–969)
  - `cv_AUC_mean` and `cv_AUC_std` stored in `results['ANN']` and `results['XGBoost']`
  - Keys: `cv_AUC_mean`, `cv_AUC_std`, `cv_AUC_scores`

- **Larger datasets (train/test split):** Cross-validation on training set (lines 1140–1155, 1222–1235)
  - `cross_val_score()` used with `scoring='roc_auc'`
  - Same keys: `cv_AUC_mean`, `cv_AUC_std`

### 2. CV-AUC is Never Persisted

- **`create_comprehensive_excel()`** (lines 2711–2930): Builds `ntcp_results.xlsx` with:
  - Summary by Organ: AUC/Brier from predictions (apparent AUC), **no CV-AUC**
  - Performance Matrix: same apparent AUC, **no CV-AUC**

- **`ml_validation.xlsx`**: Mentioned in `OUTPUT_STRUCTURE.md` for `tiered_output/`, but:
  - Created only by `tiered_ntcp_analysis.py` with different metrics (Train_AUC, Test_AUC, Overfitting_Gap)
  - **Not created by code3** in `code3_output/`

- **`create_manuscript_materials_bundle()`** (lines 5528–5560): ML_Performance sheet computes AUC/Brier from predictions only, **no CV-AUC**

### 3. Report Generator Does Not Use CV-AUC

- **code4_ntcp_output_QA_reporter.py**: Computes AUC from NTCP predictions; does not read `ml_validation.xlsx`
- **create_enhanced_summary_report()**: Uses apparent AUC from predictions; no CV-AUC
- **statistical_reporter.py**: Expects `cv_auc_mean`/`cv_auc_std` in metrics dict but these are never passed from code3

---

## Root Cause

**SCENARIO B (confirmed):** CV-AUC is computed but not saved.

The `ml_models` object holds CV-AUC in memory, but no code writes it to `ml_validation.xlsx` or any other file in the code3 output directory.

---

## Fix Implemented

1. **New function `save_ml_validation_results(ml_models, output_dir)`**
   - Extracts `cv_AUC_mean`, `cv_AUC_std` from `ml_models.models` for each organ
   - Writes `ml_validation.xlsx` with columns: Organ, Model, CV_AUC_Mean, CV_AUC_Std, N_Samples, N_Events, Validation_Method

2. **Call from `process_all_patients()`**
   - Invoked after model export and before `create_comprehensive_excel()`

3. **Update `create_comprehensive_excel()`**
   - New optional parameter `ml_models`
   - If provided, reads CV-AUC from `ml_models` and adds CV_AUC columns to Summary by Organ and Performance Matrix
   - Alternative: read from `ml_validation.xlsx` if it exists (avoids changing function signature)

4. **Update manuscript materials**
   - ML_Performance sheet in `manuscript_tables.xlsx` extended to include CV_AUC_Mean and CV_AUC_Std when available

---

## Files Modified

- `code3_ntcp_analysis_ml.py`: Add `save_ml_validation_results()`, call it, pass `ml_models` to `create_comprehensive_excel()`, update Excel sheets
