# Release notes for GitHub (py_ntcpx – post–v1.0.0 updates)

**Copy the markdown below into the GitHub release description** when publishing the latest changes (e.g. as a new release or as updated release notes for [py_ntcpx v1.0.0](https://github.com/kalyan2031990/py_ntcpx/releases/tag/py_ntcpx_v1.0.0)).

---

**Author:** K. Mondal (Medical Physicist, North Bengal Medical College, Darjeeling, India)  
**Repository:** https://github.com/kalyan2031990/py_ntcpx  
**Based on:** [Release py_ntcpx v1.0.0](https://github.com/kalyan2031990/py_ntcpx/releases/tag/py_ntcpx_v1.0.0)

---

## Summary

This update applies improvements and bug fixes to the v1.0.0 pipeline: correct ML validation metrics (Apparent vs CV/Test AUC), XGBoost constant-prediction fix for small cohorts, calibration metrics (ECE, MCE), overfitting flagging, and code5 clinical merge using code0's `patient_id`. No change to the overall pipeline structure or step order.

---

## Changes

### ML validation and overfitting

- **Apparent vs CV/Test AUC:** Tiered `ml_validation.xlsx` now reports **Apparent_AUC** (in-sample) and, when code3 has run, **CV_AUC_mean**, **CV_AUC_std**, **Test_AUC**, and **Overfitting_Gap** (Apparent − CV). **Overfitting_Flag** is set to `"High"` when Overfitting_Gap > 0.1.
- **New file:** Code3 writes **`ml_cv_metrics.xlsx`** (Organ, Model, Apparent_AUC, CV_AUC_mean, CV_AUC_std, Test_AUC, Constant_Predictor). Tiered merges this into `ml_validation.xlsx` when available.
- **Test_AUC:** Under the CV-only path (n < 100), Test_AUC in `ml_cv_metrics.xlsx` is now filled from CV_AUC_mean so it is never missing.

### XGBoost constant-prediction fix

- On small cohorts, XGBoost could collapse to a single constant prediction (AUC 0.5). **Adjusted** small-sample settings in `OverfitResistantMLModels` and in the code3 fallback: lower `min_child_weight`, higher `learning_rate`, and for n < 50 use `max_depth` 2 instead of 1 so the model can split.
- **Sanity check:** If any ML model still produces a single unique prediction, the pipeline logs a warning and sets **Constant_Predictor = True** in `ml_cv_metrics.xlsx`.

### Calibration and small-sample note

- **ECE** (Expected Calibration Error) and **MCE** (Maximum Calibration Error) are computed (10 bins) and added to **`ml_validation.xlsx`** for all models.
- A **Note** sheet in `ml_validation.xlsx` states that for small samples (n < 100) users should prefer **CV_AUC** for ML and treat results as exploratory.

### Clinical factors (code5)

- Code0 outputs **`patient_id`** in `clinical_reconciled.xlsx`. Code5 now accepts **patient_id** and maps it to PrimaryPatientID so the factors step runs without renaming columns. Error message updated to mention `patient_id`.

### Tests

- **`tests/test_ml_models.py`:** For very small sample, the expected XGBOOST `max_depth` is now 2 (was 1) to match the constant-prediction fix.

---

## Files modified

| File | Change |
|------|--------|
| `src/models/machine_learning/ml_models.py` | XGBoost small-sample settings (min_child_weight, learning_rate, max_depth) |
| `code3_ntcp_analysis_ml.py` | Export `ml_cv_metrics.xlsx`; Test_AUC fallback; Constant_Predictor; XGBoost fallback params |
| `tiered_ntcp_analysis.py` | Apparent_AUC; merge CV metrics; Overfitting_Gap/Flag; ECE/MCE; Note sheet |
| `code5_ntcp_factors_analysis.py` | Accept `patient_id` and map to PrimaryPatientID |
| `tests/test_ml_models.py` | Expect max_depth=2 for very small sample |

---

## Documentation

- **`CHANGELOG.md`** – Changelog for the repo.
- **`docs/PIPELINE_QA_AND_IMPROVEMENTS.md`** – QA notes and improvement suggestions.
- **`docs/PIPELINE_IMPROVEMENTS_AND_ANALYSIS_REPORT.md`** – Implementation and post-run analysis.

---

## Citation

**Software:** py_ntcpx – NTCP Analysis and Machine Learning Pipeline  
**Author:** K. Mondal (Medical Physicist, North Bengal Medical College, Darjeeling, India)  
**Repository:** https://github.com/kalyan2031990/py_ntcpx  
**License:** MIT

See **CITATION.cff** in the repository for machine-readable citation.
