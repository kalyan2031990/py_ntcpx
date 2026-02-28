# Changelog

All notable changes to py_ntcpx since the first public release (v1.0.0) are documented here.

**Author:** K. Mondal (Medical Physicist, North Bengal Medical College, Darjeeling, India)  
**Repository:** https://github.com/kalyan2031990/py_ntcpx

---

## [Unreleased] – 2026-03-01 (post–v1.0.0 improvements)

### Added
- **ML CV metrics export (`code3`):** New file `ml_cv_metrics.xlsx` with Apparent_AUC, CV_AUC_mean, CV_AUC_std, Test_AUC, and Constant_Predictor per organ/model (ANN, XGBoost, Random Forest).
- **Calibration metrics:** ECE (Expected Calibration Error) and MCE (Maximum Calibration Error) in tiered `ml_validation.xlsx` for all models.
- **Overfitting flag:** Column `Overfitting_Flag` in `ml_validation.xlsx` set to `"High"` when Overfitting_Gap > 0.1.
- **Small-sample note:** New sheet `Note` in `ml_validation.xlsx` advising to prefer CV_AUC for ML and that results are exploratory when n < 100.
- **Constant-predictor warning:** Console warning when an ML model produces a single unique prediction value; `Constant_Predictor` column in `ml_cv_metrics.xlsx`.

### Changed
- **XGBoost (small cohorts):** In `src/models/machine_learning/ml_models.py`, for n < 100: `min_child_weight` 5→2, `learning_rate` 0.03→0.05; for n < 50: `max_depth` 1→2 to avoid constant predictions. Fallback `train_xgboost_model` in `code3_ntcp_analysis_ml.py`: `max_depth` 2, `min_child_weight` 2.
- **Apparent vs CV/Test AUC:** Tiered `ml_validation.xlsx` now uses a single **Apparent_AUC** column (replacing duplicate Train_AUC/Test_AUC). When code3 `ml_cv_metrics.xlsx` exists, tiered merges CV_AUC_mean, CV_AUC_std, Test_AUC and sets **Overfitting_Gap** = Apparent_AUC − CV_AUC_mean.
- **Test_AUC in CV path:** When the pipeline uses the CV-only path (n < 100), Test_AUC in `ml_cv_metrics.xlsx` is now set from CV_AUC_mean so the column is always populated.
- **Clinical factors (code5):** Accept **patient_id** (from code0 reconciled output) and map to PrimaryPatientID so Step 5 merges without manual column renaming.
- **Test:** `tests/test_ml_models.py` updated so that for very small sample, expected XGBOOST `max_depth` is 2 (was 1).

### Fixed
- XGBoost producing constant predictions (single value for all patients) on small cohorts.
- Test_AUC missing (NaN) in `ml_cv_metrics.xlsx` when using 5-fold CV path.
- Code5 factors analysis failing with "Clinical data must contain PrimaryPatientID, DVH_ID, or PatientID" when clinical data used code0's `patient_id` column.
- Overfitting reported as 0 despite large Apparent–CV gap; now Overfitting_Gap and Overfitting_Flag reflect actual gap.

### Documentation
- `docs/PIPELINE_QA_AND_IMPROVEMENTS.md` – QA notes, XGBoost explanation, and improvement suggestions.
- `docs/PIPELINE_IMPROVEMENTS_AND_ANALYSIS_REPORT.md` – Implementation summary and post-run analysis.
- `docs/RELEASE_NOTES_GITHUB.md` – Release text for GitHub (copy into release description).

---

## [1.0.0] – 2026-02-28

First public release. See [Release py_ntcpx v1.0.0](https://github.com/kalyan2031990/py_ntcpx/releases/tag/py_ntcpx_v1.0.0) for full notes.

- Pipeline steps 0–8: clinical reconciliation, DVH preprocessing, NTCP (classical + ML), tiered analysis, QA, factors, publication diagrams, SHAP/LIME, publication tables.
- Overfitting and data-leakage safeguards (EPV, CCS, conservative ML).
- SHAP and LIME explainability for ANN, XGBoost, Random Forest.
- Single canonical output layout under `out2/`.

[Unreleased]: https://github.com/kalyan2031990/py_ntcpx/compare/py_ntcpx_v1.0.0...HEAD
[1.0.0]: https://github.com/kalyan2031990/py_ntcpx/releases/tag/py_ntcpx_v1.0.0
