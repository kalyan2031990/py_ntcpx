# Test Report - py_ntcpx v3.0.0

**Generated:** 2025-02-04  
**Test Runner:** `run_all_tests.py` (pytest)  
**JUnit Report:** `test_reports/pytest_report.xml`

---

## Test Summary

| Metric | Value |
|--------|-------|
| **Total Tests** | 80 |
| **Passed** | 78 |
| **Failed** | 0 |
| **Errors** | 0 |
| **Skipped** | 2 |
| **Total Time** | ~7s |

**Pass Rate:** 100% (78/78 runnable tests passed)

✅ **All tests passed successfully!**

---

## Test Results by Category

### tests/test_auc_calculator.py - TestAUCCalculator
- **Total:** 6 | **Passed:** 6 | **Failed:** 0 | **Skipped:** 0

| Test Name | Status |
|-----------|--------|
| test_auc_calculation_bootstrap | ✅ PASSED |
| test_auc_calculation_delong | ✅ PASSED |
| test_auc_ci_coverage | ✅ PASSED |
| test_auc_high_vs_low | ✅ PASSED |
| test_auc_requires_both_classes | ✅ PASSED |
| test_compare_aucs_delong | ✅ PASSED |

### tests/test_auto_feature_reducer.py - TestAutoFeatureReducer
- **Total:** 2 | **Passed:** 2 | **Failed:** 0 | **Skipped:** 0

| Test Name | Status |
|-----------|--------|
| test_auto_reduction_low_epv | ✅ PASSED |
| test_no_reduction_adequate_epv | ✅ PASSED |

### tests/test_calibration_correction.py - TestCalibrationCorrection
- **Total:** 4 | **Passed:** 4 | **Failed:** 0 | **Skipped:** 0

| Test Name | Status |
|-----------|--------|
| test_calibration_improves_slope | ✅ PASSED |
| test_calibration_slope_calculation | ✅ PASSED |
| test_isotonic_regression | ✅ PASSED |
| test_platt_scaling | ✅ PASSED |

### tests/test_clinical_safety.py - TestClinicalSafetyGuard
- **Total:** 7 | **Passed:** 7 | **Failed:** 0 | **Skipped:** 0

| Test Name | Status |
|-----------|--------|
| test_do_not_use_flag_low_ccs | ✅ PASSED |
| test_evaluate_safety_basic | ✅ PASSED |
| test_fit_training_data | ✅ PASSED |
| test_safety_guard_initialization | ✅ PASSED |
| test_safety_report_generation | ✅ PASSED |
| test_safety_report_save | ✅ PASSED |
| test_underprediction_risk_detection | ✅ PASSED |

### tests/test_data_splitter.py - TestPatientDataSplitter
- **Total:** 5 | **Passed:** 5 | **Failed:** 0 | **Skipped:** 0

| Test Name | Status |
|-----------|--------|
| test_leakage_detection | ✅ PASSED |
| test_patient_level_split | ✅ PASSED |
| test_reproducibility | ✅ PASSED |
| test_stratification | ✅ PASSED |
| test_train_test_no_overlap | ✅ PASSED |

### tests/test_dvh_validation.py - TestDVHValidation
- **Total:** 5 | **Passed:** 5 | **Failed:** 0 | **Skipped:** 0

| Test Name | Status |
|-----------|--------|
| test_dose_non_negative | ✅ PASSED |
| test_dvh_monotonicity | ✅ PASSED |
| test_gEUD_reproducibility | ✅ PASSED |
| test_v0_equals_100 | ✅ PASSED |
| test_volume_range | ✅ PASSED |

### tests/test_feature_selector.py - TestRadiobiologyGuidedFeatureSelector
- **Total:** 5 | **Passed:** 5 | **Failed:** 0 | **Skipped:** 0

| Test Name | Status |
|-----------|--------|
| test_epv_based_feature_capping | ✅ PASSED |
| test_feature_selection_max_features | ✅ PASSED |
| test_other_organs | ✅ PASSED |
| test_parotid_essential_features | ✅ PASSED |
| test_statistical_filtering | ✅ PASSED |

### tests/test_integration.py - TestEndToEndPipeline
- **Total:** 4 | **Passed:** 4 | **Failed:** 0 | **Skipped:** 0

| Test Name | Status |
|-----------|--------|
| test_complete_workflow | ✅ PASSED |
| test_feature_selection_integration | ✅ PASSED |
| test_ml_training_integration | ✅ PASSED |
| test_patient_level_split_integration | ✅ PASSED |

### tests/test_ml_models.py - TestOverfitResistantMLModels
- **Total:** 7 | **Passed:** 7 | **Failed:** 0 | **Skipped:** 0

| Test Name | Status |
|-----------|--------|
| test_ann_model_creation | ✅ PASSED |
| test_complexity_adjustment_small_sample | ✅ PASSED |
| test_epv_calculation | ✅ PASSED |
| test_epv_error_very_low_epv | ✅ PASSED |
| test_epv_warning_low_epv | ✅ PASSED |
| test_nested_cv | ✅ PASSED |
| test_xgboost_model_creation | ✅ PASSED |

### tests/test_ntcp_mathematics.py - TestNTCPMathematics
- **Total:** 4 | **Passed:** 4 | **Failed:** 0 | **Skipped:** 0

| Test Name | Status |
|-----------|--------|
| test_ntcp_at_td50 | ✅ PASSED |
| test_ntcp_bounds | ✅ PASSED |
| test_ntcp_edge_cases | ✅ PASSED |
| test_ntcp_monotonicity | ✅ PASSED |

### tests/regression/test_baseline_regression.py - TestBaselineRegression
- **Total:** 2 | **Passed:** 0 | **Failed:** 0 | **Skipped:** 2

| Test Name | Status |
|-----------|--------|
| test_baseline_exists | ⏭️ SKIPPED |
| test_classical_ntcp_outputs_unchanged | ⏭️ SKIPPED |

### test_ntcp_pipeline.py
- **TestNTCPUtils:** 3 passed
- **TestNovelModels:** 3 passed
- **TestQAModules:** 2 passed
- **TestDataValidation:** 2 passed
- **TestOutputFormats:** 2 passed
- **TestBiologicalDVH:** 2 passed
- **TestIntegration:** 1 passed

### test_data_validation.py
- **TestDVHDataValidation:** 4 passed
- **TestPatientDataValidation:** 3 passed
- **TestNTCPValueValidation:** 2 passed
- **TestOutputFileValidation:** 2 passed
- **TestEdgeCases:** 3 passed

---

## Recent Fixes Validated (v3.0.0 → v3.0.1)

The following fixes were applied and all tests continue to pass:

1. **Issue 1: ML CV-AUC Values** – `save_ml_validation_results()` now writes CV-AUC to `ml_validation.xlsx`; reports include CV-AUC columns.
2. **Issue 2: QUANTEC-RS NaN** – RS Poisson numerical stability improved; gEUD-based fallback added when DVH calculation fails.
3. **Issue 3: gEUD Verification** – gEUD formula and parameters verified correct; docstring updated.

---

## Warnings (Expected)

- **LOW EPV WARNING** in ML model tests (by design for small-sample scenarios)
- 5 warnings total; none indicate test failures

---

## Environment

- **Python:** 3.14.2
- **pytest:** 9.0.2
- **Platform:** win32

---

**End of Test Report**
