# Changelog - Version 2.1.0

## Minor Release: py_ntcpx v2.1.0 - Small Dataset Enhancements

This release enhances the pipeline to better handle small datasets (n < 100) while maintaining backward compatibility and scientific rigor.

**Release Date:** 2026-02-03

---

## 🎯 Key Enhancements

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

## 📊 Test Results

- **Total Tests**: 80
- **Passed**: 78 (100% of runnable tests)
- **Failed**: 0
- **Skipped**: 2 (baseline regression - expected)
- **Test Coverage**: All enhancements validated

---

## 🔄 Migration Guide

### For Existing Users

**No breaking changes** - all enhancements are backward compatible:

1. **Dynamic CCS**: Automatically adapts based on dataset size. No code changes needed.
2. **Clinical Factors**: Pass `clinical_data` parameter to `RadiobiologyGuidedFeatureSelector` to enable. Optional.
3. **Small Dataset Adaptations**: Automatically applied when n < 100. No configuration needed.
4. **SHAP Bootstrap**: Automatically runs for datasets < 100 samples. No changes needed.

### New Parameters

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

---

## 📝 Files Modified

- `ntcp_qa_modules.py` - Dynamic CCS threshold
- `src/features/feature_selector.py` - Clinical factor integration
- `code3_ntcp_analysis_ml.py` - Small dataset adaptations
- `shap_code7.py` - Bootstrap SHAP analysis
- `code4_ntcp_output_QA_reporter.py` - Enhanced reporting
- `src/safety/clinical_safety_guard.py` - CCS dict handling fix
- `test_report.md` - Updated test results

---

## 🎓 Scientific Improvements

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

---

## 🙏 Acknowledgments

This release focuses on making the pipeline more robust for real-world clinical datasets, which are often smaller than ideal. All changes maintain scientific rigor while providing appropriate warnings and adaptations.

---

**Next Steps**: Continue collecting data to reach recommended sample sizes (≥150 patients for parotid NTCP) for more reliable model performance.

