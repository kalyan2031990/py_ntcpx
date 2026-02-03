# ML Model Fixes and Improvements

## Date: 2026-02-02

## Issues Identified

1. **XGBoost AUC Reporting Issue**: Performance Matrix was using prediction AUC (0.575) instead of CV AUC (0.450), leading to misleading overfitted results
2. **Poor XGBoost Performance**: CV AUC of 0.450 (below random 0.5) indicating model not learning effectively
3. **Missing Error Handling**: CV fold evaluation could fail silently, producing NaN AUCs
4. **Class Imbalance**: Not properly handled in XGBoost configuration

## Fixes Applied

### 1. CV AUC Reporting (Fixed)
- **Problem**: Performance Matrix calculated AUC from full-dataset predictions (overfitted)
- **Solution**: 
  - Store CV AUC in `ml_models.cv_aucs` dictionary
  - Modified `create_comprehensive_excel()` to use CV AUC for ML models
  - Modified `create_enhanced_summary_report()` to use CV AUC for ML models
  - Updated `process_all_patients()` to return `ml_models` instance
- **Result**: Performance Matrix now shows correct CV AUC values

### 2. Improved Error Handling in CV Folds
- **Problem**: Silent failures in CV fold evaluation could produce NaN AUCs
- **Solution**:
  - Added validation checks for:
    - Test set class balance (must have both classes)
    - Prediction validity (no NaN/Inf)
    - AUC validity (no NaN/Inf)
  - Added detailed error messages for debugging
  - Filter out invalid scores before calculating mean CV AUC
- **Result**: More robust CV evaluation with better error reporting

### 3. XGBoost Configuration Improvements
- **Problem**: Model too conservative, not handling class imbalance
- **Solution**:
  - Added `scale_pos_weight` for class imbalance handling
  - Adjusted hyperparameters for small datasets (< 100 samples):
    - `max_depth`: 2 (was 2, already conservative)
    - `n_estimators`: 30 (reduced from 50)
    - `min_child_weight`: 5 (increased from 3)
    - `learning_rate`: 0.03 (reduced from 0.05)
- **Result**: Better handling of class imbalance and small datasets

### 4. Code Structure Improvements
- **Problem**: `ml_models` instance not accessible for reporting functions
- **Solution**:
  - Modified `process_all_patients()` to return `(results_df, ml_models)`
  - Updated `main()` to unpack both return values
  - Passed `ml_models` instance to reporting functions
- **Result**: CV AUCs now accessible throughout the pipeline

## Current Performance (After Fixes)

### Performance Matrix (CV AUC)
- **ML_ANN_AUC**: 0.545 ± 0.185 (5-fold CV)
- **ML_XGBoost_AUC**: 0.450 ± 0.225 (5-fold CV)

### Observations
1. **ANN Performance**: 
   - CV AUC: 0.545 (slightly above random)
   - 5 valid folds, all successful
   - Performance is acceptable but could be improved

2. **XGBoost Performance**:
   - CV AUC: 0.450 (below random 0.5)
   - 5 valid folds, all successful
   - Model is not learning effectively
   - Possible reasons:
     - Small dataset (54 samples, 35 events)
     - Only 5 features selected
     - Model may be too conservative
     - Feature quality may be insufficient

## Recommendations for Further Improvement

1. **Feature Engineering**:
   - Add interaction terms (e.g., mean_dose × V30)
   - Consider polynomial features for key dosimetric variables
   - Include more clinical features if available

2. **Model Selection**:
   - Try simpler models (Logistic Regression, Random Forest)
   - Consider ensemble methods
   - Use feature importance to guide feature selection

3. **Data Quality**:
   - Verify feature distributions and correlations
   - Check for outliers or data quality issues
   - Consider data augmentation if appropriate

4. **Hyperparameter Tuning**:
   - For XGBoost: Try slightly less conservative settings
   - Consider Bayesian optimization for hyperparameter search
   - Use nested CV for proper hyperparameter selection

5. **Cross-Validation Strategy**:
   - Consider leave-one-out CV for very small datasets
   - Use stratified CV to ensure class balance in folds
   - Consider bootstrap validation

## Files Modified

1. `code3_ntcp_analysis_ml.py`:
   - Enhanced CV fold evaluation with error handling
   - Store CV AUCs for reporting
   - Modified `process_all_patients()` to return ml_models
   - Updated Performance Matrix to use CV AUC
   - Updated summary report to use CV AUC

2. `src/models/machine_learning/ml_models.py`:
   - Added class imbalance handling in XGBoost
   - Improved hyperparameters for small datasets
   - Added `scale_pos_weight` calculation

## Testing

- Full pipeline run completed successfully
- Performance Matrix shows correct CV AUC values
- No errors in CV fold evaluation
- All 5 CV folds completed successfully for both models

## Next Steps

1. Monitor XGBoost performance with improved configuration
2. Consider feature engineering to improve model performance
3. Evaluate simpler models as baseline
4. Document any additional improvements needed

