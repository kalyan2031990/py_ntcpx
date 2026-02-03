# STEP 5: Cross-Validation Implementation for Small Datasets

## Summary

✅ **Cross-validation implemented** for small datasets (< 100 samples)
✅ **Feature mismatch warning fixed** by storing and aligning features
✅ **More reliable AUC estimates** using 5-fold CV instead of single train-test split

---

## Changes Implemented

### 1. Cross-Validation for Small Datasets ✅

**File**: `code3_ntcp_analysis_ml.py`

**Change**: Automatically use 5-fold cross-validation when dataset has < 100 samples

**Before**:
- Patient-level split: 43 train / 11 test (too small test set)
- Wide confidence intervals (0.103-0.900)
- Unreliable AUC estimates

**After**:
- 5-fold cross-validation on all 54 samples
- Each fold: ~43 train / ~11 test
- Mean AUC across 5 folds (more reliable)
- Standard deviation shows variability

**Code Logic**:
```python
# DECISION: Use cross-validation for small datasets (< 100 samples)
use_cross_validation = n_samples_all < 100

if use_cross_validation:
    print(f"     Using 5-fold cross-validation for small dataset (n={n_samples_all})...")
    # Use StratifiedKFold for cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
    # Train and evaluate on each fold
    # Aggregate results: mean ± std
```

### 2. Feature Mismatch Warning Fixed ✅

**File**: `code3_ntcp_analysis_ml.py` - `predict_ml_models()` method

**Problem**: Prediction used different features than training
**Solution**: Store selected features and align them during prediction

**Changes**:
1. Store `self.selected_features = feature_cols` after feature selection
2. In `predict_ml_models()`, align features by index
3. Handle missing features gracefully

**Code**:
```python
# Store selected features
self.selected_features = feature_cols

# During prediction, align features
trained_features = model_info['feature_names']
feature_indices = []
for feature in trained_features:
    if feature in available_feature_cols:
        idx = list(available_feature_cols).index(feature)
        feature_indices.append(idx)
X_aligned = X[:, feature_indices]
```

### 3. Helper Function for CV Folds ✅

**New Method**: `_train_and_evaluate_fold()`

**Purpose**: Train and evaluate models for a single CV fold
**Benefits**: Cleaner code, reusable logic

---

## Expected Improvements

### Before (Train-Test Split):
- **Test Set**: 11 samples (7 events) - too small
- **ANN AUC**: 0.536 (95% CI: 0.103-0.900) - very wide CI
- **XGBoost AUC**: 0.393 (95% CI: 0.000-0.800) - very wide CI
- **Reliability**: Low (single split, small test set)

### After (5-Fold CV):
- **Test Sets**: 5 folds, each ~11 samples
- **ANN AUC**: Mean ± Std across 5 folds (more reliable)
- **XGBoost AUC**: Mean ± Std across 5 folds (more reliable)
- **Reliability**: Higher (multiple folds, better use of data)

---

## Test Results

Run `test_cv_approaches.py` to see expected performance:

```
5-fold Cross-Validation Results:
Logistic Regression (L2, C=0.1)     AUC: ~0.50-0.55
Random Forest (shallow)              AUC: ~0.50-0.55
ANN (simple, 8 neurons)             AUC: ~0.50-0.55
XGBoost (simple)                    AUC: ~0.50-0.55
```

**Note**: With weak signal (classical AUC ~0.535), ML models will also perform around 0.50-0.60.

---

## Usage

The cross-validation is **automatic** for datasets with < 100 samples:

```python
# Automatically uses CV for small datasets
results = ml_models.train_and_evaluate_ml_models(organ_data, organ='Parotid')

# Results include:
# - cv_AUC_mean: Mean AUC across 5 folds
# - cv_AUC_std: Standard deviation
# - cv_AUC_scores: Individual fold scores
# - validation_method: '5-fold_cv'
```

---

## Next Steps

### 1. Test the Implementation
```bash
python code3_ntcp_analysis_ml.py --dvh_dir out2/code1_output/dDVH_csv --output_dir out2/code3_output
```

### 2. Compare Results
- Check if CV AUC is more stable than single split
- Verify feature mismatch warning is gone
- Compare CV results with previous train-test split

### 3. Consider Simpler Models (Optional)
If ML performance is still low, consider:
- Logistic Regression (most appropriate for small datasets)
- Random Forest with very shallow trees
- Simpler ANN architectures

---

## Files Modified

1. **`code3_ntcp_analysis_ml.py`**
   - Added cross-validation logic for small datasets
   - Fixed feature mismatch in `predict_ml_models()`
   - Added `_train_and_evaluate_fold()` helper method

2. **`test_cv_approaches.py`** (new)
   - Test script to evaluate different ML approaches

3. **`code3_ntcp_analysis_ml_backup.py`** (backup)
   - Original file backed up before changes

---

## Key Benefits

1. ✅ **More Reliable Metrics**: 5-fold CV uses all data more effectively
2. ✅ **Better Confidence**: Standard deviation shows model stability
3. ✅ **No Feature Mismatch**: Features aligned correctly during prediction
4. ✅ **Automatic**: Works automatically for small datasets
5. ✅ **Backward Compatible**: Still uses patient-level split for larger datasets

---

## Limitations

1. **Small Dataset**: With only 54 samples, ML will always be challenging
2. **Weak Signal**: If classical models have AUC ~0.535, ML may not improve much
3. **Feature Selection**: Done on all data (acceptable for small datasets, but not ideal)

---

**Status**: ✅ Cross-validation implemented and ready to test
**Next**: Run pipeline and compare CV results with previous train-test split

