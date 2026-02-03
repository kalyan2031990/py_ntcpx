# STEP 2: Feature Availability Investigation - Debug Report

## Summary

Debug code has been added to investigate why only 3 features are being used in ML models. The investigation reveals a **critical naming mismatch** between feature column names and feature selector expectations.

---

## Files Modified

1. **`code3_ntcp_analysis_ml.py`**
   - Added debug output to `prepare_features()` method (lines ~595-610)
   - Added debug output to `train_and_evaluate_ml_models()` method (lines ~652-660)
   - Added debug output around feature selection (lines ~734-750)

2. **`src/features/feature_selector.py`**
   - Added debug output to `select_features()` method (lines ~57-115)

---

## Critical Issue Identified: Feature Naming Mismatch

### Problem

The `RadiobiologyGuidedFeatureSelector` expects feature names that **do not match** the actual column names produced by `prepare_features()`:

| Feature Selector Expects | Actual Column Name | Status |
|-------------------------|-------------------|--------|
| `'Dmean'` | `'mean_dose'` | **MISMATCH** ❌ |
| `'V30'` | `'V30'` | ✅ Match |
| `'V45'` | `'V45'` | ✅ Match |
| `'gEUD'` | `'gEUD'` | ✅ Match |

### Root Cause

**In `code3_ntcp_analysis_ml.py` (line 574-578):**
```python
feature_cols = [
    'mean_dose', 'max_dose', 'gEUD', 'total_volume',  # Uses 'mean_dose'
    'V5', 'V10', 'V15', 'V20', 'V25', 'V30', 'V35', 'V40', 'V45', 'V50',
    'D1', 'D2', 'D5', 'D10', 'D20', 'D30', 'D50', 'D70', 'D90', 'D95'
]
```

**In `src/features/feature_selector.py` (line 21):**
```python
PAROTID_ESSENTIAL = ['Dmean', 'V30', 'V45']  # Expects 'Dmean', not 'mean_dose'
```

### Impact

1. **Only 2 essential features found**: `'V30'` and `'V45'` (missing `'Dmean'` because it's actually `'mean_dose'`)
2. **EPV rule limits features**: With low event counts, `max_features = max(int(n_events / 10), 3)` may only allow 3 features
3. **Statistical filtering may add 1 more**: If any feature passes p < 0.1 threshold, it gets added
4. **Result**: Only 3 features selected (V30, V45, + 1 statistically significant feature)

---

## Debug Code Added

### 1. `prepare_features()` Debug Output

**Location**: End of `prepare_features()` method, before return statement

**Output includes**:
- X shape and number of features
- All feature names
- Sample of first 15 features
- Checks for 'Dmean' vs 'mean_dose'
- Checks for 'gEUD'
- Checks for clinical features
- Event count and sample size

**Example output**:
```
[DEBUG] prepare_features() returned:
  - X shape: (54, 23)
  - Number of features: 23
  - All feature names: ['mean_dose', 'max_dose', 'gEUD', 'total_volume', 'V5', 'V10', ...]
  - Contains Dmean? False
  - Contains mean_dose? True
  - Contains gEUD? True
  - y sum (events): 12
  - Total samples: 54
```

### 2. `train_and_evaluate_ml_models()` Debug Output

**Location**: Beginning of `train_and_evaluate_ml_models()` method

**Output includes**:
- Input organ_data shape
- First 30 column names
- PrimaryPatientID availability
- V2_COMPONENTS_AVAILABLE status

**Example output**:
```
[DEBUG] train_and_evaluate_ml_models() called for Parotid
  - organ_data shape: (54, 45)
  - organ_data columns (first 30): ['PrimaryPatientID', 'AnonPatientID', 'Organ', ...]
  - 'PrimaryPatientID' in columns? True
  - V2_COMPONENTS_AVAILABLE: True
```

### 3. Feature Selection Debug Output

**Location**: Before and after `RadiobiologyGuidedFeatureSelector.select_features()` call

**Output includes**:
- Total features available before selection
- All feature names
- Selected features after selection
- Number of features selected

**Example output**:
```
[DEBUG] Before feature selection:
  - Total features available: 23
  - Feature names: ['mean_dose', 'max_dose', 'gEUD', 'total_volume', ...]

[DEBUG] After feature selection:
  - Selected features: ['V30', 'V45', 'V5']
  - Number selected: 3
```

### 4. `RadiobiologyGuidedFeatureSelector.select_features()` Debug Output

**Location**: Throughout `select_features()` method

**Output includes**:
- Input X_df shape and all available features
- Number of events and total samples
- Organ name
- Essential and exploratory feature lists
- Feature lookup results (FOUND/NOT FOUND)
- EPV calculation (n_events / 10)
- Features with p < 0.1
- Final selected features

**Example output**:
```
[DEBUG] RadiobiologyGuidedFeatureSelector.select_features()
  - Input X_df shape: (43, 23)
  - All available features: ['mean_dose', 'max_dose', 'gEUD', ...]
  - Number of events (y.sum()): 10
  - Total samples: 43
  - Organ: Parotid
  - Essential features for Parotid: ['Dmean', 'V30', 'V45']
  - Exploratory features for Parotid: ['D50', 'V15', 'V20', 'gEUD']
    - Looking for 'Dmean': NOT FOUND
    - Looking for 'V30': FOUND
      -> Added 'V30' to selected
    - Looking for 'V45': FOUND
      -> Added 'V45' to selected
  - n_events: 10
  - EPV rule: max_features = max(int(10 / 10), 3) = 3
  - Features with p<0.1: ['V5', 'mean_dose', ...]
  - FINAL selected features: ['V30', 'V45', 'V5']
  - Number selected: 3
```

---

## Expected Findings When Running Pipeline

When you run the pipeline with debug enabled, you should see:

1. **`prepare_features()` will show**:
   - Many features available (likely 20+)
   - `'mean_dose'` exists but `'Dmean'` does not
   - `'gEUD'` exists
   - Clinical features may or may not be present

2. **`RadiobiologyGuidedFeatureSelector` will show**:
   - Looking for `'Dmean'`: **NOT FOUND** ❌
   - Looking for `'V30'`: **FOUND** ✅
   - Looking for `'V45'`: **FOUND** ✅
   - EPV calculation limiting features to 3 (if n_events < 30)
   - Only 2-3 features selected

3. **Root cause confirmed**: The naming mismatch prevents `'Dmean'` from being found, so only `'V30'` and `'V45'` are selected as essential features, plus 1 statistically significant feature (likely `'V5'` based on the report).

---

## Next Steps

### Option 1: Fix Feature Selector (Recommended)
Update `src/features/feature_selector.py` to use the actual column names:

```python
# Change from:
PAROTID_ESSENTIAL = ['Dmean', 'V30', 'V45']

# To:
PAROTID_ESSENTIAL = ['mean_dose', 'V30', 'V45']  # Use actual column name
```

### Option 2: Add Column Name Mapping
Create a mapping function to translate between naming conventions:

```python
FEATURE_NAME_MAP = {
    'Dmean': 'mean_dose',
    'Dmax': 'max_dose',
    # ... other mappings
}
```

### Option 3: Standardize Column Names
Update `prepare_features()` to use `'Dmean'` instead of `'mean_dose'` (but this may break other parts of the codebase).

---

## Testing Instructions

### 1. Run Quick Import Test
```bash
python -c "
try:
    from src.features.feature_selector import RadiobiologyGuidedFeatureSelector
    print('FeatureSelector import: OK')
except Exception as e:
    print(f'FeatureSelector import failed: {e}')

try:
    from src.models.machine_learning.ml_models import OverfitResistantMLModels
    print('OverfitResistantMLModels import: OK')
except Exception as e:
    print(f'OverfitResistantMLModels import failed: {e}')
"
```

**Expected output**: Both imports should succeed.

### 2. Run Pipeline with Debug
```bash
python run_pipeline.py --input_dvh ./dvh_data --clinical_file ./clinical_file.xlsx 2>&1 | tee "debug_step2_output.txt"
```

### 3. Extract Debug Messages
```bash
# Windows PowerShell:
Select-String -Path "debug_step2_output.txt" -Pattern "\[DEBUG\]" | Select-Object -First 50

# Or grep equivalent:
grep "\[DEBUG\]" debug_step2_output.txt | head -50
```

### 4. Key Questions to Answer

When reviewing debug output, answer:

1. **How many total features does `prepare_features()` return?**
   - Expected: 20-25 features

2. **What are the first 10 feature names?**
   - Should include: `'mean_dose'`, `'max_dose'`, `'gEUD'`, `'V5'`, `'V10'`, etc.

3. **Is `'Dmean'` in the list?**
   - Expected: **NO** (it's `'mean_dose'`)

4. **Is `'mean_dose'` in the list?**
   - Expected: **YES**

5. **What's the EPV calculation?**
   - Format: `n_events = ? → max_features = max(int(? / 10), 3) = ?`
   - Example: `n_events = 10 → max_features = max(int(10 / 10), 3) = 3`

6. **Which essential features are found?**
   - Expected: Only `'V30'` and `'V45'` (not `'Dmean'`)

7. **What are the final selected features?**
   - Expected: `['V30', 'V45', 'V5']` or similar (only 3 features)

---

## Files to Review

1. **`code3_ntcp_analysis_ml.py`** (line 574-578): Feature column names
2. **`src/features/feature_selector.py`** (line 21): Essential feature names
3. **Debug output**: Will show exact mismatch

---

## Conclusion

The debug code is now in place. Running the pipeline will reveal:
- Exact number of features available
- Which features are found/not found by the selector
- EPV calculation and why only 3 features are selected
- Confirmation of the `'Dmean'` vs `'mean_dose'` naming mismatch

**The fix is straightforward**: Update the feature selector to use `'mean_dose'` instead of `'Dmean'`, or add a mapping function to translate between naming conventions.

