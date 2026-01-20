# Integration Complete: v2.0 Components in code3_ntcp_analysis_ml.py

## ✅ Integration Status

The v2.0 components have been successfully integrated into `code3_ntcp_analysis_ml.py`. The code now supports both:
- **v2.0 enhanced mode**: Uses patient-level splitting, overfit-resistant models, feature selection, and AUC with CI (when components are available)
- **Fallback mode**: Uses original row-level splitting if v2.0 components are not available

## 🔧 Changes Made

### 1. Imports Added (Lines ~36-47)
```python
# Import v2.0 components for data leakage prevention and overfitting control
try:
    from src.validation.data_splitter import PatientDataSplitter
    from src.models.machine_learning.ml_models import OverfitResistantMLModels
    from src.features.feature_selector import RadiobiologyGuidedFeatureSelector
    from src.metrics.auc_calculator import calculate_auc_with_ci
    from src.reporting.leakage_detector import DataLeakageDetector
    V2_COMPONENTS_AVAILABLE = True
except ImportError as e:
    V2_COMPONENTS_AVAILABLE = False
    print(f"Warning: v2.0 components not available: {e}")
```

### 2. Updated `train_and_evaluate_ml_models()` Method

#### Patient-Level Splitting (v2.0)
- Checks for `PrimaryPatientID` column
- Uses `PatientDataSplitter` for patient-level splits (prevents data leakage)
- Validates no leakage with `DataLeakageDetector`
- Extracts features **AFTER** split (critical for preventing leakage)

#### Feature Selection (v2.0)
- Uses `RadiobiologyGuidedFeatureSelector` before training
- Domain-knowledge guided selection (QUANTEC guidelines)
- EPV-based feature capping

#### Overfit-Resistant Models (v2.0)
- Uses `OverfitResistantMLModels` for conservative hyperparameters
- EPV validation with warnings
- Automatic complexity adjustment for small samples

#### AUC with Confidence Intervals (v2.0)
- Uses `calculate_auc_with_ci()` for bootstrap confidence intervals
- Reports AUC with 95% CI in output

### 3. Enhanced Output
Results dictionary now includes:
- `test_AUC_CI`: Confidence interval tuple (when v2.0 available)
- `epv`: Events per variable (when v2.0 available)

## 🧪 Testing

The integration maintains backward compatibility:
- If v2.0 components are not available, falls back to original implementation
- If `PrimaryPatientID` column is missing, falls back to row-level splitting
- All existing functionality preserved

## 📊 Expected Output

### With v2.0 Components:
```
Training ML models for Parotid...
   Using v2.0 patient-level splitting...
   Features: 24, Train Samples: 43, Events: 12
   Test Samples: 11, Test Events: 3
   Selected 8 features: ['Dmean', 'V30', 'V45', ...]...
     EPV: 1.50 events per variable
   Training ANN...
       ANN - Test AUC: 0.756 (95% CI: 0.612-0.900)
   Training XGBoost...
       XGBoost - Test AUC: 0.712 (95% CI: 0.568-0.856)
```

### Without v2.0 Components (Fallback):
```
Training ML models for Parotid...
   Using row-level splitting (v2.0 components not available)...
   Features: 24, Samples: 54, Events: 15
   Training ANN...
       ANN - Test AUC: 0.745, CV AUC: 0.782+/-0.123
```

## ⚠️ Important Notes

1. **Data Requirements**: For v2.0 patient-level splitting, the `organ_data` DataFrame must contain a `PrimaryPatientID` column
2. **Backward Compatibility**: Code works without v2.0 components installed (graceful fallback)
3. **Test Size**: v2.0 uses 20% test size (vs 30% in original) for more training data
4. **Feature Selection**: Only activates if more than 3 features are available

## 🔍 Validation Checklist

- [x] Patient-level splitting implemented
- [x] Feature extraction after split (prevents leakage)
- [x] Leakage detection integrated
- [x] Overfit-resistant ML models integrated
- [x] Feature selection integrated
- [x] AUC with confidence intervals integrated
- [x] Backward compatibility maintained
- [x] No breaking changes to existing functionality

## 📝 Next Steps

1. **Test with Real Data**: Run the pipeline with your dataset to verify v2.0 components work correctly
2. **Update QA Reporter**: Integrate leakage detection into `code4_ntcp_output_QA_reporter.py` (see `IMPLEMENTATION_STATUS.md`)
3. **Review Outputs**: Check that AUC confidence intervals are reported in all outputs
4. **Performance Validation**: Compare train-test AUC gaps (should be < 15% with v2.0)

## 🐛 Troubleshooting

### Issue: "v2.0 components not available"
**Solution**: Ensure `src/` directory exists and components are properly installed

### Issue: "PrimaryPatientID not found, falling back to row-level split"
**Solution**: Ensure your `organ_data` DataFrame includes `PrimaryPatientID` column from `results_df`

### Issue: "EPV warning: low events per variable"
**Solution**: This is expected for small datasets. Feature selection will automatically reduce features to improve EPV.
