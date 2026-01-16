# Tiered NTCP Upgrade - Implementation Summary

## Overview

This document summarizes the four-tier NTCP framework upgrade to py_ntcpx. All changes are **additive only** - no existing functionality has been removed or modified.

## What Was Added

### 1. New Module Structure

```
ntcp_models/
├── __init__.py
├── legacy_fixed.py      # Tier 1: QUANTEC LKB & RS (wraps existing)
├── legacy_mle.py        # Tier 2: MLE-refitted LKB & RS
└── modern_logistic.py   # Tier 3: Multivariable logistic NTCP
```

### 2. New Scripts

- **`tiered_ntcp_analysis.py`**: Main script that:
  - Loads existing NTCP results from code3
  - Adds Tier 2 (MLE) predictions
  - Adds Tier 3 (Logistic) predictions
  - Calculates CCS for all tiers
  - Generates dose-response curves
  - Creates ML QA validation
  - Generates master Excel report

### 3. Pipeline Integration

- **Step 3c**: New pipeline step added after Step 3b (QUANTEC Stratification)
  - Runs tiered analysis automatically
  - Integrated into `run_pipeline.py`
  - Can be skipped with `--skip 3.6` or resumed with `--resume_from step3c`

### 4. New Outputs

#### Files Created:
- `tiered_output/tiered_ntcp_results.xlsx` - Extended results with all tiers
- `tiered_output/NTCP_4Tier_Master.xlsx` - Comprehensive master report
- `tiered_output/ml_validation.xlsx` - ML QA metrics
- `tiered_output/model_parameters_mle.json` - MLE-fitted parameters
- `tiered_output/dose_response_tiers.png` - Combined dose-response curves
- `tiered_output/dose_response_{Organ}.png` - Organ-specific curves
- `output/README_outputs.md` - Output documentation

#### New Columns Added to Results:
- `NTCP_LKB_Probit_MLE` - Tier 2 MLE LKB Probit
- `NTCP_LKB_LogLogit_MLE` - Tier 2 MLE LKB LogLogit
- `NTCP_LOGISTIC` - Tier 3 multivariable logistic
- `CCS_QUANTEC` - CCS for QUANTEC models
- `CCS_MLE` - CCS for MLE models
- `CCS_Logistic` - CCS for logistic models
- `CCS_ML` - CCS for ML models

## What Was NOT Changed

### Preserved Existing Functionality:
- ✅ All existing NTCP calculations (Tier 1 and Tier 4)
- ✅ All existing output files
- ✅ All existing CLI interfaces
- ✅ All existing reports, plots, and SHAP analysis
- ✅ uNTCP and CCS implementations
- ✅ All formulas and model equations
- ✅ All output file names (except new additions)

### Files NOT Modified:
- `code3_ntcp_analysis_ml.py` - Unchanged
- `ntcp_qa_modules.py` - Unchanged
- `ntcp_novel_models.py` - Unchanged
- All existing code files - Only additions, no modifications

## Usage

### Running Tiered Analysis

#### Option 1: Full Pipeline (Recommended)
```bash
python run_pipeline.py \
    --input_txt_dir /path/to/dvh_files \
    --patient_data /path/to/patient_data.xlsx \
    --clinical_file /path/to/clinical_factors.xlsx \
    --output_dir out2
```

Step 3c (Tiered Analysis) runs automatically after Step 3b.

#### Option 2: Standalone Tiered Analysis
```bash
python tiered_ntcp_analysis.py \
    --code3_output out2/code3_output \
    --dvh_dir out2/code1_output/dDVH_csv \
    --clinical_file /path/to/clinical_factors.xlsx \
    --output_dir out2/tiered_output
```

#### Option 3: Resume from Tiered Analysis
```bash
python run_pipeline.py \
    --resume_from step3c \
    --patient_data /path/to/patient_data.xlsx \
    --clinical_file /path/to/clinical_factors.xlsx \
    --output_dir out2
```

### Skipping Tiered Analysis
```bash
python run_pipeline.py \
    --skip 3.6 \
    ...
```

## Tier Descriptions

### Tier 1: Legacy-A (QUANTEC)
- **Purpose**: Literature transportability validation
- **Models**: LKB LogLogit, LKB Probit, RS Poisson
- **Parameters**: Fixed QUANTEC literature values
- **Output**: `NTCP_LKB_LogLogit`, `NTCP_LKB_Probit`, `NTCP_RS_Poisson`

### Tier 2: Legacy-B (MLE-Refitted)
- **Purpose**: Radiobiology calibrated to cohort
- **Models**: MLE-fitted LKB Probit, MLE-fitted LKB LogLogit
- **Parameters**: Fitted via maximum likelihood estimation
- **Output**: `NTCP_LKB_Probit_MLE`, `NTCP_LKB_LogLogit_MLE`
- **Parameters File**: `model_parameters_mle.json`

### Tier 3: Modern Classical
- **Purpose**: True clinical dose-response modeling
- **Models**: Multivariable logistic regression
- **Features**: DVH metrics (mean_dose, gEUD, V30, V50, D20) + clinical factors (optional)
- **Output**: `NTCP_LOGISTIC`
- **Training**: 70/30 split, L2 regularization, bootstrap variable stability

### Tier 4: AI
- **Purpose**: Maximum achievable predictability
- **Models**: ANN, XGBoost, SHAP, uNTCP, CCS
- **Output**: `ML_ANN`, `ML_XGBoost`, `uNTCP`, `CCS`
- **Status**: Existing tier, unchanged

## Scientific Value

This upgrade enables **QUANTEC-2.0-grade science** by providing:

1. **Literature Transportability** (Tier 1): Validates if published parameters work on your cohort
2. **Cohort Calibration** (Tier 2): Shows if your patients differ from literature
3. **Clinical Dose-Response** (Tier 3): Captures true multivariable relationships
4. **Maximum Predictability** (Tier 4): Shows best achievable performance
5. **Biological Coherence** (CCS): Validates model predictions against training distribution
6. **Uncertainty Awareness** (uNTCP): Provides confidence intervals

## Output File Locations

```
out2/
├── code3_output/              # Existing (unchanged)
│   └── ntcp_results.xlsx
├── tiered_output/              # New
│   ├── tiered_ntcp_results.xlsx
│   ├── NTCP_4Tier_Master.xlsx
│   ├── ml_validation.xlsx
│   ├── model_parameters_mle.json
│   ├── dose_response_tiers.png
│   └── dose_response_{Organ}.png
└── output/                     # New
    └── README_outputs.md
```

## Dependencies

No new dependencies required. Uses existing packages:
- numpy, pandas, scipy, sklearn, matplotlib, seaborn
- openpyxl (for Excel writing)

## Testing

To verify the upgrade works:

1. Run existing pipeline to generate code3 outputs
2. Run tiered analysis:
   ```bash
   python tiered_ntcp_analysis.py \
       --code3_output out2/code3_output \
       --dvh_dir out2/code1_output/dDVH_csv \
       --output_dir out2/tiered_output
   ```
3. Check outputs:
   - `tiered_ntcp_results.xlsx` should have new columns
   - `NTCP_4Tier_Master.xlsx` should have 6 sheets
   - `dose_response_tiers.png` should exist

## Troubleshooting

### Issue: "CohortConsistencyScore not available"
- **Solution**: This is a warning, not an error. CCS calculation will be skipped.

### Issue: "Insufficient data for MLE fitting"
- **Solution**: Need at least 10 samples per organ for MLE. This is expected for small cohorts.

### Issue: "Insufficient data for logistic regression"
- **Solution**: Need at least 20 samples per organ for logistic regression. This is expected for small cohorts.

### Issue: Import errors
- **Solution**: Make sure you're running from the repository root directory, or add the repository to PYTHONPATH.

## Future Enhancements

Potential future additions (not implemented):
- RS Poisson MLE fitting (requires DVH loading in tiered script)
- Cross-validation for logistic regression
- Additional clinical features
- Ensemble models combining tiers

## Contact

For issues or questions, see the main README.md or open an issue on the repository.

