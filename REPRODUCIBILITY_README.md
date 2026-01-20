# Reproducibility Guide for py_ntcpx v2.0

This document provides instructions for reproducing all results from the py_ntcpx v2.0 pipeline.

## Random Seeds

All random operations use fixed seeds for reproducibility:

- **Main pipeline**: `random_seed = 42` (default)
- **Data splitting**: `random_seed = 42`
- **ML models**: `random_seed = 42`
- **Monte Carlo sampling**: `random_seed = 42`
- **Bootstrap sampling**: Uses `np.random.seed(42)` before each bootstrap

### Setting Random Seeds

The random seed can be configured in:
- `config/pipeline_config.yaml`: `pipeline.random_seed`
- `code3_ntcp_analysis_ml.py`: `MachineLearningModels(random_state=42)`
- `src/validation/data_splitter.py`: `PatientDataSplitter(random_seed=42)`

## Dependencies

All dependencies are specified in `requirements.txt`. For exact reproducibility, use:

```bash
pip install -r requirements.txt
```

### Key Dependencies and Versions

- Python >= 3.8
- numpy >= 1.20.0
- pandas >= 1.3.0
- scikit-learn >= 0.24.0
- xgboost >= 1.5.0 (optional, for XGBoost models)
- matplotlib >= 3.3.0
- scipy >= 1.7.0

## Data Requirements

### Input Data Format

1. **DVH Files**: CSV files named `{PrimaryPatientID}_{Organ}.csv`
   - Required columns: `Dose[Gy]`, `Volume[cm3]` or `Volume[%]`
   
2. **Clinical Data**: Excel file from `code0_clinical_reconciliation.py`
   - Required columns: `PatientID`, `Observed_Toxicity`
   - Optional: `Age`, `Chemotherapy`, `T_Stage`, etc.

### Test Data

For testing, use synthetic data (no real patient data):

```bash
python tests/test_data/generate_synthetic_data.py
```

This creates a 54-patient synthetic dataset in `tests/test_data/synthetic/`.

## Running the Pipeline

### Step 0: Clinical Reconciliation

```bash
python code0_clinical_reconciliation.py \
    --registry <registry_file.xlsx> \
    --clinical <clinical_file.xlsx> \
    --output_dir code0_output
```

### Step 1: DVH Preprocessing

```bash
python code1_dvh_preprocess.py \
    --dvh_dir <dvh_directory> \
    --output_dir code1_output
```

### Step 2: Biological DVH Calculation

```bash
python code2_bDVH.py \
    --input_dir code1_output \
    --output_dir code2_output
```

### Step 3: NTCP Analysis with ML

```bash
python code3_ntcp_analysis_ml.py \
    --dvh_dir code2_output \
    --clinical_file code0_output/clinical_reconciled.xlsx \
    --output_dir code3_output \
    --random_seed 42
```

### Step 4: QA Reporting

```bash
python code4_ntcp_output_QA_reporter.py \
    --input code3_output \
    --report_outdir QA_results
```

## Reproducing Specific Results

### Reproducing Train-Test Split

The patient-level split is deterministic with `random_seed=42`:

```python
from src.validation.data_splitter import PatientDataSplitter

splitter = PatientDataSplitter(random_seed=42, test_size=0.2)
train_df, test_df = splitter.create_splits(
    patient_df,
    patient_id_col='PrimaryPatientID',
    outcome_col='Observed_Toxicity'
)
```

### Reproducing ML Model Results

ML models use fixed random seeds:

```python
from src.models.machine_learning.ml_models import OverfitResistantMLModels

ml_model = OverfitResistantMLModels(
    n_features=10,
    n_samples=54,
    n_events=15,
    random_seed=42
)
```

### Reproducing AUC Confidence Intervals

Bootstrap CIs use fixed seed before each bootstrap:

```python
from src.metrics.auc_calculator import calculate_auc_with_ci

np.random.seed(42)  # Set seed before calculation
auc_val, auc_ci = calculate_auc_with_ci(
    y_true, y_pred,
    method='bootstrap',
    n_bootstraps=2000
)
```

## Output Files

All outputs are saved with timestamps and random seed information:

- `code3_output/ntcp_results.xlsx`: Main results
- `code3_output/figures/`: Publication-ready figures (600 DPI)
- `QA_results/comprehensive_report.docx`: QA report
- `QA_results/qa_summary_tables.xlsx`: Summary tables

## Verification

### Check Reproducibility

1. Run pipeline twice with same seed
2. Compare output files (should be identical)
3. Verify patient splits are identical
4. Verify model predictions are identical

### Expected Outputs

With `random_seed=42` and 54-patient dataset:

- Train set: ~43 patients (80%)
- Test set: ~11 patients (20%)
- EPV: ~1.5 events per variable (with 10 features)
- Feature selection: Typically selects 5-8 features

## Troubleshooting

### Different Results Between Runs

1. **Check random seed**: Ensure `random_seed=42` is set consistently
2. **Check Python version**: Use same Python version
3. **Check dependencies**: Use same package versions
4. **Check data**: Ensure input data is identical

### Platform Differences

Results may vary slightly between platforms due to:
- Floating-point precision differences
- Random number generator implementations
- Numerical library optimizations

For exact reproducibility, use:
- Same operating system
- Same Python version
- Same package versions
- Same hardware architecture

## Citation

When using this pipeline, please cite:

```
py_ntcpx v2.0: Publication-ready NTCP analysis pipeline
Random seed: 42
Reproducibility: All random operations use fixed seeds
```

## Contact

For reproducibility issues, please:
1. Check this README
2. Verify random seeds are set correctly
3. Check dependency versions
4. Review output logs for warnings

---

*Last updated: 2024*
*Version: 2.0.0*
