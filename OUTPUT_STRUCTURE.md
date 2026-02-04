# py_ntcpx - Complete Output Structure and Information

**Version**: 3.0.0 (v3.0.1–3.0.3 fixes merged)  
**Date**: February 2026

---

## Table of Contents

1. [Overview](#overview)
2. [Output Directory Structure](#output-directory-structure)
3. [Step-by-Step Outputs](#step-by-step-outputs)
4. [File Descriptions](#file-descriptions)
5. [Output Formats](#output-formats)
6. [Usage Guide](#usage-guide)

---

## Overview

The py_ntcpx v3.0.0 pipeline generates comprehensive outputs organized by processing step. All outputs are saved in the specified output directory (default: `out2/`), with each step creating its own subdirectory.

### Key Output Categories

1. **Data Processing**: DVH preprocessing, clinical reconciliation, biological DVH
2. **NTCP Analysis**: Classical models, ML models, uncertainty quantification
3. **Quality Assurance**: QA reports, CCS warnings, safety reports
4. **Clinical Analysis**: Factor associations, correlation matrices
5. **Visualizations**: High-resolution plots (600 DPI), publication diagrams
6. **Explainable AI**: SHAP explanations, LIME local interpretability (v3.0.0)
7. **Publication Materials**: Tables, figures, reproducibility information

---

## Output Directory Structure

```
out2/  (or custom --output_dir)
├── code0_output/                    # Step 0: Clinical Reconciliation
│   └── clinical_reconciled.xlsx
│
├── code1_output/                     # Step 1: DVH Preprocessing
│   ├── cDVH_csv/                     # Cumulative DVH (CSV)
│   ├── dDVH_csv/                     # Differential DVH (CSV)
│   └── processed_dvh.xlsx
│
├── code2_output/                     # Step 2: DVH Plotting & Summary
│   ├── cDVH_plots/                   # Cumulative DVH plots
│   ├── dDVH_plots/                   # Differential DVH plots
│   ├── overlay_plots/                # Overlay plots
│   └── tables/                       # Summary tables
│
├── code2_bDVH_output/                # Step 2b: Biological DVH
│   ├── bDVH_csv/                     # Biological DVH (CSV)
│   └── bDVH_plots/                   # Biological DVH plots
│
├── code3_output/                     # Step 3: NTCP Analysis with ML
│   ├── models/                       # Trained models (PKL)
│   ├── plots/                        # NTCP plots
│   ├── DR_plots/                     # Dose-response plots
│   ├── clinical_factors_analysis/     # Clinical factors analysis
│   ├── quantec_validation/           # QUANTEC validation
│   ├── manuscript_materials/         # Manuscript figures
│   ├── ntcp_results.xlsx             # Main results (incl. ML CV-AUC in Summary/Performance)
│   ├── ml_validation.xlsx            # ML validation metrics (CV-AUC for ANN, XGBoost) [v3.0.1]
│   ├── enhanced_ntcp_calculations.csv
│   ├── enhanced_summary_performance.csv
│   ├── local_biological_parameters.json
│   ├── local_classical_parameters.json
│   └── enhanced_analysis_report.txt
│
├── code4_output/                     # Step 4: QA Reporter
│   ├── tables/                       # QA tables (Table_X includes AUC_95CI in v3.0.2+)
│   ├── comprehensive_report.docx    # Comprehensive QA report
│   └── qa_summary_tables.xlsx
│
├── code6_output/                     # Step 6: Publication Diagrams
│   ├── figure_workflow.png/svg
│   ├── figure_methodology.png/svg
│   ├── figure_model_spectrum.png/svg
│   ├── figure_shap_integration.png/svg
│   ├── figure_xai_shap.png/svg
│   ├── figure_feature_taxonomy.png/svg
│   └── figure_feature_model_matrix.png/svg
│
├── code7_shap/                       # Step 7: SHAP & LIME Analysis (v3.0.0)
│   └── {Organ}/                      # Organ-specific outputs
│       ├── ANN/
│       │   ├── shap_beeswarm.png
│       │   ├── shap_bar.png
│       │   ├── shap_table.xlsx
│       │   ├── shap_stability_report.xlsx
│       │   ├── lime_explanation_{patient_id}.html
│       │   └── lime_explanation_{patient_id}.png
│       └── XGBoost/
│           ├── shap_beeswarm.png
│           ├── shap_bar.png
│           ├── shap_table.xlsx
│           ├── shap_stability_report.xlsx
│           ├── lime_explanation_{patient_id}.html
│           └── lime_explanation_{patient_id}.png
│
├── tiered_output/                    # Step 3c: Tiered NTCP Analysis
│   ├── NTCP_4Tier_Master.xlsx
│   ├── tiered_ntcp_results.xlsx
│   ├── ml_validation.xlsx
│   ├── model_parameters_mle.json
│   ├── dose_response_Parotid.png
│   └── dose_response_tiers.png
│
├── supp_results_summary_output/       # Step 8: Publication Tables
│   └── publication_tables.xlsx
│
├── publication_bundle_YYYYMMDD/      # Step 9 (v3.0.3): Aggregated publication materials
│   ├── manuscript_materials/
│   ├── tables/
│   ├── figures/
│   ├── tiered/
│   └── README.md
│
└── contracts/                        # Data Contracts (QA)
    ├── Step1_DVHRegistry.xlsx
    ├── Step2b_bDVHRegistry.xlsx
    ├── Step3_NTCPDataset.xlsx
    └── Step4_QAReport.xlsx
```

---

## Step-by-Step Outputs

### Step 0: Clinical Reconciliation (`code0_output/`)

**Purpose**: Validates and standardizes clinical data according to Clinical Contract v2.

**Outputs**:
- `clinical_reconciled.xlsx`: Standardized clinical data with validated columns

**Key Features**:
- Validates required columns
- Standardizes data types
- Creates data contract for downstream steps

---

### Step 1: DVH Preprocessing (`code1_output/`)

**Purpose**: Converts raw DVH text files to standardized CSV format.

**Outputs**:
- `cDVH_csv/`: Cumulative DVH files (one per patient-organ)
  - Format: `{PatientID}_{Organ}.csv`
  - Columns: `Dose (Gy)`, `Volume (%)`
- `dDVH_csv/`: Differential DVH files (one per patient-organ)
  - Format: `{PatientID}_{Organ}.csv`
  - Columns: `Dose (Gy)`, `Volume (%)`
- `processed_dvh.xlsx`: Combined DVH data in Excel format

**Key Features**:
- Validates DVH normalization (V0 = 100%)
- Checks monotonicity
- Handles missing data

---

### Step 2: DVH Plotting & Summary (`code2_output/`)

**Purpose**: Generates DVH visualizations and summary tables.

**Outputs**:
- `cDVH_plots/`: Cumulative DVH plots
  - Formats: PNG (600 DPI), SVG (vector)
  - One plot per patient-organ
- `dDVH_plots/`: Differential DVH plots
  - Formats: PNG (600 DPI), SVG (vector)
  - One plot per patient-organ
- `overlay_plots/`: Overlay plots (multiple DVHs on one plot)
  - Formats: PNG (600 DPI), SVG (vector)
- `tables/`: Summary tables
  - DVH metrics summary (Excel)

**Key Features**:
- High-resolution plots (600 DPI) for publication
- Vector formats (SVG) for scalability
- Comprehensive summary tables

---

### Step 2b: Biological DVH (`code2_bDVH_output/`)

**Purpose**: Generates EQD2-based biological dose-volume histograms.

**Outputs**:
- `bDVH_csv/`: Biological DVH files (CSV)
  - Format: `{PatientID}_{Organ}.csv`
  - Columns: `Dose (Gy)`, `Volume (%)` (EQD2-corrected)
- `bDVH_plots/`: Biological DVH plots
  - Formats: PNG (600 DPI)
  - One plot per patient-organ

**Key Features**:
- EQD2 conversion using α/β ratio
- Biological dose-response modeling
- Publication-ready visualizations

---

### Step 3: NTCP Analysis with ML (`code3_output/`)

**Purpose**: Comprehensive NTCP analysis including classical models, ML models, and uncertainty quantification.

#### Main Results Files

- `ntcp_results.xlsx`: **Primary output file**
  - All model predictions (classical + ML)
  - Summary by Organ and Performance Matrix (incl. ML CV-AUC columns, v3.0.1)
  - Individual patient predictions
  - CCS scores and `CCS_Warning_Flag` (v3.0.0)
  - Uncertainty quantification (uNTCP, CI)
  - Feature values used for predictions

- `ml_validation.xlsx`: **ML validation metrics** (v3.0.1)
  - CV_AUC_Mean, CV_AUC_Std for ANN and XGBoost per organ
  - N_Samples, N_Events, Validation_Method
  - Used by reports and manuscript materials

- `enhanced_ntcp_calculations.csv`: Detailed NTCP calculations (CSV format)

- `enhanced_summary_performance.csv`: Model performance summary

#### Model Files (`models/`)

- `{Organ}_ANN_model.pkl`: Trained ANN model (joblib)
- `{Organ}_XGBoost_model.pkl`: Trained XGBoost model (joblib)
- `{Organ}_scaler.pkl`: Feature scaler (StandardScaler)
- `{Organ}_feature_matrix.csv`: Feature matrix used for training
- `{Organ}_feature_registry.json`: Feature names and metadata

#### Parameter Files

- `local_biological_parameters.json`: Biological model parameters (TD50, k)
- `local_classical_parameters.json`: Classical NTCP parameters (LKB, RS Poisson)
- `model_parameters_mle.json`: MLE-refitted parameters

#### Visualizations (`plots/`)

- `Figure1_LKB_gEUD_DR_{Organ}.png/svg`: LKB dose-response curve
- `Figure2_RS_gEUD_DR_{Organ}.png/svg`: RS Poisson dose-response curve
- `Figure3_Biological_MeanDose_DR_{Organ}.png/svg`: Biological dose-response
- `Figure4_Predicted_NTCP_Comparison_{Organ}.png/svg`: Model comparison
- `Figure5_Observed_vs_Predicted_{Organ}.png/svg`: Calibration plot
- `{Organ}_ROC.png/svg`: ROC curves
- `{Organ}_calibration.png/svg`: Calibration curves
- `{Organ}_ROC_calibration_combined.png/svg`: Combined ROC and calibration

#### Clinical Factors Analysis (`clinical_factors_analysis/`)

- `categorical_factors_analysis.xlsx`: Categorical factor associations
- `continuous_factors_analysis.xlsx`: Continuous factor correlations
- `organ_specific_analysis.xlsx`: Organ-specific factor effects
- `correlation_matrix.csv`: Full correlation matrix
- `correlation_matrix.png`: Correlation heatmap (600 DPI)
- `clinical_factors_analysis_report.txt`: Comprehensive text report
- `categorical_analysis_{Factor}.png`: Categorical factor plots
- `continuous_analysis_{Factor}.png`: Continuous factor plots

#### QUANTEC Validation (`quantec_validation/`)

- `quantec_validation_{Organ}.xlsx`: QUANTEC risk stratification

#### Manuscript Materials (`manuscript_materials/`)

- `figures/Figure4_DR_Reference.png/svg`: Reference dose-response figure

**Key Features**:
- 4-tier NTCP framework (Legacy, MLE, Modern, AI)
- Adaptive CCS thresholds (v3.0.0)
- CCS warnings instead of blocking (v3.0.0)
- Comprehensive uncertainty quantification

---

### Step 4: QA Reporter (`code4_output/`)

**Purpose**: Quality assurance reporting and validation.

**Outputs**:
- `comprehensive_report.docx`: Comprehensive QA report (Word document)
  - Executive summary
  - Model performance metrics
  - CCS warnings summary (v3.0.0)
  - Clinical safety recommendations
- `qa_summary_tables.xlsx`: QA summary tables
  - Model performance metrics
  - CCS statistics
  - Safety flags
- `tables/`: Additional QA tables
  - CSV and Excel formats
  - Markdown summaries

**Key Features**:
- CCS warning summaries (v3.0.0)
- Adaptive threshold reporting
- Clinical safety recommendations

---

### Step 6: Publication Diagrams (`code6_output/`)

**Purpose**: Generates high-quality publication-ready figures.

**Outputs** (all in PNG 600 DPI and SVG formats):

- `figure_workflow.png/svg`: Pipeline workflow diagram
- `figure_methodology.png/svg`: Methodology overview
- `figure_model_spectrum.png/svg`: NTCP model spectrum
- `figure_shap_integration.png/svg`: SHAP integration diagram
- `figure_xai_shap.png/svg`: Explainable AI overview
- `figure_feature_taxonomy.png/svg`: Feature taxonomy
- `figure_feature_model_matrix.png/svg`: Feature-model matrix

**Key Features**:
- 600 DPI resolution for publication
- Vector formats (SVG) for scalability
- Journal-ready figures

---

### Step 7: SHAP & LIME Analysis (`code7_shap/`) - v3.0.0

**Purpose**: Explainable AI analysis using SHAP (global) and LIME (local).

**Outputs** (organized by organ and model):

#### SHAP Outputs

- `shap_beeswarm.png`: Beeswarm plot showing feature importance distribution
  - Color-coded by feature value
  - High-resolution (600 DPI)
  
- `shap_bar.png`: Bar plot of mean |SHAP| values
  - Sorted by importance
  - Clinical standard visualization

- `shap_table.xlsx`: Complete SHAP values table
  - Per-sample, per-feature contributions
  - Excel format for analysis

- `shap_stability_report.xlsx`: Bootstrap stability analysis (v3.0.0)
  - Feature consistency metrics
  - Stability warnings

#### LIME Outputs (v3.0.0)

- `lime_explanation_{patient_id}.html`: Interactive LIME explanation
  - Per-patient feature contributions
  - Feature value impacts
  - Opens in web browser

- `lime_explanation_{patient_id}.png`: Static LIME visualization
  - High-resolution (600 DPI)
  - Publication-ready

**Representative Patients** (automatically selected):
- Highest predicted NTCP patient
- Median predicted NTCP patient
- Lowest predicted NTCP patient

**Key Features**:
- Fixed XGBoost SHAP (model-agnostic explainer) (v3.0.0)
- Improved ANN SHAP stability warnings (v3.0.0)
- LIME local interpretability (v3.0.0)
- Complementary global (SHAP) and local (LIME) explanations

---

### Step 3c: Tiered NTCP Analysis (`tiered_output/`)

**Purpose**: Four-tier NTCP framework analysis.

**Outputs**:
- `NTCP_4Tier_Master.xlsx`: Master tiered analysis report
  - Tier 1: Legacy-A (QUANTEC fixed parameters)
  - Tier 2: Legacy-B (MLE-refitted)
  - Tier 3: Modern Classical (logistic regression)
  - Tier 4: AI Models (ANN, XGBoost)

- `tiered_ntcp_results.xlsx`: Detailed tiered results

- `ml_validation.xlsx`: ML model validation metrics

- `model_parameters_mle.json`: MLE-refitted parameters

- `dose_response_Parotid.png`: Dose-response curves

- `dose_response_tiers.png`: Tier comparison plot

**Key Features**:
- Comprehensive tier comparison
- Model performance across tiers
- Parameter stability analysis

---

### Step 8: Publication Tables (`supp_results_summary_output/`)

**Purpose**: Auto-generates publication-ready tables.

**Outputs**:
- `publication_tables.xlsx`: Comprehensive publication tables
  - Table 1: Cohort Characteristics
  - Table 2: NTCP Performance
  - Table 3: Uncertainty QA
  - Appendix A1: Model & Equation Reference
  - Appendix A2: Computational Reproducibility

**Key Features**:
- LaTeX-ready formats
- Journal submission standards
- Comprehensive metadata

---

### Data Contracts (`contracts/`)

**Purpose**: Quality assurance data contracts for each step.

**Outputs**:
- `Step1_DVHRegistry.xlsx`: DVH data contract
- `Step2b_bDVHRegistry.xlsx`: Biological DVH contract
- `Step3_NTCPDataset.xlsx`: NTCP dataset contract
- `Step4_QAReport.xlsx`: QA report contract

**Key Features**:
- Validates data integrity
- Ensures reproducibility
- Documents data transformations

---

## File Descriptions

### Excel Files (.xlsx)

**Primary Results**:
- `ntcp_results.xlsx`: Main NTCP results with all predictions
- `NTCP_4Tier_Master.xlsx`: Four-tier analysis master report
- `qa_summary_tables.xlsx`: QA summary tables
- `publication_tables.xlsx`: Publication-ready tables

**Analysis Files**:
- `enhanced_ntcp_calculations.csv`: Detailed calculations (CSV)
- `enhanced_summary_performance.csv`: Performance summary (CSV)
- `categorical_factors_analysis.xlsx`: Categorical factors
- `continuous_factors_analysis.xlsx`: Continuous factors
- `organ_specific_analysis.xlsx`: Organ-specific analysis

**SHAP/LIME Files**:
- `shap_table.xlsx`: SHAP values table
- `shap_stability_report.xlsx`: Stability analysis

### JSON Files (.json)

**Parameters**:
- `local_biological_parameters.json`: Biological parameters
- `local_classical_parameters.json`: Classical parameters
- `model_parameters_mle.json`: MLE parameters
- `{Organ}_feature_registry.json`: Feature metadata

### Image Files

**Formats**:
- **PNG**: High-resolution (600 DPI) for publication
- **SVG**: Vector format for scalability

**Categories**:
- DVH plots (cumulative, differential, biological)
- NTCP plots (dose-response, calibration, ROC)
- SHAP visualizations (beeswarm, bar plots)
- LIME visualizations (per-patient explanations)
- Publication diagrams (workflow, methodology)

### Model Files (.pkl)

**Trained Models**:
- `{Organ}_ANN_model.pkl`: ANN model (joblib)
- `{Organ}_XGBoost_model.pkl`: XGBoost model (joblib)
- `{Organ}_scaler.pkl`: Feature scaler (StandardScaler)

**Usage**:
```python
import joblib
model = joblib.load('{Organ}_ANN_model.pkl')
scaler = joblib.load('{Organ}_scaler.pkl')
```

### HTML Files (.html)

**LIME Explanations** (v3.0.0):
- `lime_explanation_{patient_id}.html`: Interactive LIME explanation
- Opens in web browser
- Shows per-patient feature contributions

### Text Files (.txt)

**Reports**:
- `enhanced_analysis_report.txt`: Enhanced analysis report
- `clinical_factors_analysis_report.txt`: Clinical factors report

### Word Documents (.docx)

**Reports**:
- `comprehensive_report.docx`: Comprehensive QA report

---

## Output Formats

### CSV Format

**Use Cases**:
- DVH data (cDVH_csv/, dDVH_csv/, bDVH_csv/)
- Enhanced calculations
- Feature matrices
- Correlation matrices

**Structure**:
- Header row with column names
- One row per sample/patient
- Standardized column names

### Excel Format (.xlsx)

**Use Cases**:
- Main results tables
- Summary tables
- Multi-sheet workbooks
- Publication tables

**Structure**:
- Multiple sheets for different analyses
- Formatted for readability
- Includes metadata

### PNG Format

**Specifications**:
- Resolution: 600 DPI
- Color space: RGB
- Bit depth: 8-bit or 16-bit

**Use Cases**:
- Publication figures
- Presentations
- Reports

### SVG Format

**Specifications**:
- Vector format (scalable)
- XML-based
- Editable in vector graphics software

**Use Cases**:
- Publication figures (scalable)
- Further editing
- Web display

### JSON Format

**Structure**:
- Hierarchical key-value pairs
- Human-readable
- Machine-parseable

**Use Cases**:
- Model parameters
- Feature metadata
- Configuration data

---

## Usage Guide

### Finding Specific Outputs

1. **Main Results**: `code3_output/ntcp_results.xlsx`
2. **SHAP Explanations**: `code7_shap/{Organ}/{Model}/`
3. **LIME Explanations**: `code7_shap/{Organ}/{Model}/lime_explanation_*.html`
4. **Publication Tables**: `supp_results_summary_output/publication_tables.xlsx`
5. **QA Reports**: `code4_output/comprehensive_report.docx`

### Interpreting Outputs

#### NTCP Results (`ntcp_results.xlsx`)

**Key Columns**:
- `NTCP_{Model}`: Model predictions (0-1 probability)
- `CCS`: Cohort Consistency Score
- `CCS_Warning_Flag`: Boolean (True if CCS below adaptive threshold) (v3.0.0)
- `uNTCP_mean`: Mean uncertainty-aware NTCP
- `uNTCP_std`: Standard deviation
- `CI_lower`, `CI_upper`: 95% confidence intervals

#### SHAP Values (`shap_table.xlsx`)

**Structure**:
- Rows: Patients
- Columns: Features
- Values: SHAP contributions (positive = risk-increasing, negative = protective)

#### LIME Explanations (`lime_explanation_*.html`)

**Interpretation**:
- Open in web browser
- Shows which features most influenced this patient's prediction
- Feature values and their impacts
- Local model behavior

### Accessing Trained Models

```python
import joblib
import pandas as pd

# Load model
model = joblib.load('code3_output/models/Parotid_ANN_model.pkl')
scaler = joblib.load('code3_output/models/Parotid_scaler.pkl')

# Load feature registry
import json
with open('code3_output/models/Parotid_feature_registry.json', 'r') as f:
    feature_registry = json.load(f)

# Prepare features
features = ['V30', 'V45', 'V5']  # Example features
X = pd.DataFrame([[values]], columns=features)
X_scaled = scaler.transform(X)

# Predict
ntcp = model.predict_proba(X_scaled)[0, 1]
```

### Publication Workflow

1. **Figures**: Use `code6_output/` for methodology figures
2. **Tables**: Use `supp_results_summary_output/publication_tables.xlsx`
3. **SHAP Plots**: Use `code7_shap/{Organ}/{Model}/shap_*.png` (600 DPI)
4. **LIME Explanations**: Include in supplementary materials
5. **Reproducibility**: Use `publication_tables.xlsx` Appendix A2

---

## Version-Specific Features (v3.0.0)

### Adaptive CCS Thresholds

- **Threshold Values**:
  - n < 30: 0.0 (disable filtering)
  - n 30-100: 0.1 (conservative)
  - n ≥ 100: 0.2 (standard)

- **Output**: `CCS_Warning_Flag` column (boolean) in `ntcp_results.xlsx`

### LIME Explanations

- **Location**: `code7_shap/{Organ}/{Model}/lime_explanation_*.{html,png}`
- **Patients**: Automatically selected (highest, median, lowest NTCP)
- **Formats**: HTML (interactive) and PNG (static)

### Enhanced SHAP

- **XGBoost**: Fixed model-agnostic explainer
- **Stability Reports**: `shap_stability_report.xlsx`
- **Improved Warnings**: CCS-aware stability checks

---

## Troubleshooting

### Missing Outputs

1. **Check Pipeline Execution**: Ensure all steps completed successfully
2. **Check Logs**: Review execution logs for errors
3. **Verify Inputs**: Ensure input files are correctly formatted

### File Size Issues

- **Large Excel Files**: Use CSV alternatives where available
- **Image Files**: PNG files are large (600 DPI) - use SVG for editing

### Model Loading

- **Dependencies**: Ensure all dependencies are installed
- **Version Compatibility**: Models saved with joblib may require compatible versions

---

## References

- **Pipeline Documentation**: See `README.md` and `ARCHITECTURE_REPORT.md`
- **Version Information**: See `CHANGELOG_v3.0.0.md`

---

**Document Version**: 3.0.0  
**Last Updated**: February 2026

