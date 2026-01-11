# NTCP Analysis and Machine Learning Pipeline for Head & Neck Cancer

A comprehensive Python pipeline for Normal Tissue Complication Probability (NTCP) modeling and machine learning prediction specifically designed for **Head & Neck (H&N) cancer** radiotherapy. The pipeline processes dose-volume histogram (DVH) data, computes traditional and novel NTCP models, applies machine learning algorithms, and includes quality assurance modules with uncertainty quantification.

Designed for reproducible research and publication-quality outputs (600 DPI figures, tidy tables).

**Maintainer:** K. Mondal  
**Version:** v1.0  
**Software Name:** py_ntcpx  
**License:** MIT

## Supported Organs at Risk (OARs)

The pipeline is optimized for the following H&N organs at risk:
- **Parotid glands** (xerostomia)
- **Larynx** (dysphagia)
- **Spinal cord** (myelopathy)
- **Oral cavity** (mucositis)
- **Submandibular glands** (xerostomia)
- **Pharyngeal constrictors** (dysphagia)
- **Mandible** (osteoradionecrosis)
- **Esophagus** (esophagitis)
- **Brachial plexus** (plexopathy)
- **Cochlea** (hearing loss)
- **Optic nerve** (neuropathy)
- **Optic chiasm** (neuropathy)
- **Brainstem** (necrosis)

## Repository Contents
- `code1_dvh_preprocess.py` — Parse TPS DVH text exports, standardize outputs; generates `cDVH_csv/` and `dDVH_csv/` plus workbook summary.
- `code2_dvh_plot_and_summary.py` — Compute dose metrics and create cDVH/dDVH plots; writes cohort summary tables.
- `code2_bDVH.py` — **NEW**: Generate biological DVH (bDVH) from physical DVH using BED/EQD2/gEUD transformations. Enables direct comparison across fractionation schemes and supports biological NTCP modeling.
- `code3_ntcp_analysis_ml.py` — Compute traditional NTCP models (LKB log-logit, LKB probit, RS Poisson), novel probabilistic models (Probabilistic gEUD, Monte Carlo NTCP), and train/evaluate ML models (ANN, XGBoost). Includes quality assurance with uncertainty-aware NTCP (uNTCP) and cohort consistency score (CCS).
- `code4_ntcp_output_QA_reporter.py` — QA the analysis outputs; flags inflated patient counts, unrealistic NTCPs, and overfitting/leakage symptoms; generates a DOCX report.
- `code5_ntcp_factors_analysis.py` — Merge clinical factors with NTCP outputs; perform categorical/continuous analyses and plots.
- `code6_publication_diagrams.py` — Generate publication-quality workflow diagrams and methodology visualizations (1200 DPI).
- `supp_results_summary.py` — **NEW**: Auto-generate all publication tables (Tables 1-4) and appendices (A1-A2) for journal submission.
- `run_pipeline.py` — **NEW**: Single-command orchestration script for complete pipeline execution.

## NTCP Models & Machine Learning Overview

### Traditional NTCP Models
1. **LKB Log-Logistic Model**: Uses generalized equivalent uniform dose (gEUD) with log-logistic link function
2. **LKB Probit Model**: Uses effective volume with probit link function
3. **RS Poisson Model**: Relative seriality model with Poisson statistics

### Novel Biological NTCP Models
1. **Probabilistic gEUD Model**: Quantifies uncertainty in NTCP predictions by sampling from parameter distributions (Niemierko 1999, Brodin 2017)
2. **Monte Carlo NTCP Model**: Incorporates DVH and parameter uncertainty through Monte Carlo sampling (Fenwick 2001)

### Machine Learning Models
- **Artificial Neural Network (ANN)**: Multi-layer perceptron with proper cross-validation
- **XGBoost**: Gradient boosting with hyperparameter optimization
- **Random Forest** (optional)
- **LightGBM** (optional)

All ML models include:
- Stratified k-fold cross-validation
- Feature importance analysis
- Overfitting prevention measures
- Performance metrics (ROC-AUC, Brier score, calibration)

### Quality Assurance Modules

#### Uncertainty-Aware NTCP (uNTCP)
Propagates parameter uncertainties into NTCP predictions using first-order Taylor expansion (Delta method):

```
σ²_NTCP = Σⱼ (∂NTCP/∂θⱼ)² × σ²_θⱼ
```

where θⱼ represents model parameters (n, TD50, m). Outputs include:
- NTCP prediction with standard deviation
- 95% confidence intervals (CI_L, CI_U)
- Per-parameter uncertainty contributions
- Clinical interpretation of uncertainty levels

#### Cohort Consistency Score (CCS)
Detects out-of-distribution patients using Mahalanobis distance:

```
CCS = exp(-½ D²_Mahalanobis)
D² = (X_new - μ)ᵀ Σ⁻¹ (X_new - μ)
```

where μ and Σ are the mean and covariance matrix of the training cohort. Interpretation:
- **CCS > 0.95**: Within cohort distribution (predictions reliable)
- **CCS 0.80-0.95**: Minor deviation (use with caution)
- **CCS 0.50-0.80**: Moderate deviation (verify parameters)
- **CCS 0.20-0.50**: Significant deviation (unreliable)
- **CCS < 0.20**: Out-of-distribution (**DO_NOT_USE** for clinical decisions)

## Install
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

## Biological DVH (bDVH) Definition

**Biological DVH (bDVH)** is defined as:

> A DVH transformed from physical dose (Gy) to a **biologically effective dose domain** using standard radiobiological formalisms, enabling direct comparison across fractionation schemes and supporting biological NTCP modeling.

### Transformation Methods

1. **BED-based bDVH**: Uses Biologically Effective Dose (BED) transformation
   - BED = nd(1 + d/(α/β))
   - Where n = number of fractions, d = dose per fraction, α/β = organ-specific ratio

2. **EQD2-based bDVH**: Uses Equivalent Dose in 2 Gy fractions (EQD2) transformation
   - EQD2 = BED / (1 + 2/(α/β))
   - Enables direct comparison to standard fractionation schemes

3. **gEUD-mapped bDVH**: Remaps dose bins via voxel-wise gEUD weighting
   - Incorporates organ-specific gEUD parameter 'a'
   - Supports parallel (negative a) and serial (positive a) organ models

### QUANTEC/RTOG Benchmarking

The pipeline includes QUANTEC (Quantitative Analyses of Normal Tissue Effects in the Clinic) and RTOG (Radiation Therapy Oncology Group) recommended dose constraints for major H&N organs:

- **Parotid**: Mean dose <26 Gy (recommended), <20 Gy (constraint)
- **Spinal Cord**: Max dose <50 Gy (recommended), <45 Gy (constraint)
- **Larynx**: Mean dose <44 Gy (recommended), <40 Gy (constraint)
- **Oral Cavity**: Mean dose <40 Gy (recommended), <35 Gy (constraint)

All dosimetric metrics are automatically benchmarked against these guidelines in publication tables.

**References:**
- Fowler JF (1989). The linear-quadratic formula and progress in fractionated radiotherapy. *Br J Radiol* 62(740):679-694.
- Bentzen SM et al. (2010). Quantitative Analyses of Normal Tissue Effects in the Clinic (QUANTEC): an introduction to the issue. *Int J Radiat Oncol Biol Phys* 76(3 Suppl):S3-S9.
- Emami B et al. (1991). Tolerance of normal tissue to therapeutic irradiation. *Int J Radiat Oncol Biol Phys* 21(1):109-122.

## Reproducibility Statement

This pipeline is designed for **reproducible research** and journal-grade analysis:

- All scripts are deterministic with fixed random seeds where applicable
- Complete computational environment is documented (OS, Python version, library versions)
- All intermediate outputs are preserved for verification
- Publication tables include full reproducibility metadata (Appendix A2)
- Single-command execution via `run_pipeline.py` ensures consistent results

## Step-by-Step Usage

### Option A: Single-Command Pipeline Execution (Recommended)

Run the complete pipeline with a single command:

```bash
python run_pipeline.py \
    --input_txt_dir ./input_txtdvh_code1 \
    --patient_data ./treatment_params_toxicity_HN_input_data.xlsx \
    --clinical_file ./treatment_params_toxicity_HN_input_data.xlsx \
    --output_dir ./out2
```

This executes all steps in the correct order:
1. DVH preprocessing
2. DVH plotting & summary
3. **Biological DVH generation** (NEW)
4. NTCP analysis with ML
5. QA reporting
6. Clinical factors analysis
7. Publication diagrams
8. SHAP analysis
9. **Publication tables generation** (NEW)

### Option B: Manual Step-by-Step Execution

### Step 1: Preprocess DVH Files
Convert TPS-exported DVH text files to standardized CSV format:
```bash
python code1_dvh_preprocess.py --src ./input_txtdvh_code1 --dst ./out2/code1_output
```
**Output**: `cDVH_csv/`, `dDVH_csv/`, and `processed_dvh.xlsx`

### Step 2: Generate Dose Metrics and Plots
Calculate dose-volume metrics (gEUD, Dmean, Dmax, Vxx, Dxx) and create publication-quality DVH plots:
```bash
python code2_dvh_plot_and_summary.py --cdvh_dir ./out2/code1_output/cDVH_csv --outdir ./out2/code2_output
```
**Output**: Dose metrics tables, cumulative/differential DVH plots (PNG & SVG, 600 DPI)

### Step 2b: Generate Biological DVH (NEW)
Transform physical DVH to biological DVH using radiobiological formalisms:

```bash
python code2_bDVH.py \
    --input_dir ./out2/code1_output/dDVH_csv \
    --output_dir ./out2/code2_bDVH_output \
    --clinical_file ./treatment_params_toxicity_HN_input_data.xlsx \
    --method EQD2
```

**Methods available**: `BED`, `EQD2`, `gEUD`, or `all` (generates all three)

**Output**:
- `bDVH_csv/`: Biological DVH CSV files (BED/EQD2/gEUD transformed)
- `bDVH_plots/`: Comparison plots (physical vs biological DVH, 600 DPI)
- `bDVH_summary.xlsx`: Summary metrics including transformation ratios

**Note**: bDVH is derived **only from code1 outputs**, not re-parsed from raw DVH files.

### Step 3: NTCP Analysis with Machine Learning
Compute traditional and novel NTCP models, train ML models, and apply QA modules:
```bash
python code3_ntcp_analysis_ml.py \
    --dvh_dir ./out2/code1_output/dDVH_csv \
    --patient_data ./treatment_params_toxicity_HN_input_data.xlsx \
    --output_dir ./out2/code3_output
```
**Required columns in `patient_data.xlsx`**:
- `PatientID` (or `Patient ID`, `Patient_ID`)
- `Organ` (or `OAR`)
- `Observed_Toxicity` (binary: 0 or 1)
- Optional: `PatientName`, `TotalDose`, `NumFractions`, `DosePerFraction`

**Output**:
- `enhanced_ntcp_calculations.csv`: All NTCP predictions (traditional, probabilistic, Monte Carlo, ML)
- `ntcp_results.xlsx`: Comprehensive Excel workbook with multiple sheets
- `enhanced_summary_performance.csv`: Model performance metrics per organ
- `plots/`: Publication-ready figures (ROC curves, calibration plots, dose-response curves, 600 DPI)

**Output fields include**:
- Traditional NTCP: `NTCP_LKB_LogLogit`, `NTCP_LKB_Probit`, `NTCP_RS_Poisson`
- Novel models: `ProbNTCP_Mean`, `ProbNTCP_CI_L`, `ProbNTCP_CI_U`, `MC_NTCP_Mean`, `MC_NTCP_CI_L`, `MC_NTCP_CI_U`
- ML models: `NTCP_ML_ANN`, `NTCP_ML_XGBoost`
- QA metrics: `uNTCP`, `uNTCP_STD`, `uNTCP_CI_L`, `uNTCP_CI_U`, `CCS`, `CCS_Warning`, `CCS_Safety`

### Step 4: Quality Assurance Report
Generate QA report identifying potential issues:
```bash
python code4_ntcp_output_QA_reporter.py \
    --input ./out2/code3_output \
    --report_outdir ./out2/code4_output
```
**Output**: `qa_report.docx` with flags for data quality issues, overfitting, and leakage

### Step 5: Clinical Factors Analysis
Analyze associations between clinical factors and NTCP predictions:
```bash
python code5_ntcp_factors_analysis.py \
    --input_file ./treatment_params_toxicity_HN_input_data.xlsx \
    --enhanced_output_dir ./out2/code3_output
```
**Output**: Clinical factor analysis plots and tables

### Step 6: Generate Publication Diagrams (Optional)
Create workflow and methodology diagrams for manuscripts:
```bash
python code6_publication_diagrams.py --output_dir ./out2/code6_output --dpi 1200
```
**Output**: Publication-quality workflow, feature mapping, and methodology diagrams (PNG & SVG, 1200 DPI)

### Step 7: SHAP Supplementary Analysis (Optional)
Generate SHAP (SHapley Additive exPlanations) plots for ML model interpretability:
```bash
python shap_suppl.py \
    --input_dir ./out2/code3_output \
    --output_dir ./out2/code_shap_full
```
**Output**: SHAP plots and feature importance analysis per organ and model

### Step 8: Generate Publication Tables (NEW)
Auto-generate all publication tables and appendices for journal submission:
```bash
python supp_results_summary.py \
    --code1_output ./out2/code1_output \
    --code2_output ./out2/code2_output \
    --code3_output ./out2/code3_output \
    --code4_output ./out2/code4_output \
    --code5_output ./out2/code5_output \
    --clinical_file ./treatment_params_toxicity_HN_input_data.xlsx \
    --output_dir ./out2/supp_results_summary_output
```

**Output**: `publication_tables.xlsx` with the following sheets:

- **Table 1: Cohort, Treatment & DVH Characteristics** (per organ)
  - Cohort demographics (n, age, gender, comorbidity)
  - Treatment technique distribution (%)
  - Diagnosis distribution (%)
  - Dosimetric & biological metrics (QUANTEC-aligned)
  - Expanded Vx (V5, V10, V15, V20, V30, V40, V50, V60)
  - Expanded Dx (Dmean, Dmax, D2, D10, D50)
  - QUANTEC/RTOG benchmark columns

- **Table 2: NTCP Model Performance** (Internal Validation)
  - Per-organ performance metrics
  - AUC, Brier score, calibration
  - Best model identification

- **Table 3: Uncertainty-Aware & QA Metrics** (Expanded)
  - Mean uNTCP, CI width
  - % DO_NOT_USE (CCS)
  - % QA flags raised
  - Dominant QA reason

- **Table 4: Clinical Factors vs NTCP**
  - Factor associations with NTCP
  - p-values and statistical tests
  - Directionality (↑ risk / ↓ risk)

- **Appendix A1: Model & Equation Reference**
  - All NTCP models with equations
  - Parameter definitions
  - Source citations

- **Appendix A2: Computational Reproducibility**
  - OS, CPU, RAM
  - Python and library versions
  - Total runtime
  - QA flags summary

## Intended Journal Submission Use

This pipeline is designed for **journal-grade, uncertainty-aware radiobiological evaluation** suitable for submission to:

- **International Journal of Radiation Oncology, Biology, Physics (IJROBP)**
- **Physics in Medicine & Biology (PMB)**
- **Medical Physics**
- **Radiotherapy & Oncology**

### Key Features for Publication

1. **Explicit Biological DVH (bDVH) Generation**: Enables direct comparison across fractionation schemes
2. **Expanded Cohort & Dosimetric Characterization**: Comprehensive QUANTEC-aligned metrics
3. **Integrated Uncertainty + QA-Aware Reporting**: Demonstrates clinical risk containment
4. **Single-Command Reproducible Execution**: Full computational reproducibility documentation

### Scientific Rationale

The extended pipeline clearly supports:

- **Biological relevance**: bDVH vs physical DVH comparison
- **Failure modes**: Classical NTCP limitations
- **Added safety**: Uncertainty-aware metrics superiority
- **Clinical interpretability**: SHAP + QA + CCS integration

## Citation (recommend Zenodo DOI)
Archive a GitHub release with Zenodo and cite the DOI in your manuscript.
```
Mondal, K., et al. (2025). py_ntcpx: NTCP Analysis and Machine Learning Pipeline (v1.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.xxxxxxx
```





[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16786956.svg)](https://doi.org/10.5281/zenodo.16786956)

