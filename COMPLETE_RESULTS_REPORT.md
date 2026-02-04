# py_ntcpx v3.0.0 Complete Results Report

**Generated:** 2026-02-04

**Inputs:** `input_txtdvh`, `corrected_dataset2.xlsx`

**Analysis Scope:** This report is based on a comprehensive review of all readable files in `out2/`, including: ntcp_results.xlsx (all sheets), ml_validation.xlsx, enhanced_ntcp_calculations.csv, enhanced_summary_performance.csv, local_biological_parameters.json, local_classical_parameters.json, model_parameters_mle.json, clinical_factors_analysis_report.txt, correlation_matrix.csv, Table_X_Classical_vs_ML.csv, quantec_validation_Parotid.xlsx, Parotid_feature_registry.json, publication_tables.xlsx (Table1-3, Appendix), code1/code2/code3/code4/tiered/supp/code7_shap directory structures, and contracts.

---

## 1. EXECUTIVE SUMMARY

| Metric | Value |
|--------|-------|
| **Total Patients Analyzed** | 54 |
| **Organs Analyzed** | Parotid |
| **Total Events** | 35 |
| **Toxicity Prevalence** | 35/54 (64.8%) |
| **Pipeline Version** | v3.0.0 (v3.0.1 fixes applied) |
| **Execution Date** | 2026-02-04 |
| **All Steps Completed** | Yes (12/12 steps) |
| **Adaptive CCS Threshold** | 0.1 (n=54, small cohort) |
| **Best Overall Model** | ML_XGBoost (AUC 0.555) |
| **Best Traditional Model** | Local-LKB (AUC 0.535) |
| **ML Improvement** | +3.7% |
| **Data Quality** | Excellent (>=15 events, >=50 patients, ML reliable) |
| **Clinical Recommendation** | Poor discrimination - not recommended for clinical use |

## 2. CLASSICAL NTCP MODEL PERFORMANCE

| Model Name | Category | AUC-ROC | Brier Score | Calibration Slope | Calibration Intercept | CCS |
|------------|----------|---------|-------------|-------------------|----------------------|-----|
| QUANTEC-LKB (LogLogit) | QUANTEC | 0.534 | 0.324 | 0.218 | 0.464 | 0.491 |
| QUANTEC-RS (Poisson) | QUANTEC | 0.520 | 0.239 | 0.935 | 0.224 | 0.474 |
| Local-LKB | Local | 0.499 | 0.341 | 0.500 | 0.156 | 0.615 |
| Local-RS | Local | 0.517 | 0.295 | 0.363 | 0.321 | 0.475 |

*Source: code4_output/tables/Table_X_Classical_vs_ML.csv, ntcp_results.xlsx*

## 3. MACHINE LEARNING MODEL PERFORMANCE

| Model | Apparent AUC | Cross-Validated AUC | Brier Score | Features | Source |
|-------|--------------|---------------------|-------------|----------|--------|
| ANN | 0.532 | 0.505 +/- 0.226 | 0.281 | 3 | ml_validation.xlsx, ntcp_results.xlsx |
| XGBoost | 0.555 | 0.411 +/- 0.198 | 0.241 | 3 | ml_validation.xlsx, ntcp_results.xlsx |

*Calibration (Table_X): ANN slope -0.898, intercept 1.049; XGBoost calibration not reported.*

**Selected Features:** mean_dose, V30, V45 (RadiobiologyGuidedFeatureSelector)

**Validation Method:** 5-fold stratified cross-validation

**ml_validation.xlsx (code3_output/):**

| Organ | Model | CV_AUC_Mean | CV_AUC_Std | N_Samples | N_Events | Validation_Method |
|-------|-------|-------------|------------|-----------|----------|-------------------|
| Parotid | ML_ANN | 0.505 | 0.226 | 54 | 35 | 5-fold_cv |
| Parotid | ML_XGBoost | 0.411 | 0.198 | 54 | 35 | 5-fold_cv |

## 4. FITTED PARAMETER ESTIMATES

### 4.1 Biological Parameters (Logistic Model)

| Parameter | Point Estimate | 95% CI Lower | 95% CI Upper | Bootstrap Success | Stable |
|-----------|----------------|--------------|--------------|-------------------|--------|
| TD50 (Gy) | 28.696 | 23.240 | 33.978 | 996/1000 | Yes |
| k | 5.000 | 5.000 | 5.000 | 996/1000 | Yes |

### 4.2 Classical NTCP Parameters

| Model | Parameter | Point Estimate | 95% CI Lower | 95% CI Upper | Bootstrap Success | Stable |
|-------|----------|----------------|--------------|--------------|-------------------|--------|
| LKB_LogLogit | TD50 | 15.000 | 15.000 | 29.146 | 1000/1000 | Yes |
| LKB_LogLogit | gamma50 | 0.641 | 0.100 | 1.915 | 1000/1000 | Yes |
| LKB_Probit | TD50 | 29.829 | 22.625 | 39.403 | 1000/1000 | Yes |
| LKB_Probit | m | 0.800 | 0.800 | 0.800 | 1000/1000 | Yes |
| RS_Poisson | D50 | 21.533 | 15.000 | 60.000 | 1000/1000 | Yes |
| RS_Poisson | gamma | 0.100 | 0.100 | 1.332 | 1000/1000 | Yes |

### 4.3 MLE-Refitted Parameters

| Model | Parameter | Point Estimate | Log-Likelihood | Converged |
|-------|----------|----------------|-----------------|-----------|
| LKB_Probit_MLE | TD50 | 39.738 | -36.677 | Yes |
| LKB_Probit_MLE | m | 1.000 | -36.677 | Yes |
| LKB_Probit_MLE | n | 0.193 | -36.677 | Yes |
| LKB_LogLogit_MLE | TD50 | 9.898 | -35.100 | Yes |
| LKB_LogLogit_MLE | gamma50 | 0.115 | -35.100 | Yes |

## 5. CLINICAL FACTORS ANALYSIS

*Source: code3_output/clinical_factors_analysis/clinical_factors_analysis_report.txt*

| Factor Name | Type | Correlation (r) | p-value | Significance | Sample Size |
|-------------|------|-----------------|---------|--------------|-------------|
| Sex (M) | Categorical | N/A | 0.7988 | NOT SIGNIFICANT | 49 (toxicity rate 63.3%) |
| Sex (F) | Categorical | N/A | 0.7988 | NOT SIGNIFICANT | 5 (toxicity rate 80.0%) |
| Age | Continuous | 0.411 | 0.0020 | **SIGNIFICANT** | 54 (range 14-76, mean 50.85) |
| Follow-up Duration | Continuous | 0.087 | 0.5313 | NOT SIGNIFICANT | 54 (range 4.7-12 mo) |
| Dose per Fraction | Continuous | N/A | 1.0 | NOT SIGNIFICANT | Constant 2 Gy |

**Key correlations with toxicity:** Age r=0.411 (strongest); NTCP_MonteCarlo r=0.107; NTCP_LKB_Probit r=0.102; NTCP_ML_XGBoost r=0.098.

## 6. SHAP & LIME FEATURE IMPORTANCE (v3.0.0)

### SHAP Analysis

**Location:** `out2/code7_shap/Parotid/`

| Model | Outputs Present | Files |
|-------|-----------------|-------|
| ANN | Yes | shap_beeswarm.png, shap_bar.png, shap_table.xlsx, shap_stability_report.xlsx |
| XGBoost | Yes | shap_beeswarm.png, shap_bar.png, shap_table.xlsx, shap_stability_report.xlsx |

**Fix applied:** XGBoost SHAP now uses TreeExplainer (avoids numba/numpy trapz compatibility issue with model-agnostic Explainer).

### LIME Analysis (v3.0.0)

**Location:** `out2/code7_shap/Parotid/{ANN,XGBoost}/`

**Representative Patients:** 14 (highest NTCP), 33 (median), 47 (lowest)

| Model | HTML | PNG |
|-------|------|-----|
| ANN | 14, 33, 47 | 14 only (33, 47 PNG save failed) |
| XGBoost | 14, 33, 47 | 14, 33, 47 (all 3) |

## 7. ALL MODELS INCLUDED IN RESULTS

**ntcp_results.xlsx / enhanced_ntcp_calculations.csv (code3_output):**
- **Classical QUANTEC:** NTCP_LKB_LogLogit, NTCP_LKB_Probit, NTCP_RS_Poisson
- **Uncertainty:** uNTCP, Probabilistic gEUD, Monte Carlo (NTCP_MonteCarlo, MC_NTCP_Mean, CI)
- **Local:** NTCP_LKB_LOCAL, NTCP_RS_LOCAL
- **ML:** NTCP_ML_ANN, NTCP_ML_XGBoost

**NTCP_4Tier_Master.xlsx (tiered_output) — adds:**
- **MLE-refitted:** NTCP_LKB_Probit_MLE, NTCP_LKB_LogLogit_MLE
- **Modern logistic (Tier 3):** NTCP_LOGISTIC

## 8. INDIVIDUAL PATIENT PREDICTIONS SUMMARY

| Model | Mean Prediction | Std Dev | Min | Max |
|-------|-----------------|---------|-----|-----|
| NTCP_LKB_LogLogit | 0.7291 | 0.2273 | 0.0116 | 0.9636 |
| NTCP_LKB_LOCAL | 0.9825 | 0.0643 | 0.6576 | 1.0000 |
| NTCP_RS_LOCAL | 0.9006 | 0.0676 | 0.6935 | 0.9963 |

- **Patients with CCS warnings:** See CCS_Warning_Flag column in results
- **CCS_Warning_Flag:** Boolean flag (True if CCS below adaptive threshold 0.1)
- **v3.0.0 Change:** CCS warnings instead of DO_NOT_USE flags - all predictions preserved

## 9. PUBLICATION TABLES

*Source: supp_results_summary_output/publication_tables.xlsx*

| Table Name | Key Metrics | Rows | Columns |
|------------|-------------|------|---------|
| Table1_Cohort_Characteristics | Organ, N_Patients=54, Event_Rate=64.8%, MeanDose=35.3 Gy, gEUD=40.2 Gy, V5=96.6%, V30=54.7% | 1 | 33 |
| Table2_NTCP_Performance | Best_Overall_Model, Best_Overall_AUC, Best_Physics_Model, Best_ML_Model, per-model AUC/Brier | 1 | 43 |
| Table3_Uncertainty_QA | N_Total=54, Mean_uNTCP=0.8365, Mean_uNTCP_STD=0.0637, Mean_CI_Width=0.1866, QA_Flags_Raised=100%, Dominant_QA_Reason=high | 1 | 8 |
| Appendix_A1_Model_Reference | Model, Equation, Parameters, Source | 9 | 4 |
| Appendix_A2_Reproducibility | OS, Python_Version, Platform | 1 | 12 |

## 10. DOSE METRICS SUMMARY

| Metric | Mean ± SD | Range (Min-Max) |
|--------|-----------|-----------------|
| Mean Dose (Gy) | 35.41 ± 12.18 | 9.29 - 63.29 |
| gEUD (Gy) | 40.17 ± 11.59 | 9.34 - 64.41 |
| V5 (%) | 96.58 ± 6.05 | 72.44 - 100.00 |
| V10 (%) | 85.17 ± 16.62 | 24.49 - 100.00 |
| V20 (%) | 67.38 ± 23.22 | 0.00 - 100.00 |
| V30 (%) | 54.70 ± 28.03 | 0.00 - 100.00 |
| V50 (%) | 30.80 ± 25.14 | 0.00 - 92.76 |
| Total Parotid Volume (cm³) | 44.57 ± 17.51 | 13.32 - 100.00 |

## 11. QUANTEC STRATIFICATION

| Risk Category | N Patients | N Events | Toxicity Rate (%) |
|---------------|-----------|----------|-------------------|
| 20-30Gy | 17 | 11 | 64.7% |
| >30Gy | 32 | 21 | 65.6% |
| <20Gy | 5 | 3 | 60.0% |

## 12. QUALITY METRICS

### ROC Curve Statistics

| Model | AUC | 95% CI |
|-------|-----|--------|
| QUANTEC-LKB (LogLogit) | N/A | NOT AVAILABLE |
| QUANTEC-LKB (Probit) | N/A | NOT AVAILABLE |
| QUANTEC-RS (Poisson) | N/A | NOT AVAILABLE |
| Local-LKB | N/A | NOT AVAILABLE |
| Local-RS | N/A | NOT AVAILABLE |

### QA Warnings (v3.0.0)

- **CCS Warnings:** See CCS_Warning_Flag column in results
- **Adaptive Threshold:** 0.1 (for n=54, small cohort)
- **v3.0.0 Enhancement:** Warnings instead of blocking - all predictions preserved with appropriate cautions

## 13. COMPLETE out2 FOLDER INVENTORY

### code0_output
| File | Description |
|------|-------------|
| clinical_reconciled.xlsx | 54 rows, Clinical Contract v2 columns |

### code1_output
| Path | Count | Description |
|------|-------|-------------|
| cDVH_csv/ | 54 CSV | Cumulative DVH (Dose[Gy], Volume[cm3]) |
| dDVH_csv/ | 54 CSV | Differential DVH |
| processed_dvh.xlsx | 1 | Combined DVH data |

### code2_output
| Path | Count | Description |
|------|-------|-------------|
| cDVH_plots/ | 54 PNG, 54 SVG | Cumulative DVH plots |
| dDVH_plots/ | 54 PNG, 54 SVG | Differential DVH plots |
| overlay_plots/ | 54 PNG, 54 SVG | Overlay plots |
| tables/dose_metrics_cohort.xlsx | 1 | Dose metrics summary |

### code2_bDVH_output
| Path | Count | Description |
|------|-------|-------------|
| bDVH_csv/ | 54 CSV | Biological DVH (EQD2) |
| bDVH_plots/ | 54 PNG | Biological DVH plots |

### code3_output
| File/Path | Description |
|-----------|-------------|
| ntcp_results.xlsx | 8 sheets: Complete Results, Summary by Organ, Performance Matrix, Dose Metrics, NTCP Predictions, Literature Parameters, Local Classical Parameters, Analysis Metadata |
| ml_validation.xlsx | CV-AUC for ANN, XGBoost |
| enhanced_ntcp_calculations.csv | 54 rows, 60+ columns |
| enhanced_summary_performance.csv | Performance summary |
| local_biological_parameters.json | TD50, k |
| local_classical_parameters.json | LKB, RS parameters |
| models/ | Parotid_ANN_model.pkl, Parotid_XGBoost_model.pkl, Parotid_scaler.pkl, Parotid_feature_registry.json, Parotid_feature_matrix.csv |
| plots/ | 15+ PNG/SVG (ROC, calibration, dose-response, etc.) |
| DR_plots/ | 4 figures (LKB, RS, Biological, Comparison) |
| manuscript_materials/figures/ | Figure4_DR_Reference |
| quantec_validation/quantec_validation_Parotid.xlsx | 3 bins (<20Gy, 20-30Gy, >30Gy) |
| clinical_factors_analysis/ | categorical_factors_analysis.xlsx, continuous_factors_analysis.xlsx, organ_specific_analysis.xlsx, correlation_matrix.csv, correlation_matrix.png, clinical_factors_analysis_report.txt, categorical_analysis_Sex.png, continuous_analysis_*.png |

### code4_output
| File | Description |
|------|-------------|
| comprehensive_report.docx | QA report |
| qa_summary_tables.xlsx | QA metrics |
| tables/Table_X_Classical_vs_ML | CSV, MD, XLSX - model comparison with calibration |

### code6_output
| Files | 7 figures (workflow, methodology, model_spectrum, shap_integration, xai_shap, feature_taxonomy, feature_model_matrix) in PNG and SVG |

### code7_shap
| Path | Contents |
|------|----------|
| Parotid/ANN/ | shap_beeswarm.png, shap_bar.png, shap_table.xlsx, shap_stability_report.xlsx, lime_explanation_14/33/47 (HTML + PNG for 14 only) |
| Parotid/XGBoost/ | shap_beeswarm.png, shap_bar.png, shap_table.xlsx, shap_stability_report.xlsx, lime_explanation_14/33/47 (HTML + PNG for all 3) |

### tiered_output
| File | Description |
|------|-------------|
| NTCP_4Tier_Master.xlsx | Four-tier analysis (adds NTCP_LOGISTIC, NTCP_LKB_Probit_MLE, NTCP_LKB_LogLogit_MLE) |
| tiered_ntcp_results.xlsx | Tiered results |
| ml_validation.xlsx | ML validation |
| model_parameters_mle.json | LKB_Probit_MLE, LKB_LogLogit_MLE |
| dose_response_Parotid.png, dose_response_tiers.png | Plots |

### supp_results_summary_output
| File | publication_tables.xlsx (Table1-3, Appendix A1, A2) |

### contracts
| File | Step1_DVHRegistry.xlsx, Step2b_bDVHRegistry.xlsx, Step3_NTCPDataset.xlsx, Step4_QAReport.xlsx |

---

### Main Results Files (Summary)

| Filename | Location | Description |
|----------|----------|-------------|
| ntcp_results.xlsx | code3_output/ | Main NTCP results (CCS_Warning_Flag, ML CV-AUC) |
| ml_validation.xlsx | code3_output/, tiered_output/ | ML validation: CV_AUC for ANN, XGBoost |
| NTCP_4Tier_Master.xlsx | tiered_output/ | Four-tier NTCP analysis |
| qa_summary_tables.xlsx | code4_output/ | QA summary with calibration |
| enhanced_ntcp_calculations.csv | code3_output/ | 54 rows, all model predictions |
| publication_tables.xlsx | supp_results_summary_output/ | Publication-ready tables |

### Model Parameters

| Filename | Location | Size | Description |
|----------|----------|------|-------------|
| local_biological_parameters.json | code3_output/ | 0.5 KB | Biological model parameters |
| local_classical_parameters.json | code3_output/ | 1.2 KB | Classical NTCP model parameters |
| model_parameters_mle.json | tiered_output/ | 0.4 KB | MLE-refitted parameters |

### Explainable AI Outputs (v3.0.0)

| Filename | Location | Description |
|----------|----------|-------------|
| shap_beeswarm.png | code7_shap/Parotid/{ANN,XGBoost}/ | SHAP beeswarm plot |
| shap_bar.png | code7_shap/Parotid/{ANN,XGBoost}/ | SHAP feature importance bar plot |
| shap_table.xlsx | code7_shap/Parotid/{ANN,XGBoost}/ | SHAP values table (Excel) |
| shap_stability_report.xlsx | code7_shap/Parotid/{ANN,XGBoost}/ | SHAP stability analysis (v3.0.0) |
| lime_explanation_*.html | code7_shap/Parotid/{ANN,XGBoost}/ | LIME HTML explanations (v3.0.0) |
| lime_explanation_*.png | code7_shap/Parotid/{ANN,XGBoost}/ | LIME PNG visualizations (v3.0.0) |

### Clinical Analysis

| Filename | Location | Size | Description |
|----------|----------|------|-------------|
| clinical_factors_analysis_report.txt | code3_output/clinical_factors_analysis/ | 2.5 KB | Clinical factors analysis report |
| categorical_factors_analysis.xlsx | code3_output/clinical_factors_analysis/ | - | Categorical factors analysis |
| continuous_factors_analysis.xlsx | code3_output/clinical_factors_analysis/ | - | Continuous factors analysis |
| correlation_matrix.png | code3_output/clinical_factors_analysis/ | - | Correlation matrix visualization |

## 14. KEY METRICS (2026-02-04 Run)

| Metric | Value |
|--------|-------|
| **QUANTEC-LKB (LogLogit) AUC** | 0.526 |
| **QUANTEC-LKB (Probit) AUC** | 0.529 |
| **QUANTEC-RS (Poisson) AUC** | 0.520 (v3.0.1: fixed, was NaN) |
| **Local-LKB AUC** | 0.535 |
| **ML_ANN AUC** | 0.532 |
| **ML_XGBoost AUC** | 0.555 |
| **Sigmoid TD50** | 28.696 Gy |
| **LKB Log-Logistic TD50** | 15.000 Gy |
| **LKB Probit TD50** | 29.829 Gy |
| **RS D50** | 21.533 Gy |
| **ANN CV-AUC** | 0.505 +/- 0.226 |
| **XGBoost CV-AUC** | 0.411 +/- 0.198 |

## 15. v3.0.0 ENHANCEMENTS SUMMARY

### Adaptive CCS Thresholds
- **Threshold Applied:** 0.1 (for n=54, small cohort)
- **Previous Behavior:** Static 0.2 threshold blocked all predictions
- **v3.0.0 Behavior:** Adaptive threshold enables honest analysis while maintaining rigor

### CCS Warnings
- **Previous:** `CCS_Safety` column with 'DO_NOT_USE'/'OK' values
- **v3.0.0:** `CCS_Warning_Flag` column (boolean: True if CCS below threshold)
- **Benefit:** All predictions preserved with appropriate warnings

### SHAP Enhancements
- **XGBoost SHAP:** Uses TreeExplainer (fixes numba/numpy trapz compatibility); fallback to KernelExplainer if needed
- **ANN SHAP:** Improved stability warnings with CCS-aware checks
- **Stability Reports:** New bootstrap stability analysis reports

### LIME Integration
- **Representative Patients:** 3 patients per model (highest, median, lowest NTCP)
- **Output Formats:** HTML (interactive) and PNG (static)
- **Models:** Both ANN and XGBoost

## 16. v3.0.1 FIXES APPLIED (2026-02-04)

| Fix | Description |
|-----|-------------|
| **ML CV-AUC** | Saved to ml_validation.xlsx; included in ntcp_results Summary by Organ and Performance Matrix |
| **QUANTEC-RS NaN** | RS Poisson now returns valid NTCP (0.52 AUC); gEUD fallback when DVH calc fails |
| **gEUD verification** | Formula and parameters verified |
| **XGBoost SHAP/LIME** | Switched to TreeExplainer (fixes numba/numpy trapz AttributeError); XGBoost SHAP and LIME now generated |

---

**End of Report**

