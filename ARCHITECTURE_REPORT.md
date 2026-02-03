# py_ntcpx v2.0.0 - Complete Architecture Report

**Version**: 2.0.0  
**Date**: 2024  
**Status**: Production Ready ✅

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Pipeline Architecture](#pipeline-architecture)
3. [Radiobiological Models](#radiobiological-models)
4. [Machine Learning Models](#machine-learning-models)
5. [Explainable AI (SHAP)](#explainable-ai-shap)
6. [Clinical Factors Analysis](#clinical-factors-analysis)
7. [Statistical Methods](#statistical-methods)
8. [Uncertainty Quantification](#uncertainty-quantification)
9. [Working Functionality](#working-functionality)
10. [Test Report](#test-report)
11. [Capabilities Summary](#capabilities-summary)

---

## Executive Summary

**py_ntcpx v2.0.0** is a comprehensive, publication-ready pipeline for Normal Tissue Complication Probability (NTCP) analysis in head & neck cancer radiotherapy. The system integrates classical radiobiological models, modern machine learning approaches, explainable AI (SHAP), clinical factor analysis, and rigorous statistical validation.

### Key Features
- **9-step pipeline** from DVH preprocessing to publication-ready outputs
- **4-tier NTCP model framework** (Legacy, MLE-refitted, Modern Classical, AI)
- **Patient-level data splitting** preventing data leakage
- **EPV-aware ML training** with overfitting prevention
- **SHAP-based interpretability** for clinical-grade model explanations
- **Comprehensive statistical validation** with confidence intervals
- **49 unit and integration tests** - All passing ✅

---

## Pipeline Architecture

### Overall Structure

The pipeline follows a modular, step-by-step architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                    py_ntcpx v2.0.0 Pipeline                 │
└─────────────────────────────────────────────────────────────┘

Step 0: Clinical Reconciliation
    ↓
Step 1: DVH Preprocessing (code1_dvh_preprocess.py)
    ↓
Step 2: DVH Plotting & Summary (code2_dvh_plot_and_summary.py)
    ↓
Step 2b: Biological DVH Generation (code2_bDVH.py)
    ↓
Step 3: NTCP Analysis with ML (code3_ntcp_analysis_ml.py)
    ├─ Classical NTCP Models
    ├─ Machine Learning Models
    └─ Uncertainty Quantification
    ↓
Step 3b: QUANTEC Stratification
    ↓
Step 3c: Tiered NTCP Analysis
    ↓
Step 4: QA Reporter (code4_ntcp_output_QA_reporter.py)
    ↓
Step 5: Clinical Factors Analysis (code5_ntcp_factors_analysis.py)
    ↓
Step 6: Publication Diagrams (code6_publication_diagrams.py)
    ↓
Step 7: SHAP Analysis (shap_code7.py)
    ↓
Step 8: Publication Tables Summary (supp_results_summary.py)
```

### Core Components

#### 1. Data Processing Layer
- **DVH Preprocessing**: Converts raw DVH text files to standardized CSV format
- **Clinical Reconciliation**: Validates and standardizes clinical data (Clinical Contract v2)
- **Biological DVH**: Generates EQD2-based biological dose-volume histograms

#### 2. Model Layer (`src/models/`)
- **Traditional Models** (`src/models/traditional/`): Classical NTCP implementations
- **Machine Learning** (`src/models/machine_learning/`): ANN, XGBoost with overfitting prevention
- **Uncertainty** (`src/models/uncertainty/`): Monte Carlo NTCP, uncertainty propagation

#### 3. Validation Layer (`src/validation/`)
- **Data Splitting**: Patient-level splitting with stratification
- **Leakage Detection**: Automated data leakage audit
- **Calibration Correction**: Platt scaling, isotonic regression
- **Nested CV**: Unbiased performance estimation

#### 4. Feature Engineering (`src/features/`)
- **Feature Selection**: Radiobiology-guided feature selection
- **Auto Reduction**: EPV-based automatic feature reduction

#### 5. Metrics & Reporting (`src/metrics/`, `src/reporting/`)
- **AUC Calculation**: With bootstrap confidence intervals
- **Statistical Reporter**: Comprehensive statistical summaries
- **Leakage Detector**: Patient overlap detection

#### 6. Safety Layer (`src/safety/`)
- **Clinical Safety Guard**: Underprediction risk detection
- **Cohort Consistency Score**: QA validation

---

## Radiobiological Models

### Tier 1: Legacy-A (QUANTEC Fixed Parameters)

**Location**: `ntcp_models/legacy_fixed.py`

Fixed QUANTEC parameters from literature, no refitting.

#### Models Implemented:

1. **LKB Log-Logistic Model**
   - **Equation**: `NTCP = 1 / (1 + (TD50/gEUD)^(4*γ50))`
   - **Parameters**: TD50, γ50, α/β
   - **Source**: Lyman 1985, Niemierko 1999
   - **Use Case**: Baseline comparison, literature validation

2. **LKB Probit Model**
   - **Equation**: `NTCP = Φ((D - TD50)/(m*TD50))`
   - **Parameters**: TD50, m
   - **Source**: Lyman 1985
   - **Use Case**: Alternative sigmoid model

3. **RS Poisson Model**
   - **Equation**: `NTCP = 1 - exp(-exp(γ*(D/D50)))`
   - **Parameters**: D50, γ, s
   - **Source**: Källman 1992
   - **Use Case**: Poisson-based dose-response

### Tier 2: Legacy-B (MLE-Refitted)

**Location**: `ntcp_models/legacy_mle.py`

Maximum Likelihood Estimation (MLE) refitting of classical models to local data.

#### Features:
- **MLE Optimization**: Scipy-based parameter estimation
- **Bootstrap Confidence Intervals**: 1000 bootstrap samples
- **Convergence Validation**: Checks optimization success
- **Parameter Stability**: Bootstrap-based stability assessment

#### Refitted Models:
- LKB Log-Logistic (MLE)
- LKB Probit (MLE)
- RS Poisson (MLE)

### Tier 3: Modern Classical

**Location**: `ntcp_models/modern_logistic.py`

Multivariable logistic regression (de Vette / CITOR style).

#### Features:
- **L2-Regularized Logistic Regression**: Prevents overfitting
- **Bootstrap Variable Stability**: Feature importance via bootstrap
- **Calibration Curves**: Model calibration assessment
- **DVH + Clinical Features**: Combines dose metrics with clinical factors

#### Implementation:
```python
class ModernLogisticNTCP:
    - LogisticRegression with L2 regularization (C=0.1)
    - StandardScaler for feature normalization
    - Bootstrap stability analysis (1000 iterations)
    - Calibration curve generation
```

### Tier 4: AI Models (Already in Step 3)

**Location**: `code3_ntcp_analysis_ml.py`, `src/models/machine_learning/ml_models.py`

Machine learning models with overfitting prevention.

#### Models:
1. **Artificial Neural Network (ANN)**
   - Architecture: (16, 8) hidden layers (adaptive based on sample size)
   - Solver: Adam optimizer
   - Regularization: L2 (alpha=0.01)
   - Early Stopping: Enabled

2. **XGBoost**
   - Trees: 50 (reduced for small datasets)
   - Max Depth: 2 (conservative)
   - Regularization: L1 (0.5) + L2 (2.0)
   - Learning Rate: 0.05

### Biological Dose-Response Refitting

**Location**: `biological_refitting.py`

Biologically interpretable dose-response refitting using mean dose (Gy).

#### Models:
1. **LKB Log-Logit Biological**
   - `NTCP(D) = 1 / (1 + (TD50 / D) ** gamma50)`
   - Uses mean dose, not gEUD

2. **LKB Probit Biological**
   - `NTCP(D) = Phi((D - TD50) / (m * TD50))`
   - Uses mean dose, not gEUD

3. **RS Poisson Biological**
   - `NTCP(D) = 1 - exp(-(D / D50) ** gamma)`
   - Uses mean dose, not gEUD

**Note**: Refitted parameters are for interpretability ONLY and do not overwrite prediction pipelines.

---

## Machine Learning Models

### Architecture Overview

The ML pipeline implements rigorous methodology to prevent overfitting and data leakage:

```
┌─────────────────────────────────────────────────────────────┐
│              ML Training Pipeline (v2.0.0)                 │
└─────────────────────────────────────────────────────────────┘

1. Patient-Level Data Splitting
   ├─ PatientDataSplitter (prevents leakage)
   ├─ Stratification by outcome
   └─ Leakage detection audit

2. Feature Selection
   ├─ RadiobiologyGuidedFeatureSelector
   ├─ Domain knowledge (literature-validated features)
   ├─ Statistical filtering (p < 0.1)
   └─ EPV-based capping (max_features = n_events / 10)

3. Model Training
   ├─ OverfitResistantMLModels wrapper
   ├─ EPV calculation and enforcement
   ├─ Dynamic complexity adjustment
   └─ Cross-validation (5-fold stratified)

4. Evaluation
   ├─ AUC with 95% CI (bootstrap)
   ├─ Brier score
   ├─ Calibration curves
   └─ DeLong test for AUC comparison
```

### Key Components

#### 1. Patient-Level Data Splitting

**Location**: `src/validation/data_splitter.py`

```python
class PatientDataSplitter:
    - Patient-level splitting (not row-level)
    - Stratification by outcome
    - Test size: 20% (default)
    - Leakage detection built-in
```

**Benefits**:
- Prevents data leakage when patients have multiple organs
- Ensures train/test independence
- Maintains outcome distribution balance

#### 2. Feature Selection

**Location**: `src/features/feature_selector.py`

```python
class RadiobiologyGuidedFeatureSelector:
    - Starts with literature-validated features
      * Parotid: Dmean, V30, V45 (essential)
      * Clinical: Chemotherapy, T_Stage, Diabetes (exploratory)
    - Adds features with univariate p < 0.1 (Mann-Whitney U)
    - Caps at EPV rule: max_features = max(int(n_events / 10), 3)
```

**Feature Selection Strategy**:
1. **Domain Knowledge**: Literature-validated features first
2. **Statistical Filtering**: Univariate tests (p < 0.1)
3. **EPV Rule**: Maximum features based on events per variable
4. **Organ-Specific**: Different essential features per organ

#### 3. Overfit-Resistant ML Models

**Location**: `src/models/machine_learning/ml_models.py`

```python
class OverfitResistantMLModels:
    - EPV calculation: n_events / n_features
    - Refuses training if EPV < 5
    - Dynamic complexity adjustment:
      * n_samples < 50: Single layer (8 neurons)
      * n_samples < 100: Single layer (16 neurons)
      * Otherwise: (16, 8) layers
    - Conservative hyperparameters
```

**ANN Configuration**:
- **Hidden Layers**: (16, 8) - reduced from (20, 10)
- **Solver**: Adam (changed from lbfgs)
- **Learning Rate**: Adaptive
- **Max Iterations**: 500 (reduced from 1000)
- **Regularization**: L2 (alpha=0.01)

**XGBoost Configuration**:
- **n_estimators**: 50
- **max_depth**: 2 (reduced from 3)
- **learning_rate**: 0.05 (reduced from 0.1)
- **subsample**: 0.7 (reduced from 0.8)
- **reg_alpha**: 0.5 (increased from 0.1)
- **reg_lambda**: 2.0 (increased from 1.0)
- **min_child_weight**: 3 (added)

#### 4. Cross-Validation

**Implementation**: 5-fold StratifiedKFold

```python
cv_scores = cross_val_score(
    model, X_train, y_train,
    cv=min(5, len(X_train)//3),
    scoring='roc_auc'
)
```

**Adaptive Folds**: Adjusts number of folds based on training set size.

#### 5. AUC Calculation with Confidence Intervals

**Location**: `src/metrics/auc_calculator.py`

```python
def calculate_auc_with_ci(y_true, y_pred, method='bootstrap', n_bootstrap=1000):
    - Bootstrap method: 1000 resamples
    - DeLong method: Analytical CI
    - Returns: (auc, (ci_lower, ci_upper))
```

**Methods**:
- **Bootstrap**: Non-parametric, robust
- **DeLong**: Analytical, faster

---

## Explainable AI (SHAP)

### Implementation

**Location**: `shap_code7.py`

True-model SHAP analysis that explains the exact ML models trained in Step 3.

### Features

1. **Model Loading**
   - Loads saved models from Step 3 (`code3_output/models/`)
   - Supports organ-specific models
   - Handles ANN and XGBoost separately

2. **SHAP Explainer Selection**
   - **ANN**: KernelExplainer (model-agnostic)
   - **XGBoost**: TreeExplainer (model-specific, faster, exact)

3. **Visualizations Generated**

   a. **Beeswarm Plot**
      - Shows feature importance distribution
      - Color-coded by feature value
      - Publication-ready (600 DPI)

   b. **Global Feature Importance Bar Plot**
      - Mean |SHAP| values
      - Clinical standard visualization
      - Sorted by importance

   c. **SHAP Values Table**
      - Excel export of all SHAP values
      - Per-sample, per-feature contributions

### Output Structure

```
code7_shap/
├── Parotid/
│   ├── ANN/
│   │   ├── shap_beeswarm.png
│   │   ├── shap_bar.png
│   │   └── shap_table.xlsx
│   └── XGBoost/
│       ├── shap_beeswarm.png
│       ├── shap_bar.png
│       └── shap_table.xlsx
└── [Other organs]/
```

### Clinical Interpretation

SHAP values provide:
- **Feature Importance**: Which features drive predictions
- **Direction**: Positive (risk-increasing) vs negative (protective)
- **Magnitude**: Strength of feature contribution
- **Individual Explanations**: Per-patient feature contributions

**Example Interpretation**:
- `V30: -0.124` → Higher V30 associated with lower toxicity risk (protective)
- `Dmean: +0.089` → Higher mean dose increases toxicity risk

---

## Clinical Factors Analysis

### Implementation

**Location**: `code5_ntcp_factors_analysis.py`

### Analysis Types

#### 1. Categorical Factors

**Statistical Tests**:
- **Chi-square test**: For association with observed toxicity
- **Kruskal-Wallis test**: For NTCP prediction differences across categories

**Factors Analyzed**:
- Sex (M/F)
- Treatment Technique
- Chemotherapy (Yes/No)
- T Stage
- N Stage
- Diabetes
- Smoking Status

**Output**:
- Contingency tables
- Toxicity rates by category
- Statistical significance (p-values)
- NTCP prediction distributions

#### 2. Continuous Factors

**Statistical Tests**:
- **Pearson correlation**: For linear relationships
- **Mann-Whitney U test**: For differences between toxicity groups
- **Kruskal-Wallis test**: For multi-group comparisons

**Factors Analyzed**:
- Age
- Dose per Fraction
- Total Dose
- Treatment Duration
- Follow-up Duration

**Output**:
- Correlation coefficients (r)
- P-values
- Scatter plots (NTCP vs factor)
- Box plots (toxicity groups)

#### 3. Organ-Specific Effects

**Analysis**:
- Stratified by organ (Parotid, Larynx, etc.)
- Organ-specific factor associations
- Organ-specific NTCP model performance

**Output**:
- Organ-specific summary tables
- Organ-specific correlation matrices
- Organ-specific plots

#### 4. Correlation Matrix

**Implementation**:
- Pearson correlation for continuous variables
- Spearman correlation for ordinal variables
- Heatmap visualization (600 DPI)

**Output**:
- Correlation matrix (CSV)
- Correlation heatmap (PNG, 600 DPI)
- Statistical significance annotations

### Output Files

1. **categorical_factors_analysis.xlsx**
   - Chi-square results
   - Toxicity rates by category
   - NTCP predictions by category

2. **continuous_factors_analysis.xlsx**
   - Correlation coefficients
   - P-values
   - Statistical test results

3. **organ_specific_analysis.xlsx**
   - Organ-stratified results
   - Organ-specific associations

4. **correlation_matrix.png / .csv**
   - Full correlation matrix
   - Publication-ready visualization

5. **clinical_factors_analysis_report.txt**
   - Comprehensive text summary
   - Clinical recommendations

---

## Statistical Methods

### Applied Statistical Tests

#### 1. Hypothesis Testing

**Chi-square Test** (`scipy.stats.chi2_contingency`)
- **Use**: Categorical factor association with toxicity
- **Output**: χ² statistic, p-value, degrees of freedom

**Mann-Whitney U Test** (`scipy.stats.mannwhitneyu`)
- **Use**: Continuous factor differences between toxicity groups
- **Output**: U statistic, p-value

**Kruskal-Wallis Test** (`scipy.stats.kruskal`)
- **Use**: Multi-group comparisons (non-parametric ANOVA)
- **Output**: H statistic, p-value

**Pearson Correlation** (`scipy.stats.pearsonr`)
- **Use**: Linear relationships between continuous variables
- **Output**: Correlation coefficient (r), p-value

**Spearman Correlation** (`scipy.stats.spearmanr`)
- **Use**: Monotonic relationships (ordinal/continuous)
- **Output**: Correlation coefficient (ρ), p-value

#### 2. Confidence Intervals

**Bootstrap Confidence Intervals**
- **Method**: 1000 bootstrap resamples
- **Applied to**:
  - AUC (95% CI)
  - NTCP predictions
  - Model parameters
  - Performance metrics

**DeLong Test** (`src/metrics/auc_calculator.py`)
- **Use**: AUC comparison between models
- **Method**: Analytical confidence intervals
- **Correction**: Bonferroni for multiple comparisons

#### 3. Model Evaluation Metrics

**AUC (Area Under ROC Curve)**
- **Calculation**: `sklearn.metrics.roc_auc_score`
- **Confidence Intervals**: Bootstrap (1000 samples) or DeLong
- **Interpretation**: Discrimination ability

**Brier Score**
- **Calculation**: `sklearn.metrics.brier_score_loss`
- **Interpretation**: Calibration (lower is better)

**Log Loss**
- **Calculation**: `sklearn.metrics.log_loss`
- **Interpretation**: Probabilistic prediction quality

**Calibration Slope**
- **Calculation**: `src/validation/calibration_correction.py`
- **Interpretation**: Perfect calibration = 1.0

#### 4. Parameter Estimation

**Maximum Likelihood Estimation (MLE)**
- **Method**: `scipy.optimize.minimize` or `differential_evolution`
- **Objective**: Negative log-likelihood
- **Applied to**: LKB, RS Poisson parameter refitting

**Bootstrap Parameter Stability**
- **Method**: 1000 bootstrap resamples
- **Output**: Parameter distributions, 95% CI
- **Stability Check**: Convergence rate

#### 5. Cross-Validation

**Stratified K-Fold Cross-Validation**
- **Folds**: 5 (adaptive: min(5, len(X_train)//3))
- **Stratification**: By outcome class
- **Metric**: ROC-AUC
- **Output**: Mean ± std across folds

**Nested Cross-Validation** (`src/validation/nested_cv.py`)
- **Outer Loop**: 5-fold (performance estimation)
- **Inner Loop**: 5-fold (hyperparameter tuning)
- **Use**: Unbiased performance estimation

#### 6. Multiple Comparisons Correction

**Bonferroni Correction**
- **Use**: DeLong test for multiple model comparisons
- **Method**: α_adjusted = α / n_comparisons
- **Applied**: AUC comparison across models

#### 7. Calibration Correction

**Platt Scaling** (`src/validation/calibration_correction.py`)
- **Method**: Logistic regression on predictions
- **Use**: Post-hoc probability calibration

**Isotonic Regression**
- **Method**: Non-parametric monotonic transformation
- **Use**: More flexible calibration correction

---

## Uncertainty Quantification

### Methods Implemented

#### 1. Monte Carlo NTCP

**Location**: `src/models/uncertainty/monte_carlo_ntcp.py`

**Method**: Parameter uncertainty propagation via Monte Carlo sampling

```python
class MonteCarloNTCPCorrect:
    - Samples parameters from multivariate normal distribution
    - Uses parameter covariance matrix
    - Calculates NTCP distribution across samples
    - Returns: mean, std, 95% CI
```

**Features**:
- **Samples**: 10,000 (default)
- **Parameter Distribution**: Multivariate normal
- **Covariance**: From MLE fitting
- **Output**: Mean NTCP, standard deviation, 95% CI

#### 2. Probabilistic gEUD Model

**Location**: `ntcp_novel_models.py`

**Method**: Parameter distributions from literature

```python
class ProbabilisticgEUDModel:
    - Parameter distributions (normal) from literature
    - Monte Carlo sampling (1000 samples)
    - gEUD calculation with parameter uncertainty
    - Returns: mean, std, 95% CI, full distribution
```

**Organ-Specific Parameters**:
- Parotid: n ~ N(0.45, 0.1), TD50 ~ N(28.4, 5.0)
- Larynx: n ~ N(1.0, 0.2), TD50 ~ N(44.0, 8.0)
- Spinal Cord: n ~ N(0.03, 0.01), TD50 ~ N(66.5, 10.0)

#### 3. Uncertainty-Aware NTCP (uNTCP)

**Location**: `ntcp_qa_modules.py`

**Method**: First-order Taylor expansion (Delta method)

```python
class UncertaintyAwareNTCP:
    - Calculates partial derivatives (numerical)
    - Propagates uncertainty via Delta method
    - Returns: NTCP, std, 95% CI, uncertainty contributions
```

**Uncertainty Sources**:
- Parameter uncertainty (n, TD50, m)
- DVH uncertainty (systematic + random error)
- Model uncertainty

#### 4. Cohort Consistency Score (CCS)

**Location**: `ntcp_qa_modules.py`

**Method**: Mahalanobis distance-based consistency check

```python
class CohortConsistencyScore:
    - Calculates Mahalanobis distance from cohort mean
    - Flags outliers (distance > threshold)
    - Returns: CCS score, outlier flags
```

**Use**: Quality assurance, outlier detection

---

## Working Functionality

### Pipeline Steps

#### Step 0: Clinical Reconciliation ✅
- **File**: `code0_clinical_reconciliation.py`
- **Function**: Validates and standardizes clinical data
- **Output**: `clinical_reconciled.xlsx`
- **Status**: Working

#### Step 1: DVH Preprocessing ✅
- **File**: `code1_dvh_preprocess.py`
- **Function**: Converts DVH text files to CSV
- **Output**: `cDVH_csv/`, `dDVH_csv/`, `processed_dvh.xlsx`
- **Status**: Working

#### Step 2: DVH Plotting & Summary ✅
- **File**: `code2_dvh_plot_and_summary.py`
- **Function**: Generates DVH plots and summary tables
- **Output**: Plots (PNG, SVG), summary tables (Excel)
- **Status**: Working

#### Step 2b: Biological DVH ✅
- **File**: `code2_bDVH.py`
- **Function**: Generates EQD2-based biological DVH
- **Output**: `bDVH_csv/`, `bDVH_plots/`
- **Status**: Working

#### Step 3: NTCP Analysis with ML ✅
- **File**: `code3_ntcp_analysis_ml.py`
- **Function**: Comprehensive NTCP analysis
  - Classical models (LKB, RS Poisson)
  - ML models (ANN, XGBoost)
  - Uncertainty quantification
- **Output**: `ntcp_results.xlsx`, models (PKL), plots (PNG)
- **Status**: Working

#### Step 3b: QUANTEC Stratification ✅
- **File**: `quantification/quantec_stratifier.py`
- **Function**: QUANTEC-based risk stratification
- **Output**: Stratification tables and plots
- **Status**: Working

#### Step 3c: Tiered NTCP Analysis ✅
- **File**: `tiered_ntcp_analysis.py`
- **Function**: Four-tier NTCP framework
- **Output**: Tiered results (Excel, JSON)
- **Status**: Working

#### Step 4: QA Reporter ✅
- **File**: `code4_ntcp_output_QA_reporter.py`
- **Function**: Quality assurance reporting
- **Output**: QA summary tables (Excel, DOCX)
- **Status**: Working

#### Step 5: Clinical Factors Analysis ✅
- **File**: `code5_ntcp_factors_analysis.py`
- **Function**: Clinical factor association analysis
- **Output**: Factor analysis tables, correlation matrices
- **Status**: Working

#### Step 6: Publication Diagrams ✅
- **File**: `code6_publication_diagrams.py`
- **Function**: Publication-ready figures (600 DPI)
- **Output**: High-resolution plots (PNG, SVG)
- **Status**: Working

#### Step 7: SHAP Analysis ✅
- **File**: `shap_code7.py`
- **Function**: True-model SHAP explanations
- **Output**: SHAP plots (beeswarm, bar), SHAP tables (Excel)
- **Status**: Working

#### Step 8: Publication Tables Summary ✅
- **File**: `supp_results_summary.py`
- **Function**: Publication-ready tables
- **Output**: LaTeX tables, Excel summaries
- **Status**: Working

### Key Features Status

| Feature | Status | Notes |
|---------|--------|-------|
| Patient-level splitting | ✅ | Prevents data leakage |
| Feature selection | ✅ | Radiobiology-guided |
| EPV enforcement | ✅ | Refuses training if EPV < 5 |
| Overfitting prevention | ✅ | Conservative ML architectures |
| AUC with CI | ✅ | Bootstrap and DeLong methods |
| Calibration correction | ✅ | Platt scaling, isotonic regression |
| SHAP explanations | ✅ | True-model SHAP |
| Uncertainty quantification | ✅ | Monte Carlo, probabilistic models |
| Clinical safety guard | ✅ | Underprediction risk detection |
| Publication outputs | ✅ | 600 DPI figures, LaTeX tables |

---

## Test Report

### Test Suite Overview

**Total Tests**: 49  
**Status**: All Passing ✅  
**Framework**: pytest  
**Location**: `tests/`

### Test Categories

#### 1. Data Validation Tests

**File**: `tests/test_dvh_validation.py`

| Test | Status | Description |
|------|--------|-------------|
| `test_v0_equals_100` | ✅ | Validates DVH volume normalization |
| `test_dvh_monotonicity` | ✅ | Checks dose-volume monotonicity |
| `test_dose_non_negative` | ✅ | Ensures non-negative doses |
| `test_volume_range` | ✅ | Validates volume in [0, 100] |
| `test_gEUD_reproducibility` | ✅ | Checks gEUD calculation consistency |

#### 2. Data Splitting Tests

**File**: `tests/test_data_splitter.py`

| Test | Status | Description |
|------|--------|-------------|
| `test_train_test_no_overlap` | ✅ | Ensures no patient overlap |
| `test_patient_level_split` | ✅ | Validates patient-level splitting |
| `test_leakage_detection` | ✅ | Tests leakage detection |
| `test_stratification` | ✅ | Validates outcome stratification |
| `test_reproducibility` | ✅ | Checks random seed reproducibility |

#### 3. ML Model Tests

**File**: `tests/test_ml_models.py`

| Test | Status | Description |
|------|--------|-------------|
| `test_epv_calculation` | ✅ | Validates EPV calculation |
| `test_epv_error_very_low_epv` | ✅ | Tests EPV < 5 error |
| `test_epv_warning_low_epv` | ✅ | Tests EPV warning (5-10) |
| `test_ann_model_creation` | ✅ | Tests ANN model creation |
| `test_xgboost_model_creation` | ✅ | Tests XGBoost model creation |
| `test_complexity_adjustment_small_sample` | ✅ | Tests dynamic complexity adjustment |
| `test_nested_cv` | ✅ | Tests nested cross-validation |

#### 4. Feature Selection Tests

**File**: `tests/test_feature_selector.py`

| Test | Status | Description |
|------|--------|-------------|
| `test_parotid_essential_features` | ✅ | Validates Parotid essential features |
| `test_feature_selection_max_features` | ✅ | Tests EPV-based feature capping |
| `test_epv_based_feature_capping` | ✅ | Tests feature reduction |
| `test_statistical_filtering` | ✅ | Tests univariate filtering |
| `test_other_organs` | ✅ | Tests organ-specific features |

#### 5. AUC Calculator Tests

**File**: `tests/test_auc_calculator.py`

| Test | Status | Description |
|------|--------|-------------|
| `test_auc_calculation_bootstrap` | ✅ | Tests bootstrap AUC CI |
| `test_auc_calculation_delong` | ✅ | Tests DeLong AUC CI |
| `test_auc_high_vs_low` | ✅ | Tests AUC discrimination |
| `test_auc_requires_both_classes` | ✅ | Tests binary class requirement |
| `test_compare_aucs_delong` | ✅ | Tests DeLong comparison |
| `test_auc_ci_coverage` | ✅ | Tests CI coverage |

#### 6. Clinical Safety Tests

**File**: `tests/test_clinical_safety.py`

| Test | Status | Description |
|------|--------|-------------|
| `test_safety_guard_initialization` | ✅ | Tests safety guard setup |
| `test_fit_training_data` | ✅ | Tests training data fitting |
| `test_evaluate_safety_basic` | ✅ | Tests basic safety evaluation |
| `test_do_not_use_flag_low_ccs` | ✅ | Tests DO_NOT_USE flag |
| `test_underprediction_risk_detection` | ✅ | Tests underprediction detection |
| `test_safety_report_generation` | ✅ | Tests safety report generation |
| `test_safety_report_save` | ✅ | Tests report saving |

#### 7. Calibration Correction Tests

**File**: `tests/test_calibration_correction.py`

| Test | Status | Description |
|------|--------|-------------|
| `test_platt_scaling` | ✅ | Tests Platt scaling |
| `test_isotonic_regression` | ✅ | Tests isotonic regression |
| `test_calibration_slope_calculation` | ✅ | Tests calibration slope |
| `test_calibration_improves_slope` | ✅ | Tests calibration improvement |

#### 8. Auto Feature Reducer Tests

**File**: `tests/test_auto_feature_reducer.py`

| Test | Status | Description |
|------|--------|-------------|
| `test_auto_reduction_low_epv` | ✅ | Tests automatic feature reduction |
| `test_no_reduction_adequate_epv` | ✅ | Tests no reduction when EPV adequate |

#### 9. NTCP Mathematics Tests

**File**: `tests/test_ntcp_mathematics.py`

| Test | Status | Description |
|------|--------|-------------|
| `test_ntcp_bounds` | ✅ | Tests NTCP in [0, 1] |
| `test_ntcp_monotonicity` | ✅ | Tests dose-response monotonicity |
| `test_ntcp_at_td50` | ✅ | Tests NTCP = 0.5 at TD50 |
| `test_ntcp_edge_cases` | ✅ | Tests edge cases (zero dose, high dose) |

#### 10. Integration Tests

**File**: `tests/test_integration.py`

| Test | Status | Description |
|------|--------|-------------|
| `test_patient_level_split_integration` | ✅ | Tests patient-level split integration |
| `test_feature_selection_integration` | ✅ | Tests feature selection integration |
| `test_ml_training_integration` | ✅ | Tests ML training integration |
| `test_complete_workflow` | ✅ | Tests end-to-end workflow |

#### 11. Regression Tests

**File**: `tests/regression/test_baseline_regression.py`

| Test | Status | Description |
|------|--------|-------------|
| `test_baseline_exists` | ✅ | Validates baseline capture |
| `test_classical_ntcp_outputs_unchanged` | ✅ | Tests classical NTCP stability |

### Test Execution

**Command**: `pytest -q` or `python run_all_tests.py`

**Output**: JUnit XML report (`test_reports/pytest_report.xml`)

**Coverage**:
- Data validation: ✅
- Data splitting: ✅
- ML models: ✅
- Feature selection: ✅
- AUC calculation: ✅
- Clinical safety: ✅
- Calibration: ✅
- NTCP mathematics: ✅
- Integration: ✅
- Regression: ✅

---

## Capabilities Summary

### Core Capabilities

1. **DVH Processing**
   - Raw DVH text → Standardized CSV
   - Cumulative and differential DVH
   - Biological DVH (EQD2)

2. **NTCP Modeling**
   - 4-tier framework (Legacy, MLE, Modern, AI)
   - 8+ NTCP models
   - Parameter refitting (MLE)
   - Uncertainty quantification

3. **Machine Learning**
   - ANN and XGBoost
   - Overfitting prevention
   - EPV-aware training
   - Feature selection

4. **Explainable AI**
   - SHAP explanations
   - Feature importance
   - Individual predictions

5. **Clinical Analysis**
   - Factor association analysis
   - Statistical testing
   - Correlation analysis
   - Organ-specific effects

6. **Statistical Validation**
   - Bootstrap confidence intervals
   - DeLong test for AUC comparison
   - Calibration correction
   - Nested cross-validation

7. **Quality Assurance**
   - Data leakage detection
   - Clinical safety guard
   - Cohort consistency score
   - QA reporting

8. **Publication Outputs**
   - 600 DPI figures
   - LaTeX tables
   - Excel summaries
   - Comprehensive reports

### Performance Characteristics

- **Reproducibility**: ✅ Global random seed management
- **Scalability**: Handles datasets with 50-500+ patients
- **Robustness**: Error handling, fallback mechanisms
- **Documentation**: Comprehensive API and methodology docs

### Limitations

1. **Sample Size Requirements**
   - EPV < 5: Training refused
   - Recommended: EPV ≥ 10 for reliable ML

2. **Organ Support**
   - Optimized for head & neck organs
   - Extensible to other sites

3. **Clinical Factors**
   - Requires standardized clinical data format
   - Clinical Contract v2 compliance

### Future Enhancements

- Additional ML models (Random Forest, SVM)
- Deep learning architectures
- Multi-organ joint modeling
- Real-time prediction API
- Web-based interface

---

## Conclusion

**py_ntcpx v2.0.0** represents a comprehensive, publication-ready pipeline for NTCP analysis with:

- ✅ **Rigorous methodology**: Patient-level splitting, EPV enforcement, overfitting prevention
- ✅ **Comprehensive models**: 4-tier NTCP framework, ML models, uncertainty quantification
- ✅ **Explainable AI**: SHAP-based model interpretation
- ✅ **Statistical validation**: Bootstrap CI, DeLong test, calibration correction
- ✅ **Clinical integration**: Factor analysis, safety guard, QA reporting
- ✅ **Production quality**: 49 tests passing, comprehensive documentation

The pipeline is ready for clinical research and publication use.

---

**Report Generated**: 2024  
**Version**: 2.0.0  
**Status**: Production Ready ✅

