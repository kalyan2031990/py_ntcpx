py_ntcpx v3.0.1 – Documentation, Pipeline Orchestration, and Reporting Refinements
===============================================================================

**Release Date:** 2026-02-27  
**Version:** 3.0.1  
**Based on:** [py_ntcpx v3.0.0 – Uncertainty-aware NTCP Framework with Machine Learning Interpretability](https://github.com/kalyan2031990/py_ntcpx/releases/tag/v3.0.0)

---

New Enhancements in v3.0.1
--------------------------

This release is a **polishing and alignment** update on top of v3.0.0. All core scientific methods, CCS logic, and ML/XAI behaviour from v3.0.0 are preserved; v3.0.1 focuses on:

1. **Versioning & Documentation Clean‑up**
   - Updated all top‑level documents and headers to reflect **py_ntcpx v3.0.1**:
     - `ARCHITECTURE_REPORT.md` – version header, pipeline diagrams, and conclusion now labelled v3.0.1.
     - `OUTPUT_STRUCTURE.md` – document version set to v3.0.1; overview text updated from “v3.0.0 pipeline” to “v3.0.1 pipeline”.
     - Script headers and “Software: py_ntcpx …” banners in:
       - `run_pipeline.py`
       - `shap_code7.py`
       - `code0_clinical_reconciliation.py`
       - `code2_bDVH.py`
       - `code3_ntcp_analysis_ml.py`
       - `code6_publication_diagrams.py`
       - `quantification/quantec_stratifier.py`
       - `supp_results_summary.py`
       - Local test harnesses (`test_data_validation.py`, `test_ntcp_pipeline.py`)
   - Ensures that when users inspect logs, CLI help, or reports, the **software version is consistently reported as v3.0.1**, avoiding ambiguity between the v3.0.0 and v3.0.1 code drops.

2. **Architecture & Manuscript Support Materials**
   - Added and integrated new documentation and helper scripts for manuscript preparation and reproducible analysis:
     - `architecture.md` – concise overview of the v3.0.x architecture, explicitly connecting the fixed py_ntcpx implementation to the NTCP_Analysis_Pipeline v1.0.1 and the four‑tier framework.
     - `INSPECTION_REPORT_py_ntcpx_v3_alignment.md` – detailed inspection report documenting:
       - RS formulation corrections (Källman voxel form, removal of `RS_LOCAL` shortcuts).
       - Alignment of dose–response plots and formulas with the J Med Phys 2026 manuscript and NTCP_Analysis_Pipeline v1.0.1.
       - What was changed vs left as a separate “interpretive” layer (biological mean‑dose fits).
     - `build_master_csvs.py` – utility for building:
       - `structured_output/core_results/comprehensive_master_data.csv` – direct export of `ntcp_results.xlsx`.
       - `complete_data.csv` – compact per‑patient CSV with key DVH, NTCP and model outputs, mirrored into `structured_output/manuscript_materials/tables/` for direct use in figure/table scripts.
     - `manuscript_preparation_materials.md` – guide for using `structured_output/` and `complete_data.csv` to regenerate figures and tables for the manuscript.
     - `py_ntcpx_test_report.txt` and `test_report_prev.md` – preserved test summaries for local runs, anchoring the 80‑test pytest suite used in v3.0.0.
   - These additions make **manuscript‑oriented work** (tables, figures, supplementary material) easier and more traceable, while keeping all clinical logic unchanged.

3. **Pipeline Orchestration & Contracts (run_pipeline.py)**
   - Hardened and documented the complete pipeline orchestrator for **Windows‑friendly, contract‑validated runs**:
     - Confirms **clinical contract v2** via `code0_clinical_reconciliation.py` before NTCP analysis:
       - Validates that `clinical_reconciled.xlsx` exists and contains `patient_id`.
       - Logs match statistics between `Step1_DVHRegistry.xlsx` and reconciled clinical data when `ContractValidator` is available.
     - Enforces contract‑based gating before key QA stages:
       - Step‑2 DVH summary now writes a `Step2_DVHSummary` contract when successful.
       - Step‑3 NTCP analysis can generate `Step3_NTCPDataset.xlsx` as a contract snapshot from `ntcp_results.xlsx`.
       - Step‑4 QA reporter validates `Step3_NTCPDataset` against `Step1_DVHRegistry` before computing ROC/calibration.
     - Adds a **structured output view**:
       - `structured_output/core_results/`, `structured_output/tiered_ntcp/`, `structured_output/qa_reports/`, `structured_output/clinical_factors/`, `structured_output/figures/`, and `structured_output/manuscript_materials/…`.
       - Curates the most important Excel/CSV results and 600‑DPI figures for manuscript work without touching raw outputs.
     - Provides a more explicit CLI:
       - Help text now labels this as the **py_ntcpx v3.0.1** pipeline orchestrator.
       - Adds resume/skip options and clear logging around which steps are executed, skipped, or resumed.
   - Net effect: the **end‑to‑end pipeline is easier to run reproducibly**, with contract checks surfaced more clearly and a manuscript‑friendly structure built automatically at the end of a successful run.

4. **SHAP + LIME and XAI Tooling (shap_code7.py)**
   - Keeps all v3.0.0 XAI behaviour, while clarifying versioning and CLI:
     - Header now declares **Software: py_ntcpx v3.0.1**.
     - CLI description updated to:  
       `"Step 7: True-Model SHAP Analysis (Clinical Grade) + LIME (v3.0.1)"`.
   - All v3.0.0 enhancements remain intact:
     - Model‑agnostic `shap.Explainer` for serialized XGBoost/RandomForest models with `base_score` fix.
     - ANN SHAP stability analysis with CCS‑aware warnings.
     - Per‑patient LIME explanations (HTML + PNG) for highest/median/lowest NTCP patients.

5. **Small Refinements to Supporting Scripts**
   - `code0_clinical_reconciliation.py`, `code2_bDVH.py`, `code6_publication_diagrams.py`, `quantification/quantec_stratifier.py`, `supp_results_summary.py`:
     - All now consistently report **Software: py_ntcpx v3.0.1** in logs and headers.
     - Preserve all v3.0.0 behaviour (no changes to equations, CCS thresholds, or QA logic).
   - Local test harnesses (`test_data_validation.py`, `test_ntcp_pipeline.py`) updated to reference v3.0.1, keeping the wording consistent with the current library version while the underlying test content remains unchanged from v3.0.0.

6. **No Scientific or API‑Level Breaking Changes**
   - v3.0.1 is **intentionally conservative**:
     - **No changes** to:
       - CCS adaptive thresholds or `CCS_Warning_Flag` semantics.
       - EPV enforcement / overfitting safeguards.
       - Classical NTCP equations, RS fixes, or uncertainty models.
       - SHAP/LIME algorithms and outputs.
     - The public API, folder structure under `out2/`, and all key file names (`ntcp_results.xlsx`, `NTCP_4Tier_Master.xlsx`, `publication_tables.xlsx`, etc.) are unchanged.
   - This release is safe to adopt wherever v3.0.0 was previously used; it simply provides **clearer documentation, more robust orchestration, and better manuscript tooling**.

7. **Testing Notes**
   - v3.0.1 **inherits the full v3.0.0 test suite** (80 tests, 78 runnable, 2 baseline regression tests skipped by design).
   - Local test reports (`test_report.md`, `py_ntcpx_test_report.txt`, `test_report_prev.md`) document:
     - 100% pass rate for runnable tests in the v3.0.0 environment.
     - Expected EPV warnings from deliberately small synthetic cohorts.
   - In environments where `pytest` is not installed as a command‑line entry point, tests can still be executed via the provided Python scripts (e.g. `run_all_tests.py`), subject to having the same Python and dependency stack used for v3.0.0.

---

All Features from v2.0.0 and v2.1.0 (Inherited from v3.0.0)
-----------------------------------------------------------

> The following sections are **copied from the v3.0.0 release** ([link](https://github.com/kalyan2031990/py_ntcpx/releases/tag/v3.0.0)) to provide a complete view of the framework. v3.0.1 **does not change any of these capabilities**; it only adds the refinements listed above.

### Data Integrity & Leakage Prevention

* ✅ Patient-level data splitting with stratification  
* ✅ LeakageAudit utility for automated leakage detection  
* ✅ StandardScaler fit only on training data  
* ✅ Split-before-transform enforcement  

### Overfitting Prevention & Model Containment

* ✅ EPV (Events Per Variable) enforcement (refuses to train if EPV < 5)  
* ✅ Auto feature reduction when EPV < 5  
* ✅ Conservative ML architectures (ANN, XGBoost)  
* ✅ Dynamic model complexity adjustment  
* ✅ Domain-guided feature selection  

### Statistical Rigor

* ✅ Correct Monte Carlo NTCP with parameter uncertainty  
* ✅ Bootstrap confidence intervals for all metrics  
* ✅ DeLong test for AUC comparison with Bonferroni correction  
* ✅ Nested cross-validation for unbiased performance estimation  
* ✅ Calibration correction (Platt scaling, isotonic regression)  

### Clinical Safety Layer

* ✅ ClinicalSafetyGuard with underprediction risk detection  
* ✅ Cohort Consistency Score (CCS) integration  
* ✅ Adaptive CCS thresholds (v3.0.0 enhancement)  
* ✅ CCS warnings instead of DO_NOT_USE flags (v3.0.0)  
* ✅ Automated safety reports  

### Model Documentation

* ✅ Auto-generated model cards  
* ✅ EXPLORATORY labels for ML models  
* ✅ Intended use, limitations, and failure modes documented  

### Outputs

* ✅ 600 DPI figures  
* ✅ LaTeX tables for manuscript  
* ✅ Statistical reporting with confidence intervals  
* ✅ Comprehensive documentation  

### Reproducibility

* ✅ Global random seed management  
* ✅ YAML configuration management  
* ✅ Dependency locking  
* ✅ Baseline capture for regression testing  

### Small Dataset Support (from v2.1.0)

* ✅ Dynamic CCS threshold (enhanced in v3.0.0)  
* ✅ Clinical factor integration  
* ✅ Small dataset adaptations  
* ✅ Robust SHAP analysis (enhanced in v3.0.0)  
* ✅ Enhanced reporting  

---

Key Enhancements Introduced in v3.0.0 (Retained in v3.0.1)
---------------------------------------------------------

The full v3.0.0 release notes – including **adaptive CCS thresholds, CCS warnings, fixed XGBoost SHAP, improved ANN SHAP stability, and LIME integration** – continue to apply in v3.0.1. For details, see the original v3.0.0 release text at:

- `https://github.com/kalyan2031990/py_ntcpx/releases/tag/v3.0.0`

v3.0.1 should be viewed as:

> **“py_ntcpx v3.0.0 + better documentation, orchestration, and manuscript tooling, with the same scientific core.”**

