# py_ntcpx — changelog

## v1.1.1 — correctness release (2026-09-02)

Eleven defects were found while re-executing the published v1.1.0 pipeline from the raw
dose–volume histograms, reconciling every published value against that run, and responding
to peer review. All are correctness or reporting fixes; no study design, model,
hyperparameter or dataset changed.

### Fixed

- **Monte Carlo samplers were unseeded (uNTCP not reproducible).**
  `ProbabilisticgEUDModel.calculate_ntcp_distribution()` and
  `MonteCarloNTCPModel.calculate_ntcp_with_uncertainty()` accepted a `random_state`
  argument and silently discarded it, and `code3_ntcp_analysis_ml.py` called both without
  a seed. Five repeat calls on one patient gave Monte Carlo CI widths from 0.007 to 0.026.
  The uNTCP point estimate was stable to about ±0.004, but the uncertainty band — the
  purpose of the metric — was not reproducible between runs. `random_state` is now
  forwarded through both wrappers and both call sites are seeded (42). Two independent
  full pipeline runs now produce bit-identical uNTCP.

- **Tier-4 "apparent AUC" was the first cross-validation fold's model applied to every
  patient.** `code3_ntcp_analysis_ml.py` stored `ann_models[0]`, `xgb_models[0]` and
  `rf_models[0]` for prediction, which is neither an apparent nor an out-of-fold figure.
  Each model is now refit on the full cohort. Effect on the reported values:
  ANN apparent 0.547 → 0.511, XGBoost 0.619 → 0.783, random forest 0.749 → 0.891;
  overfitting gaps −0.003 → −0.040, +0.248 → +0.402 and +0.449 → +0.590. XGBoost moves
  from HIGH to CRITICAL, which is what both the documented rule and `code4:212` always
  implied.

- **Summary metrics were computed from 4-decimal-rounded predictions.**
  `tiered_ntcp_analysis.load_existing_results()` preferred `ntcp_results.xlsx`, whose NTCP
  columns are rounded to four decimals by `code3`, over the full-precision
  `enhanced_ntcp_calculations.csv`. The rounding manufactured 12 ties among 54 patients and
  shifted the LKB probit AUC from 0.5358 to 0.5407 — the source of the long-standing
  disagreement between the summary tables and the per-patient data. The full-precision file
  is now loaded first.

- **Tier-3 cross-validation standardised on the whole cohort.**
  `ModernLogisticNTCP.predict_ntcp_cv()` called `StandardScaler().fit_transform(X)` before
  the leave-one-out loop. The scaler is now fitted inside each fold. LOO-CV AUC
  0.6788 → 0.6732; overfitting gap 0.1823 → 0.1879.

- **C9 — SHAP and LIME were un-seeded.** `calculate_bootstrap_shap` drew its 100 bootstrap
  resamples from the global NumPy state and `LimeTabularExplainer` was constructed without
  `random_state`, so feature rank means, rank standard deviations and LIME attribution
  weights changed between otherwise identical runs. Both are now seeded (42), matching the
  Monte Carlo fix above.

- **C10 — event rates above 100%.** `code4_ntcp_output_QA_reporter` builds its working frame
  by concatenating several result files, so each patient appears several times. Patient
  count was de-duplicated but the event count was not, giving `n 54, n_rows 162, events 93,
  event_rate 172.2%` in `qa_summary_tables.xlsx`. Events are now counted per patient.

- **C11 — the software recommended clinical use on the basis of apparent performance.**
  `get_clinical_recommendation` received the best *apparent* AUC and, at 0.891 (the
  full-cohort random forest), emitted "Excellent discrimination - highly suitable for
  clinical use". That is the exact failure mode this package exists to detect. The function
  no longer produces clinical-use language and states that the figure is apparent only.

- **C12 — data-quality rating asserted "ML reliable"** at >=50 patients and >=15 events.
  Event count alone does not establish reliability; events per variable does. Reworded.

- **C13 — CCS was written to four identical "tier-specific" columns.**
  `CCS_QUANTEC`, `CCS_MLE`, `CCS_Logistic` and `CCS_ML` were assigned from one vector, while
  the manuscript described them as computed against tier-specific reference populations.
  CCS depends on the patient's dosimetric profile and the reference cohort, not on the NTCP
  model, so one column `CCS_cohort` is emitted. A `ccs_specification.json` now records the
  feature set actually used (`mean_dose, gEUD, V30, V50, D20, max_dose` — six metrics, not
  the V5-V70 set the manuscript described), the covariance treatment, and the fact that the
  0.10 threshold is an adaptive heuristic keyed to cohort size rather than a published
  cutoff. Separately, `code3` computed a *different* CCS on the Tier-4 ML feature subset and
  stored it as `CCS`; the two differ by up to 0.72, so it is renamed `CCS_ml_features`.

- **C16 — a third, contradictory overfitting-grading rule.**
  `tiered_ntcp_analysis` wrote `Overfitting_Flag` as `'High' if gap > 0.1 else ''`, so
  `ml_validation.xlsx` graded the random forest (gap 0.590, cross-validated AUC 0.300) the
  same as a mild 0.11 gap and never emitted CRITICAL, while
  `code4_ntcp_output_QA_reporter.flag_overfitting` implements the documented four-level
  scheme. The workbook therefore contradicted both the documentation and the other QA
  module. `tiered_ntcp_analysis` now calls `flag_overfitting` directly, so one rule
  produces every grade: CRITICAL when gap > 0.30 or cross-validated AUC < 0.50, HIGH when
  gap > 0.15 or (cross-validated AUC < 0.55 and apparent AUC > 0.65), MODERATE when
  gap > 0.10, otherwise NONE. Under the single rule the Tier-3 logistic model (gap 0.188)
  grades HIGH, not MODERATE.

- **C14 — version strings hard-coded to `v1.0`** in the reproducibility appendix. Now read
  from `VERSION`.

- **C15 — the clinical file had to be passed twice.** Omitting `--clinical_file` silently
  dropped the `age_over_50` indicator from the Tier-3 model, changing the reported result.
  `run_pipeline` now falls back to `--patient_data` and logs that it has done so.

### Also

- `VERSION` and `config/pipeline_config.yaml` said `1.0.0` while the tag, the Zenodo release
  and the manuscript said v1.1.0. Both now read `1.1.1`.
- `REPRODUCIBILITY_README.md` omitted `--clinical_file`. Without it the Tier-3 model drops
  the `age_over_50` indicator, giving 24 features, EPV 1.3 and a LOO-CV AUC of 0.536 instead
  of 0.673. The required invocation is documented.

### Data archive

Zenodo release 1.1.0 shipped a clinical workbook that is not the analysis dataset: it holds
34 events against the 31 analysed, and disagrees with the analysis data on 25 of 54 toxicity
labels, 51 of 54 ages, 10 of 54 sex entries and 8 of 54 follow-up times. Release 1.1.1
replaces it with the workbook that reproduces the published results.


## v1.1.0 (Tiered NTCP + Honest Evaluation)

### Model Completeness and EPV Safety

- **Tier 2 RS Poisson MLE (G1)**  
  - Wired `LegacyMLENTCP.fit_rs_poisson_mle` into `tiered_ntcp_analysis.py`.  
  - DVH arrays are passed from `DVHProcessor` (or an optional pre-loaded `dvh_dict`) so that RS Poisson MLE parameters are fitted and per-patient `NTCP_RS_Poisson_MLE` predictions are added.  
  - `model_parameters_mle.json` now includes an `RS_Poisson_MLE` entry when the fit converges.

- **Shared EPV checker (G2)**  
  - Added `EPVError` and `check_epv()` to `ntcp_models.__init__` for reusable Events-Per-Variable checks across classical and ML models.  
  - `check_epv` raises `EPVError` when EPV falls below a configurable minimum (default 10).

- **Tier 3 Logistic EPV-aware auto-reduction (G2, G11)**  
  - `ModernLogisticNTCP` now:
    - Tracks EPV per organ via `self.epv_` and whether auto-reduction was applied via `self.epv_reduced_`.  
    - Uses `check_epv` on the training set; if EPV < 10, it auto-reduces to the top `⌊events/10⌋` features using univariate correlation, then revalidates EPV.  
  - Clinical covariates are EPV-gated:
    - Introduces `CLINICAL_CANDIDATES` (`age`, `sex_binary`, `tobacco_exposure`, `chemotherapy`, `hpv_status`, `baseline_xerostomia`) and `CLINICAL_PRIORITY` (`age`, `baseline_xerostomia`, `tobacco_exposure`).  
    - Allocates an EPV budget `⌊events/10⌋ - n_dvh_features` for adding clinical factors in priority order.  
    - Optionally encodes `age_over_50` in addition to continuous `age` when `include_age=True`.

- **Tier 4 ML strict EPV flag (G2)**  
  - `OverfitResistantMLModels` now imports `check_epv` / `EPVError` and accepts `strict_epv: bool`.  
  - When `strict_epv=True`, EPV < 10 raises `EPVError` (hard gate).  
  - When `strict_epv=False` (default for backward compatibility), EPV < 10 emits a warning but does not block training; overfitting and low-EPV models are flagged downstream by QA and `NTCPEvaluator`.

### Honest Tier 3 Performance (Cross-Validation) (G3)

- **New `ModernLogisticNTCP.predict_ntcp_cv()`**  
  - Provides cross-validated NTCP predictions with:
    - `LOO` for `n < 100`  
    - Stratified 5-fold for `n ≥ 100`  
  - Returns:
    - `predictions_cv`, `predictions_apparent`  
    - `cv_auc`, `cv_auc_std`, `loo_auc`, `apparent_auc`, `overfitting_gap`  
    - `cv_strategy`, `epv`, `n_features`, `feature_names`, `fold_aucs`.

- **Tier 3 integration in `tiered_ntcp_analysis.py`**  
  - Replaced in-sample logistic predictions with:
    - `NTCP_MV_Logistic_apparent` (full-data model)  
    - `NTCP_MV_Logistic_cv` (CV predictions, honest AUC).  
  - Preserves legacy `NTCP_LOGISTIC` as an alias to the apparent predictions.  
  - Stores per-organ Tier 3 metrics in `results_df.attrs['tier3_metrics']` for unified evaluation.

### Boundary Detection for Classical Refits (G13)

- **Biological refitting**  
  - `biological_refitting.py` already implemented robust bootstrap confidence intervals and unstable-fit flags; v1.1.0 leverages these in the unified evaluator instead of altering the fitting internals.

## Unified Evaluation and Outputs

### NTCPEvaluator (G4, G8, G9, G10)

- **New module `src/metrics/ntcp_evaluator.py`**  
  - Defines a `ModelMetrics` dataclass capturing discrimination, calibration, EPV, and QA flags for any NTCP model tier.  
  - `NTCPEvaluator.evaluate()`:
    - Computes apparent AUC + 95% bootstrap CI (all models).  
    - Incorporates CV AUC when CV predictions are supplied (e.g., Tier 3 / Tier 4).  
    - Calculates Brier score, Expected Calibration Error (ECE), Maximum Calibration Error (MCE), and simple calibration slope/intercept.  
    - Derives EPV and sets `epv_flag` if EPV < 10 for multi-feature models.  
    - Flags overfitting when:
      - `gap > 0.10`, or  
      - `(cv_auc < 0.55 and apparent_auc > 0.65)`.
  - `NTCPEvaluator.to_dataframe()` and `save_performance_table()`:
    - Produce a publication-ready DataFrame and Excel workbook with:
      - `All_Models`, `Performance_Summary`, `Calibration`, and `QA_Flags` sheets.  
    - Apply simple conditional formatting to highlight low CV AUC and QA flags.

### Unified Tiered Evaluation & uNTCP (G3, G4, G6, G7, G9, G10, G15)

- **Tiered metrics integration**  
  - `tiered_ntcp_analysis.py` now:
    - Aggregates per-organ/per-model metrics across:
      - Tier 1A: `NTCP_T1A_LKB_LogLogistic`, `NTCP_T1A_LKB_Probit`, `NTCP_T1A_RS_Poisson`.  
      - Tier 2: `NTCP_T2_LKB_Probit_MLE`, `NTCP_T2_LKB_LogLogistic_MLE`, `NTCP_T2_RS_Poisson_MLE`.  
      - Tier 3: `NTCP_T3_MV_Logistic` (apparent + CV).  
      - Tier 4: `NTCP_T4_ANN_apparent`, `NTCP_T4_XGBoost_apparent`, `NTCP_T4_RF_apparent` (where available).  
    - Calls `NTCPEvaluator` to compute `ModelMetrics` for each model/organ pair.  
    - Writes a canonical `performance_summary_v1.1.xlsx` in the tiered output directory via `NTCPEvaluator.save_performance_table()`.

- **uNTCP computation inside tiered pipeline (G6)**  
  - Added `compute_untcp(results_df)` to `tiered_ntcp_analysis.py`.  
  - Combines:
    - Probabilistic gEUD: `NTCP_Probabilistic_gEUD` / `ProbNTCP_Mean` / `Prob_gEUD_mean`  
    - Monte Carlo NTCP: `MC_NTCP_Mean` (with SD approximated from `MC_NTCP_CI_L`/`_U` where available).  
  - Uses inverse-variance weighting:
    - `uNTCP = (μ_p w_p + μ_m w_m) / (w_p + w_m)` with `w = 1 / σ²` where σ is available.  
    - Falls back to the arithmetic mean when variances are unavailable.  
    - If probabilistic or Monte Carlo NTCP are missing, falls back to `NTCP_LKB_Probit`.  
  - Stores:
    - `uNTCP`, `uNTCP_STD`, `uNTCP_CI_L`, and `uNTCP_CI_U` on `results_df`.

- **Canonical column deduplication (G7)**  
  - `ntcp_utils.deduplicate_ntcp_columns(df, canonical_map=None)`:
    - Normalizes Monte Carlo columns to `MC_NTCP_Mean`, `MC_NTCP_Std`, `MC_NTCP_CI_L`, `MC_NTCP_CI_U`.  
    - Normalizes probabilistic gEUD columns to `NTCP_Probabilistic_gEUD`, `NTCP_Probabilistic_gEUD_std`, and CI bounds.  
    - Maps `NTCP_LOGISTIC` → `NTCP_MV_Logistic_apparent` while preserving a `_DUPLICATE_CHECK` copy if values differ.  
  - Called in `tiered_ntcp_analysis.main()` before final Excel writes, ensuring a clean, non-duplicated per-patient table.

### QA Reporter and Overfitting Flags (G5, G8, G17)

- **Correct patient counting in `code4_ntcp_output_QA_reporter.py` (G5, G17)**  
  - Per-organ summary now distinguishes:
    - `n_rows`: row count.  
    - `n`: unique patient count via `PrimaryPatientID`/`PatientID` where available.  
  - Global stats:
    - `global_rows` now uses `sum(n_rows)` rather than `sum(n)`.  
    - `global_patients` uses the unique patient ID set across all files.  
  - DOCX report text has been aligned to reflect unique patient counts and total rows via the updated columns.

- **Gap-based overfitting flag (G8)**  
  - Introduced `flag_overfitting(apparent_auc, cv_auc, overfitting_gap_threshold=0.10)`:
    - Returns `(flag: bool, severity: str, message: str)` with severity levels:
      - `CRITICAL`: `gap > 0.30` or `cv_auc < 0.50`.  
      - `HIGH`: `gap > 0.15` or `(cv_auc < 0.55 and apparent_auc > 0.65)`.  
      - `MODERATE`: `gap > 0.10`.  
      - `NONE`: otherwise.  
    - For fixed-parameter models (`cv_auc` absent), returns `'N/A'`.
  - ML QA sheet (`ml_validation.xlsx`) now:
    - Merges CV AUC metrics from `ml_cv_metrics.xlsx`.  
    - Computes `Overfitting_Gap`, `Overfitting_Flag_bool`, `Overfitting_Severity`, and `Overfitting_Message` per model.

## SHAP, Figures, and Output Structure (G7, G12, G14, G15)

- **SHAP indexing by patient ID (G12)**  
  - `shap_code7.py` now saves `shap_table.xlsx` with:
    - Index set to `AnonPatientID` if present in the feature matrix, otherwise `PrimaryPatientID`, and finally a simple index fallback.  
    - Ensures SHAP values can be directly joined to per-patient NTCP rows.

- **Figure deduplication (G14)**  
  - Added `save_figure_canonical(fig, output_dir, base_name, organ="", dpi=600, formats=None)` to `code6_publication_diagrams.py`.  
  - All `fig.savefig()` calls now go through `save_figure_canonical`, which:
    - Always writes an organ-suffixed (or base) file.  
    - Removes non-suffixed duplicates for organ-specific plots.  
  - Prevents unintentional duplication of figure variants in the `code6_output` directory.

- **Tiered master report and performance summary (G15)**  
  - `tiered_ntcp_analysis.py` now:
    - Writes the canonical performance workbook:
      - `performance_summary_v1.1.xlsx` via `NTCPEvaluator.save_performance_table(all_metrics, ...)`.  
    - Leaves the existing `NTCP_4Tier_Master.xlsx` structure intact for backward compatibility.  
    - Ensures the per-patient table includes both new canonical and legacy prediction names.

## Pipeline Orchestration & CLI (Versioning and Flags)

- **Version bump and metadata**  
  - Updated docstrings and logs to `py_ntcpx_v1.1.0` in:
    - `run_pipeline.py`  
    - `code6_publication_diagrams.py`  
    - `test_ntcp_pipeline.py` header text (where applicable).

- **New CLI flags in `run_pipeline.py`**  
  - `--strict_epv`:  
    - Plumbed to the pipeline (for now used conceptually to configure strict EPV behaviour; Tier 4 models accept a `strict_epv` argument).  
  - `--cv_strategy`:  
    - For Tier 3 logistic CV predictions in `tiered_ntcp_analysis.py`, accepts `'auto'`, `'LOO'`, or `'5-fold'`.  
  - `--include_age`:  
    - Controls whether age/age_over_50 are allowed as Tier 3 covariates (subject to EPV budget).  
  - `--resume_from`, `--skip` and other legacy arguments are preserved unchanged for backward compatibility.

## New Tests (v1.1.0)

- **Planned test cases (`tests/test_ntcp_evaluator.py`)**  
  - `test_epv_error_raised_when_too_low`: ensures `EPVError` is raised when logistic models attempt fitting with EPV < 10 under strict settings.  
  - `test_rs_mle_produces_predictions`: validates that RS Poisson MLE produces predictions when DVH arrays are provided (using synthetic test data).  
  - `test_t3_cv_predictions_stored`: checks that `NTCP_T3_MV_Logistic_cv` differs from apparent and uses LOO when `n < 100`.  
  - `test_patient_count_not_inflated`: asserts that `code4`’s per-organ `n` equals the unique patient count rather than row count.  
  - `test_overfitting_flag_gap_based`: confirms that models with `gap > 0.10` are flagged regardless of absolute AUC.  
  - `test_untcp_assembled_in_tiered`: ensures `uNTCP` exists in tiered output even if absent from code3 output.  
  - `test_no_duplicate_mc_columns`: verifies that the final master output contains exactly the canonical three MC columns (Mean, CI_L, CI_U).  
  - `test_shap_indexed_by_patient_id`: checks that SHAP outputs use `AnonPatientID` (or `PrimaryPatientID`) as index.  
  - `test_unified_performance_table_all_tiers`: ensures `performance_summary_v1.1.xlsx` contains rows spanning all major model tiers.

> Note: v1.1.0 changes are strictly additive and maintain backward compatibility with the `run_pipeline.py` entry point, step IDs, and legacy column names. Existing downstream scripts depending on `NTCP_LOGISTIC`, `MC_NTCP_Mean`, etc., continue to function, while new canonical names and unified evaluation outputs are provided for publication-grade reporting.

# Changelog

All notable changes to py_ntcpx since the first public release (v1.0.0) are documented here.

**Author:** K. Mondal (Medical Physicist, North Bengal Medical College, Darjeeling, India)  
**Repository:** https://github.com/kalyan2031990/py_ntcpx

---

## [Unreleased] – 2026-03-01 (post–v1.0.0 improvements)

### Added
- **ML CV metrics export (`code3`):** New file `ml_cv_metrics.xlsx` with Apparent_AUC, CV_AUC_mean, CV_AUC_std, Test_AUC, and Constant_Predictor per organ/model (ANN, XGBoost, Random Forest).
- **Calibration metrics:** ECE (Expected Calibration Error) and MCE (Maximum Calibration Error) in tiered `ml_validation.xlsx` for all models.
- **Overfitting flag:** Column `Overfitting_Flag` in `ml_validation.xlsx` set to `"High"` when Overfitting_Gap > 0.1.
- **Small-sample note:** New sheet `Note` in `ml_validation.xlsx` advising to prefer CV_AUC for ML and that results are exploratory when n < 100.
- **Constant-predictor warning:** Console warning when an ML model produces a single unique prediction value; `Constant_Predictor` column in `ml_cv_metrics.xlsx`.

### Changed
- **XGBoost (small cohorts):** In `src/models/machine_learning/ml_models.py`, for n < 100: `min_child_weight` 5→2, `learning_rate` 0.03→0.05; for n < 50: `max_depth` 1→2 to avoid constant predictions. Fallback `train_xgboost_model` in `code3_ntcp_analysis_ml.py`: `max_depth` 2, `min_child_weight` 2.
- **Apparent vs CV/Test AUC:** Tiered `ml_validation.xlsx` now uses a single **Apparent_AUC** column (replacing duplicate Train_AUC/Test_AUC). When code3 `ml_cv_metrics.xlsx` exists, tiered merges CV_AUC_mean, CV_AUC_std, Test_AUC and sets **Overfitting_Gap** = Apparent_AUC − CV_AUC_mean.
- **Test_AUC in CV path:** When the pipeline uses the CV-only path (n < 100), Test_AUC in `ml_cv_metrics.xlsx` is now set from CV_AUC_mean so the column is always populated.
- **Clinical factors (code5):** Accept **patient_id** (from code0 reconciled output) and map to PrimaryPatientID so Step 5 merges without manual column renaming.
- **Test:** `tests/test_ml_models.py` updated so that for very small sample, expected XGBOOST `max_depth` is 2 (was 1).

### Fixed
- XGBoost producing constant predictions (single value for all patients) on small cohorts.
- Test_AUC missing (NaN) in `ml_cv_metrics.xlsx` when using 5-fold CV path.
- Code5 factors analysis failing with "Clinical data must contain PrimaryPatientID, DVH_ID, or PatientID" when clinical data used code0's `patient_id` column.
- Overfitting reported as 0 despite large Apparent–CV gap; now Overfitting_Gap and Overfitting_Flag reflect actual gap.

### Documentation
- `docs/PIPELINE_QA_AND_IMPROVEMENTS.md` – QA notes, XGBoost explanation, and improvement suggestions.
- `docs/PIPELINE_IMPROVEMENTS_AND_ANALYSIS_REPORT.md` – Implementation summary and post-run analysis.
- `docs/RELEASE_NOTES_GITHUB.md` – Release text for GitHub (copy into release description).

---

## [1.0.0] – 2026-02-28

First public release. See [Release py_ntcpx v1.0.0](https://github.com/kalyan2031990/py_ntcpx/releases/tag/py_ntcpx_v1.0.0) for full notes.

- Pipeline steps 0–8: clinical reconciliation, DVH preprocessing, NTCP (classical + ML), tiered analysis, QA, factors, publication diagrams, SHAP/LIME, publication tables.
- Overfitting and data-leakage safeguards (EPV, CCS, conservative ML).
- SHAP and LIME explainability for ANN, XGBoost, Random Forest.
- Single canonical output layout under `out2/`.

[Unreleased]: https://github.com/kalyan2031990/py_ntcpx/compare/py_ntcpx_v1.0.0...HEAD
[1.0.0]: https://github.com/kalyan2031990/py_ntcpx/releases/tag/py_ntcpx_v1.0.0
