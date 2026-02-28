# Pipeline QA: Realism, Clinical Plausibility & Improvements

## 1. Are the pipeline outputs realistic and correct?

**Short answer: The methodology is correct; some numbers reflect data limitations, not bugs.**

- **Classical models (LKB, Logistic, MLE):** Implementations and reported metrics are consistent. AUCs ~0.53–0.70 for Parotid xerostomia are in line with small cohorts and single-organ endpoints.
- **ML validation:** Apparent vs CV-AUC is now correctly separated. Random Forest shows **large overfitting** (Apparent 0.75 vs CV 0.30) — the pipeline correctly reports this via `Overfitting_Gap`.
- **XGBoost:** In the current run, XGBoost outputs a **single constant value** (0.577) for all patients, so it has no discriminative power. That is why:
  - **Apparent AUC = CV-AUC = 0.5** (they “match” because both are 0.5).
  - **CV_AUC_std = 0** (every fold gives AUC 0.5).
  This is a **model behaviour / configuration issue**, not a bug in the metric logic.
- **Test_AUC** is NaN in `ml_cv_metrics.xlsx` because the path that runs a train/test split and stores `test_AUC` is not the one used in this pipeline run (e.g. only the CV-only path runs). Filling Test_AUC when a holdout is used would improve interpretability.

---

## 2. Are the results clinically plausible?

**Generally yes for classical NTCP; ML needs caution.**

- **Endpoint (grade ≥2 xerostomia, Parotid):** Standard endpoint; incidence and dose–response are in a plausible range for H&N RT.
- **LKB / Logistic parameters:** TD50 and shape parameters are in ranges reported in the literature for parotid and xerostomia.
- **AUCs ~0.5–0.7:** For a single organ, limited features, and n≈54, moderate discriminative performance is expected. AUC ~0.5 for some models indicates no discriminative ability (e.g. constant predictor), not necessarily wrong code.
- **Random Forest:** Apparent AUC 0.75 is **not** a reliable performance estimate; CV-AUC ~0.30 shows the model does not generalize. For clinical use, **CV-AUC (and external validation)** should be the basis, not Apparent AUC.
- **Calibration:** Not fully assessed here; adding calibration curves and metrics (e.g. ECE, MCE) would strengthen clinical plausibility.

---

## 3. Why does XGBoost Apparent AUC exactly match CV-AUC?

**Because XGBoost is predicting the same value for every patient.**

- In `enhanced_ntcp_calculations.csv`, **NTCP_ML_XGBoost** has **one unique value** (0.5767667) for all 54 rows.
- A model that does not separate events vs non-events has:
  - **Apparent AUC = 0.5**
  - **CV AUC = 0.5** in every fold → **CV_AUC_mean = 0.5**, **CV_AUC_std = 0**.
- So the “exact match” is from **no discrimination**, not from a special property of the pipeline. Possible causes:
  - Strong regularization / small data (e.g. default XGBoost with n=54).
  - Feature set or preprocessing (e.g. constant or near-constant features).
  - Class imbalance or random seed leading to a single-leaf or constant solution.
- **Recommendation:** Diagnose XGBoost (learning curves, feature importance, predictions distribution). Consider simpler baselines (e.g. logistic on same features) and document that ML is exploratory at n=54.

---

## 4. Suggested improvements (overall pipeline)

| Area | Suggestion |
|------|------------|
| **ML validation** | Ensure the code path that computes a holdout **Test_AUC** runs when a train/test split is used, and write it into `ml_cv_metrics.xlsx` so tiered/QA can show it. |
| **XGBoost** | Investigate constant predictions: check regularization, number of trees, max depth, and feature matrix; add a sanity check (e.g. warn if predictions have only one unique value). |
| **Small sample** | Add a clear note in reports that n≈54 is small for ML; prefer **CV-AUC** and **confidence intervals** (e.g. DeLong or bootstrap) for AUC; consider reporting EPV (events per variable). |
| **Calibration** | Add calibration metrics (e.g. Brier already present; ECE, MCE, calibration curves) and reference them in the QA report. |
| **Overfitting** | Use **Overfitting_Gap = Apparent_AUC − CV_AUC** (already done when CV metrics exist); consider flagging models with gap > 0.1 as “high overfitting” in the report. |
| **Reproducibility** | Fix seed for train/test and CV splits and document it; ensure one canonical split when reporting Test_AUC. |
| **Documentation** | In `COMPLETE_RESULTS_REPORT.md` (or equivalent), state that **Apparent_AUC** is in-sample and **CV-AUC** is the preferred metric for ML; explain that XGBoost 0.5 reflects no discrimination when it occurs. |
| **Clinical factors** | Resolve the “Clinical data must contain PrimaryPatientID, DVH_ID, or PatientID” merge error in the factors step so that step runs cleanly. |

---

*Generated from pipeline QA review (py_ntcpx).*
