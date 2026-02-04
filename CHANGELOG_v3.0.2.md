# Changelog - Version 3.0.2

## Patch Release: py_ntcpx v3.0.2 - AUC 95% CI and LIME Fix

**Release Date:** 2026-02-04

---

## Fixes

### 1. AUC 95% CI in Table X (Classical vs ML)
- **Enhancement:** Table X now includes `AUC_95CI` column with bootstrap 95% confidence intervals.
- **Implementation:** Uses `src.metrics.auc_calculator.calculate_auc_with_ci` (bootstrap method).
- **Output:** Format `lower-upper` (e.g., `0.52-0.78`) for each model in `tables/Table_X_Classical_vs_ML.xlsx`.

### 2. LIME PNG for ANN (Patients 33, 47)
- **Issue:** `explanation.as_pyplot_figure()` raised `KeyError: 1` when label not in `local_exp`.
- **Fix:** Pass explicit `label` from `explanation.local_exp.keys()` to `as_pyplot_figure(label=label)`.

---

## Files Modified

- `code4_ntcp_output_QA_reporter.py` - AUC 95% CI in Table X
- `shap_code7.py` - LIME PNG label fix
- `VERSION` - 3.0.2
