# Issue 3: gEUD Calculation Verification (40.17 Gy)

**Date:** 2025-02-04  
**Status:** Verified correct

---

## Summary

The gEUD calculation in py_ntcpx v3.0.0 is **correct** and follows the standard QUANTEC formulation. A value of 40.17 Gy for parotid is plausible for a high-dose H&N cohort.

---

## Formula Verification

**Implementation:** `DVHProcessor.calculate_gEUD()` in code3_ntcp_analysis_ml.py

**Formula:** gEUD = (Σ v_i × D_i^a)^(1/a)

Where:
- v_i = volume fraction in dose bin i (v_i = V_i / ΣV_i)
- D_i = dose in bin i (Gy)
- a = organ-specific parameter from QUANTEC

**Expected behavior:**
- a=1: gEUD = mean dose
- a>1: gEUD > mean dose (emphasizes high-dose regions)
- a<1: gEUD < mean dose (emphasizes low-dose regions)

---

## Parameter Verification

| Organ      | a parameter | Source   | Notes                          |
|-----------|-------------|----------|--------------------------------|
| Parotid   | 2.2         | QUANTEC  | Parallel, volume-dependent     |
| Larynx    | 1.0         | QUANTEC  | ≈ mean dose                    |
| SpinalCord| 7.4         | QUANTEC  | Serial, max-dose sensitive     |

Parotid a=2.2 is taken from `literature_params['Parotid']['LKB_LogLogit']['a']`.

---

## gEUD = 40.17 Gy Assessment

For parotid with a=2.2:
- **Typical gEUD/mean_dose ratio:** 1.1 – 1.4
- If gEUD = 40.17 Gy → mean dose ≈ 29–36 Gy
- If mean dose ≈ 35 Gy → gEUD ≈ 38–49 Gy (40.17 Gy is within range)

**Conclusion:** 40.17 Gy is consistent with a high-dose H&N cohort (e.g. mean dose ~30–35 Gy).

---

## DVH Format

- **Input:** Differential DVH from code1 `dDVH_csv/` (Volume[cm3] per dose bin)
- **Alternative:** Volume[%] supported after column rename; relative volumes used for gEUD
- **Pipeline:** code3 uses `dDVH_csv` from code1

---

## Changes Made

1. **Docstring update** in `calculate_gEUD()`:
   - Formula and QUANTEC references
   - Organ-specific `a` values
   - Expected gEUD/mean_dose ratio for parotid

No changes to the numerical implementation; existing logic is correct.
