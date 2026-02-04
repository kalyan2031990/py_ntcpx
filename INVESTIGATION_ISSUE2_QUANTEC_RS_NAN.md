# Issue 2: QUANTEC-RS Model Returns NaN – Investigation & Fix

**Date:** 2025-02-04  
**Status:** Fixed

---

## Summary

The RS Poisson (QUANTEC-RS) model could return NaN due to:
1. **Numerical instability** in the full DVH-based formula (overflow/underflow with small s, e.g. 0.01 for Parotid)
2. **DVH column mismatch** – `Volume[%]` not supported (e.g. synthetic/test data)
3. **No fallback** when the primary calculation failed

---

## Root Causes

### 1. Numerical Instability
- `np.power(1.0 - product_term, 1.0/s)` with s=0.01 → exponent 100; small rounding can make `(1 - product_term)` negative → NaN
- `exp(gamma * (1 - D/D50))` can overflow for extreme dose ratios
- No clipping of intermediate values

### 2. DVH Format
- Loader only supported `Dose[Gy]` + `Volume[cm3]` or `Dose` + `Volume`
- `Volume[%]` (e.g. synthetic data) caused KeyError → exception → 0.0 or NaN depending on path

### 3. No Fallback
- When the DVH-based RS Poisson failed, there was no alternative
- The gEUD-based formula NTCP = 1 - exp(-(gEUD/D50)^γ) is a standard QUANTEC formulation and is numerically stable

---

## Fix Implemented

### 1. DVH Loading (`load_dvh_file`)
- Support for `Volume[%]` in addition to `Volume[cm3]`
- Fallback column detection for columns containing "dose" and "volume"

### 2. `ntcp_rs_poisson` – Numerical Stability
- Clip exponent in `2^(-exp(...))` to [-50, 50] to avoid overflow
- Use `s_safe = max(s, 1e-6)` to avoid extreme exponents
- Clip `product_term` to `[0, 1 - 1e-10]` before `(1 - product_term)^(1/s)` to avoid NaN from negative base
- Cap `1/s` to 100 to reduce underflow
- Return `np.nan` (instead of 0.0) on failure so the caller can use the fallback

### 3. gEUD-Based Fallback in `calculate_all_ntcp_models`
- If `ntcp_rs_poisson` returns NaN or fails, use: NTCP = 1 - exp(-(gEUD_eqd2/D50)^γ)
- Same formula as in biological_refitting and common QUANTEC usage
- Ensures a valid NTCP whenever gEUD is available

---

## Files Modified

- `code3_ntcp_analysis_ml.py`:
  - DVH column handling in `load_dvh_file`
  - Numerical stability in `ntcp_rs_poisson`
  - gEUD fallback in `calculate_all_ntcp_models`

---

## Verification

- Test 1: Full DVH calculation → valid NTCP (e.g. 0.40)
- Test 2: Invalid DVH columns → gEUD fallback → valid NTCP (e.g. 0.71)
- Test 3: Normal DVH → valid NTCP (no NaN)
