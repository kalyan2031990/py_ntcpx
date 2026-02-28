#!/usr/bin/env python3
"""
Build the 'manuscript material' folder: one integrated, non-redundant set for
research and clinical use.

Outputs (no redundancy):
- manuscript_aggregated_summary.xlsx — Single workbook: Code0–Code7 + Tiered + SHAP summary + LIME summary.
- publication_tables.xlsx — Tables 1–4 and appendices (Step 8).
- SupplementaryTables_S1-S4.xlsx — One file with 4 sheets (S1–S4) for submission.
- README_MANUSCRIPT_MATERIAL.md, py_ntcpx_v1.0.0_execution_log.txt,
  OUTPUT_STRUCTURE.md, COMPLETE_RESULTS_REPORT.md, ARCHITECTURE_REPORT.md.

No standalone summary_*.csv (all data in the aggregated workbook).
SHAP results: consolidated from code7_shap into one sheet (mean |SHAP| per feature per model).

Run from repo root. Run build_supplementary_tables.py first so supp has S1–S4 (we merge them).
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd


def _safe_sheet_name(name: str, max_len: int = 31) -> str:
    s = name.replace("/", "-").replace("\\", "-")[:max_len]
    return s or "Sheet"


def _add_sheet(writer: pd.ExcelWriter, df: pd.DataFrame, sheet_name: str) -> None:
    if df is None or df.empty:
        return
    df.to_excel(writer, sheet_name=_safe_sheet_name(sheet_name), index=False)


def _build_shap_summary(code7: Path) -> pd.DataFrame | None:
    """Consolidate SHAP feature importance from Parotid/ANN, XGBoost, RandomForest."""
    rows = []
    parotid = code7 / "Parotid"
    if not parotid.exists():
        return None
    for model_dir in ["ANN", "XGBoost", "RandomForest"]:
        shap_file = parotid / model_dir / "shap_table.xlsx"
        if not shap_file.exists():
            continue
        try:
            df = pd.read_excel(shap_file, sheet_name=0)
            # df: columns = features, rows = samples
            mean_abs = df.abs().mean()
            for feat in mean_abs.index:
                rows.append({"Organ": "Parotid", "Model": model_dir, "Feature": feat, "Mean_abs_SHAP": round(mean_abs[feat], 6)})
        except Exception:
            continue
    if not rows:
        return None
    return pd.DataFrame(rows)


def main() -> int:
    repo = Path(__file__).resolve().parent
    out2 = repo / "out2"
    manuscript = out2 / "manuscript material"
    if manuscript.exists():
        shutil.rmtree(manuscript)
    manuscript.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%Y-%m-%d")
    print("Building manuscript material (integrated, no redundancy)...\n")

    agg_path = manuscript / "manuscript_aggregated_summary.xlsx"
    sheets_added = []

    with pd.ExcelWriter(agg_path, engine="openpyxl") as writer:
        # ---- Code0 ----
        code0 = out2 / "code0_output"
        if code0.exists():
            for x in code0.glob("*.xlsx"):
                try:
                    df = pd.read_excel(x, sheet_name=0)
                    _add_sheet(writer, df, "Code0_ClinicalReconciliation")
                    sheets_added.append("Code0_ClinicalReconciliation")
                    print("  Code0: clinical reconciliation")
                except Exception as e:
                    print(f"  Code0 skip: {e}")
                break

        # ---- Code1 ----
        code1 = out2 / "code1_output"
        if code1.exists() and (code1 / "processed_dvh.xlsx").exists():
            try:
                df = pd.read_excel(code1 / "processed_dvh.xlsx", sheet_name=0)
                _add_sheet(writer, df, "Code1_DVH_Summary")
                sheets_added.append("Code1_DVH_Summary")
                print("  Code1: DVH summary")
            except Exception as e:
                print(f"  Code1 skip: {e}")

        # ---- Code2 ----
        tab = out2 / "code2_output" / "tables" / "dose_metrics_cohort.xlsx"
        if tab.exists():
            try:
                df = pd.read_excel(tab, sheet_name=0)
                _add_sheet(writer, df, "Code2_DoseMetricsCohort")
                sheets_added.append("Code2_DoseMetricsCohort")
                print("  Code2: dose metrics cohort")
            except Exception as e:
                print(f"  Code2 skip: {e}")

        # ---- Code2b ----
        code2b = out2 / "code2_bDVH_output"
        contracts = out2 / "contracts"
        done_2b = False
        if code2b.exists():
            for f in code2b.glob("*.xlsx"):
                try:
                    df = pd.read_excel(f, sheet_name=0)
                    _add_sheet(writer, df, "Code2b_bDVH_Summary")
                    sheets_added.append("Code2b_bDVH_Summary")
                    print("  Code2b: bDVH summary")
                    done_2b = True
                except Exception as e:
                    print(f"  Code2b skip: {e}")
                break
        if not done_2b and contracts.exists():
            f = contracts / "Step2b_bDVHRegistry.xlsx"
            if f.exists():
                try:
                    df = pd.read_excel(f, sheet_name=0)
                    _add_sheet(writer, df, "Code2b_bDVH_Summary")
                    sheets_added.append("Code2b_bDVH_Summary")
                    print("  Code2b: bDVH summary (contracts)")
                except Exception as e:
                    print(f"  Code2b skip: {e}")

        # ---- Code3 ----
        code3 = out2 / "code3_output"
        ntcp_file = code3 / "ntcp_results.xlsx" if code3.exists() else None
        if ntcp_file and ntcp_file.exists():
            try:
                xl = pd.ExcelFile(ntcp_file)
                for sh in ["Complete Results", "Summary by Organ", "Performance Matrix", "Dose Metrics"]:
                    if sh in xl.sheet_names:
                        df = xl.parse(sh)
                        _add_sheet(writer, df, _safe_sheet_name(f"Code3_{sh.replace(' ', '')}"))
                        sheets_added.append(sh)
                for name in ["enhanced_summary_performance.csv", "enhanced_ntcp_calculations.csv"]:
                    f = code3 / name
                    if f.exists():
                        df = pd.read_csv(f)
                        _add_sheet(writer, df, _safe_sheet_name("Code3_" + name.replace(".csv", "").replace(" ", "")))
                        sheets_added.append(name)
                print("  Code3: NTCP results + enhanced summaries")
            except Exception as e:
                print(f"  Code3 skip: {e}")

        # ---- Code4 ----
        code4 = out2 / "code4_output"
        qa_file = code4 / "qa_summary_tables.xlsx" if code4.exists() else None
        if qa_file and qa_file.exists():
            try:
                xl = pd.ExcelFile(qa_file)
                for i, sh in enumerate(xl.sheet_names):
                    df = xl.parse(sh)
                    _add_sheet(writer, df, _safe_sheet_name(f"Code4_{i+1}_{sh.replace(' ', '')[:18]}"))
                    sheets_added.append(sh)
                tx = code4 / "tables" / "Table_X_Classical_vs_ML.xlsx"
                if not tx.exists():
                    tx = code4 / "tables" / "Table_X_Classical_vs_ML.csv"
                if tx.exists():
                    df = pd.read_csv(tx) if tx.suffix == ".csv" else pd.read_excel(tx, sheet_name=0)
                    _add_sheet(writer, df, "Code4_TableX_ClassicalVsML")
                print("  Code4: QA + Table X")
            except Exception as e:
                print(f"  Code4 skip: {e}")

        # ---- Code5 ----
        code5 = out2 / "code5_output"
        if code5.exists():
            for stem in ["categorical_factors_analysis", "continuous_factors_analysis", "organ_specific_analysis"]:
                f = code5 / f"{stem}.xlsx"
                if f.exists():
                    try:
                        df = pd.read_excel(f, sheet_name=0)
                        _add_sheet(writer, df, _safe_sheet_name("Code5_" + stem[:20]))
                        sheets_added.append(stem)
                    except Exception as e:
                        print(f"  Code5 {stem} skip: {e}")
            print("  Code5: clinical factors")

        # ---- Code7: LIME + SHAP ----
        code7 = out2 / "code7_shap"
        if code7.exists():
            lime_file = code7 / "LIME_summary_all_organs.xlsx"
            if lime_file.exists():
                try:
                    df = pd.read_excel(lime_file, sheet_name=0)
                    _add_sheet(writer, df, "Code7_LIME_Summary")
                    sheets_added.append("Code7_LIME_Summary")
                    print("  Code7: LIME summary")
                except Exception as e:
                    print(f"  Code7 LIME skip: {e}")
            shap_df = _build_shap_summary(code7)
            if shap_df is not None and not shap_df.empty:
                _add_sheet(writer, shap_df, "Code7_SHAP_Summary")
                sheets_added.append("Code7_SHAP_Summary")
                print("  Code7: SHAP summary (mean |SHAP| per feature per model)")

        # ---- Tiered ----
        tiered = out2 / "tiered_output"
        if tiered.exists():
            for name, prefix in [
                ("NTCP_4Tier_Master.xlsx", "Tiered_4Tier_Master"),
                ("ml_validation.xlsx", "Tiered_ML_Validation"),
                ("tiered_ntcp_results.xlsx", "Tiered_ntcp_results"),
            ]:
                f = tiered / name
                if f.exists():
                    try:
                        df = pd.read_excel(f, sheet_name=0)
                        _add_sheet(writer, df, prefix)
                        sheets_added.append(prefix)
                    except Exception as e:
                        print(f"  Tiered {name} skip: {e}")
            print("  Tiered: 4-tier master, ML validation, tiered results")

    print(f"\n  Wrote {agg_path.name} ({len(sheets_added)} sheets)")

    # ---- Publication tables (one file) ----
    supp = out2 / "supp_results_summary_output"
    pub = supp / "publication_tables.xlsx"
    if pub.exists():
        shutil.copy2(pub, manuscript / "publication_tables.xlsx")
        print("  publication_tables.xlsx")

    # ---- Supplementary S1–S4 as ONE workbook (no redundancy) ----
    supp_one = manuscript / "SupplementaryTables_S1-S4.xlsx"
    with pd.ExcelWriter(supp_one, engine="openpyxl") as w:
        if supp.exists():
            pub_xl = pd.ExcelFile(supp / "publication_tables.xlsx")
            if "Appendix_A1_Model_Reference" in pub_xl.sheet_names:
                pd.read_excel(supp / "publication_tables.xlsx", sheet_name="Appendix_A1_Model_Reference").to_excel(w, sheet_name="S1_ModelEquations", index=False)
            ntcp_path = out2 / "code3_output" / "ntcp_results.xlsx"
            if ntcp_path.exists():
                df = pd.read_excel(ntcp_path, sheet_name=0)
                s2_cols = [c for c in ["patient_id", "PrimaryPatientID", "AnonPatientID", "Organ", "Observed_Toxicity", "mean_dose", "gEUD", "V30", "V45",
                    "NTCP_LKB_LogLogit", "NTCP_LKB_Probit", "NTCP_RS_Poisson", "NTCP_LKB_LOCAL", "NTCP_ML_ANN", "NTCP_ML_XGBoost", "NTCP_ML_RandomForest",
                    "uNTCP", "uNTCP_STD", "CCS", "CCS_Warning_Flag"] if c in df.columns]
                if s2_cols:
                    df[s2_cols].to_excel(w, sheet_name="S2_PatientNTCP", index=False)
            if "Table2_NTCP_Performance" in pub_xl.sheet_names:
                pd.read_excel(supp / "publication_tables.xlsx", sheet_name="Table2_NTCP_Performance").to_excel(w, sheet_name="S3_Validation", index=False)
            ml_val = out2 / "tiered_output" / "ml_validation.xlsx"
            if ml_val.exists():
                pd.read_excel(ml_val).to_excel(w, sheet_name="S4_FoldPerformance", index=False)
    print("  SupplementaryTables_S1-S4.xlsx (one file, 4 sheets)")

    # ---- Docs ----
    def copy_doc(src_name: str, add_header: bool = False) -> None:
        src = repo / src_name
        if not src.exists():
            return
        dest = manuscript / src_name
        if add_header:
            dest.write_text(f"*Manuscript material — {date_str}.*\n\n" + src.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
        else:
            shutil.copy2(src, dest)
        print(f"  {src_name}")

    log_src = repo / "py_ntcpx_v1.0.0_execution_log.txt"
    if log_src.exists():
        content = log_src.read_text(encoding="utf-8", errors="replace")
        (manuscript / "py_ntcpx_v1.0.0_execution_log.txt").write_text(f"Manuscript material — {date_str}.\n\n" + content, encoding="utf-8")
        print("  py_ntcpx_v1.0.0_execution_log.txt")

    struct_src = repo / "OUTPUT_STRUCTURE.md"
    if struct_src.exists():
        extra = f"""

---

## Manuscript Material (this folder)

**Generated**: {date_str}

Single integrated set (no redundant files): **manuscript_aggregated_summary.xlsx** (all pipeline summaries + SHAP + LIME), **publication_tables.xlsx**, **SupplementaryTables_S1-S4.xlsx**, and documentation. Export sheets to CSV as needed.
"""
        (manuscript / "OUTPUT_STRUCTURE.md").write_text(struct_src.read_text(encoding="utf-8", errors="replace") + extra, encoding="utf-8")
        print("  OUTPUT_STRUCTURE.md")

    copy_doc("COMPLETE_RESULTS_REPORT.md", add_header=True)
    copy_doc("ARCHITECTURE_REPORT.md", add_header=False)

    readme = f"""# Manuscript Material — Integrated Summary (No Redundancy)

**Generated**: {date_str}

Single set for manuscript preparation, tables/figures, and research or clinical use. Redundancy removed; SHAP and LIME included in the aggregated workbook.

## Files (8 total)

| File | Description |
|------|--------------|
| **manuscript_aggregated_summary.xlsx** | One workbook: Code0 (clinical), Code1 (DVH), Code2 (dose metrics), Code2b (bDVH), Code3 (NTCP complete + summary + performance + dose + enhanced), Code4 (QA + Table X), Code5 (factors), Code7 (LIME summary + **SHAP summary**), Tiered (4-tier master, ML validation, tiered results). Export any sheet to CSV for R/Python. |
| **publication_tables.xlsx** | Tables 1–4 and appendices (Step 8). |
| **SupplementaryTables_S1-S4.xlsx** | One file, 4 sheets: S1 Model equations, S2 Patient NTCP, S3 Validation, S4 Fold performance. |
| **COMPLETE_RESULTS_REPORT.md** | Full results report (includes Random Forest, SHAP, LIME). |
| **ARCHITECTURE_REPORT.md** | Pipeline architecture. |
| **OUTPUT_STRUCTURE.md** | Output layout. |
| **README_MANUSCRIPT_MATERIAL.md** | This file. |
| **py_ntcpx_v1.0.0_execution_log.txt** | Pipeline run log. |

## SHAP and LIME

- **Code7_SHAP_Summary** (in manuscript_aggregated_summary.xlsx): Mean |SHAP| per feature per model (Parotid: ANN, XGBoost, Random Forest). Source: out2/code7_shap/Parotid/{{ANN,XGBoost,RandomForest}}/shap_table.xlsx.
- **Code7_LIME_Summary**: LIME summary across organs/models (same workbook).

## Figures

Figures (ROC, calibration, dose–response, SHAP/LIME PNGs) live in out2/ (code3_output/plots, code6_output, tiered_output, code7_shap). This folder is tables-only for portability.
"""
    (manuscript / "README_MANUSCRIPT_MATERIAL.md").write_text(readme, encoding="utf-8")
    print("  README_MANUSCRIPT_MATERIAL.md")

    print(f"\nDone. Manuscript material: {manuscript} (reduced, no redundant CSVs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
