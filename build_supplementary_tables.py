#!/usr/bin/env python3
"""
build_supplementary_tables.py
==============================

Create journal-style supplementary tables from the current `out2` outputs:

- SupplementaryTable_S1_ModelEquations.xlsx
- SupplementaryTable_S2_PatientNTCP.xlsx
- SupplementaryTable_S3_Validation.xlsx
- SupplementaryTable_S4_FoldPerformance.xlsx

All tables are written into: out2/supp_results_summary_output/
using the latest contents of:
- out2/supp_results_summary_output/publication_tables.xlsx
- out2/code3_output/ntcp_results.xlsx
- out2/tiered_output/ml_validation.xlsx (if available)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> int:
    base = Path("out2")
    supp_dir = base / "supp_results_summary_output"
    supp_dir.mkdir(parents=True, exist_ok=True)

    pub = supp_dir / "publication_tables.xlsx"
    if not pub.exists():
        raise FileNotFoundError(f"publication_tables.xlsx not found at {pub}")

    xl = pd.ExcelFile(pub)

    # ---- S1: Model equations (appendix A1) ----
    if "Appendix_A1_Model_Reference" in xl.sheet_names:
        s1 = xl.parse("Appendix_A1_Model_Reference")
        s1_out = supp_dir / "SupplementaryTable_S1_ModelEquations.xlsx"
        s1.to_excel(s1_out, index=False)
        print(f"Wrote {s1_out}")
    else:
        print("WARNING: Appendix_A1_Model_Reference sheet not found in publication_tables.xlsx")

    # ---- S2: Patient-level NTCP table ----
    ntcp_path = base / "code3_output" / "ntcp_results.xlsx"
    if not ntcp_path.exists():
        raise FileNotFoundError(f"ntcp_results.xlsx not found at {ntcp_path}")

    # Complete Results sheet (sheet 0)
    ntcp_df = pd.read_excel(ntcp_path, sheet_name=0)

    s2_cols = [
        "patient_id",
        "PrimaryPatientID",
        "AnonPatientID",
        "Organ",
        "Observed_Toxicity",
        "mean_dose",
        "gEUD",
        "V30",
        "V45",
        "NTCP_LKB_LogLogit",
        "NTCP_LKB_Probit",
        "NTCP_RS_Poisson",
        "NTCP_LKB_LOCAL",
        "NTCP_ML_ANN",
        "NTCP_ML_XGBoost",
        "NTCP_ML_RandomForest",
        "uNTCP",
        "uNTCP_STD",
        "uNTCP_CI_L",
        "uNTCP_CI_U",
        "CCS",
        "CCS_Warning_Flag",
    ]
    s2_cols = [c for c in s2_cols if c in ntcp_df.columns]
    s2_df = ntcp_df[s2_cols].copy()
    s2_out = supp_dir / "SupplementaryTable_S2_PatientNTCP.xlsx"
    s2_df.to_excel(s2_out, index=False)
    print(f"Wrote {s2_out}")

    # ---- S3: Validation summary (per-model performance) ----
    if "Table2_NTCP_Performance" in xl.sheet_names:
        s3 = xl.parse("Table2_NTCP_Performance")
        s3_out = supp_dir / "SupplementaryTable_S3_Validation.xlsx"
        s3.to_excel(s3_out, index=False)
        print(f"Wrote {s3_out}")
    else:
        print("WARNING: Table2_NTCP_Performance sheet not found in publication_tables.xlsx")

    # ---- S4: Fold performance / model validation details ----
    # Use tiered_output/ml_validation.xlsx if available.
    ml_val_path = base / "tiered_output" / "ml_validation.xlsx"
    if ml_val_path.exists():
        ml_df = pd.read_excel(ml_val_path)
        s4_out = supp_dir / "SupplementaryTable_S4_FoldPerformance.xlsx"
        ml_df.to_excel(s4_out, index=False)
        print(f"Wrote {s4_out}")
    else:
        print(f"WARNING: ml_validation.xlsx not found at {ml_val_path}; "
              f"SupplementaryTable_S4_FoldPerformance.xlsx not created.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

