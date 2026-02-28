from pathlib import Path

import pandas as pd


def main() -> None:
    base = Path(r"C:\Users\Sampa\OneDrive\Desktop\NTCP_Analysis_Pipeline\py_ntcpx_output_v3")
    code3 = base / "code3_output"
    ntcp_xlsx = code3 / "ntcp_results.xlsx"

    if not ntcp_xlsx.exists():
        raise FileNotFoundError(f"ntcp_results.xlsx not found at {ntcp_xlsx}")

    out_core = base / "code3_output"
    out_core.mkdir(parents=True, exist_ok=True)

    # 1) comprehensive_master_data.csv: direct export of ntcp_results.xlsx
    ntcp_df = pd.read_excel(ntcp_xlsx)
    comp_path = out_core / "comprehensive_master_data.csv"
    ntcp_df.to_csv(comp_path, index=False)

    # 2) complete_data.csv: subset for figure/table generation
    cols_needed = []
    for c in [
        "PrimaryPatientID",
        "AnonPatientID",
        "PatientID",
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
    ]:
        if c in ntcp_df.columns:
            cols_needed.append(c)

    complete_df = ntcp_df[cols_needed].copy()
    complete_main = base / "complete_data.csv"
    complete_df.to_csv(complete_main, index=False)

    # Mirror into manuscript_materials/tables under code3_output
    mm_tables = base / "code3_output" / "manuscript_materials" / "tables"
    mm_tables.mkdir(parents=True, exist_ok=True)
    complete_mm = mm_tables / "complete_data.csv"
    complete_df.to_csv(complete_mm, index=False)

    print("Wrote:")
    print(" -", comp_path)
    print(" -", complete_main)
    print(" -", complete_mm)


if __name__ == "__main__":
    main()

