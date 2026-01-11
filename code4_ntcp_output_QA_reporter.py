#!/usr/bin/env python3
"""
NTCP Output QA Reporter
-----------------------
Reads the output folder (or zip) created by enhanced_ntcp_analysis.py / enhanced_ntcp_ml.py
and generates:
  1) comprehensive_report.docx – human‑readable QA report
  2) qa_summary_tables.xlsx – per‑organ summary metrics + unique patient list

It flags:
- Inflated patient counts (files vs unique PatientID).
- Unrealistic NTCP values: NaNs, const predictions, outside [0,1].
- Low-n / low-event instability.
- Potential ML overfitting / leakage (AUC≥0.90 with n<40 or events<8).
- Traditional model optimism when events<5 but AUC≥0.85.

Usage
-----
python ntcp_output_QA_reporter.py --input <out_dir_or_zip> --report_outdir <outdir>

Examples
--------
python ntcp_output_QA_reporter.py --input enhanced_ntcp_analysis_ml_out.zip --report_outdir QA_results
python ntcp_output_QA_reporter.py --input ./enhanced_ntcp_analysis_ml_out --report_outdir QA_results
"""

import argparse
import os, re, sys, zipfile, math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

# Windows-safe encoding configuration
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except (AttributeError, ValueError):
    pass

import numpy as np
import pandas as pd

# Optional deps: if unavailable, the script will print a helpful error.
try:
    from sklearn.metrics import roc_curve, auc, brier_score_loss
except Exception as e:
    print("[WARN] scikit‑learn not available; AUC/Brier will be NaN. Install scikit‑learn to compute metrics.", file=sys.stderr)
    roc_curve = None
    auc = None
    brier_score_loss = None

try:
    from docx import Document
except Exception as e:
    Document = None
    print("[WARN] python‑docx not available; will skip DOCX report. Install python‑docx.", file=sys.stderr)


def unzip_if_needed(input_path: Path, workdir: Path) -> Path:
    """If input is a .zip, extract to workdir and return the root folder; else return input_path."""
    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(input_path, 'r') as z:
            z.extractall(workdir)
        # Heuristic: if the zip contained a single top-level folder, return that; else return workdir
        entries = [p for p in workdir.iterdir()]
        if len(entries) == 1 and entries[0].is_dir():
            return entries[0]
        return workdir
    return input_path


def discover_files(root: Path) -> List[Path]:
    files = []
    for r, d, fs in os.walk(root):
        for f in fs:
            files.append(Path(r) / f)
    return files


def load_table(path: Path) -> Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]]:
    """Load CSV or XLSX (all sheets)."""
    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        if path.suffix.lower() == ".xlsx":
            with pd.ExcelFile(path) as xl:
                return {s: xl.parse(s) for s in xl.sheet_names}
    except Exception as e:
        print(f"[WARN] Failed to load {path}: {e}", file=sys.stderr)
    return None


def likely_results_file(p: Path) -> bool:
    low = p.name.lower()
    if not low.endswith((".csv", ".xlsx")):
        return False
    keys = ["result", "summary", "by_organ", "metrics", "ntcp", "calc"]
    return any(k in low for k in keys)


def is_patient_level_df(df: pd.DataFrame) -> bool:
    cols = [c.strip().lower() for c in df.columns]
    key_cols = {"patient", "patientid", "id", "mrn"}
    outcome_cols = {"observed_toxicity", "toxicity", "event", "grade", "label"}
    pred_cols = [c for c in cols if c.startswith("ntcp_") or c.startswith("ml_") or "lkb" in c or "rs_" in c]
    return (any(k in cols for k in key_cols) and any(o in cols for o in outcome_cols)) or (len(pred_cols) >= 1 and "organ" in cols)


def harmonize(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    # Rename common columns
    rename_map = {}
    for c in df2.columns:
        lc = c.strip().lower()
        if lc in ("patient", "patientid", "ptid", "id"):
            rename_map[c] = "PatientID"
        elif lc == "organ":
            rename_map[c] = "Organ"
        elif lc in ("observed_toxicity", "toxicity", "event", "label"):
            rename_map[c] = "Observed_Toxicity"
        elif lc == "grade":
            rename_map[c] = "Grade"
    if rename_map:
        df2 = df2.rename(columns=rename_map)

    # Standardize types
    if "Organ" in df2.columns:
        df2["Organ"] = df2["Organ"].astype(str).str.strip()

    # Derive Observed_Toxicity if missing: treat Grade>=2 as positive
    if "Observed_Toxicity" not in df2.columns and "Grade" in df2.columns:
        g = pd.to_numeric(df2["Grade"], errors="coerce")
        df2["Observed_Toxicity"] = (g >= 2).astype(float)

    if "Observed_Toxicity" in df2.columns:
        df2["Observed_Toxicity"] = pd.to_numeric(df2["Observed_Toxicity"], errors="coerce")
        df2["Observed_Toxicity"] = df2["Observed_Toxicity"].fillna(0).clip(0, 1)

    # Normalize prediction columns (keep original names too)
    for c in list(df2.columns):
        lc = c.lower()
        if lc in ("lkb_loglogit", "lkb_probit", "rs_poisson"):
            df2.rename(columns={c: f"NTCP_{c}"}, inplace=True)
        if lc in ("ml_ann", "ml_xgboost"):
            # Standardize ML columns as NTCP_ML_ANN / NTCP_ML_XGBoost
            suffix = "ANN" if "ann" in lc else "XGBoost"
            df2.rename(columns={c: f"NTCP_ML_{suffix}"}, inplace=True)

    return df2


def auc_safe(y_true, y_pred) -> float:
    if roc_curve is None or auc is None:
        return float("nan")
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) < 5 or len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        return float(auc(fpr, tpr))
    except Exception:
        return float("nan")


def brier_safe(y_true, y_pred) -> float:
    if brier_score_loss is None:
        return float("nan")
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) < 5:
        return float("nan")
    try:
        return float(brier_score_loss(y_true, y_pred))
    except Exception:
        return float("nan")


def flag_unrealistic(ntcp_vals: pd.Series) -> List[str]:
    ntcp_vals = pd.to_numeric(ntcp_vals, errors="coerce")
    flags = []
    if ntcp_vals.isna().all():
        flags.append("All NTCP are NaN")
    if (ntcp_vals < 0).any() or (ntcp_vals > 1).any():
        flags.append("NTCP outside [0,1] range")
    if ntcp_vals.nunique(dropna=True) <= 1:
        flags.append("No variation in NTCP predictions")
    return flags


def main():
    ap = argparse.ArgumentParser(description="QA Reporter for NTCP outputs")
    ap.add_argument("--input", required=True, help="Path to output folder or zip produced by enhanced_ntcp_analysis.py / enhanced_ntcp_ml.py")
    ap.add_argument("--report_outdir", required=False, default="QA_results", help="Directory to write the report and tables")
    args = ap.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    report_dir = Path(args.report_outdir).expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    # Prepare working directory for zip extraction if needed
    workdir = report_dir / "_unpacked"
    workdir.mkdir(exist_ok=True)

    root = unzip_if_needed(input_path, workdir) if input_path.exists() else None
    if root is None or not root.exists():
        print(f"[ERROR] Input not found: {input_path}", file=sys.stderr)
        sys.exit(2)

    files = discover_files(root)
    # Focus on likely results
    candidate_paths = [p for p in files if likely_results_file(p)]
    tables = {}
    for p in candidate_paths:
        tables[str(p)] = load_table(p)

    # Gather patient-level frames
    patient_frames: List[Tuple[str, pd.DataFrame]] = []
    for path, obj in tables.items():
        if obj is None:
            continue
        if isinstance(obj, dict):
            for sname, df in obj.items():
                if isinstance(df, pd.DataFrame) and df.shape[0] > 0 and is_patient_level_df(df):
                    patient_frames.append((f"{path}::{sname}", df))
        elif isinstance(obj, pd.DataFrame):
            if obj.shape[0] > 0 and is_patient_level_df(obj):
                patient_frames.append((path, obj))

    # Harmonize and combine
    harm_frames: List[Tuple[str, pd.DataFrame]] = []
    for name, df in patient_frames:
        try:
            hf = harmonize(df)
            if "Organ" in hf.columns and ("Observed_Toxicity" in hf.columns or "Grade" in hf.columns):
                # Ensure Observed_Toxicity exists
                if "Observed_Toxicity" not in hf.columns and "Grade" in hf.columns:
                    g = pd.to_numeric(hf["Grade"], errors="coerce")
                    hf["Observed_Toxicity"] = (g >= 2).astype(float)
                harm_frames.append((name, hf))
        except Exception as e:
            print(f"[WARN] Harmonize failed for {name}: {e}", file=sys.stderr)

    combined = pd.concat([hf for _, hf in harm_frames], ignore_index=True, sort=False) if harm_frames else None

    # Determine unique patients
    def get_pid_cols(df: pd.DataFrame) -> List[str]:
        return [c for c in df.columns if c.lower() in ("patientid", "patient", "id")]
    patient_ids = set()
    if combined is not None:
        pid_cols = get_pid_cols(combined)
        if pid_cols:
            for pid in combined[pid_cols[0]].astype(str).str.strip().unique():
                if pid:
                    patient_ids.add(pid)
        else:
            # fallback via filename patterns: e.g., PID_Organ.csv
            for p in files:
                m = re.search(r"[\\/](\w+)_([A-Za-z]+)\.csv$", str(p))
                if m:
                    patient_ids.add(m.group(1))

    # Compute per-organ metrics and flags
    issues: List[str] = []
    report_rows: List[Dict[str, Union[str, int, float]]] = []
    if combined is not None and "Organ" in combined.columns:
        organs = sorted([o for o in combined["Organ"].dropna().unique()])
        model_cols = [c for c in combined.columns if c.startswith("NTCP_")]
        # Allow some common alt names
        aliases = {"NTCP_LKB_LogLogit":"NTCP_lkb_loglogit", "NTCP_LKB_Probit":"NTCP_lkb_probit", "NTCP_RS_Poisson":"NTCP_rs_poisson"}
        for organ in organs:
            sub = combined[combined["Organ"] == organ].copy()
            n = len(sub)
            if "Observed_Toxicity" in sub.columns:
                events = int(pd.to_numeric(sub["Observed_Toxicity"], errors="coerce").fillna(0).sum())
            else:
                events = np.nan
            event_rate = (events/n*100 if n>0 and not np.isnan(events) else np.nan)

            # Metrics
            perf = {}
            for col in model_cols:
                if col in sub.columns:
                    y_true = pd.to_numeric(sub["Observed_Toxicity"], errors="coerce").fillna(0).values if "Observed_Toxicity" in sub.columns else None
                    y_pred = pd.to_numeric(sub[col], errors="coerce").values
                    perf[col] = {
                        "AUC": auc_safe(y_true, y_pred) if y_true is not None else float("nan"),
                        "Brier": brier_safe(y_true, y_pred) if y_true is not None else float("nan"),
                        "Flags": flag_unrealistic(sub[col])
                    }

            # Data quality flags
            if n < 20:
                issues.append(f"{organ}: Low sample size (n={n})")
            if not np.isnan(events) and events < 5:
                issues.append(f"{organ}: Few events (events={events})")

            # ML overfitting/leakage heuristic
            for ml_key in ["NTCP_ML_ANN", "NTCP_ML_XGBoost", "NTCP_ML_XGBOOST"]:
                if ml_key in perf:
                    auc_v = perf[ml_key]["AUC"]
                    if not (auc_v is None or np.isnan(auc_v)):
                        if (n < 40 or (not np.isnan(events) and events < 8)) and auc_v >= 0.90:
                            issues.append(f"{organ}: Potential ML overfitting/leakage (AUC={auc_v:.3f}, n={n}, events={events})")

            # Traditional models optimism under very low events
            for trad in ["NTCP_LKB_LogLogit", "NTCP_LKB_Probit", "NTCP_RS_Poisson"]:
                if trad in perf:
                    auc_v = perf[trad]["AUC"]
                    if not (auc_v is None or np.isnan(auc_v)):
                        if (not np.isnan(events) and events < 5) and auc_v >= 0.85:
                            issues.append(f"{organ}: Traditional model AUC={auc_v:.3f} with events={events} (unstable)")

            # Tabulate
            best_model, best_auc = None, -1
            for k, v in perf.items():
                if v["AUC"] is not None and not np.isnan(v["AUC"]) and v["AUC"] > best_auc:
                    best_auc = v["AUC"]
                    best_model = k

            row = {
                "Organ": organ, "n": n, "events": events,
                "event_rate_%": (round(event_rate, 1) if isinstance(event_rate, (int, float)) and not np.isnan(event_rate) else np.nan),
                "best_model": best_model, "best_auc": (round(best_auc, 3) if best_auc >= 0 else np.nan),
            }
            for k, v in perf.items():
                row[f"AUC|{k}"] = (round(v["AUC"], 3) if v["AUC"] == v["AUC"] else np.nan)
                row[f"Brier|{k}"] = (round(v["Brier"], 3) if v["Brier"] == v["Brier"] else np.nan)
                row[f"Flags|{k}"] = "; ".join(v["Flags"]) if v["Flags"] else ""
            report_rows.append(row)
    else:
        issues.append("Could not locate a patient-level results table or Organ column in outputs.")

    summary_df = pd.DataFrame(report_rows) if report_rows else pd.DataFrame()

    # Global stats
    global_rows = int(summary_df["n"].sum()) if "n" in summary_df else np.nan
    global_patients = len(patient_ids) if patient_ids else np.nan

    # Save tables
    excel_path = report_dir / "qa_summary_tables.xlsx"
    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        if not summary_df.empty:
            summary_df.to_excel(writer, index=False, sheet_name="PerOrganSummary")
        if patient_ids:
            pd.DataFrame(sorted(list(patient_ids)), columns=["PatientID"]).to_excel(writer, index=False, sheet_name="UniquePatients")

    # Save DOCX report
    docx_path = report_dir / "comprehensive_report.docx"
    if Document is None:
        print(f"[WARN] python-docx missing; skipping DOCX. Tables saved at: {excel_path}")
    else:
        doc = Document()
        doc.add_heading("Comprehensive QA Report – NTCP Outputs", 0)
        p = doc.add_paragraph()
        p.add_run("Generated on: ").bold = True
        p.add_run(pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))

        doc.add_heading("1) Summary & Data Integrity", level=1)
        doc.add_paragraph(f"• Estimated unique patients: {global_patients}")
        doc.add_paragraph(f"• Total rows across organs (sum of per‑organ n): {global_rows}")
        doc.add_paragraph("Note: Counting per‑organ files as patients inflates totals. This report estimates unique patients using PatientID (if available) or filename patterns.")

        if not summary_df.empty:
            doc.add_heading("2) Per‑Organ Metrics", level=1)
            show_cols = ["Organ","n","events","event_rate_%","best_model","best_auc"]
            show_cols += [c for c in sorted(summary_df.columns) if c.startswith("AUC|")]
            show_cols += [c for c in sorted(summary_df.columns) if c.startswith("Brier|")]
            table = doc.add_table(rows=1, cols=len(show_cols))
            hdr = table.rows[0].cells
            for i, c in enumerate(show_cols):
                hdr[i].text = c
            for _, row in summary_df[show_cols].iterrows():
                cells = table.add_row().cells
                for i, c in enumerate(show_cols):
                    val = row[c]
                    cells[i].text = "" if (isinstance(val, float) and np.isnan(val)) else str(val)
        else:
            doc.add_paragraph("No per‑organ metrics could be constructed from the provided outputs.")

        doc.add_heading("3) Detected Inconsistencies & Risk Flags", level=1)
        if issues:
            for it in issues:
                doc.add_paragraph(f"• {it}")
        else:
            doc.add_paragraph("No critical issues detected by the heuristic checks.")

        if not summary_df.empty:
            doc.add_heading("4) Model‑Specific Flags by Organ", level=1)
            flags_cols = [c for c in summary_df.columns if c.startswith("Flags|")]
            for _, r in summary_df.iterrows():
                sub_flags = []
                for c in flags_cols:
                    txt = r[c]
                    if isinstance(txt, str) and txt.strip():
                        sub_flags.append(f"{c.split('|',1)[1]} -> {txt}")
                if sub_flags:
                    doc.add_paragraph(f"{r['Organ']}:")
                    for f in sub_flags:
                        doc.add_paragraph(f"• {f}")

        doc.add_heading("5) ML Overfitting/Leakage Heuristics (Applied)", level=1)
        doc.add_paragraph("Flagged when AUC ≥ 0.90 with n < 40 or events < 8. High discrimination in small/low‑event cohorts suggests optimism or leakage.")

        doc.save(docx_path)

    print(f"[OK] Saved report: {docx_path}")
    print(f"[OK] Saved tables: {excel_path}")


if __name__ == "__main__":
    main()
