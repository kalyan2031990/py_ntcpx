#!/usr/bin/env python3
"""code1_dvh_preprocess.py  ⟶  processed_dvh.xlsx  +  cDVH_csv/ +  dDVH_csv/
================================================================================
Reads every *.txt DVH in a source folder, extracts header metadata, detects
whether the curve is *cumulative* or *differential*, converts as needed, and
writes **both** variants:

    • cDVH_csv/<AnoID>_<Organ>.csv   – cumulative DVH (volume column = cm³)
    • dDVH_csv/<AnoID>_<Organ>.csv   – differential DVH (cm³ per‑bin)

The script keeps the workbook summary (processed_dvh.xlsx) unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import re
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Tuple, List, Dict
import sys

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

# Import utility functions
try:
    from ntcp_utils import normalize_columns
except ImportError:
    # Fallback if module not available
    def normalize_columns(df):
        return df

# Import contract validator
try:
    from contract_validator import ContractValidator
except ImportError:
    ContractValidator = None

# ── canonical organs ───────────────────────────────────────────────────────────

def canon(raw: str) -> str:
    s = re.sub(r"[ _\\-]", "", raw.lower())
    if any(tag in s for tag in ["pd", "prtd", "parot", "parotid"]):
        return "Parotid"
    if "cord" in s:
        return "SpinalCord"
    if "larynx" in s:
        return "Larynx"
    if "oral" in s or "mucosa" in s:
        return "OralCavity"
    return raw.title().replace(" ", "")

ORG_TYPE = {
    "SpinalCord": "serial",
    "Parotid": "parallel",
    "OralCavity": "parallel",
    "Larynx": "mixed",
}

# ── header regex table ─────────────────────────────────────────────────────────
RX = {
    "pid": re.compile(r"Patient\s+ID\s*[:=]\s*(.+)", re.I),
    "pname": re.compile(r"Patient\s+Name\s*[:=]\s*(.+)", re.I),
    "agesx": re.compile(r"(\d{1,3})\s*YRS?[/\s]*(M|F)", re.I),
    "sex": re.compile(r"\b(Male|Female|M\.|F\.)\b", re.I),
    "age": re.compile(r"(\d{1,3})\s*YRS?", re.I),
    "diag": re.compile(r"Ca\.\s*([A-Za-z ]+)", re.I),
    "min": re.compile(r"Min\s*dose.*[:=]\s*([\d.]+)", re.I),
    "max": re.compile(r"Max\s*dose.*[:=]\s*([\d.]+)", re.I),
    "mean": re.compile(r"Mean\s*dose.*[:=]\s*([\d.]+)", re.I),
    "tpd": re.compile(r"Prescribed.*dose.*[:=]\s*([\d.]+)", re.I),
    "dpf": re.compile(r"(?:Dose\s*per\s*Fraction|DPF).*[:=]\s*([\d.]+)", re.I),
}

# ── helpers ───────────────────────────────────────────────────────────────────-

def to_gy(x: float) -> float:
    """Original TPS often exports cGy; convert >150 => assume cGy -> Gy."""
    return x / 100 if x > 150 else x


def uniq_id(stem: str) -> str:
    return f"ID_{hashlib.md5(stem.encode()).hexdigest()[:6]}"


# ── DVH type detection & conversion ───────────────────────────────────────────

def is_cumulative(vol: np.ndarray) -> bool:
    """Heuristic: cumulative DVH is monotonically non‑increasing."""
    return np.all(np.diff(vol) <= 1e-6)


def cum_to_diff(vol_cum: np.ndarray) -> np.ndarray:
    diff = np.empty_like(vol_cum)
    diff[:-1] = vol_cum[:-1] - vol_cum[1:]
    diff[-1] = vol_cum[-1]
    return diff


def diff_to_cum(vol_diff: np.ndarray) -> np.ndarray:
    return vol_diff[::-1].cumsum()[::-1]


# ── main txt‑parser ───────────────────────────────────────────────────────────

def parse_txt(path: Path):
    meta: Dict[str, object] = defaultdict(lambda: np.nan)
    dose: List[float] = []
    vol: List[float] = []

    for raw in path.read_text(errors="ignore").splitlines():
        # Header parsing
        for k, rx in RX.items():
            if (m := rx.search(raw)):
                if k in ("min", "max", "mean", "tpd", "dpf"):
                    meta[k] = to_gy(float(m.group(1)))
                elif k == "agesx":
                    meta["Age"] = float(m.group(1))
                    meta["Sex"] = m.group(2).upper()
                elif k == "sex":
                    meta["Sex"] = m.group(1)[0].upper()
                else:
                    meta[k] = m.group(1).strip()

        if raw.lower().startswith("structure"):
            meta["OrganRaw"] = raw.split(":", 1)[-1].strip()
            continue

        line = raw.lstrip()
        if not line or (not line[0].isdigit() and line[0] != "-"):
            continue
        parts = re.split(r"[,\s]+", line)
        try:
            dose.append(float(parts[0]))
            vol.append(float(parts[-1]))
        except ValueError:
            continue

    if not dose:
        return None

    D = np.array(dose)
    V = np.array(vol)
    if D.max() > 150:  # cGy -> Gy
        D = D / 100.0

    organ = canon(meta.get("OrganRaw", path.stem))

    # Clean patient name
    if "pname" in meta:
        name = meta["pname"]
        name = re.split(r"[,/()]", name, 1)[0]
        name = re.sub(r"\d.*", "", name)
        meta["pname"] = name.strip()

    return meta, organ, D, V


# ── derived DVH metrics (unchanged) ───────────────────────────────────────────

def metrics(cum_D: np.ndarray, cum_V: np.ndarray, meta):
    Vr = cum_V / cum_V[0] * 100.0
    hdr_mean = pd.to_numeric(meta.get("mean"), errors="coerce")
    meanD = (
        hdr_mean
        if not np.isnan(hdr_mean)
        else np.trapz(Vr * cum_D, cum_D) / np.trapz(Vr, cum_D)
    )
    hdr_max = pd.to_numeric(meta.get("max"), errors="coerce")
    vmax = hdr_max if not np.isnan(hdr_max) else cum_D.max()
    hdr_min = pd.to_numeric(meta.get("min"), errors="coerce")
    first_drop = cum_D[np.where(Vr < 99.5)[0][0]] if (Vr < 99.5).any() else cum_D.min()
    vmin = hdr_min if (not np.isnan(hdr_min) and hdr_min <= vmax) else first_drop
    modal = cum_D[np.argmax(-np.diff(np.r_[Vr[0], Vr]))]
    median = np.interp(50, Vr[::-1], cum_D[::-1]) if Vr.min() <= 50 else np.nan
    return {
        "MeanDose(Gy)": meanD,
        "MaxDose(Gy)": vmax,
        "MinDose(Gy)": vmin,
        "MedianDose(Gy)": median,
        "ModalDose(Gy)": modal,
    }


# ── main builder ─────────────────────────────────────────────────────────────

def build(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    cdir = dst / "cDVH_csv"  # cumulative
    ddir = dst / "dDVH_csv"  # differential
    cdir.mkdir(exist_ok=True)
    ddir.mkdir(exist_ok=True)

    rows = []
    idmap: Dict[str, str] = OrderedDict()
    n = 1

    for txt in sorted(src.glob("*.txt")):
        parsed = parse_txt(txt)
        if parsed is None:
            print(f"  Warning:  {txt.name}: no DVH rows -> skipped")
            continue
        meta, org, D, V_raw = parsed

        # Detect DVH type & convert
        if is_cumulative(V_raw):
            V_cum = V_raw.copy()
            V_diff = cum_to_diff(V_cum)
        else:
            V_diff = V_raw.copy()
            V_cum = diff_to_cum(V_diff)

        # Extract PrimaryPatientID from DVH header or filename
        # This is the REAL patient ID (e.g., "2020-734"), not anonymized
        primary_patient_id = meta.get("pid")
        if not primary_patient_id or primary_patient_id.strip() == "":
            # Fallback: extract from filename stem (before first underscore or space)
            stem_parts = txt.stem.split('_')[0].split()
            primary_patient_id = stem_parts[0] if stem_parts else uniq_id(txt.stem)
        
        # Clean PrimaryPatientID: remove whitespace, normalize
        primary_patient_id = str(primary_patient_id).strip()
        
        # Generate anonymized ID for display only (PT001, PT002, etc.)
        # Map same PrimaryPatientID to same AnonPatientID
        if primary_patient_id not in idmap:
            idmap[primary_patient_id] = f"PT{n:03d}"
            n += 1
        anon_patient_id = idmap[primary_patient_id]
        
        # Use PrimaryPatientID in filenames for matching
        # This ensures matching works with real patient IDs
        filename_base = f"{primary_patient_id}_{org}"
        
        # Write CSVs using PrimaryPatientID
        pd.DataFrame({"Dose[Gy]": D, "Volume[cm3]": V_cum}).to_csv(
            cdir / f"{filename_base}.csv", index=False
        )
        pd.DataFrame({"Dose[Gy]": D, "Volume[cm3]": V_diff}).to_csv(
            ddir / f"{filename_base}.csv", index=False
        )

        rows.append(
            {
                "PrimaryPatientID": primary_patient_id,  # REAL patient ID for matching
                "AnonPatientID": anon_patient_id,  # Anonymized for display only
                "PatientName": meta.get("pname", np.nan),
                "Sex": meta.get("Sex", np.nan),
                "Age": meta.get("Age", np.nan),
                "Diagnosis": meta.get("diag", np.nan),
                "Organ": org,
                "OrganType": ORG_TYPE.get(org, "mixed"),
                **metrics(D, V_cum, meta),
                "TPD(Gy)": meta.get("tpd", np.nan),
                "DPF(Gy)": meta.get("dpf", np.nan),
            }
        )

    # Workbook summary
    df = pd.DataFrame(rows)
    if len(df) > 0 and "PrimaryPatientID" in df.columns:
        df = df.sort_values(["PrimaryPatientID", "Organ"])
    elif len(df) > 0 and "Organ" in df.columns:
        df = df.sort_values(["Organ"])
    # Use context manager for Windows-safe Excel writing
    excel_file = dst / "processed_dvh.xlsx"
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='DVH_Data')

    organs = sorted(set(r["Organ"] for r in rows))
    primary_patients = sorted(set(r["PrimaryPatientID"] for r in rows))
    print(
        f"  processed_dvh.xlsx  +  {len(rows)}x2 CSVs -> {dst}\n"
        f" Patients: {len(primary_patients)}   Organs: {organs}"
    )
    
    # Generate Step1_DVHRegistry contract (authoritative source)
    # Registry uses PrimaryPatientID (real patient ID) as source of truth
    if ContractValidator is not None:
        contracts_dir = dst.parent / "contracts"
        validator = ContractValidator(contracts_dir)
        registry_path = validator.get_contract_path("Step1_DVHRegistry")
        
        # Build registry from rows data (PrimaryPatientID + AnonPatientID mapping)
        # Use validator's create_step1_registry which will use processed_dvh.xlsx
        registry_df = validator.create_step1_registry(dst, registry_path)
        
        print(f"\n[CONTRACT] Step1_DVHRegistry.xlsx created: {len(registry_df)} DVH entries")
        print(f"  PrimaryPatientID range: {len(primary_patients)} unique patients")
        anon_count = len(set(r['AnonPatientID'] for r in rows)) if 'AnonPatientID' in registry_df.columns else 0
        print(f"  AnonPatientID range: {anon_count} anonymized IDs")
        validator.log_match_statistics(registry_df)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="DVH .txt -> Excel + cumulative & differential CSVs"
    )
    ap.add_argument("txt_folder")
    ap.add_argument("--outdir", default="DVH_Preproc_Out")
    args = ap.parse_args()

    build(Path(args.txt_folder).expanduser().resolve(), Path(args.outdir).expanduser().resolve())
