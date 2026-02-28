#!/usr/bin/env python3
"""code2_dvh_plot_and_summary_fixed.py – rev 2025‑07‑13‑b
========================================================
Complete, runnable version after truncation issues.

Highlights (compared with 07‑12 build)
--------------------------------------
* **No‑overlap cDVH labels** for D̄, Dₘₐₓ, D₀.₁ cc.
* **dDVH annotations**: verticals at D̄, Dₘₐₓ plus D₂ cc for serial organs.
* **Overlay plots for every patient**; duplicate legends removed.
* **Extra metrics**: V20, V25, V50 [%] (all organs) + D₂ cc, D₀.₀₁ cc [Gy] (serial organs).
* **Tidy Excel** workbook `tables/dose_metrics_cohort.xlsx` with PerStructure & CohortSummary.
"""
from __future__ import annotations

import argparse, sys, math, logging
from pathlib import Path
from typing import Tuple, Dict, List

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
import matplotlib.pyplot as plt

# ── CONFIG ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 300,
})
STYLE = dict(linewidth=1.6)
SERIAL_ORGANS = {"SpinalCord", "Brainstem", "OpticNerve"}
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

# ── HELPERS ───────────────────────────────────────────────────────────

def canon(raw: str) -> str:
    """Return a canonical organ name for tidy grouping."""
    if not isinstance(raw, str):
        return str(raw)
    s = raw.lower()
    mapping = {
        "parotid": "Parotid", "cord": "SpinalCord", "larynx": "Larynx",
        "oral": "OralCavity", "brain": "Brainstem", "optic": "OpticNerve",
        "mandible": "Mandible", "cochlea": "Cochlea", "submandibular": "Submandibular",
    }
    for k, v in mapping.items():
        if k in s:
            return v
    return raw.title().replace(" ", "")


def load_csv(csv: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load Code‑1 cDVH CSV → Dose[Gy], AbsVol[cm³] (ascending dose)."""
    df = pd.read_csv(csv)
    cols = {c.lower().replace(" ", ""): c for c in df.columns}
    dose_col = next(c for c in cols.values() if "dose" in c.lower())
    vol_col = next(c for c in cols.values() if "volume" in c.lower())
    D = df[dose_col].to_numpy(float)
    V = df[vol_col].to_numpy(float)
    if D[0] > D[-1]:  # ensure ascending
        D, V = D[::-1], V[::-1]
    return D, V


def prepare_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

# ── METRICS ───────────────────────────────────────────────────────────

def _dose_at_volume(D: np.ndarray, V_abs: np.ndarray, cc: float) -> float:
    mask = V_abs <= cc
    return D[mask][0] if mask.any() else float("nan")


def _vol_at_dose(D: np.ndarray, Vr: np.ndarray, gy: float) -> float:
    idx = np.searchsorted(D, gy)
    return float(Vr[idx]) if idx < Vr.size else 0.0


def dvh_metrics(D: np.ndarray, V_abs: np.ndarray, organ: str) -> Dict[str, float]:
    Vr = V_abs / V_abs[0] * 100.0
    dv_rel = -np.diff(Vr) / 100.0
    dV = dv_rel * V_abs[0]
    midD = (D[:-1] + D[1:]) / 2.0
    m = min(len(dV), len(midD))
    D_mean = (midD[:m] * dV[:m]).sum() / V_abs[0]

    metrics: Dict[str, float] = {
        "MeanDose(Gy)": round(D_mean, 2),
        "Dmax(Gy)": round(_dose_at_volume(D, V_abs, 0.1), 2),
        "D0.1cc(Gy)": round(_dose_at_volume(D, V_abs, 0.1), 2),
        "V20Gy(%)": round(_vol_at_dose(D, Vr, 20.0), 1),
        "V25Gy(%)": round(_vol_at_dose(D, Vr, 25.0), 1),
        "V50Gy(%)": round(_vol_at_dose(D, Vr, 50.0), 1),
    }
    if organ in SERIAL_ORGANS:
        metrics.update({
            "D2cc(Gy)": round(_dose_at_volume(D, V_abs, 2.0), 2),
            "D0.01cc(Gy)": round(_dose_at_volume(D, V_abs, 0.01), 2),
        })
    return metrics

# ── PLOTTING ──────────────────────────────────────────────────────────

def _stack_y(y0: float, used: List[float], margin: float = 3) -> float:
    y = y0
    while any(abs(y - yy) < margin for yy in used):
        y += margin
    used.append(y)
    return y


def _annotate_cdvh(ax, D, Vr, metrics: Dict[str, float]):
    used: List[float] = []
    for key, label in [("Dmax(Gy)", "Dₘₐₓ"), ("MeanDose(Gy)", "D̄"), ("D0.1cc(Gy)", "D₀.₁cc")]:
        d = metrics.get(key, math.nan)
        if math.isnan(d):
            continue
        v = float(np.interp(d, D, Vr))
        y = _stack_y(v, used)
        ax.scatter(d, v, s=12, zorder=5)
        ax.text(d, y, f"{label}={d:.1f}", ha="center", va="bottom", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7), zorder=6)


def plot_cdvh(D, Vr, stem: str, metrics: Dict[str, float], outdir: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(D, Vr, **STYLE, label=stem)
    ax.set_xlim(left=0); ax.set_ylim(0, 100)
    ax.set_xlabel("Dose (Gy)"); ax.set_ylabel("Volume (%)")
    _annotate_cdvh(ax, D, Vr, metrics)
    ax.legend()
    for ext in (".png", ".svg"):
        fig.savefig(outdir / f"cDVH_{stem}{ext}", bbox_inches="tight")
    plt.close(fig)


def plot_ddvh(D, Vr, stem: str, metrics: Dict[str, float], is_serial: bool, outdir: Path):
    diff = -np.diff(Vr) / np.diff(D)
    midD = (D[:-1] + D[1:]) / 2.0
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(midD, diff, **STYLE, label=stem)
    for key, ls in [("MeanDose(Gy)", "-"), ("Dmax(Gy)", "--")]:
        d = metrics.get(key, math.nan)
        if not math.isnan(d):
            ax.axvline(d, linestyle=ls, linewidth=0.8, color="grey")
    if is_serial:
        d2 = metrics.get("D2cc(Gy)")
        if d2 and not math.isnan(d2):
            ax.axvline(d2, linestyle=":", linewidth=0.8, color="grey")
    ax.set_xlabel("Dose (Gy)"); ax.set_ylabel("-dV/dD (% / Gy)")
    ax.legend(); ax.set_xlim(left=0)
    for ext in (".png", ".svg"):
        fig.savefig(outdir / f"dDVH_{stem}{ext}", bbox_inches="tight")
    plt.close(fig)


def plot_overlay(patient: str, entries: List[Dict], outdir: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    seen: set[str] = set()
    for e in entries:
        lbl = e["Organ"] if e["Organ"] not in seen else "_nolegend_"
        ax.plot(e["Dose"], e["Vr"], **STYLE, label=lbl)
        seen.add(e["Organ"])
        dmax = e["Metrics"].get("Dmax(Gy)")
        if not math.isnan(dmax):
            ax.axvline(dmax, linestyle="--", linewidth=0.8)
    ax.set_xlim(left=0); ax.set_ylim(0, 100)
    ax.set_xlabel("Dose (Gy)"); ax.set_ylabel("Volume (%)")
    ax.legend(title=patient)
    for ext in (".png", ".svg"):
        fig.savefig(outdir / f"overlay_{patient}{ext}", bbox_inches="tight")
    plt.close(fig)

# ── CORE ─────────────────────────────────────────────────────────────

def process(code1_dir: Path, outdir: Path):
    code1_dir = code1_dir.resolve()
    excel_path = code1_dir / "processed_dvh.xlsx"
    cdvh_dir = code1_dir / "cDVH_csv"
    if not excel_path.exists():
        sys.exit("processed_dvh.xlsx missing – run Code‑1 first")

    cplot_dir = prepare_dir(outdir / "cDVH_plots")
    dplot_dir = prepare_dir(outdir / "dDVH_plots")
    oplot_dir = prepare_dir(outdir / "overlay_plots")
    table_dir = prepare_dir(outdir / "tables")

    master = pd.read_excel(excel_path)
    master["Organ"] = master["Organ"].astype(str).map(canon)

    rows: List[Dict] = []
    overlay_buf: Dict[str, List[Dict]] = {}

    for _, r in master.iterrows():
        # Identity-safe: use PrimaryPatientID for file matching, AnonPatientID for display only
        primary_id = r.get('PrimaryPatientID', None)
        if primary_id is None or pd.isna(primary_id):
            logging.warning(f"Row missing PrimaryPatientID - skipped")
            continue
        
        anon_id = r.get('AnonPatientID', None)
        
        organ = r.Organ
        
        # File matching uses PrimaryPatientID (identity-safe)
        stem = f"{primary_id}_{organ}"
        csv_path = cdvh_dir / f"{stem}.csv"
        if not csv_path.exists():
            logging.warning(f"{csv_path.name} missing – skipped")
            continue
        try:
            D, V = load_csv(csv_path)
        except Exception as exc:
            logging.warning(f"{stem}: {exc}")
            continue
        Vr = V / V[0] * 100.0
        metrics = dvh_metrics(D, V, organ)

        # Display uses AnonPatientID if available, otherwise PrimaryPatientID
        display_id = anon_id if anon_id and not pd.isna(anon_id) else primary_id
        display_stem = f"{display_id}_{organ}"

        # plots - use display_id for labels (AnonPatientID for display)
        plot_cdvh(D, Vr, display_stem, metrics, cplot_dir)
        plot_ddvh(D, Vr, display_stem, metrics, organ in SERIAL_ORGANS, dplot_dir)

        overlay_buf.setdefault(display_id, []).append({"Dose": D, "Vr": Vr, "Organ": organ, "Metrics": metrics})
        # Store both for output
        rows.append({"PrimaryPatientID": primary_id, "AnonPatientID": anon_id, "Organ": organ, **metrics})

    # overlay plots - display_id is used (AnonPatientID if available)
    for display_id, entries in overlay_buf.items():
        plot_overlay(display_id, entries, oplot_dir)

    if not rows:
        logging.error("No DVHs processed – aborting Excel write")
        return

    per_df = pd.DataFrame(rows)
    num_cols = per_df.select_dtypes(include=[np.number]).columns
    summary = per_df.groupby("Organ")[num_cols].agg(["mean", "std"]).round(2)
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]

    xlsx = table_dir / "dose_metrics_cohort.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
        per_df.to_excel(xw, sheet_name="PerStructure", index=False)
        summary.to_excel(xw, sheet_name="CohortSummary")
    logging.info(f"Excel -> {xlsx.relative_to(outdir)}")

# ── CLI ──────────────────────────────────────────────────────────────

def parse_cli():
    p = argparse.ArgumentParser(description="Generate DVH plots and cohort metrics from Code‑1 output")
    p.add_argument("code1_dir", type=Path, help="Path to Code‑1 result folder")
    p.add_argument("--outdir", "-o", type=Path, default=None, help="Output folder (default: <code1_dir>/code2_out)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_cli()
    out = args.outdir or args.code1_dir / "code2_out"
    process(args.code1_dir, out)
    logging.info("Done [OK]")
