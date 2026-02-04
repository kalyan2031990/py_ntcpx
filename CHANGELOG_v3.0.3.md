# Changelog - Version 3.0.3

## Patch Release: py_ntcpx v3.0.3 - Publication Bundle, Config Centralization, Reproducibility

**Release Date:** 2026-02-04

---

## New Features

### 1. Publication Bundle (Step 9)
- **New script**: `scripts/create_publication_bundle.py` creates `publication_bundle_YYYYMMDD/`
- **Contents**: manuscript_materials, tables (publication_tables, Table_X, ntcp_results), figures (publication_diagrams, SHAP/LIME), tiered outputs, QA report
- **Pipeline**: Step 9 added to `run_pipeline.py`; runs automatically after Step 8

### 2. Centralized Configuration
- **Config loader**: `src/utils/config_loader.py` reads `config/pipeline_config.yaml`
- **Random seed**: Code3 `MachineLearningModels` now reads `pipeline.random_seed` from config
- **Default output dir**: Added `pipeline.default_output_dir: "out2"` to config

### 3. Seed Documentation
- **REPRODUCIBILITY_README.md**: Added "Seed Locations" table documenting all random seed usages
- **Source of truth**: `config/pipeline_config.yaml` → `pipeline.random_seed`

### 4. Conda Environment
- **environment.yml**: Conda environment for py_ntcpx (conda-forge + pip fallbacks)
- **Usage**: `conda env create -f environment.yml`

### 5. Dependencies
- **pyyaml**: Added to requirements.txt (required for config loader)

---

## Files Added/Modified

- `scripts/create_publication_bundle.py` - NEW
- `src/utils/config_loader.py` - NEW
- `environment.yml` - NEW
- `config/pipeline_config.yaml` - Added default_output_dir
- `code3_ntcp_analysis_ml.py` - MachineLearningModels reads random_seed from config
- `run_pipeline.py` - Step 9 publication bundle, resume_from step9
- `REPRODUCIBILITY_README.md` - Seed locations table
- `requirements.txt` - Added pyyaml
- `VERSION` - 3.0.3
