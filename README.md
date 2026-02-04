# py_ntcpx — NTCP Analysis Pipeline (v3.0.0)

NTCP (Normal Tissue Complication Probability) analysis and ML pipeline for head & neck cancer: DVH preprocessing, classical and logistic NTCP models, uncertainty quantification, SHAP-based interpretability, and LIME explanations.

## Version governance

| Version | Status | Branch/Tag |
|---------|--------|------------|
| **v3.0.0** | **Default public version** (Latest) | `main` |
| v3.0.1 | Patch fixes (ML CV-AUC, QUANTEC-RS, gEUD) | Included in `main`; see [FIXES_v3.0.0_to_v3.0.1.md](FIXES_v3.0.0_to_v3.0.1.md) |
| **v2.1.0** | Previous stable release | `v2.1.0` tag |
| **v2.0.0** | Previous stable release | `v2.0.0` tag |
| v1.2.1 | Archived, legacy-stable | `v1.2.1` tag |
| v1.2.0 | Archived, legacy | `v1.2.0` tag |
| v1.1.0 | Archived, legacy | `v1.1.0` tag |
| v1.0.0 | Archived, legacy | `v1.0.0` tag |

**Active versions**: v3.0.0 (main) and v2.1.0 are fully supported. For the latest features, use `main` (v3.0.0). For legacy workflows, use `git checkout v2.1.0` or `v2.0.0`. Archived versions (v1.x) are available via git tags for reproducibility only.

## Quick start

1. Clone:
   ```bash
   git clone https://github.com/kalyan2031990/py_ntcpx.git
   cd py_ntcpx
   ```
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   # macOS / Linux:  source .venv/bin/activate
   # Windows:       .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. Run tests:
   ```bash
   python run_all_tests.py
   # or: pytest -q
   ```
5. Run pipeline (DVH text directory and patient Excel required):
   ```bash
   python run_pipeline.py --input_txt_dir <path/to/dvh_txt> --patient_data <path/to/patient_data.xlsx> [--output_dir out2]
   ```

## Repository layout

| Path | Description |
|------|-------------|
| `src/` | Validation, models, reporting, features |
| `config/` | `pipeline_config.yaml` |
| `ntcp_models/`, `quantification/` | NTCP and QUANTEC modules |
| `scripts/` | Publication checklist, SHAP supplementary |
| `tests/` | Pytest tests |
| `archive/` | ARCHIVED — non-canonical release scripts (v1.0.0, v1.1.0) for reproducibility only |
| `requirements.txt`, `pyproject.toml` | Dependencies and tooling |
| `CITATION.cff`, `LICENSE` | Citation and license |

## Key outputs (code3)

- `ntcp_results.xlsx` — NTCP predictions, Summary by Organ, Performance Matrix (incl. ML CV-AUC)
- `ml_validation.xlsx` — ML validation metrics (CV-AUC for ANN, XGBoost)
- `manuscript_materials/` — Publication-ready figures and tables

## CI and maintenance

- Tests: GitHub Actions (`.github/workflows/ci.yml`)
- Style: black, ruff (see `pyproject.toml`)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for workflow, tests, and code style. Use a topic branch and open a PR against `main`.

## Privacy and data

Clinical datasets must be fully de-identified and shared only under appropriate approvals. See CONTRIBUTING.md and LICENSE for details.

## License and citation

- License: see [LICENSE](LICENSE).
- Citation: [CITATION.cff](CITATION.cff).

## Contact

Maintained by **kalyan2031990**. Open an issue for feature requests or bug reports.
