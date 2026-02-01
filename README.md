# py_ntcpx — NTCP Analysis Pipeline (v2.0.0)

NTCP (Normal Tissue Complication Probability) analysis and ML pipeline for head & neck cancer: DVH preprocessing, classical and logistic NTCP models, uncertainty quantification, and SHAP-based interpretability.

## Version governance

| Version | Status | Branch/Tag |
|---------|--------|------------|
| **v2.0.0** | Default public version | `main` |
| **v1.2.1** | Frozen, legacy-stable | `v1.2.1` tag |

Only v1.2.1 and v2.0.0 are canonical. For legacy workflows, use `git checkout v1.2.1`. Non-canonical releases are archived in `archive/` for reproducibility only.

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
   pytest -q
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
