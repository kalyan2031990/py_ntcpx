# NTCP Analysis Pipeline

Short description
- Purpose: NTCP (Normal Tissue Complication Probability) analysis and ML pipeline for head & neck cancer. Includes uncertainty features and SHAP interpretability pipeline.

Quickstart
1. Clone:
   git clone https://github.com/kalyan2031990/py_ntcpx.git
   cd py_ntcpx
2. Create a virtual environment:
   python -m venv .venv
   source .venv/bin/activate   # macOS / Linux
   .venv\Scripts\activate      # Windows (PowerShell)
3. Install:
   pip install --upgrade pip
   pip install -r requirements.txt
4. Run tests:
   pytest -q
5. Run example pipeline:
   python run_pipeline.py --config configs/example.yaml

Repository layout (recommended)
- py_ntcpx/                  â€” Python package (refactor large scripts here)
- scripts/                   â€” CLI wrappers and convenience scripts
- tests/                     â€” Tests (pytest)
- docs/                      â€” Documentation and examples
- requirements.txt           â€” pinned deps for quick install
- CITATION.cff               â€” citation file
- LICENSE                    â€” license text

Maintenance & CI
- Tests run via GitHub Actions (.github/workflows/ci.yml)
- Style & linting: black, ruff (see pyproject.toml)

Contributing
- Please read CONTRIBUTING.md for workflow, tests, and code style.
- Use a topic branch and open a PR against main.

Privacy & data
- Clinical datasets must be fully de-identified and shared only under appropriate approvals. See PRIVACY_CHECKLIST.md and CODE_AVAILABILITY.md in the repository for details.

License & citation
- This repository is distributed under the terms in the LICENSE file.
- Cite this work using CITATION.cff.

Contact / Maintainers
- Maintained by: kalyan2031990
- Open an issue to request features or report bugs.
