# Contributing

Thanks for your interest in contributing.

How to contribute
1. Fork the repository and create a topic branch from main:
   git checkout -b myfix/short-description
2. Run tests & linters before committing:
   pip install -r requirements.txt
   pytest
   black .
   ruff check .
3. Commit and push to your fork and open a pull request describing your change.

Pull Request guidelines
- Keep PRs focused and small.
- Include tests for new functionality or bug fixes.
- Reference any related issue numbers.

Code style
- Use black for formatting and ruff for linting.
- Follow PEP 8 and add type hints where helpful.

Import policy
- Archived contents (archive/) are excluded from installation and distribution.
- Do not import or depend on archived materials for any analysis or results.

Community
- Be respectful and constructive. See CODE_OF_CONDUCT.md.
