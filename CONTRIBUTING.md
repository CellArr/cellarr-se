# Contributing

Welcome to `cellarr-se`. Contributions of all kinds are appreciated — bug reports, documentation improvements, and code changes.

All contributors are expected to be open, considerate, reasonable, and respectful. When in doubt, the [Python Software Foundation's Code of Conduct](https://www.python.org/psf/conduct/) is a good reference.

## Issue Reports

If you encounter a bug or have a feature request, please open an issue on the [issue tracker](https://github.com/CellArr/cellarr-se/issues). Include:

- Your operating system and Python version
- Steps to reproduce the problem
- A minimal example that demonstrates the issue

## Documentation Improvements

Documentation lives in the `docs/` folder alongside the source code. To build it locally:

```bash
tox -e docs
python3 -m http.server --directory docs/_build/html
```

Then open `http://localhost:8000` in your browser.

## Code Contributions

### Set up a development environment

```bash
git clone https://github.com/CellArr/cellarr-se.git
cd cellarr-se
pip install -e ".[testing]"
```

Or with micromamba/conda:

```bash
micromamba create -n cellarr-se -c conda-forge python=3.11 pip -y
micromamba run -n cellarr-se pip install -e ".[testing]"
```

### Workflow

1. Open an issue to discuss non-trivial changes before starting work.
2. Create a branch:
   ```bash
   git checkout -b my-feature
   ```
3. Make your changes. Add unit tests for new behaviour.
4. Run the test suite:
   ```bash
   pytest tests/
   # or via tox
   tox
   ```
5. Run the linter:
   ```bash
   ruff check src/ tests/
   ruff format --check src/ tests/
   ```
6. Commit and push, then open a pull request against `main`.

### Releases (maintainers)

1. Ensure all tests pass on `main`.
2. Tag the commit: `git tag v0.1.0 && git push origin v0.1.0`
3. The CI pipeline will build the wheel and publish to PyPI automatically on tag push.
   Requires `TWINE_USERNAME` and `TWINE_PASSWORD` set as CI/CD variables
   (`TWINE_USERNAME=__token__`, `TWINE_PASSWORD=<your-pypi-token>`).

## Note

This project was scaffolded with [PyScaffold](https://pyscaffold.org/) and [BiocSetup](https://github.com/biocpy/biocsetup).

[repository]: https://github.com/CellArr/cellarr-se
[issue tracker]: https://github.com/CellArr/cellarr-se/issues
