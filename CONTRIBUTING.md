# Contributing to PixPoint

Thank you for considering a contribution. Bug reports, questions, documentation
fixes and code are all welcome. This page says how to get set up and what a
change needs before it can be merged.

## Questions and bug reports

Open an issue on GitHub. For a bug, the issue template asks for the PixPoint
version, the platform (CPU or GPU, CUDA/CuPy versions), a short script that
reproduces the problem and the full error message. A pair file or a map is
rarely needed: most problems reproduce on a synthetic map of a few thousand
pixels, and a small reproducer is what gets a bug fixed quickly.

## Development setup

```bash
git clone https://github.com/D-Gebauer/PixPoint.git
cd PixPoint
pip install -e ".[dev]"          # package + pytest, black, isort, flake8, mypy, treecorr
pip install -e ".[docs]"         # to build the documentation
```

Useful commands (see the `Makefile`):

| command | what it does |
|---|---|
| `make test-fast` | the fast local loop, about a minute (no GPU, no slow exactness gates) |
| `pytest -m "not gpu"` | everything CI runs, about six minutes |
| `make lint` | black, isort, flake8 and `mypy --strict`, exactly as CI runs them |
| `make format` | apply black and isort |
| `make docs` | build the documentation into `docs/_build/html` |

The GPU tests (`pytest -m gpu`) need an NVIDIA GPU with CuPy; CI does not run
them, so say in a pull request whether you ran them. Their coverage reaches
Codecov separately: the maintainer measures it on an A100 and uploads it as the
`gpu` flag (`benchmarks/static_treecode/gpu_coverage.sh`, see `codecov.yml`).

## What a change needs

1. **Tests.** PixPoint is an estimator library: a measurement that is subtly
   wrong is worse than one that crashes. Every estimator has reference tests
   that compare it against brute-force sums, TreeCorr or an independent
   implementation at machine precision, and those tolerances are not
   loosened. A new feature comes with a test of the same kind; a bug fix
   comes with the test that would have caught it.
2. **Both backends.** Every CUDA kernel has a NumPy twin in
   `tests/cuda_emulation.py` and a Numba counterpart in `pixpoint/backend/`,
   so the GPU code paths run in CI on CPU-only machines.
   `tests/test_kernel_signatures.py` enforces that every kernel has its twin.
3. **Reproducibility.** Default settings must keep producing bitwise the same
   numbers as the previous release unless the changelog says otherwise, and
   every change that moves a result, however slightly, is a changelog entry
   that says by how much.
4. **Style.** `make lint` must pass: black (88 columns), isort, flake8 and
   `mypy --strict`. The package is fully typed; device-or-host arrays are
   typed as `Any`.
5. **Documentation.** User-facing changes update the relevant page under
   `docs/`, the docstrings and `CHANGELOG.md`.

## Pull requests

Work on a branch, keep the pull request to one topic, and describe what
changed and how you checked it (which tests, on which hardware). CI runs the
lint job and the test matrix on every pull request. Small, focused pull
requests are reviewed quickly; a large one is easier to review if it comes
with a short design note in the description.

## Code of conduct

This project follows the [Contributor Covenant](https://github.com/D-Gebauer/PixPoint/blob/main/CODE_OF_CONDUCT.md). Be kind.

## Licence

By contributing you agree that your contribution is licensed under the
project's MIT licence.
