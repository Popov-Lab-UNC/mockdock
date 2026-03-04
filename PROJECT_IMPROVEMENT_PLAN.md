# FCGMB Project Improvement Plan

Based on the [Setting up Python Packages blog post by Jonathon Vandezande](https://jevandezande.github.io/blog/setting-up-python-packages/), here are detailed suggestions to structure, clean up, and polish the FCGMB package for publishing to the broader scientific community. While the blog advocates for a "flat" layout, this plan acknowledges your preference for a `src/` layout (which is also the default recommended by the Python Packaging Authority).

## 1. Project Layout (The `src/` Layout)
The repository currently mixes source code (`fcgmb/`), standalone scripts (`pipeline/`, `analyze_variance_runs.py`), notebooks (`notebooks/`), compute configurations (`.sbatch` files), and documentation (`draft.md`, `README.md`) in the root directory. Adopting the `src/` layout and strictly organizing the root directory will greatly improve maintainability.

### Recommended Directory Structure:
```
fcgmb/
├── .github/              # GitHub workflows (CI/CD, publishing)
├── docs/                 # Documentation (Sphinx/MkDocs, draft.md if needed)
├── notebooks/            # Jupyter/marimo notebooks
├── results/              # Output from running the package (add to .gitignore)
├── scripts/              # Useful scripts for managing the package/experiments
│   ├── experiments/      # e.g., analyze_variance_runs.py, run_variance.py
│   ├── pipeline/         # Move the top-level pipeline/ directory here
│   └── slurm/            # Move all *.sbatch and *.sh scripts here
├── src/                  # Source code directory
│   └── fcgmb/            # The actual Python package
│       ├── __init__.py
│       ├── analysis.py
│       └── ...
├── tests/                # Unit tests
├── static/               # Holds input data (if any outside of package data)
├── .coveragerc           # Information on how to assess code coverage
├── .editorconfig         # Information for the editor on how to format files
├── .env                  # Private environment variables (add to .gitignore)
├── .envrc                # Handles automatic loading of the environment (direnv)
├── .gitignore            # Tells Git which files to ignore
├── LICENSE               # License indicating how code can be used
├── prek.toml             # Git hooks for pre-commit/pre-push (or .pre-commit-config.yaml)
├── pyproject.toml        # Package setup, tool management, etc.
├── README.md             # Information about your package and how to use it
└── uv.lock               # Precise set of packages to load
```

### Action Items for Layout Cleanup:
- **Move Source Code**: Move the `fcgmb/` package directory into `src/fcgmb/`. Update `pyproject.toml`'s `[tool.setuptools.packages.find]` to `where = ["src"]` and remove complex excludes, as `src/` inherently isolates your package code from `data/`, `logs/`, etc.
- **Move Top-Level Scripts**: Move `add_docking_baselines.py`, `analyze_variance_runs.py`, `run_variance.py`, `run_workflow.py`, and `validation_test.py` into `scripts/` or `scripts/experiments/`. If `run_workflow.py` is meant to be a CLI tool for users, expose it via `[project.scripts]` in `pyproject.toml` instead.
- **Move SLURM Scripts**: Move `run_variance.sbatch`, `run_workflow_cpu.sbatch`, `run_workflow_longleaf.sbatch`, and `validation_test.sbatch` to `scripts/slurm/`.
- **Move Data/Configs**: Ensure that top-level folders like `configs/`, `custom_configs/`, `custom_data/`, and `exps/` are clearly separated. If they are user/experimental outputs, add them to `.gitignore` so they don't pollute the repository. If they are examples, move them to an `examples/` directory.
- **Move the `pipeline/` Folder**: The `pipeline/` directory contains data mining scripts (e.g., `compute_mcs.py`, `fetch_chembl_targets.py`). These are distinct from the core user-facing package. Move this directory to `scripts/pipeline/`.
- **Move Drafts**: Move `draft.md` to a new `docs/` folder or remove it if it's no longer needed in the repository.

## 2. Package Manager & Project Isolation
You are already using `uv` and `pyproject.toml` with a `uv.lock` file, which is fantastic and exactly what the blog post recommends.

### Action Items:
- Consider adding a `.envrc` file to use `direnv` for automatic environment loading when navigating into the project directory:
  ```bash
  watch_file uv.lock
  dotenv_if_exists .env
  uv sync --frozen --dev
  source .venv/bin/activate
  ```
- **Build Backend**: You are using `setuptools.build_meta`. The blog recommends `hatchling`. Switching is optional, but `hatchling` requires significantly less configuration (especially when using the `src/` layout, it just works without explicit `find` blocks).

## 3. Project Tools (Formatting, Linting, Typing)
Your `pyproject.toml` shows you use `ruff` and `black`, but as the blog post notes, "Ruff has since replaced Black as my formatting tool due to its incredible speed (it maintains the same formatting style)."

### Action Items:
- **Remove Black**: Remove the `[tool.black]` configuration block from `pyproject.toml`. Rely entirely on `ruff format` and `ruff check`.
- **Typing**: You have `ty` in your dev dependencies (as recommended by the blog). Run it systematically (`ty .`). Ensure all new functions, especially in public APIs (`oracle.py`, `docking.py`), have precise type hints.
- **Remove `TODO` Comments**: A quick `grep -rn "TODO"` or `# TODO` shows leftover comments (e.g., in git hooks if they were committed). Review all source files and remove/resolve unnecessary `TODO` comments to present a polished, finished product.

## 4. Testing
You have a `tests/` directory and `pytest` configured in `pyproject.toml`, but the blog post emphasizes comprehensive unit testing.

### Action Items:
- Add `pytest-cov` to your dev dependencies to measure test coverage.
- Create a `.coveragerc` file to define how coverage is measured (e.g., omitting `tests/` from the coverage calculation).
- Ensure that core components (`analysis.py`, `data.py`, `docking.py`, `oracle.py`) have robust test suites. Run tests using `pytest --cov=src/fcgmb`.
- **Doctests**: Consider adding simple doctests to your docstrings (as shown in the blog) to provide runnable examples that double as smoke tests.

## 5. Git Flow (CI/CD and Pre-commit)
Currently, there appear to be no automated checks on the repository.

### Action Items:
- **Pre-commit Hooks**: Add a `prek.toml` (recommended by the blog) or `.pre-commit-config.yaml` to enforce `ruff format`, `ruff check`, and `ty` on every commit. This ensures you (or LLM agents) don't introduce trivial formatting/linting errors.
- **GitHub Actions**: Create `.github/workflows/ci.yml`. This workflow should run on Pull Requests and pushes to `main`. It should:
  1. Install `uv`.
  2. Install the project (`uv sync`).
  3. Run formatting checks (`ruff format --check`).
  4. Run linting (`ruff check`).
  5. Run type checking (`ty src/fcgmb`).
  6. Run tests (`pytest`).

## 6. Publishing Packages
The blog post emphasizes making the publishing process frictionless.

### Action Items:
- Create a GitHub Action `.github/workflows/publish.yml` that triggers on a new Release or a specific tag (e.g., `v*`).
- This action should use `uv build` (or `python -m build`) and then `uv publish` (or `twine upload`) to automatically push the package to PyPI, removing the need for manual uploads.

## 7. Documentation
Your `README.md` is functional but could be expanded.

### Action Items:
- **Docstrings**: Ensure all functions use a consistent docstring style (e.g., Google-style). Drop type information from the docstrings (since you use static typing) and make argument descriptions terse.
- **Development Guide**: Add a "Development" or "Contributing" section to the `README.md` explaining how to set up the dev environment (`uv sync --dev`), run tests, and format code.
- **Formal Documentation**: If the package logic is complex, consider setting up Sphinx or MkDocs to automatically generate a documentation website from your docstrings.

## 8. Licensing Code
You already have a `LICENSE` file. Verify that it is a standard, permissive open-source license (like MIT or BSD-3-Clause) as recommended by the blog post to maximize adoption by both academic and commercial entities.

## Summary of Files to Delete or Rewrite:
- **Delete**: `.black` configurations from `pyproject.toml`.
- **Move/Rewrite**: `draft.md` (move to `docs/` or delete).
- **Move**: `pipeline/` -> `scripts/pipeline/`.
- **Move**: `.sbatch` files -> `scripts/slurm/`.
- **Move**: Root-level python scripts (`analyze_*.py`, `run_*.py`) -> `scripts/` or `scripts/experiments/`.
- **Rewrite**: Update `pyproject.toml` to support the `src/` layout.
