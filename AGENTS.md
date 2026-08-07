# Repository Guidelines

## Project Structure & Module Organization

Core Python code lives under `src/psi/`: models are in `models/`, training logic in `trainers/`, dataset adapters in `data/`, and experiment definitions in `config/train/`. `src/openpi/` and `src/gr00t/` contain integrated model stacks; avoid broad refactors there unless the change specifically targets those integrations. Runnable workflows live in `scripts/`, usage examples in `examples/`, and comparison implementations in `baselines/`. Tests are primarily under `tests/`, with some colocated `*_test.py` files. Static figures and normalization metadata belong in `assets/`. Do not commit generated `data/`, `checkpoints/`, `outputs/`, or `.runs/` content.

## Build, Test, and Development Commands

This project requires Python 3.10 and uses `uv`:

```bash
uv venv .venv-psi --python 3.10
source .venv-psi/bin/activate
GIT_LFS_SKIP_SMUDGE=1 uv sync --group serve --group viz --group psi --active
python -c "import psi; print(psi.__version__)"
pytest -q
```

Initialize submodules with `git submodule update --init --recursive` before SIMPLE-related work. Run focused tests during iteration, for example `pytest -q tests/test_simple_datagen_curobo_missing.py`. Training and deployment are launched through task-specific scripts such as `scripts/train/psi0/finetune-real-psi0.sh <task>` and `scripts/deploy/serve_psi0-rtc.sh`; review GPU, checkpoint, and data-path settings before running them.

## Coding Style & Naming Conventions

Follow existing Python conventions: four-space indentation, type hints for public interfaces, `snake_case` for modules/functions/variables, and `PascalCase` for classes. Keep configuration modules descriptive (for example, `finetune_simple_psi0_config.py`) and shell scripts action-oriented. Prefer small, localized changes and preserve established patterns in third-party-derived code. No repository-wide formatter is configured, so keep imports ordered and code PEP 8 compliant.

## Testing Guidelines

Tests use `pytest`. Name new files `test_<feature>.py` and test functions `test_<behavior>()`. Add regression coverage for bug fixes, especially around imports, data layouts, configuration resolution, and model interfaces. Hardware-heavy tests should document required GPUs, submodules, datasets, or checkpoints and fail with an actionable message when prerequisites are absent.

## Commit & Pull Request Guidelines

Recent history favors short, imperative subjects such as `fix bug: ...`, `add ...`, and `update ...`. Keep each commit focused and state the affected workflow when useful. Pull requests should summarize behavior changes, list commands run, link relevant issues, and call out required data or hardware. Include logs or screenshots for visualization, simulation, or deployment changes, and never include credentials from `.env`.
