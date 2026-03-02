# Repository Guidelines

## Project Structure & Module Organization
- `ordinalclip/`: core package.
  - `models/`: `OrdinalCLIP`, image encoders, and prompt learners.
  - `runner/`: Lightning training loop, data module, optimizer/scheduler setup.
  - `utils/`: registry and logging helpers.
- `scripts/`: runnable entry points (`run.py`, experiment sweep/generation, result parsing, ordinality visualization).
- `configs/`: layered OmegaConf YAMLs (`default.yaml` + `base_cfgs/...`) for reproducible experiments.
- `tests/`: pytest suites for CLIP loading, prompt learners, and runner behavior.
- `CLIP/`: OpenAI CLIP submodule (installed editable).
- Runtime artifacts are expected in `.cache/` (weights), `data/` (datasets), and `results/` (outputs).

## Build, Test, and Development Commands
- `conda env create -f environment.yaml && conda activate ordinalclip`: create the baseline environment.
- `pip install -r requirements.txt && pip install -e CLIP/ && pip install -e .`: install dependencies and editable packages.
- `python scripts/run.py --config configs/default.yaml --config <override.yaml>`: run train/eval with merged configs.
- `python scripts/run.py ... --test_only`: evaluate without training.
- `pytest tests/` or `pytest tests/test_runner.py`: run all tests or a targeted file.
- `pre-commit run --all-files`: run lint/format/security hooks before pushing.

## Coding Style & Naming Conventions
- Python style: 4-space indentation, max line length `120` (`pyproject.toml`).
- Format/import order: `black` + `isort`; lint with `flake8`; optional modernization with `flynt`.
- Use `snake_case` for functions/files/config keys, `PascalCase` for classes, and keep config filenames descriptive (example: `num-shots-16.yaml`).

## Testing Guidelines
- Framework: `pytest`.
- Place tests under `tests/` as `test_*.py`; keep one behavioral concern per test function.
- For model-related tests, ensure required checkpoints exist in `.cache/clip`.
- No strict coverage gate is defined; new features should add or update focused tests.

## Commit & Pull Request Guidelines
- Existing history uses short imperative subjects (example: `Update README.md`, `fix submodule of CLIP`).
- Prefer concise commit titles, one logical change per commit, and include config/data assumptions in the body when relevant.
- PRs should include: purpose, key changes, exact run/test commands used, and representative outputs (metrics/log snippets) for training changes.

## Security & Configuration Tips
- Never commit dataset contents, checkpoints, or secrets.
- Keep machine-specific paths and credentials out of YAML; pass overrides via config files or CLI options.

## Agent-Specific Instructions
- Default assistant response language is Chinese (`中文`).
- Switch to another language only when the user explicitly requests it.
