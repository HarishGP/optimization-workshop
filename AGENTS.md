# AGENTS.md

## Purpose
This repository supports a 2-hour undergraduate workshop on optimization in neural network training.
The codebase is designed for teaching, not for production deployment or package publishing.
Students mainly interact with notebooks; supporting Python modules should remain simple, readable, and stable.

## Ground rules
- Prefer clarity over cleverness.
- Keep dependencies minimal and CPU-friendly.
- Do not introduce packaging/build complexity unless explicitly requested.
- Do not add GPU-specific code or CUDA-only dependencies.
- Avoid large framework additions unless they are essential to the workshop goals.
- Preserve a beginner-friendly structure and naming style.

## Project structure
- `notebooks/`: student-facing workshop notebooks.
- `workshoplib/`: reusable helper code imported by notebooks.
- `smoke_test.py`: quick import/environment check.
- `pyproject.toml`: uv-managed dependencies for a non-packaged project.

## Coding guidelines
- Write small, readable functions with explicit names.
- Add short docstrings to nontrivial functions.
- Prefer pure functions where possible.
- Keep notebook cells simple; move reusable logic into `workshoplib/`.
- Avoid unnecessary abstraction, metaprogramming, or deep class hierarchies.
- Use comments to explain teaching intent, not obvious syntax.

## Agent workflow
Before making changes:
1. Read `PROJECT_BRIEF.md`.
2. Preserve the notebook-first teaching workflow.
3. Prefer minimal edits over broad refactors.
4. If changing dependencies, explain why the new dependency is necessary for the workshop.
5. If modifying imports or file layout, keep notebook usage simple.

## Dependencies
- Use `uv`.
- Assume the project is not installed as a package.
- Keep PyTorch CPU-only unless explicitly told otherwise.
- Do not add heavy optional tools by default.

## Testing and verification
- For code changes, prefer lightweight verification.
- Use `uv run python smoke_test.py` for quick import checks.
- If editing notebook-facing helpers, ensure imports remain simple:
  - `from workshoplib.datagen import ...`
  - `from workshoplib.model import ...`

## Style expectations
- Write code that an undergraduate with basic Python can read.
- Favor explicit tensors, loops, and optimizer steps over overly compact idioms.
- Keep examples small enough to run quickly on student laptops.
- Make failure modes easy to diagnose.

## Avoid
- Hidden magic in notebook setup.
- Path hacks unless absolutely necessary.
- Overengineering.
- Unused dependencies.
- Breaking changes to notebook imports without a strong reason.