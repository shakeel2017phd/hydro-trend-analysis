# Contributing

Thanks for your interest in improving hydro-trend-analysis.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Before you open a pull request

- **Format & lint:** `ruff format . && ruff check .`
- **Types:** `mypy src`
- **Tests:** `pytest` (add tests for any new behaviour)

All of the above also run in CI, and `pre-commit` runs them on each commit.

## Commit & versioning

- Keep commits focused; write imperative subject lines ("Add Sen slope helper").
- **Do not edit a version by hand.** Releases are cut by pushing a `vX.Y.Z`
  git tag; `setuptools_scm` derives the version from it.
- Add a note under `## [Unreleased]` in `CHANGELOG.md`.

## Reporting bugs

Open an issue with a minimal reproducible example and your Python/OS versions.