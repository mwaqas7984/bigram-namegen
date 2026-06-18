# Contributing

Contributions are welcome — bug fixes, new sample datasets, additional models (e.g. an MLP or attention-based extension), or documentation improvements.

## Getting set up

```bash
git clone https://github.com/<your-username>/bigram-namegen.git
cd bigram-namegen
pip install -e ".[dev,plot]"
pytest
```

## Guidelines

- Keep functions small and focused; this project favors clarity over cleverness since it's meant to be readable.
- Add or update tests in `tests/` for any behavior change.
- Run `pytest` before opening a pull request and make sure it passes.
- Match the existing code style (type hints on public functions, docstrings explaining the *why*, not just the *what*).

## Reporting issues

Please open a GitHub issue with a clear description, the command you ran, and the full error/output if applicable.
