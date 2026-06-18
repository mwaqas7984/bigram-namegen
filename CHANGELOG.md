# Changelog

All notable changes to this project are documented here.
This project follows [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-06-18

### Added
- Initial release.
- Counting-based bigram model (`counting_model.py`) with Laplace smoothing, name sampling, and NLL loss evaluation.
- Single-layer neural bigram model (`neural_model.py`) trained via gradient descent.
- Loss-curve plotting (`plotting.py`).
- Command-line interface (`python -m bigram_namegen`).
- Example dataset of 50 fantasy villain names.
- Unit tests for vocabulary, counting model, and neural model.
- Original exploratory notebook preserved under `notebooks/`.
