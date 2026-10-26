# Contributing

ARGUS is an early, small engine. Focused fixes, reproducible issues, provider
adapters, and honest evaluations are welcome.

1. Fork the repository and create a branch.
2. Install Python 3.10+ and run `python -m pip install -e '.[dev]'`.
3. Make a focused change and add a regression test when behavior changes.
4. Run `ruff check .`, `pytest -q`, `argus-engine demo`, and `python -m build`.
5. Open a pull request describing the behavior, tradeoffs, and validation.

Tests must run without cloud credentials, private footage, weight downloads,
or paid inference. Mock the model boundary for unit tests. Keep optional SDKs
and detectors out of core imports. Preserve exact-verdict parsing and the
conservative treatment of uncertain or failed requests.

Never commit keys, `.env` files, private camera URLs, images of people,
customer data, recordings, model weights, generated outputs, or local databases.
Use synthetic examples or footage whose redistribution rights and consent
are documented. Do not paste provider request bodies into public issues.

For evaluation changes, include labeled ground truth and separate false merges
from missed matches. Do not turn a synthetic demo into an accuracy claim.
See [the evaluation plan](docs/EVALUATION.md).

By contributing, you agree that your original contributions are licensed under
the repository's MIT license. Preserve applicable third-party notices.
