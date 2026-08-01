# Reproducibility

1. Create a Python 3.10+ virtual environment.
2. Install dependencies and the repository in editable mode.
3. Run `pytest`.
4. Place compatible datasets under `data/` or pass absolute paths to the scripts.
5. Execute each script from the repository root.

The implementations use fixed random seeds. Autoencoder results can still vary slightly across PyTorch versions, hardware, and numerical backends.
