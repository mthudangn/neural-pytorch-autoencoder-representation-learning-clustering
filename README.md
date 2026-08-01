# Representation Learning & Clustering

A Python research prototype combining probabilistic document clustering, neural decision-boundary analysis, and self-taught representation learning from labelled and unlabelled data.

## Core components

- Multinomial mixture model for document clustering with soft and hard Expectation–Maximisation.
- Log-domain probability calculations and log-sum-exp normalisation for numerical stability.
- Count-vector text preprocessing and PCA cluster visualisation.
- Perceptron hyperparameter search with and without early stopping.
- Multi-layer perceptron comparison across hidden-layer widths, learning rates, and L2 strengths.
- PyTorch autoencoders trained on combined labelled and unlabelled data.
- Self-taught classifiers using concatenated raw and latent features.

## Repository structure

```text
representation-learning-clustering/
├── src/representation_learning/
│   ├── document_em.py
│   ├── decision_boundaries.py
│   └── autoencoder.py
├── scripts/
│   ├── 01_document_clustering.py
│   ├── 02_decision_boundaries.py
│   └── 03_self_taught_learning.py
├── data/README.md
├── docs/
├── tests/
└── results/reference_metrics.csv
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest
```

The scripts accept user-supplied datasets through command-line arguments. Expected schemas are documented in `data/README.md`.

## Reference findings

Soft EM converged in 77 iterations in the reference run, while hard EM converged in 10. Both recovered the same broad PCA structure, but soft assignments represented uncertainty in overlapping regions whereas hard assignments produced sharper boundaries.

On a non-linearly separable binary dataset, the best Perceptron variants produced test error near 0.12. A one-hidden-layer neural network reduced test error to 0.0010 with hidden width 15, learning rate 0.1, and weak L2 regularisation. Stronger regularisation increased test error to 0.0135 and produced a visibly stiffer boundary.

The autoencoder experiment used 50 labelled samples, 1,500 unlabelled samples, 500 test samples, and 784 input features. Reconstruction error decreased as the bottleneck expanded. At hidden width 180, the baseline classifier produced test error 0.5280, while latent-feature augmentation reduced it to 0.4540. Benefits were strongest at moderate bottleneck sizes and inconsistent at very small or very large representations.

## Scope

This repository demonstrates modelling behaviour and representation-learning trade-offs. Included reference numbers are dataset-specific experimental outputs, not universal benchmarks.

## License

MIT
