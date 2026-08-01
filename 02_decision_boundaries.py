#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from representation_learning import grid_search_mlp, grid_search_perceptron, plot_decision_boundary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare linear and neural decision boundaries.")
    parser.add_argument("train_csv", type=Path)
    parser.add_argument("test_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/decision_boundaries"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(args.train_csv)
    test = pd.read_csv(args.test_csv)
    feature_columns = [column for column in train.columns if column != "label"][:2]
    X_train, y_train = train[feature_columns].to_numpy(), train["label"].to_numpy()
    X_test, y_test = test[feature_columns].to_numpy(), test["label"].to_numpy()

    perceptron_results, perceptron_models = grid_search_perceptron(X_train, y_train, X_test, y_test)
    mlp_results, mlp_models = grid_search_mlp(X_train, y_train, X_test, y_test)
    perceptron_results.to_csv(args.output_dir / "perceptron_grid.csv", index=False)
    mlp_results.to_csv(args.output_dir / "mlp_grid.csv", index=False)

    models = list(perceptron_models.items()) + [(f"MLP alpha={alpha}", model) for alpha, model in mlp_models.items()]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), tight_layout=True)
    for ax, (name, model) in zip(axes.ravel(), models):
        plot_decision_boundary(model, X_test, y_test, ax=ax, title=name)
    fig.savefig(args.output_dir / "decision_boundaries.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
