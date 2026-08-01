#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from representation_learning import compare_self_taught


def split_labelled(frame: pd.DataFrame):
    label_column = frame.columns[0]
    return frame.drop(columns=[label_column]).to_numpy(), frame[label_column].to_numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline and self-taught neural classifiers.")
    parser.add_argument("labelled_csv", type=Path)
    parser.add_argument("unlabelled_csv", type=Path)
    parser.add_argument("test_csv", type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("results/self_taught"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    labelled = pd.read_csv(args.labelled_csv)
    unlabelled = pd.read_csv(args.unlabelled_csv)
    test = pd.read_csv(args.test_csv)
    X_labelled, y_labelled = split_labelled(labelled)
    X_test, y_test = split_labelled(test)

    results = compare_self_taught(
        X_labelled,
        y_labelled,
        unlabelled.to_numpy(),
        X_test,
        y_test,
        epochs=args.epochs,
    )
    results.to_csv(args.output_dir / "self_taught_metrics.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(results["hidden_size"], results["baseline_test_error"], marker="o", label="Baseline")
    plt.plot(results["hidden_size"], results["self_taught_test_error"], marker="s", label="Self-taught")
    plt.xlabel("Hidden size")
    plt.ylabel("Test classification error")
    plt.title("Baseline versus latent-feature augmentation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_dir / "classification_error.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(results["hidden_size"], results["reconstruction_error"], marker="o")
    plt.xlabel("Bottleneck size")
    plt.ylabel("Mean reconstruction distance")
    plt.title("Autoencoder compression–reconstruction trade-off")
    plt.tight_layout()
    plt.savefig(args.output_dir / "reconstruction_error.png", dpi=160)
    plt.close()
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
