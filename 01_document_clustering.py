#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer

from representation_learning import MultinomialMixtureEM, vectorize_documents


def load_documents(path: Path) -> tuple[list[str], list[str]]:
    labels, documents = [], []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        if "\t" in line:
            label, document = line.split("\t", 1)
        else:
            label, document = "unknown", line
        labels.append(label.strip())
        documents.append(document.strip())
    return labels, documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster documents with soft and hard multinomial EM.")
    parser.add_argument("documents", type=Path)
    parser.add_argument("--clusters", type=int, default=4)
    parser.add_argument("--min-df", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("results/document_clustering"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _, documents = load_documents(args.documents)
    X, vectorizer = vectorize_documents(documents, min_df=args.min_df)
    soft = MultinomialMixtureEM(args.clusters, mode="soft", random_state=0).fit(X)
    hard = MultinomialMixtureEM(args.clusters, mode="hard", random_state=0).fit(X)

    pd.DataFrame({"soft_cluster": soft.labels_, "hard_cluster": hard.labels_}).to_csv(
        args.output_dir / "cluster_assignments.csv", index=False
    )
    pd.DataFrame({"soft_log_likelihood": soft.log_likelihood_history_}).to_csv(
        args.output_dir / "soft_log_likelihood.csv", index=False
    )
    pd.DataFrame({"hard_log_likelihood": hard.log_likelihood_history_}).to_csv(
        args.output_dir / "hard_log_likelihood.csv", index=False
    )

    X_normalized = Normalizer(norm="l2").fit_transform(X)
    X_2d = PCA(n_components=2, random_state=0).fit_transform(X_normalized.toarray())
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), tight_layout=True)
    axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=soft.labels_, s=15)
    axes[0].set_title("Soft EM")
    axes[1].scatter(X_2d[:, 0], X_2d[:, 1], c=hard.labels_, s=15)
    axes[1].set_title("Hard EM")
    fig.savefig(args.output_dir / "pca_clusters.png", dpi=160)
    plt.close(fig)
    print({"soft_iterations": soft.n_iter_, "hard_iterations": hard.n_iter_, "vocabulary_size": len(vectorizer.vocabulary_)})


if __name__ == "__main__":
    main()
