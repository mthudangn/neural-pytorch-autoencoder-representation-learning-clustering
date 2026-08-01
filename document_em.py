"""Multinomial mixture model for document clustering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.special import logsumexp
from scipy.sparse import csr_matrix, issparse
from sklearn.feature_extraction.text import CountVectorizer


def vectorize_documents(
    documents: Iterable[str],
    min_df: int = 1,
    stop_words: str | None = "english",
) -> tuple[csr_matrix, CountVectorizer]:
    vectorizer = CountVectorizer(lowercase=True, stop_words=stop_words, min_df=min_df)
    matrix = vectorizer.fit_transform(list(documents)).astype(np.float64)
    return matrix.tocsr(), vectorizer


class MultinomialMixtureEM:
    """Mixture of multinomials fitted using soft or hard EM."""

    def __init__(
        self,
        n_clusters: int = 4,
        mode: str = "soft",
        max_iter: int = 200,
        tol: float = 1e-6,
        smoothing: float = 1e-3,
        random_state: int = 0,
    ):
        self.n_clusters = n_clusters
        self.mode = mode
        self.max_iter = max_iter
        self.tol = tol
        self.smoothing = smoothing
        self.random_state = random_state

    def _validate_X(self, X):
        matrix = X.tocsr().astype(np.float64) if issparse(X) else csr_matrix(np.asarray(X, dtype=np.float64))
        if matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError("X must contain at least one document and one feature")
        if matrix.min() < 0:
            raise ValueError("Multinomial counts must be non-negative")
        return matrix

    def _joint_log_probability(self, X: csr_matrix) -> np.ndarray:
        return np.asarray(X @ np.log(self.word_probabilities_).T) + np.log(self.cluster_priors_)[None, :]

    def _responsibilities(self, X: csr_matrix) -> tuple[np.ndarray, float]:
        joint = self._joint_log_probability(X)
        normalizer = logsumexp(joint, axis=1, keepdims=True)
        soft = np.exp(joint - normalizer)
        log_likelihood = float(normalizer.sum())
        if self.mode == "hard":
            hard = np.zeros_like(soft)
            hard[np.arange(len(soft)), np.argmax(soft, axis=1)] = 1.0
            return hard, log_likelihood
        return soft, log_likelihood

    def fit(self, X) -> "MultinomialMixtureEM":
        X = self._validate_X(X)
        if self.mode not in {"soft", "hard"}:
            raise ValueError("mode must be 'soft' or 'hard'")
        if int(self.n_clusters) < 2:
            raise ValueError("n_clusters must be at least 2")

        rng = np.random.default_rng(self.random_state)
        responsibilities = rng.dirichlet(np.ones(int(self.n_clusters)), size=X.shape[0])
        self.log_likelihood_history_: list[float] = []
        previous_labels = None

        for iteration in range(int(self.max_iter)):
            effective_counts = responsibilities.sum(axis=0) + self.smoothing
            self.cluster_priors_ = effective_counts / effective_counts.sum()

            weighted_word_counts = responsibilities.T @ X
            weighted_word_counts = np.asarray(weighted_word_counts) + self.smoothing
            self.word_probabilities_ = weighted_word_counts / weighted_word_counts.sum(axis=1, keepdims=True)

            updated, log_likelihood = self._responsibilities(X)
            self.log_likelihood_history_.append(log_likelihood)

            if self.mode == "hard":
                labels = np.argmax(updated, axis=1)
                if previous_labels is not None and np.array_equal(labels, previous_labels):
                    responsibilities = updated
                    self.n_iter_ = iteration + 1
                    break
                previous_labels = labels
            elif iteration > 0 and abs(self.log_likelihood_history_[-1] - self.log_likelihood_history_[-2]) <= self.tol:
                responsibilities = updated
                self.n_iter_ = iteration + 1
                break

            responsibilities = updated
        else:
            self.n_iter_ = int(self.max_iter)

        self.responsibilities_ = responsibilities
        self.labels_ = np.argmax(responsibilities, axis=1)
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X) -> np.ndarray:
        X = self._validate_X(X)
        joint = self._joint_log_probability(X)
        return np.exp(joint - logsumexp(joint, axis=1, keepdims=True))

    def predict(self, X) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)
