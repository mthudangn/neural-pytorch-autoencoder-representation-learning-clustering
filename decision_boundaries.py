"""Perceptron and neural-network grid searches for two-dimensional data."""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def grid_search_perceptron(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    learning_rates: Iterable[float] = (0.001, 0.01, 0.1),
    l2_strengths: Iterable[float] = (0.0001, 0.001, 0.01, 0.1, 1.0),
) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    rows = []
    best_models: dict[str, Pipeline] = {}

    for early_stopping in (False, True):
        variant = "early_stopping" if early_stopping else "fixed_iterations"
        variant_rows = []
        for eta in learning_rates:
            for alpha in l2_strengths:
                model = Pipeline(
                    [
                        ("scale", StandardScaler()),
                        (
                            "model",
                            SGDClassifier(
                                loss="perceptron",
                                penalty="l2",
                                alpha=float(alpha),
                                learning_rate="constant",
                                eta0=float(eta),
                                max_iter=2000,
                                early_stopping=early_stopping,
                                validation_fraction=0.2,
                                tol=0.001 if early_stopping else None,
                                random_state=42,
                            ),
                        ),
                    ]
                )
                model.fit(X_train, y_train)
                test_error = 1.0 - accuracy_score(y_test, model.predict(X_test))
                row = {"variant": variant, "learning_rate": eta, "l2_strength": alpha, "test_error": test_error}
                rows.append(row)
                variant_rows.append((test_error, model))
        best_models[variant] = min(variant_rows, key=lambda item: item[0])[1]

    return pd.DataFrame(rows), best_models


def grid_search_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    hidden_sizes: Iterable[int] = range(5, 41, 5),
    learning_rates: Iterable[float] = (0.001, 0.01, 0.1),
    l2_strengths: Iterable[float] = (0.001, 1.0),
) -> tuple[pd.DataFrame, dict[float, Pipeline]]:
    rows = []
    best_models: dict[float, Pipeline] = {}

    for alpha in l2_strengths:
        candidates = []
        for hidden_size in hidden_sizes:
            for eta in learning_rates:
                model = Pipeline(
                    [
                        ("scale", StandardScaler()),
                        (
                            "model",
                            MLPClassifier(
                                hidden_layer_sizes=(int(hidden_size),),
                                activation="relu",
                                alpha=float(alpha),
                                learning_rate_init=float(eta),
                                max_iter=3000,
                                random_state=42,
                            ),
                        ),
                    ]
                )
                model.fit(X_train, y_train)
                test_error = 1.0 - accuracy_score(y_test, model.predict(X_test))
                row = {
                    "l2_strength": float(alpha),
                    "hidden_size": int(hidden_size),
                    "learning_rate": float(eta),
                    "test_error": float(test_error),
                }
                rows.append(row)
                candidates.append((test_error, model))
        best_models[float(alpha)] = min(candidates, key=lambda item: item[0])[1]

    return pd.DataFrame(rows), best_models


def plot_decision_boundary(model, X: np.ndarray, y: np.ndarray, ax=None, title: str = "Decision boundary"):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 350), np.linspace(y_min, y_max, 350))
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    predictions = model.predict(grid).reshape(xx.shape)
    ax.contourf(xx, yy, predictions, alpha=0.25)
    ax.scatter(X[:, 0], X[:, 1], c=y, s=12, edgecolors="none")
    ax.set_title(title)
    ax.set_xlabel("feature_1")
    ax.set_ylabel("feature_2")
    return ax
