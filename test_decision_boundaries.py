import numpy as np
from representation_learning import grid_search_perceptron


def test_perceptron_grid_returns_both_variants():
    X = np.array([[-2, -1], [-1, -2], [1, 2], [2, 1], [-1.5, -1.5], [1.5, 1.5]], dtype=float)
    y = np.array([0, 0, 1, 1, 0, 1])
    results, models = grid_search_perceptron(
        X, y, X, y, learning_rates=[0.01], l2_strengths=[0.001]
    )
    assert set(models) == {"fixed_iterations", "early_stopping"}
    assert len(results) == 2
