import numpy as np
from representation_learning import MultinomialMixtureEM


def test_document_em_fits_count_matrix():
    X = np.array([
        [8, 1, 0, 0],
        [7, 2, 0, 0],
        [0, 0, 8, 1],
        [0, 0, 7, 2],
    ])
    model = MultinomialMixtureEM(n_clusters=2, mode="soft", random_state=2).fit(X)
    assert model.predict_proba(X).shape == (4, 2)
    assert np.allclose(model.predict_proba(X).sum(axis=1), 1.0)
