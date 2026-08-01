"""Representation-learning and clustering utilities."""

from .document_em import MultinomialMixtureEM, vectorize_documents
from .decision_boundaries import grid_search_perceptron, grid_search_mlp, plot_decision_boundary
from .autoencoder import Autoencoder, train_autoencoder, encode_features, compare_self_taught

__all__ = [
    "MultinomialMixtureEM",
    "vectorize_documents",
    "grid_search_perceptron",
    "grid_search_mlp",
    "plot_decision_boundary",
    "Autoencoder",
    "train_autoencoder",
    "encode_features",
    "compare_self_taught",
]
