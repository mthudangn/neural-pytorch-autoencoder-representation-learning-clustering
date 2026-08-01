import numpy as np
from representation_learning.autoencoder import train_autoencoder, encode_features


def test_autoencoder_latent_shape():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 8)).astype(np.float32)
    model = train_autoencoder(X, hidden_size=3, epochs=2, batch_size=10)
    assert encode_features(model, X).shape == (20, 3)
