"""Autoencoder and self-taught feature-augmentation utilities."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class Autoencoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU())
        self.decoder = nn.Linear(hidden_size, input_size)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(X))


def train_autoencoder(
    X: np.ndarray,
    hidden_size: int,
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    random_state: int = 2025,
    device: str | None = None,
) -> Autoencoder:
    torch.manual_seed(random_state)
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = Autoencoder(X.shape[1], int(hidden_size)).to(target_device)
    loader = DataLoader(TensorDataset(torch.from_numpy(X.astype(np.float32))), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    model.train()
    for _ in range(int(epochs)):
        for (batch,) in loader:
            batch = batch.to(target_device)
            optimizer.zero_grad()
            loss = criterion(model(batch), batch)
            loss.backward()
            optimizer.step()
    return model


def encode_features(model: Autoencoder, X: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        return model.encoder(torch.from_numpy(X.astype(np.float32)).to(device)).cpu().numpy()


def reconstruction_error(model: Autoencoder, X: np.ndarray) -> float:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(X.astype(np.float32)).to(device)
        reconstructed = model(tensor)
        return float(torch.linalg.vector_norm(tensor - reconstructed, dim=1).mean().cpu())


def compare_self_taught(
    X_labeled: np.ndarray,
    y_labeled: np.ndarray,
    X_unlabeled: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    hidden_sizes: Iterable[int] = (20, 60, 100, 140, 180, 220),
    epochs: int = 100,
) -> pd.DataFrame:
    """Compare raw-feature MLPs with raw-plus-latent self-taught MLPs."""
    scaler = StandardScaler().fit(np.vstack([X_labeled, X_unlabeled]))
    X_labeled_scaled = scaler.transform(X_labeled).astype(np.float32)
    X_unlabeled_scaled = scaler.transform(X_unlabeled).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)
    X_autoencoder = np.vstack([X_labeled_scaled, X_unlabeled_scaled])

    rows = []
    for hidden_size in hidden_sizes:
        autoencoder = train_autoencoder(X_autoencoder, int(hidden_size), epochs=epochs)
        latent_train = encode_features(autoencoder, X_labeled_scaled)
        latent_test = encode_features(autoencoder, X_test_scaled)

        baseline = MLPClassifier(hidden_layer_sizes=(int(hidden_size),), max_iter=1000, random_state=42)
        baseline.fit(X_labeled, y_labeled)
        baseline_error = 1.0 - accuracy_score(y_test, baseline.predict(X_test))

        augmented_train = np.column_stack([X_labeled, latent_train])
        augmented_test = np.column_stack([X_test, latent_test])
        self_taught = MLPClassifier(hidden_layer_sizes=(int(hidden_size),), max_iter=1000, random_state=42)
        self_taught.fit(augmented_train, y_labeled)
        self_taught_error = 1.0 - accuracy_score(y_test, self_taught.predict(augmented_test))

        rows.append(
            {
                "hidden_size": int(hidden_size),
                "reconstruction_error": reconstruction_error(autoencoder, X_test_scaled),
                "baseline_test_error": float(baseline_error),
                "self_taught_test_error": float(self_taught_error),
                "improvement": float(baseline_error - self_taught_error),
            }
        )
    return pd.DataFrame(rows)
