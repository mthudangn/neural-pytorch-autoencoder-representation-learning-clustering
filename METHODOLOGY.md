# Methodology

## Document clustering

Documents are converted to sparse count vectors after lowercasing, English stop-word removal, and minimum-document-frequency filtering. A mixture of multinomials models each cluster with a prior probability and a word-probability distribution.

Soft EM uses fractional posterior responsibilities. Hard EM converts those responsibilities to one-hot assignments before the M-step. Joint probabilities are computed in the log domain, and log-sum-exp normalisation prevents numerical overflow and underflow.

## Decision-boundary analysis

A linear Perceptron is tuned over learning rate and L2 strength with and without early stopping. A one-hidden-layer ReLU network is tuned over hidden width, learning rate, and regularisation. The comparison focuses on test error and the geometric fit of each decision boundary.

## Self-taught learning

Autoencoders are fitted on the union of labelled and unlabelled features. The supervised classifier is trained only on labelled rows. Self-taught models concatenate the raw feature vector with the learned bottleneck representation. This permits unlabelled observations to influence representation learning without supplying pseudo-labels.
