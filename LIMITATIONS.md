# Limitations

- EM optimises a non-convex objective and can converge to local optima.
- Bag-of-words counts discard word order and semantic context.
- PCA visualisations are two-dimensional projections and do not preserve every cluster relationship.
- Test-set grid selection is suitable for controlled experiments but should be replaced with a validation or nested-CV protocol in deployment-oriented work.
- Autoencoder reconstruction quality does not guarantee discriminative latent features.
- The reference self-taught experiment contains very few labelled examples, so estimates have substantial variance.
