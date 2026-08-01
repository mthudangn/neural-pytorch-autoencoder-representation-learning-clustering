# Reference Results

## Document clustering

- Soft EM convergence: 77 iterations.
- Hard EM convergence: 10 iterations.
- Soft EM final reported log-likelihood: approximately -2,380,383.8.
- Hard EM final marginal log-likelihood: approximately -2,397,669.6.

## Decision boundaries

| Model | Configuration | Test error |
|---|---|---:|
| Perceptron with early stopping | learning rate 0.01, L2 1.0 | 0.1210 |
| Perceptron without early stopping | learning rate 0.001, L2 0.0001 | 0.1205 |
| Neural network, weak L2 | hidden width 15, learning rate 0.1, L2 0.001 | 0.0010 |
| Neural network, strong L2 | hidden width 25, learning rate 0.1, L2 1.0 | 0.0135 |

## Self-taught learning

| Hidden size | Reconstruction error | Baseline error | Self-taught error |
|---:|---:|---:|---:|
| 20 | 21.505257 | 0.6300 | 0.7760 |
| 60 | 18.082838 | 0.5500 | 0.5300 |
| 100 | 16.084755 | 0.5780 | 0.4660 |
| 140 | 14.588547 | 0.5660 | 0.5740 |
| 180 | 13.523723 | 0.5280 | 0.4540 |
| 220 | 12.604636 | 0.5380 | 0.5800 |
