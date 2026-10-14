# Model Selection

Every candidate was fitted on the **train** games and scored on the **validation** games.
The model with the lowest validation log-loss was selected and refitted on train + validation.
`predict_48_ms` is the time to score 48 positions, roughly one full AI turn.

| algorithm | params | val log-loss | val accuracy | val ROC-AUC | fit s | predict_48 ms |
|---|---|---|---|---|---|---|
| hist_gradient_boosting | `{"learning_rate": 0.05, "max_leaf_nodes": 63, "max_iter": 400, "l2_regularization": 1.0}` | 0.4449 | 0.7679 | 0.8679 | 18.3 | 4.6 |
| hist_gradient_boosting | `{"learning_rate": 0.1, "max_leaf_nodes": 31, "max_iter": 400, "l2_regularization": 1.0}` | 0.4464 | 0.7680 | 0.8671 | 12.7 | 6.0 |
| random_forest | `{"n_estimators": 150, "max_depth": 20, "min_samples_leaf": 20}` | 0.4560 | 0.7644 | 0.8628 | 11.2 | 29.8 |
| random_forest | `{"n_estimators": 150, "max_depth": 12, "min_samples_leaf": 20}` | 0.4586 | 0.7650 | 0.8625 | 8.6 | 28.2 |
| logistic_regression | `{"C": 0.1}` | 0.4695 | 0.7565 | 0.8529 | 1.0 | 1.1 |
| logistic_regression | `{"C": 1.0}` | 0.4695 | 0.7565 | 0.8529 | 1.0 | 0.7 |
| mlp | `{"hidden_layer_sizes": [128, 64], "alpha": 0.0001}` | 0.6749 | 0.7349 | 0.8217 | 75.3 | 0.7 |
| mlp | `{"hidden_layer_sizes": [256, 128], "alpha": 0.001}` | 1.1540 | 0.7240 | 0.8032 | 153.4 | 1.0 |

**Selected:** `hist_gradient_boosting` with `{"learning_rate": 0.05, "max_leaf_nodes": 63, "max_iter": 400, "l2_regularization": 1.0}`. Model version `v20261014-082937`.
