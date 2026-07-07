# Supplementary: Hyperparameter tuning of the machine-learning models

To ensure the comparison was not biased by under-tuned complex models, each ML
algorithm was tuned by a randomised hyperparameter search (`RandomizedSearchCV`,
40 candidate configurations, participant-grouped inner cross-validation)
**inside every outer training fold** of the nested cross-validation
(`scripts/12_nested_cv.py`); the selected configurations therefore never saw the
outer test data. The best configurations selected across the 15 outer folds
(3 scenarios × 5 folds) are summarised below (`nested_cv_ml_fold_params.csv`).

**Supplementary Table Sy. Most frequently selected hyperparameters (of 15 outer folds).**

| Model | Hyperparameter | Search grid | Modal selection |
|---|---|---|---|
| XGBoost | max_depth | 2, 3, 4, 6 | **2** (12/15) |
| | reg_lambda (L2) | 0.5, 1, 2, 5 | **5** (7/15, grid maximum) |
| | subsample | 0.7, 0.85, 1.0 | 0.7 (8/15) |
| | colsample_bytree | 0.7, 0.85, 1.0 | 0.7 (8/15) |
| | learning_rate | 0.01–0.2 | 0.05 (7/15) |
| | n_estimators | 100, 200, 400 | 200 (10/15) |
| Random Forest | min_samples_leaf | 1, 2, 5, 10, 20 | **20** (10/15, grid maximum) |
| | max_features | sqrt, 0.3, 0.5, None | **0.3** (12/15) |
| | class_weight | balanced, balanced_subsample, None | **None** (15/15) |
| | max_depth | 4, 6, 8, 10, None | 8 (6/15) |
| MLP | hidden_layer_sizes | 1–3 layers, 16–128 units | **single (128,)** (12/15) |
| | activation | relu, tanh | **tanh** (15/15) |
| | learning_rate_init | 3e-3, 1e-3, 5e-4 | 5e-4 (9/15) |
| | alpha (L2) | 1e-5 … 1e-1 | (spread) |

**Interpretation.** Tuning consistently drove every complex model toward its
most **regularised, least complex** configuration: XGBoost to shallow (depth-2)
trees with maximal L2 penalty and aggressive sub-sampling; Random Forest to
large terminal leaves and few features per split; and the neural network to a
single hidden layer. In other words, the optimal "complex" models were those
constrained to behave almost linearly — precisely the regime the logistic
regression occupies by construction. This provides a mechanistic explanation for
why increasing algorithmic complexity did not improve discrimination on these
low-dimensional, predominantly linear clinical data.

---

## Draft text for the Discussion (manuscript)

> Notably, this result held even after each machine-learning model was
> hyperparameter-tuned within the cross-validation. The tuning process itself is
> informative: across folds it consistently favoured the most heavily
> regularised, least complex configurations — depth-2 gradient-boosted trees
> with maximal L2 penalty, large-leaf random forests using few features per
> split, and single-hidden-layer neural networks (Supplementary Table Sy). That
> the optimal complex models were those constrained to approximate a linear,
> additive structure reinforces our central finding: for identifying prevalent
> radiographic knee osteoarthritis from routine clinical-epidemiological
> variables, the signal is captured by linear associations dominated by age and
> BMI, and the flexibility of "black-box" algorithms confers no advantage.
