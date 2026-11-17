# Model card

**Model:** Group-calibrated random forest for UCI Iranian Churn.

**Purpose:** Educational churn-risk estimation and demonstration of evaluation/decision trade-offs.

## Training data and target

UCI Iranian Churn (2020), https://doi.org/10.24432/C5JW3Z, CC BY 4.0. 3,150 rows; no missing cells in the provided CSV. Feature names are normalized by collapsing repeated spaces. The target is `Churn`: 1 indicates churn and 0 non-churn. The source describes month-1–9 feature aggregates and churn status at month 12. No additional temporal or population assumptions are imposed.

Nine inputs are used. Status, Customer Value, Age and Age Group are excluded. Charge Amount is treated as an ordinal band, not currency. Seven records have band 10 despite the source card's 0–9 description. All records are retained; no synthetic oversampling is used.

## Algorithm and selection

Candidate logistic regressions use StandardScaler fitted within each training fold and C values 0.1 or 1. Candidate random forests use 120 trees, depth 6/minimum leaf 6 or depth 10/minimum leaf 5. Random seed is 42. Candidates are compared with three-fold stratified grouped training CV, scored by average precision.

The selected depth-10 forest is wrapped in three-fold sigmoid calibration with an ensemble of calibrated fold estimators. Training contains 1,892 rows / 1,668 unique selected-feature groups; validation 627 / 563; test 631 / 554.

## Threshold and reported performance

The default 0.10 threshold maximizes validation F2 on a 0.05–0.95 grid in 0.01 increments. Ties favor precision and then the higher threshold. All preprocessing, model selection and calibration occur within training; threshold selection uses validation.

Held-out test: AP 0.8132, ROC AUC 0.9603, Brier 0.0614, precision 0.6093, recall 0.9293, F2 0.8410. Confusion matrix: TP 92, FP 59, FN 7, TN 473. Performance is row-weighted; repeated profiles are retained but isolated across partitions.

Bootstrap intervals resample whole test profile groups 400 times with a fixed seed. They are conditional on the fitted model and split. They do not quantify all sources of uncertainty.

## Explanations

Validation permutation importance measures the average decrease in AP over eight independent shuffles per feature. Correlated features may share importance.

Workbench sensitivity is the current prediction minus the prediction obtained after replacing one input with its rounded training median. Changes are evaluated one at a time. They do not sum to the model score, do not establish causal relationships, and can produce implausible combinations of correlated inputs.

## Portable inference

The app carries the learned tree parameters, positive-class leaf probabilities and sigmoid coefficients for each calibrated fold estimator. Forest traversal casts feature values to float32, matching sklearn tree prediction behavior. Predictions are averaged across calibrated folds.

The server serves static assets only. The app does not send entered profile values to an inference API. Its portable parameters and historical test records are downloadable assets from a public-source-data project, not a confidential production model.

## Not evaluated / not claimed

- Future telecom populations, temporal generalization or drift robustness.
- Fairness across demographic groups; removing age is not a fairness guarantee.
- Campaign response, retention uplift, causal impact, profit or customer lifetime value.
- Guaranteed probability calibration after deployment.
- Person-level identity isolation beyond identical selected-feature grouping, since no source identifiers are available.

Operational use would require prospective validation, explicit ownership, drift/calibration monitoring and an intervention evaluation. This app performs none of those account actions.
