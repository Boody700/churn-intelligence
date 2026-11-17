# Churn Intelligence: make prediction useful and evaluation honest

## Problem and scope

Retention teams have limited review and contact capacity. A churn model can rank accounts, but a score alone hides practical questions: how many churners are missed, how many alerts are false, and whether the evaluation reflects genuinely unseen profiles.

This historical portfolio project uses the UCI Iranian Churn dataset to demonstrate a complete modeling and delivery workflow. It is not a client engagement or an operational customer-retention system.

## Data understanding and feature decisions

The source contains 3,150 rows. The source card describes features aggregated over nine months and churn status at the end of month 12, with a three-month planning gap. I retained all records because there is no source customer ID that would establish whether repeated behavior profiles represent duplicate records or different people.

There are 300 exactly repeated full rows and 2,785 distinct profiles across the nine selected inputs. A naive row split could place identical behavior vectors into both train and test. Instead, I grouped identical selected-feature profiles and kept each group intact throughout evaluation.

The model uses call failures, complaint presence, subscription length, charge band, voice seconds, call count, SMS count, distinct called numbers and tariff plan. I omitted Status as a conservative account-state shortcut exclusion, Customer Value because its derivation is not fully documented, and both age fields. Excluding age does not demonstrate fairness; behavior may remain a proxy.

The source's charge-band description says 0–9, but seven observed rows contain 10. I retained the observed ordinal category and documented the discrepancy rather than inventing a monetary meaning or silently deleting those rows.

## Evaluation design

Stratified grouped folds assign approximately 60% train, 20% validation and 20% test: 1,892, 627 and 631 rows respectively. Two scaled logistic-regression variants and two random-forest variants are compared through three-fold grouped cross-validation inside the training partition. Model choice is based on mean average precision, appropriate for examining positive-class ranking in an imbalanced problem.

The selected depth-10 random forest is sigmoid-calibrated through grouped training folds. The validation partition chooses an alert threshold that maximizes F2, reflecting a stated preference for recall. This is an evaluation assumption, not an estimated business utility function. The untouched test partition is then scored once for the reported results.

## Results and their meaning

On 631 held-out test rows, including 99 churners, the model achieves average precision 0.813, ROC AUC 0.960 and Brier score 0.061. The constant training-prevalence baseline yields test AP 0.157, AUC 0.500 and Brier 0.132.

At the validation-selected 0.10 threshold, the model flags 151 records: 92 true positives and 59 false positives. It misses 7 churners, resulting in 92.9% recall and 60.9% precision. Reporting these counts makes the trade-off concrete; saying only “high accuracy” would hide it.

Four hundred bootstrap resamples of whole test profile groups give a 95% AP interval of 0.720–0.881 and ROC AUC interval of 0.943–0.975. They describe sampling variability conditional on this model and partition, not model-selection uncertainty or future population shift.

## Product and engineering

The application contains an evaluation overview, a live profile workbench, an interactive threshold lab and a model card. It exports the actual calibrated ensemble into portable JSON. The JavaScript engine follows each decision tree and applies the learned sigmoid calibration, averaging fold-level probabilities.

All 631 test predictions are checked against Python to machine precision. The app performs inference locally and requires no external model API. The workbench shows sensitivity to replacing one input with its rounded training median. Those differences are explicitly not causal effects, SHAP values or additive decompositions.

## Limits and next steps

No customer IDs or dates are available for a longitudinal test. Related but nonidentical profiles can still cross partitions. The model has not been tested prospectively or in a current telecom population. Calibration may change after deployment. A risk score does not identify customers who will respond to a retention offer.

The next step would be a prospective cohort test with a prespecified metric and operating policy. Any retention intervention should then be assessed with a randomized holdout and cost/margin data, not inferred from historical predictive performance.

## Skills demonstrated

Data auditing, leakage-aware grouped evaluation, sklearn pipelines, baseline comparison, model calibration, threshold selection, confidence intervals, explainability limits, portable inference, interactive product design and reproducible testing.
