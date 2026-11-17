# Explain this project confidently

## A short introduction

“I built a telecom churn prediction project using UCI data. I noticed repeated behavior profiles, so I grouped identical model inputs before splitting to prevent exact-profile leakage. I compared logistic regression and random forests through grouped training cross-validation, calibrated the selected model, and chose a threshold using validation data. On the separate 631-row test set, it found 92 of 99 churners with 59 false alerts. I then exported the actual ensemble into a browser inference engine and built a workbench and threshold lab so the trade-offs are visible.”

Use this after studying the implementation. Be transparent about AI assistance and explain what you personally reviewed, tested and changed.

## Questions worth preparing

**Why not split rows randomly?** Identical selected-feature profiles could occur in both train and test. Grouping preserves exact-profile isolation while retaining records that might represent different customers.

**Why average precision?** The positive class is around 16%. AP evaluates ranking across recall levels and can be compared with the prevalence baseline. It is not accuracy or simply a trapezoidal area under the PR curve.

**Why calibrate?** Ranking quality and probability quality are different. Sigmoid calibration uses training-held-out folds, and a separate test calibration curve and Brier score check the resulting probabilities. Calibration is not assumed to transfer to a new population.

**Why is the threshold only 0.10?** It was selected by validation F2, which emphasizes recall. At that operating point, 151 test examples are flagged, including 59 false positives. The threshold is not a universal production recommendation.

**Why not use Status?** Even though the source describes features as prior to the outcome, account state can be a strong shortcut with ambiguous operational meaning. I excluded it conservatively and documented the choice rather than claiming it was proven leakage.

**Why omit Customer Value and age?** Customer Value's calculation is not fully documented in the source card. Age fields are omitted from prediction; that alone does not establish fairness because behavioral proxies may remain.

**What do local explanations mean?** They show model-score sensitivity when one input is replaced with a rounded training reference. They are not causal effects or additive attributions.

**How does inference work without an API?** The app traverses the exported decision trees, averages their leaf probabilities, applies each fold's sigmoid calibration and then averages calibrated scores. All held-out predictions are compared with sklearn outputs.

**How would you improve it?** Collect identifiers and dates for a longitudinal split; validate on a fresh population; monitor calibration and feature distributions; and test an intervention with a randomized holdout before claiming retention value.

## Suggested demo

1. Model overview: explain AP versus accuracy and the exact alert counts.
2. Show the grouped split and candidate-model comparison.
3. Load a low-score and high-score example in the workbench; explain sensitivity limits.
4. Move the threshold and explain the changing confusion matrix.
5. Open the model card and explain one exclusion and one deployment limitation.
6. Show the train script and the Python/JavaScript prediction parity check.

## Résumé wording

After studying and adapting the work, a defensible description is:

“Built a calibrated telecom churn model and browser inference application using Python, scikit-learn and JavaScript; implemented group-isolated train/validation/test evaluation and achieved 0.813 test average precision, with explicit threshold and false-alert analysis.”

Do not claim 96% accuracy, a retention improvement or a real-client business result. ROC AUC is not accuracy, and no intervention was measured.
