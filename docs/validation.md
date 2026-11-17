# Validation record

## Analytical checks

- Full source: 3,150 rows; no missing CSV cells; archive SHA-256 recorded in metrics and model card.
- 2,785 unique nine-feature profiles; no profile-group overlap among training, validation and test.
- Split membership can be regenerated exactly from source features and seed.
- Model inputs omit target, status, customer value and age fields.
- Threshold selection can be regenerated from saved validation scores and matches the F2-maximizing grid choice.
- Test metrics and confusion-matrix counts reconcile to the saved test scores and labels.
- sklearn predictions, portable Python reconstruction and JavaScript inference agree for all 631 test records; JavaScript maximum absolute difference is approximately 1.11e-16.
- Threshold recall is monotonic, class counts conserve the population, and invalid inference inputs are rejected.
- Rounded training reference inputs are valid integers and produce zero sensitivity when used as their own replacement reference.
- Eight Python unit/integration checks pass, plus the JavaScript inference suite.

## Browser checks

The app was inspected in a real browser at desktop and 390 × 844 phone width. The low/high held-out examples, calculation action and arrow-key threshold changes were exercised. Moving the threshold from 10% to 11% changed flagged records from 151 to 149 while preserving the 92 true positives, as expected from test scores. No browser console errors or warnings were observed during these checks.

The UI includes native form validation, input range limits, clear last-calculation status, focus styles, a skip link, labeled controls, reduced-motion support and status announcements. These measures do not replace a full screen-reader/assistive-technology audit.

The final scoring workflow was rechecked after the input-label and update-status refinements. A desktop screenshot is included at `reports/app-desktop.jpg`. The evaluation figure was generated and visually inspected for readable labels, curve interpretation and calibration scales.

Native browser download handling may depend on host settings. The CSV generation path is implemented and its threshold filtering is covered by inference/metric checks; no external system or customer account is modified.

## Refinement checks

The additional app-level JavaScript suite checks blank-input preservation across routes, exports of the last calculated inputs rather than unsaved edits, the reference-sensitivity empty state, chart-table alternatives, model metadata export and threshold `aria-valuetext`. Blank preservation and reference/high-score behavior were also exercised in the browser. The predictive model and its reported test scores were not retrained or tuned during this polish pass.

## Reproduce

```text
python scripts/train.py
python -m unittest discover -s tests -v
node tests/inference.test.cjs
node --check dist/app.js
python scripts/serve.py
```

Python tests expect a completed training run because the raw archive and joblib model are intentionally not included in source control. Never load a joblib artifact from an untrusted source.
