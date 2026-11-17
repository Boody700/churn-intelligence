"use strict";
const D = window.CHURN_DATA, E = window.ChurnEngine, content = document.querySelector("#content");
if (!D || !E) {
  content.innerHTML = '<div class="error-block" role="alert">The model assets could not load. Refresh the page, or rebuild with scripts/train.py.</div>';
} else {
  let chart = function(points, type) {
    const W = 560, H = 270, L = 43, R = 15, T = 16, B = 34, x = (v) => L + v * (W - L - R), y = (v) => H - B - v * (H - T - B);
    let out = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-labelledby="${type}-title ${type}-desc"><title id="${type}-title">${type === "pr" ? "Held-out precision\u2013recall curve" : "Calibration on held-out test examples"}</title><desc id="${type}-desc">${type === "pr" ? "Average precision " + dec(M.test.average_precision) + ". Test prevalence " + pct(M.splits.test.prevalence) : "Mean predicted risk versus observed churn, grouped into quantile bins. The diagonal represents perfect calibration."}</desc>`;
    for (let i = 0; i <= 4; i++) {
      const v = i / 4;
      out += `<line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}" stroke="#eee9f4"/><text x="${L - 9}" y="${y(v) + 4}" text-anchor="end">${pct(v)}</text><text x="${x(v)}" y="${H - 12}" text-anchor="middle">${pct(v)}</text>`;
    }
    if (type === "pr") out += `<line x1="${L}" x2="${W - R}" y1="${y(M.splits.test.prevalence)}" y2="${y(M.splits.test.prevalence)}" stroke="#c0b6ce" stroke-dasharray="5 5"/>`;
    else out += `<line x1="${x(0)}" y1="${y(0)}" x2="${x(1)}" y2="${y(1)}" stroke="#c0b6ce" stroke-dasharray="5 5"/>`;
    out += `<path d="${points.map(([a, b], i) => `${i ? "L" : "M"}${x(a).toFixed(2)},${y(b).toFixed(2)}`).join(" ")}" fill="none" stroke="#8d72c7" stroke-width="2.5"/>`;
    if (type !== "pr") out += points.map(([a, b]) => `<circle cx="${x(a)}" cy="${y(b)}" r="4" fill="#8d72c7"><title>Predicted ${pct(a)}; observed ${pct(b)}</title></circle>`).join("");
    return out + `</svg><details><summary>Read the chart data</summary><div class="table-scroll chart-data"><table><thead><tr><th>${type === "pr" ? "RECALL" : "MEAN PREDICTED RISK"}</th><th class="num">${type === "pr" ? "PRECISION" : "OBSERVED CHURN"}</th></tr></thead><tbody>${points.map(([a, b]) => `<tr><td>${pct(a)}</td><td class="num">${pct(b)}</td></tr>`).join("")}</tbody></table></div></details>`;
  }, overview = function() {
    const top = D.testRows.slice(0, Math.ceil(D.testRows.length * 0.2)), topRate = top.reduce((a, r) => a + r.actual, 0) / top.length, lift = topRate / M.splits.test.prevalence;
    return `<div class="metrics">${metric("Average precision", dec(M.test.average_precision), `95% grouped-bootstrap interval: ${dec(M.intervals.average_precision[0], 2)}\u2013${dec(M.intervals.average_precision[1], 2)}`)}${metric("Churn recall", pct(M.test.recall), "Known churners identified at the validation-selected threshold")}${metric("Alert precision", pct(M.test.precision), "Flagged test examples that actually churned")}${metric("Held-out examples", num(M.splits.test.rows), `${num(M.splits.test.churners)} churners \xB7 ${num(M.splits.test.groups)} isolated input profiles`)}</div>
    <div class="grid"><section class="panel">${heading("Finding churners without hiding false alarms.", "Precision\u2013recall across thresholds on the held-out test set.", "TEST PARTITION")}<div class="legend"><span>Calibrated model</span><span>Test prevalence ${pct(M.splits.test.prevalence)}</span></div>${chart(D.curves.pr, "pr")}<p class="chart-note">Horizontal axis: recall. Vertical axis: precision. Average precision summarizes this curve; it is not classification accuracy.</p></section><section class="panel">${heading("A measured starting point.", "A useful score needs an honest evaluation.")}<div class="steps"><div class="step"><span>TRAIN</span><strong>${num(M.splits.train.rows)}</strong><small>Model selection<br>+ calibration</small></div><div class="step"><span>VALIDATE</span><strong>${num(M.splits.validation.rows)}</strong><small>Choose alert<br>threshold</small></div><div class="step"><span>TEST</span><strong>${num(M.splits.test.rows)}</strong><small>Report final<br>performance</small></div></div><p>Identical nine-feature profiles stay in one partition, including repeated records. The split is approximately 60 / 20 / 20 and stratified by outcome.</p><div class="importance-row"><div class="importance-top"><span>Risk concentration in top 20%</span><strong>${dec(lift, 1)}\xD7 lift</strong></div><div class="importance-bar"><span style="width:${topRate * 100}%"></span></div></div><p class="fine-print">${top.reduce((a, r) => a + r.actual, 0)} churners among the ${top.length} highest-scored test examples. Lift compares their observed churn rate with overall test prevalence.</p></section></div>
    <aside class="callout"><span class="icon" aria-hidden="true">\u21C4</span><div><h3>The threshold is a decision, not a magic number.</h3><p>At the validation-selected ${pct(M.threshold)} cutoff, the model flags ${M.test.alerts} test examples, identifies ${M.test.tp} churners and produces ${M.test.fp} false alerts. Explore how the balance changes.</p></div><a href="#threshold">Open threshold lab \u2197</a></aside>
    <div class="two-grid"><section class="panel">${heading("A simple baseline earns its place.", "Model selection uses grouped cross-validation within training.")}<div class="table-scroll"><table><thead><tr><th>MODEL</th><th class="num">TRAIN CV AP</th><th class="num">FOLD SD</th></tr></thead><tbody>${D.comparison.map((c) => `<tr class="${c.name === M.selected_model ? "winner" : ""}"><td class="model-name">${esc(c.name)}${c.name === M.selected_model ? '<span class="badge">SELECTED</span>' : ""}</td><td class="num">${dec(c.cv_ap)}</td><td class="num">${dec(c.cv_std)}</td></tr>`).join("")}</tbody></table></div><p class="chart-note">The selected model is then sigmoid-calibrated on grouped training folds. Test AP is ${dec(M.test.average_precision)}; a constant-risk baseline has AP ${dec(M.baseline.average_precision)}.</p></section><section class="panel">${heading("Which inputs matter to the model?", "Permutation importance on validation data.", "AP DECREASE")}${D.fields.slice().sort((a, b) => b.importance - a.importance).slice(0, 5).map((f) => `<div class="importance-row"><div class="importance-top"><span>${esc(f.label)}</span><strong>${dec(f.importance)}</strong></div><div class="importance-bar"><span style="width:${Math.max(0, f.importance) / Math.max(...D.fields.map((f2) => f2.importance), 1e-3) * 100}%"></span></div></div>`).join("")}<p class="fine-print">Mean AP decrease over 8 shuffles. Correlated inputs can share importance. This is model reliance, not a causal explanation.</p></section></div>`;
  }, scoreResult = function() {
    const risk = E.predict(D.model, state.scored), flag = risk >= M.threshold, percentile = D.testRows.filter((r) => r.score <= risk).length / D.testRows.length;
    return `${heading("Estimated churn risk", "Calibrated model output.")}<div class="risk-ring"><svg viewBox="0 0 200 200" aria-hidden="true"><circle cx="100" cy="100" r="84" fill="none" stroke="#493e62" stroke-width="9"/><circle cx="100" cy="100" r="84" fill="none" stroke="#c8acf0" stroke-width="9" stroke-linecap="round" stroke-dasharray="${risk * 527.79} 527.79"/></svg><div class="inside"><span class="risk-number">${pct(risk)}</span><span>estimated risk</span></div></div><div><span class="risk-pill">${flag ? "Above" : "Below"} the ${pct(M.threshold)} evaluation threshold</span></div><p>This profile ${flag ? "would" : "would not"} be flagged at the validation-selected cutoff. A score is an estimate, not certainty about a person\u2019s behavior.</p><div class="risk-context">At or above ${pct(percentile)} of held-out test scores.<br><span>A dataset comparison, not an extra probability.</span></div><div class="risk-foot">Runs entirely in your browser.<br>Inputs are not sent to an inference API.</div>`;
  }, sensitivities = function() {
    const rows = E.sensitivity(D.model, state.scored, D.fields).filter((r) => Math.abs(r.delta) >= 5e-4).slice(0, 5), max = Math.max(...rows.map((r) => Math.abs(r.delta)), 0.01);
    return `${heading("What changes this score?", "One input at a time, compared with its rounded training median.", "SENSITIVITY CHECK")}${rows.map((r) => `<div class="sensitivity-row"><div>${esc(r.label)}<div class="reference">Reference: ${num(r.reference)}</div></div><div class="track"><span style="width:${Math.abs(r.delta) / max * 100}%;background:${r.delta < 0 ? "#94b8a6" : "#a293ca"}"></span></div><span class="delta">${r.delta > 0 ? "+" : ""}${(r.delta * 100).toFixed(1)} pp</span></div>`).join("") || '<p class="reference-empty">No input has a score difference of at least 0.05 percentage points relative to its reference value.</p>'}<p class="fine-print">Positive values mean this input raises the model score relative to replacing it with the rounded training median. These effects do not add up to the prediction and do not estimate the effect of an intervention. Correlated inputs and unrealistic combinations can distort this diagnostic.</p>`;
  }, workbench = function() {
    return `${notice("Enter nine-month behavior aggregates. Predictions refer to the source dataset\u2019s three-month planning gap. This historical model has not been validated on a current telecom population.")}<div class="toolbar"><p>Start from a held-out example, or explore a hypothetical profile.</p><div class="button-group"><button class="button" id="low-example">Low-score example</button><button class="button" id="high-example">High-score example</button><button class="button" id="reset-profile">Training medians \u21BA</button><button class="button" id="export-profile">\u2193 Export calculation</button></div></div><div class="workbench-grid"><section class="panel">${heading("Build a behavior profile.", "All fields are required. Ranges come from the training partition.")}<form id="risk-form"><div class="form-grid">${D.fields.map((f, i) => `<label class="input-group" for="feature-${i}"><span class="input-label">${esc(f.label)}</span>${i === 1 || i === 8 ? `<select id="feature-${i}" aria-label="${esc(f.label)}" aria-describedby="help-${i}" required>${(i === 1 ? [[0, "No complaint"], [1, "Complaint recorded"]] : [[1, "Pay as you go"], [2, "Contractual"]]).map(([v, l]) => `<option value="${v}" ${state.x[i] === v ? "selected" : ""}>${l}</option>`).join("")}</select>` : `<input type="number" aria-label="${esc(f.label)}" aria-describedby="help-${i}" id="feature-${i}" value="${Number.isFinite(state.x[i]) ? state.x[i] : ""}" min="${f.min}" max="${f.max}" step="1" required inputmode="numeric">`}<small id="help-${i}">${esc(f.unit)}<br>Training range: ${num(f.min)}\u2013${num(f.max)}</small></label>`).join("")}</div><div class="form-footer"><span>Behavior only. Age, customer value<br>and account status are excluded.</span><button class="button primary" type="submit">Calculate risk \u2197</button></div><p id="profile-status" class="fine-print" style="margin-top:14px" role="status">Integer references use rounded training medians.</p><p id="form-error" class="error" role="alert"></p></form></section><section class="panel risk-panel" id="risk-output" aria-live="polite">${scoreResult()}</section></div><section class="panel section-space" id="sensitivities">${sensitivities()}</section>`;
  }, thresholdResults = function() {
    const m = E.metrics(D.testRows, state.threshold), flagged = D.testRows.filter((r) => r.score >= state.threshold);
    return `<div class="metrics">${metric("Flagged for review", num(m.alerts), `${pct(m.alerts / D.testRows.length)} of held-out test examples`)}${metric("Alert precision", pct(m.precision), `${m.tp} true alerts out of ${m.alerts} total alerts`)}${metric("Churn recall", pct(m.recall), `${m.tp} of ${M.splits.test.churners} known churners identified`)}${metric("Missed churners", num(m.fn), "Actual churners below the current cutoff")}</div><div class="two-grid"><section class="panel">${heading("Every alert has a trade-off.", "Confusion matrix at the current cutoff.", "TEST PARTITION")}<div class="matrix"><div class="matrix-cell positive"><span class="matrix-label">TRUE POSITIVES</span><div class="matrix-value">${m.tp}</div><p>Flagged and actually churned</p></div><div class="matrix-cell warn"><span class="matrix-label">FALSE POSITIVES</span><div class="matrix-value">${m.fp}</div><p>Flagged but did not churn</p></div><div class="matrix-cell warn"><span class="matrix-label">FALSE NEGATIVES</span><div class="matrix-value">${m.fn}</div><p>Not flagged, but actually churned</p></div><div class="matrix-cell"><span class="matrix-label">TRUE NEGATIVES</span><div class="matrix-value">${m.tn}</div><p>Not flagged and did not churn</p></div></div></section><section class="panel prose">${heading("Choose for a real decision.", "A threshold should reflect costs and capacity.")}<p>Lower cutoffs generally identify more churners but create more false alerts. Higher cutoffs narrow the review queue and can miss more churners.</p><p>The default <strong>${pct(M.threshold)}</strong> cutoff maximizes F2 on validation data, which weights recall more strongly than precision. F2 is a modeling assumption, not a measured business objective.</p><p>This slider displays <strong>held-out test outcomes</strong> for learning. Do not use it to retune the model and then claim the same test results are independent.</p><p>Before deployment, define contact capacity, costs, expected incremental benefit, and a fresh evaluation plan. No customer outreach or revenue uplift was measured.</p></section></div><section class="panel section-space">${heading("A ranked review queue.", "Highest-scored flagged test examples. These are historical dataset records.", `${flagged.length} FLAGGED`)}<div class="table-scroll"><table><thead><tr><th>EXAMPLE</th><th class="num">MODEL RISK</th><th>OBSERVED OUTCOME</th><th class="num">COMPLAINT</th><th class="num">CALL FAILURES</th></tr></thead><tbody>${flagged.slice(0, 12).map((r) => `<tr><td class="model-name">${r.id}</td><td class="num"><div class="queue-risk"><span class="risk-track"><span style="width:${r.score * 100}%"></span></span>${pct(r.score)}</div></td><td>${r.actual ? "Churned" : "Did not churn"}</td><td class="num">${r.x[1] ? "Yes" : "No"}</td><td class="num">${num(r.x[0])}</td></tr>`).join("") || '<tr><td colspan="5">No test examples meet this cutoff. Lower the threshold to populate the queue.</td></tr>'}</tbody></table></div><p class="chart-note">Showing up to 12 records; export includes every flagged test example. Example IDs are generated row labels, not customer identifiers.</p></section>`;
  }, threshold = function() {
    return `${notice("The default cutoff was selected on validation data before test scoring. The results below are an educational test-set exploration, not a new threshold-selection exercise.")}<section class="panel section-space"><div class="threshold-control"><label for="threshold">Flag examples with estimated risk of at least</label><input id="threshold" type="range" min="5" max="95" step="1" value="${Math.round(state.threshold * 100)}" aria-describedby="threshold-help"><output for="threshold" class="threshold-value" id="threshold-value">${pct(state.threshold)}</output></div><div class="toolbar" style="margin-bottom:0"><p id="threshold-help">Adjust with the slider or arrow keys. All test examples stay fixed.</p><div class="button-group"><button class="button" id="reset-threshold">Validation default \u21BA</button><button class="button" id="export-queue">\u2193 Export flagged examples</button></div></div></section><div id="threshold-results">${thresholdResults()}</div>`;
  }, modelCard = function() {
    return `<div class="toolbar"><p>Evaluation metadata, source checksum and model settings in one file.</p><button class="button" id="export-model-card">\u2193 Download model metadata</button></div><div class="two-grid"><section class="panel prose">${heading("01 / Intended use", "An educational churn prediction case study.")}<p>Estimate churn risk from nine months of historical telecom behavior, demonstrate model evaluation, and explore the operational trade-off between recall and false alerts.</p><p><strong>Source:</strong> <a href="https://doi.org/10.24432/C5JW3Z" target="_blank" rel="noopener">UCI Iranian Churn (2020)</a>, CC BY 4.0. The source describes 3,150 telecom records, nine-month feature aggregates and churn status at the end of month 12.</p><p>No current-population or production validation has been performed. No account actions, financial decisions or customer outreach are carried out by this application.</p><p><strong>Selected model:</strong> ${esc(M.selected_model)}. A three-fold sigmoid-calibrated ensemble is fitted only on training data. Model choice uses mean training CV average precision.</p></section><section class="panel prose">${heading("02 / Split integrity", "Prevent identical profiles leaking across partitions.")}<p>There are ${num(M.source_rows)} rows, ${num(M.unique_profiles)} distinct nine-feature profiles and ${num(M.exact_duplicate_rows)} exactly repeated full rows. With no customer ID, repeated profiles might be different people, so all rows are retained.</p><p>All identical selected-feature profiles share a group. Stratified grouped folds allocate entire groups to approximately 60% train, 20% validation and 20% test.</p><p>Preprocessing, model selection and sigmoid calibration stay within training folds. The validation set selects a threshold by F2. The test partition is scored afterwards.</p><p>Without customer identifiers or collection dates, this is not a longitudinal or temporal generalization test. Similar but nonidentical profiles may remain related.</p></section></div><div class="two-grid"><section class="panel">${heading("03 / Probability quality", "Calibration is measured, not assumed.", "HELD-OUT TEST")}${chart(D.curves.calibration, "calibration")}<p class="chart-note">Horizontal: mean predicted risk. Vertical: observed churn fraction. Quantile bins have limited sample sizes; this is a diagnostic, not a guarantee of calibrated probabilities in another population.</p><div class="table-scroll"><table><thead><tr><th>METRIC</th><th class="num">MODEL</th><th class="num">CONSTANT BASELINE</th></tr></thead><tbody><tr><td>Average precision \u2191</td><td class="num">${dec(M.test.average_precision)}</td><td class="num">${dec(M.baseline.average_precision)}</td></tr><tr><td>ROC AUC \u2191</td><td class="num">${dec(M.test.roc_auc)}</td><td class="num">${dec(M.baseline.roc_auc)}</td></tr><tr><td>Brier score \u2193</td><td class="num">${dec(M.test.brier)}</td><td class="num">${dec(M.baseline.brier)}</td></tr></tbody></table></div></section><section class="panel prose">${heading("04 / Feature policy", "Conservative choices, explicitly documented.")}<p><strong>Included:</strong> call failures, complaints, subscription length, charge band, voice seconds, call count, SMS count, distinct called numbers and tariff plan.</p><p><strong>Status is excluded:</strong> an ambiguous account-state shortcut is not needed for this case study, even though the source describes feature collection before the outcome.</p><p><strong>Customer Value is excluded:</strong> the calculation is not fully documented in the source card. Age and Age Group are also excluded from prediction.</p><p>Excluding age does not demonstrate fairness; behavior can act as a proxy. Subgroup and prospective evaluation are still needed before operational use.</p><p>The source has no missing cells. Inputs are numeric and the complaint/plan categories are restricted by the interface. Browser inference uses the exact exported calibrated ensemble.</p></section></div><div class="two-grid"><section class="panel prose">${heading("05 / Uncertainty & limitations", "Keep the claim proportional to the evidence.")}<p>Test ROC AUC: <strong>${dec(M.test.roc_auc)}</strong>, with a 95% grouped-bootstrap interval of ${dec(M.intervals.roc_auc[0])}\u2013${dec(M.intervals.roc_auc[1])}. Test AP: <strong>${dec(M.test.average_precision)}</strong>, interval ${dec(M.intervals.average_precision[0])}\u2013${dec(M.intervals.average_precision[1])}.</p><p>Intervals use 400 resamples of test profile groups. They describe sampling variability conditional on this trained model and split; they do not capture model-selection uncertainty or population shift.</p><p>Scores rank observed churn risk. They do not measure responsiveness to a campaign, profit, retention uplift, or causal influence of a behavior.</p><p>A prospective test on new data, calibration monitoring and an intervention experiment are appropriate next steps.</p></section><section class="panel prose">${heading("06 / Reproduction & verification", "A small app backed by a complete ML workflow.")}<p>Run <code>python scripts/train.py</code> to download, split, train, calibrate, evaluate and regenerate the portable model, test scores and figures. Run <code>python -m unittest discover -s tests -v</code> and <code>node tests/inference.test.cjs</code> to validate.</p><p>The exported inference format is checked against scikit-learn predictions. Python reconstruction maximum absolute error: <strong>${M.browser_parity_max_error.toExponential(2)}</strong>.</p><p>Case study, split membership, metric JSON, training code and an interview guide are included in the source project. No external inference API, account or paid service is required.</p><p><strong>Dataset archive SHA-256</strong></p><p style="overflow-wrap:anywhere;font-family:monospace;font-size:10px">${M.source_sha256}</p></section></div>`;
  }, render = function() {
    const focus = document.activeElement?.id;
    document.querySelectorAll("[data-page]").forEach((a) => {
      const active = a.dataset.page === state.page;
      a.classList.toggle("active", active);
      if (active) a.setAttribute("aria-current", "page");
      else a.removeAttribute("aria-current");
    });
    document.querySelector("#page-title").innerHTML = titles[state.page][0];
    document.querySelector("#subtitle").textContent = titles[state.page][1];
    document.title = { overview: "Model overview", workbench: "Risk workbench", threshold: "Threshold lab", "model-card": "Model card" }[state.page] + " \u2014 Churn Intelligence";
    content.innerHTML = { overview, workbench, threshold, "model-card": modelCard }[state.page]();
    bind();
    if (focus) document.getElementById(focus)?.focus({ preventScroll: true });
  }, readInputs = function() {
    return D.fields.map((_, i) => document.getElementById("feature-" + i).value.trim() === "" ? NaN : Number(document.getElementById("feature-" + i).value));
  }, validProfile = function(x) {
    return x.every((v, i) => Number.isFinite(v) && Number.isInteger(v) && v >= D.fields[i].min && v <= D.fields[i].max);
  }, example = function(high) {
    const eligible = D.testRows.filter((r) => validProfile(r.x));
    state.x = (high ? eligible[0] : eligible.at(-1)).x.slice();
    state.scored = state.x.slice();
    render();
    document.querySelector("#status").textContent = (high ? "High" : "Low") + "-score held-out example loaded.";
  }, updateThreshold = function() {
    document.querySelector("#threshold").setAttribute("aria-valuetext", pct(state.threshold) + " estimated-risk cutoff");
    document.querySelector("#threshold-value").textContent = pct(state.threshold);
    document.querySelector("#threshold-results").innerHTML = thresholdResults();
    const m = E.metrics(D.testRows, state.threshold);
    document.querySelector("#status").textContent = `Threshold ${pct(state.threshold)}. ${m.alerts} flagged, ${m.tp} true alerts, ${m.fn} missed churners.`;
  }, saveJSON = function(name, value) {
    const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2)], { type: "application/json" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = name;
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1e3);
    document.querySelector("#status").textContent = "Exported " + name;
  }, bind = function() {
    const on = (id, event, fn) => document.getElementById(id)?.addEventListener(event, fn);
    on("export-profile", "click", () => saveJSON("churn-calculation.json", { model: M.selected_model, source_sha256: M.source_sha256, inputs: Object.fromEntries(D.fields.map((f, i) => [f.name, state.scored[i]])), risk: E.predict(D.model, state.scored), evaluation_threshold: M.threshold, interpretation: "Historical dataset model estimate; not a production or causal claim." }));
    on("export-model-card", "click", () => saveJSON("churn-model-card.json", M));
    on("low-example", "click", () => example(false));
    on("high-example", "click", () => example(true));
    on("reset-profile", "click", () => {
      state.x = D.fields.map((f) => f.reference);
      state.scored = state.x.slice();
      render();
    });
    D.fields.forEach((_, i) => on("feature-" + i, "input", () => {
      state.x = readInputs();
      document.querySelector("#profile-status").textContent = "Inputs changed. Calculate risk to update the score.";
    }));
    on("risk-form", "submit", (event) => {
      event.preventDefault();
      const x = readInputs();
      if (!validProfile(x)) {
        document.querySelector("#form-error").textContent = "Enter whole numbers within each training range.";
        return;
      }
      state.x = x;
      state.scored = x.slice();
      document.querySelector("#profile-status").textContent = "Score updated for the displayed inputs.";
      document.querySelector("#form-error").textContent = "";
      document.querySelector("#risk-output").innerHTML = scoreResult();
      document.querySelector("#sensitivities").innerHTML = sensitivities();
      document.querySelector("#status").textContent = "Estimated risk " + pct(E.predict(D.model, x));
    });
    on("threshold", "input", (event) => {
      state.threshold = Number(event.target.value) / 100;
      updateThreshold();
    });
    on("reset-threshold", "click", () => {
      state.threshold = M.threshold;
      document.querySelector("#threshold").value = Math.round(M.threshold * 100);
      updateThreshold();
    });
    on("export-queue", "click", () => {
      const rows = [["example_id", "model_risk", "threshold", "observed_churn", ...D.fields.map((f) => f.name)], ...D.testRows.filter((r) => r.score >= state.threshold).map((r) => [r.id, r.score.toFixed(8), state.threshold, r.actual, ...r.x])];
      const csv = "\uFEFF" + rows.map((row) => row.map((v) => '"' + String(v).replace(/"/g, '""') + '"').join(",")).join("\r\n");
      const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
      const link = document.createElement("a");
      link.href = url;
      link.download = "churn-flagged-test-examples.csv";
      document.body.append(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1e3);
      const toast = document.querySelector("#toast");
      toast.textContent = "Flagged test examples exported.";
      toast.classList.add("show");
      setTimeout(() => toast.classList.remove("show"), 3e3);
    });
  }, route = function() {
    const page = location.hash.slice(1);
    state.page = Object.hasOwn(titles, page) ? page : "overview";
    render();
  };
  const M = D.meta;
  const state = { page: "overview", threshold: M.threshold, x: D.fields.map((f) => f.reference), scored: D.fields.map((f) => f.reference) };
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  const pct = (n) => new Intl.NumberFormat("en-GB", { style: "percent", maximumFractionDigits: 1 }).format(n);
  const num = (n) => new Intl.NumberFormat("en-GB", { maximumFractionDigits: 0 }).format(n);
  const dec = (n, d = 3) => Number(n).toFixed(d);
  const metric = (label, value, caption) => `<article class="metric"><div class="label">${label}<span class="tiny" aria-hidden="true">\u25C8</span></div><div class="value">${value}</div><p>${caption}</p></article>`;
  const heading = (title, description, tag = "") => `<div class="panel-head"><div><h2>${title}</h2><p>${description}</p></div>${tag ? `<span class="tag">${tag}</span>` : ""}</div>`;
  const notice = (s) => `<aside class="notice">${s}</aside>`;
  const titles = { overview: ["Spot churn risk.<br><span>Understand the trade-off.</span>", "A model is only useful when you know what its numbers mean."], workbench: ["Explore a profile.<br><span>See the model respond.</span>", "Nine behavior signals. One calibrated score. Clear limitations."], threshold: ["Choose your cutoff.<br><span>See what it costs.</span>", "Explore the balance between finding churners and creating false alerts."], "model-card": ["Inspect the evidence.<br><span>Know the boundaries.</span>", "The model, the data, the evaluation and the decisions behind them."] };
  window.addEventListener("hashchange", () => {
    if (location.hash === "#main") {
      document.querySelector("#main").focus();
      return;
    }
    route();
    window.scrollTo({ top: 0, behavior: "instant" });
    document.querySelector("#main").focus({ preventScroll: true });
  });
  route();
}
