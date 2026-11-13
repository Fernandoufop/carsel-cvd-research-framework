# Proposed validation before scientific submission

This is a **plan**, not evidence already collected. All results must be
computed and documented on actual labeled records.

## Reproducible data protocol

1. Identify dataset provenance, inclusion criteria, ethics/data license,
   target meaning, sample count and group structure.
2. Confirm the `sex`, `cholesterol`, `glucose`, `smoking`, `alcohol`,
   `physical_activity` encodings. Record age units explicitly.
3. Keep outcome labels inaccessible to preprocessing, inference and route
   selection on the untouched holdout.
4. Remove duplicate patients and use patient/group-level splits if multiple
   observations per person exist.
5. Use nested or multiple outer CV splits. Select architecture, gates, and
   hyperparameters inside training folds only. Record the random seeds.
6. Report class distribution, performance uncertainty, calibration, subgroup
   behavior, subgroup sample sizes, and any threshold-dependent tradeoffs.
7. If possible, evaluate a separate external cohort or a disjoint institution
   or time period. Don't call a split of one dataset "external validation".

## Baselines and ablations

Required baselines to consider: RF, NB, GB, KNN, logistic regression, soft
voting, and a stacking approach with an appropriate OOF meta-learner.

Suggested ablations: no residual specialization, single stage, no calibration,
fixed vs optimized gates, no KNN fallback, and confidence-based vs original
label-informed *training-only* diagnostics. Do not compare method variants
on the basis of post-hoc tuned test scores.

## Research and software evidence

- Accuracy, sensitivity, specificity, precision, F1, balanced accuracy, MCC
- ROC-AUC with 95% CI, PR-AUC/AP with 95% CI (imbalance-sensitive)
- Brier score, reliability diagrams, expected calibration error if suitable
- Route coverage, false-negative incidence in exited cases, abstention/fallback
- Training runtime, inference latency, RAM, CPU hardware, code version
- Installation from a clean environment and container reproduction
- Continuous integration, test coverage, license, readme, changelog, DOI
- Independent rerun of all manuscript tables and figures

## SoftwareX manuscript evidence package

- Software archive with release tag, complete dependencies and license
- Software metadata table and source-code permanent identifier
- Concise motivation and software description
- Screenshot of real interface showing reproducible functionality
- Documented examples and companion datasets where licensing permits
- Comparison showing why the software is useful beyond simply importing
  standard estimators from scikit-learn
- Impact and reuse discussion; known limitations and ethical boundaries

## No fabrication

Never fill benchmark cells with invented performance values. Use `NR` / `N/A`
or explicitly labeled hypothetical examples until real experiments have run.
Do not use the artificial test dataset as clinical validation.
