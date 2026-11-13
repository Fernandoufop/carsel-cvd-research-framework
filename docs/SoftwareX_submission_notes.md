# Preparing a SoftwareX submission — CaRSEL-CVD

**Readiness:** functional research prototype; **not** a submission-ready scientific validation package or validated clinical tool.

## Proposed paper title

CaRSEL-CVD: An open-source Python framework for reproducible cascaded ensemble learning with confidence-based routing

## Software contribution

The software provides a modular Python implementation of residual-specialized classifiers, development-only confidence routing, independent test reporting, single- and multi-record inference, and a plain HTML/CSS/JS interface. The key contribution should be described as a **reusable and auditable software workflow**, not merely higher predictive accuracy or clinical utility.

## Required experimental evidence before submission

- Confirm dataset provenance, license, feature coding and the meaning of `cardio=1`.
- Reproduce the original CaRSEL paper's exact algorithm separately if equivalence is claimed. Specifically reconcile argmax stage selection and EAWV probability voting.
- Run real training and repeated external or nested validation and report variability and confidence intervals.
- Benchmark independent holdout performance, calibration, class imbalance, routing errors, KNN fallback and computational cost on representative hardware.
- Compare with scikit-learn baselines and existing cascade/routing software; run ablations on stage selection, gate thresholds, and residual specialization.
- Demonstrate that changing heldout labels **after fitting** cannot affect inference outputs.
- Document missing-value treatment, excluded records, hyperparameter search and exact package versions.
- Document privacy controls. The current server is **localhost research use only**.

## Publication package

- A public repository with verified license and authorship, stable release tag and immutable archived DOI (e.g. archival deposit).
- Runnable source, installation documentation, minimal example, CI tests and reproducible dataset/code instructions.
- A paper written with the current SoftwareX template and verified journal requirements.
- Figures generated from real outputs, not from the illustrative mockups.

## Proposed article sections

1. Motivation and significance
2. Software description (architecture, model, modular API, implementation)
3. Illustrative examples (dataset, actual measured results, screenshots)
4. Impact and reuse across tabular problems
5. Conclusions and limitations
6. Code metadata, open-source license, archival repository, reproducibility statement

**Do not fabricate** experiments, model metrics, external validation, feature-attribution values, runtime benchmarks, or patient-level recommendations. Screenshots of conceptual UI are design artifacts only.
