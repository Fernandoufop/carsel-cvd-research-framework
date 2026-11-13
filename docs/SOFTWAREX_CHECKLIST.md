# SoftwareX publication-readiness checklist

**Project release target: v0.2.0 — research prototype.** The journal will decide suitability; compliance with any checklist does not guarantee peer review or acceptance. Verify the current journal instructions and code metadata template before submission.

## Implemented and software-tested

- [x] Python CaRSEL experimental ensemble and independent label-free inference
- [x] FastAPI backend, browser frontend in English (HTML/CSS/vanilla JS)
- [x] All nine functional navigation areas (including prediction results)
- [x] Machine-readable contract for the eleven encoded features
- [x] CSV validation, single and batch inference, aggregated outputs
- [x] Development-only OOF candidate selection and residual specialization
- [x] Development-only gate thresholds and a pretrained KNN fallback
- [x] Independent one-shot holdout report; measured ROC, PR and calibration plots
- [x] CLI training, protected opt-in HTTP training and archived experiments
- [x] Automated tests; GitHub Actions CI definition; local Docker setup
- [x] MIT license for the newly authored prototype code
- [x] Concept-art figures separated from actual software screens

## Open items before an actual manuscript submission

- [ ] Recover the original CaRSEL source and establish exact equivalence (or explicitly distinguish variants)
- [ ] Reconcile original EAWV weighted voting with original argmax stage selector
- [ ] Confirm source-code copyright, authors and affiliations
- [ ] Execute reproducible experiments on legitimately accessible cardiovascular datasets
- [ ] Run repeated/nested, independent and preferably external evaluations
- [ ] Compare baseline classifiers and conduct ablations of the cascade and gate
- [ ] Evaluate class-wise error, calibration, routing of high-confidence false negatives, uncertainty, data shift, subgroup behavior
- [ ] Benchmark runtime, throughput, memory and installation reproducibility
- [ ] Publish a public GitHub repository and release tag; archive a DOI
- [ ] Create immutable model/dataset manifests and exact dependency lockfile
- [ ] Add robust authentication, encryption and privacy governance before any nonlocal or identifiable-patient deployment
- [ ] Prepare submission-quality figures from **real** experiments only
- [ ] Fill SoftwareX code metadata, citation, statement of software impact and author declarations

## Evidence and caveats

The UI mockup figures are not empirical screenshots. They use hypothetical patient cases and numbers. They also depict cholesterol/glucose in physical units, whereas the implemented contract uses ordinal categorical codes. Such visuals must **not** be used as measured results or a truthful technical specification. The `docs/actual_screens/` directory illustrates the genuine untrained interface.
