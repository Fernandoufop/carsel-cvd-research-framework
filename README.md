# CaRSEL-CVD — Cardiovascular Research Software

Research prototype (v0.2.0) for leakage-controlled cascaded ensemble learning in cardiovascular tabular datasets. The project combines a Python/FastAPI backend with an English-language HTML, CSS, and JavaScript web interface.

> **Research use only.** This is not a validated diagnostic device or clinically calibrated risk score. The repository contains no patient records or pretrained medical model. Performance statistics must come from reproducible held-out experiments, never from UI mockups.

The source distribution includes the eleven fields `age`, `height`, `weight`, `sex`, `ap_hi`, `ap_lo`, `cholesterol`, `glucose`, `smoking`, `alcohol`, and `physical_activity`. The encoded features `cholesterol` and `glucose` are categories (1–3), not measured concentrations.

Source code, documentation, tests, and setup instructions will be provided in this repository. See the research validation protocol before using the software for scientific claims.
