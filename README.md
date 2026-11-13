# CaRSEL-CVD — open-source research software

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/) [![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE) [![Status: research](https://img.shields.io/badge/status-research%20prototype-orange)](docs/SOFTWAREX_CHECKLIST.md)

**CaRSEL-CVD v0.2.0** is an English-language research platform combining a Python implementation of cascaded residual ensemble classification, a FastAPI backend, and a complete frontend written in plain HTML, CSS and JavaScript.

**Medical safety notice:** Not a medical device or validated diagnostic test. Do not use to diagnose disease, triage patients, choose treatment, or make patient-care decisions. Its class-1 probability comes from an experimental model trained on an uploaded dataset and is **not a clinically calibrated risk estimate**.

This repository **does not contain a clinical dataset, patient records, or a pretrained cardiovascular model**. Synthetic data are used only within automated tests. All charts and accuracy metrics in the running application are empty until a real training/evaluation run is completed. The concept-art screens in `docs/mockups/` contain illustrative UI figures and *hypothetical* statistics; do not treat those figures as experimental results.

## Screens & modules

| Page | Function |
|---|---|
| Dashboard | Real model status, recorded holdout metrics, experiments, pipeline overview |
| Single Prediction | 11 validated fields, inference for one record |
| Prediction Results | Predicted class, class-1 probability, confidence routing trace, JSON export |
| Batch Prediction | UTF-8 CSV upload, validation, up to 100,000 records, predictions CSV export |
| Model Training | Authorized CSV training with OOF stage selection and development-only gate tuning |
| Training Results | Measured ROC, precision–recall and calibration curves, confusion matrix, model diagnostics |
| Model Registry | Locally archived experiments and settings |
| API & Docs | REST documentation, machine-readable schema, interactive OpenAPI at `/docs` |
| Settings | Browser-local UI preferences (not a fictitious server administration console) |

The UI uses **English throughout**. Nine *actual interface screenshots* (untrained state) are included under `docs/actual_screens/`; nine *design mockups* under `docs/mockups/` are separately labeled as conceptual illustrations.

## Quick start

**Prerequisite:** Python 3.10+ (Python 3.11 recommended). An internet connection may be required on first installation to download dependencies.

### Windows

Extract the repository and double-click `start_windows.bat`, or use PowerShell:

```powershell
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
.venv\Scripts\python.exe -m uvicorn carsel.api:app --host 127.0.0.1 --port 8000
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
uvicorn carsel.api:app --host 127.0.0.1 --port 8000
```

Then open **http://127.0.0.1:8000** (web app) or **http://127.0.0.1:8000/docs** (OpenAPI). The model will correctly be reported as *not trained* until you train it on a properly labeled dataset.

### Docker

```bash
docker compose up --build
```

The included Compose configuration maps the app only to `127.0.0.1:8000`; **do not expose it publicly** without additional authentication, HTTPS, privacy and security controls.

## Dataset contract

Exactly 11 required predictors, in this order:

```text
age,height,weight,sex,ap_hi,ap_lo,cholesterol,glucose,smoking,alcohol,physical_activity
```

For training, also supply a binary label column (default `cardio`, 0 = negative, 1 = positive). The normal API uses age **in years**; a training CSV can use days if declared explicitly.

| Variable | Accepted values | Meaning |
|---|---|---|
| `age` | 18–110 years | Age in API; source CSV may be days |
| `height` | 100–230 cm | Height |
| `weight` | 25–350 kg | Weight |
| `sex` | 1 or 2 | Confirm source coding (commonly 1 = female, 2 = male) |
| `ap_hi` | 60–300 mmHg | Systolic pressure |
| `ap_lo` | 30–200 mmHg | Diastolic pressure; must be strictly below `ap_hi` |
| `cholesterol` | 1, 2, 3 | **Ordinal encoded category**, not mmol/L |
| `glucose` | 1, 2, 3 | **Ordinal encoded category**, not mmol/L |
| `smoking` | 0 or 1 | Binary code |
| `alcohol` | 0 or 1 | Binary code |
| `physical_activity` | 0 or 1 | Binary code |

Training CSV aliases are supported for `gender→sex`, `gluc→glucose`, `smoke→smoking`, `alco→alcohol`, and `active→physical_activity`. This aliases-only rule does **not** silently recode category values. See `/api/schema` for the machine-readable contract.

## Training via CLI

```bash
carsel-train --csv /path/to/cardio_train.csv --age-unit days --target cardio
# or
carsel-train --csv /path/to/my_data.csv --age-unit years --target outcome
```

`--drop-invalid` is opt-in; otherwise malformed rows stop the run. A row-level invalidity summary is retained in the JSON report. The CLI saves the active model and an archived experiment in the local `artifacts/` directory. This folder is `.gitignore`d because trained objects can include private information and should not be published blindly.

### Training from the web page

HTTP training is deliberately **disabled by default**. Enable it for private local use with an admin token, then open the **Model Training** page:

```bash
export CARSEL_ENABLE_TRAINING=true
export CARSEL_ADMIN_TOKEN='choose-a-long-random-secret'
uvicorn carsel.api:app --host 127.0.0.1 --port 8000
```

On PowerShell use `$env:CARSEL_ENABLE_TRAINING="true"` and `$env:CARSEL_ADMIN_TOKEN="..."`. The frontend never stores the training token in browser preferences. Training jobs are local, in-process jobs and do not survive application restarts; use one Uvicorn worker.

## Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | Service health |
| GET | `/api/schema` | Feature and coding contract |
| GET | `/api/model` | Active model status & aggregate report |
| POST | `/api/predict` | Single inference with routing trace |
| POST | `/api/predict/batch` | JSON inference for 1–500 records |
| POST | `/api/predict/csv` | CSV inference up to 100,000 records; CSV download |
| POST | `/api/train` | Authenticated, opt-in training |
| GET | `/api/jobs/{job_id}` | Authenticated training status |
| GET | `/api/models` | List locally archived runs |
| GET | `/api/experiments/{run_id}` | Aggregate metrics, model metadata and curves |

Example:

```bash
curl -X POST http://127.0.0.1:8000/api/predict \
 -H 'Content-Type: application/json' \
 -d '{"age":55,"height":170,"weight":75,"sex":1,"ap_hi":135,"ap_lo":85,"cholesterol":1,"glucose":1,"smoking":0,"alcohol":0,"physical_activity":1}'
```

If there is no trained model, inference returns HTTP 503 **instead of fabricated predictions**.

## How the implemented model works

1. Hold out an independent stratified test partition before model construction.
2. Within the fit partition, compare RF, Gaussian NB and Gradient Boosting through stratified out-of-fold predictions. Optimize out-of-fold **balanced accuracy** (not label-informed test selection).
3. Construct additional specialist stages from the preceding stage's *out-of-fold errors* inside the fit partition.
4. Fit probability transformation using development-only out-of-fold scores.
5. Select confidence thresholds on a separate development selection partition, with an empirical exit-error cap (not a clinical guarantee).
6. Freeze the pipeline. At inference, route by score confidence without knowing any true label; use a pretrained KNN for final fallback.
7. After freezing, score the untouched test partition once and archive metrics, aggregated diagnostic curves and reproducibility settings.

**Research integrity:** This is a scientifically motivated, leakage-controlled *variant* of the proposed CaRSEL-CVD protocol. It is **not proven to be exactly equivalent** to the article's original architecture. In particular, the manuscript's EAWV label implies weighted voting, while the supplied equation is an argmax over correct classifications. This implementation does not falsely claim to implement weighted EAWV or reproduce reported article accuracy. The exact original model can only be ported and verified after comparing original source code, preprocessing, feature codes, and experiment logs.

## Reproducibility and testing

```bash
python -m pip install -e '.[dev]'
pytest -q
```

Tests use artificial tabular samples solely for checking behavior. The suite covers schema validation, reproducibility structure, label-free inference, API responses, CSV batch inference and the availability of all English screens. A CI workflow in `.github/workflows/ci.yml` runs tests on pull requests and pushes. For methodology details see `docs/validation_protocol.md`, and for publication gaps see `docs/SOFTWAREX_CHECKLIST.md`.

## Screenshots and publication figures

The `docs/actual_screens/` images show the implemented untrained frontend, with no invented model performance. The `docs/mockups/` images are design concepts only: they are **not** screenshots of live measured clinical results and must not be used as experimental figures in a manuscript. For a SoftwareX submission, generate and cite new screenshots/plots directly from the released software using real, licensed datasets, with a stable released version and exact settings.

## License, citation, privacy

MIT license applies to the newly authored project code; contributors must confirm licensing for any future imported original CaRSEL source. Do not commit patient data, original private CSVs, access tokens, or serialized models. Authorship, affiliations, final software version, archived repository URL/DOI and peer-reviewed evidence must be confirmed prior to journal submission. See `docs/SoftwareX_submission_notes.md`.
