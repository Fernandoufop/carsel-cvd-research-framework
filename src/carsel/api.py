"""Single-origin FastAPI backend + static frontend for local research use."""
from __future__ import annotations

from functools import lru_cache
import logging
import io
import csv
import json
import os
from pathlib import Path
import secrets
from threading import Lock
import uuid

from fastapi import BackgroundTasks, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import pandas as pd

from . import __version__
from .data import DatasetError, load_dataset
from .schema import PatientFeatures, public_schema, FEATURES
from .store import load_model, load_report, model_path, save_artifacts, archive_run, list_runs, load_run_report
from .training import TrainingSettings, train_carsel

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
WEB_DIR = Path(__file__).resolve().parent / "web"
app = FastAPI(
    title="CaRSEL-CVD Research API",
    version=__version__,
    description="Experimental cascade learning. Not validated for clinical diagnosis.",
)
app.mount("/assets", StaticFiles(directory=WEB_DIR / "assets"), name="assets")
_jobs: dict[str, dict] = {}
_jobs_lock = Lock()


def _training_enabled() -> bool:
    return os.environ.get("CARSEL_ENABLE_TRAINING", "false").lower() == "true"


@lru_cache(maxsize=2)
def _cached_model(path_string: str, mtime_ns: int):
    # Loading pickle/joblib from arbitrary sources is dangerous. This only
    # reads the app's own model path, never a user-uploaded joblib.
    return load_model(Path(path_string))


def _model_or_none():
    path = model_path()
    if not path.is_file():
        return None
    return _cached_model(str(path.resolve()), path.stat().st_mtime_ns)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
def health():
    return {"service": "carsel-cvd", "version": __version__, "status": "ok"}


@app.get("/api/schema")
def schema():
    return public_schema()


@app.get("/api/model")
def model_status():
    try:
        model = _model_or_none()
        report = load_report(model_path()) if model is not None else None
    except Exception:
        return {"trained": False, "training_enabled": _training_enabled(),
                "error": "Model artifact could not be loaded. Check server logs and local dependencies."}
    return {
        "trained": model is not None,
        "training_enabled": _training_enabled(),
        "software_version": __version__,
        "summary": model.model_summary() if model is not None else None,
        "report": report,
        "warning": "Research software only. Not for patient care or clinical decision-making.",
    }


@app.post("/api/predict")
def predict(payload: PatientFeatures):
    try:
        model = _model_or_none()
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Could not load local model artifact") from exc
    if model is None:
        raise HTTPException(status_code=503,
                            detail="No fitted model available. Train CaRSEL on a labeled CSV first.")
    frame = pd.DataFrame([payload.model_dump()])
    result = model.predict_with_trace(frame)[0]
    return {
        **result,
        "interpretation": "Predicted probability for dataset class 1; not a validated clinical risk.",
        "model_version": model.version,
        "research_only": True,
    }


class BatchPredictionRequest(BaseModel):
    records: list[PatientFeatures] = Field(min_length=1, max_length=500)


@app.post("/api/predict/batch")
def predict_batch(payload: BatchPredictionRequest):
    try:
        model = _model_or_none()
    except Exception as exc:
        raise HTTPException(503, "Could not load local model artifact") from exc
    if model is None:
        raise HTTPException(503, "No fitted model available. Train CaRSEL on a labeled CSV first.")
    frame = pd.DataFrame([item.model_dump() for item in payload.records])
    results = model.predict_with_trace(frame)
    return {"n": len(results), "predictions": results, "research_only": True,
            "interpretation": "Experimental class-1 predictions, not clinical risk estimates."}



@app.post("/api/predict/csv")
async def predict_csv(dataset: UploadFile = File(...)):
    """Validated, in-memory batch inference; no original records in output.

    25 MB input limit, 100,000 row cap and 2,000-row inference chunks.
    Output includes row indices, predictions, and routing traces, not input data.
    """
    if not (dataset.filename or "").lower().endswith(".csv"):
        raise HTTPException(400, "Expected a .csv file")
    payload = await dataset.read(MAX_UPLOAD_BYTES + 1)
    await dataset.close()
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "CSV exceeds the 25 MB limit")
    if not payload:
        raise HTTPException(400, "CSV is empty")
    try:
        frame = pd.read_csv(io.BytesIO(payload), encoding="utf-8-sig")
    except Exception as exc:
        raise HTTPException(400, "Cannot parse UTF-8 CSV") from exc
    if not len(frame) or len(frame) > 100_000:
        raise HTTPException(400, "Expected between 1 and 100,000 records")
    if len(set(frame.columns)) != len(frame.columns):
        raise HTTPException(400, "Duplicate CSV column names")
    missing = [col for col in FEATURES if col not in frame.columns]
    if missing:
        raise HTTPException(422, "Missing input features: " + ", ".join(missing))
    if any(col not in FEATURES for col in frame.columns):
        # Do not risk uploading an outcome or identifier column when predicting.
        raise HTTPException(422, "Batch CSV must contain only the 11 documented feature columns")
    try:
        model = _model_or_none()
    except Exception as exc:
        raise HTTPException(503, "Could not load the local trained model") from exc
    if model is None:
        raise HTTPException(503, "No trained model. Train on a labeled CSV first")
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["row_index", "predicted_class", "positive_probability", "decision_layer", "routing_trace"])
    for start in range(0, len(frame), 2000):
        sub = frame.iloc[start:start + 2000]
        try:
            records = [PatientFeatures.model_validate(row).model_dump()
                       for row in sub.to_dict(orient="records")]
        except Exception as exc:
            # Do not echo supplied patient data in HTTP errors.
            raise HTTPException(422, f"Invalid feature encoding/range in rows {start+1}–{min(start+2000,len(frame))}; see /api/schema") from exc
        results = model.predict_with_trace(pd.DataFrame(records))
        for i, result in enumerate(results):
            writer.writerow([start + i + 1, result["predicted_class"],
                            result["positive_probability"], result["decision_layer"],
                            json.dumps(result["trace"], separators=(",", ":"))])
    return Response(content=output.getvalue(), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": "attachment; filename=carsel_predictions.csv",
                             "X-Rows-Processed": str(len(frame))})

def _set_job(job_id: str, **fields):
    with _jobs_lock:
        _jobs[job_id] = {**_jobs.get(job_id, {}), **fields}


def _run_training(job_id: str, payload: bytes, age_unit: str, target: str,
                  drop_invalid: bool, max_layers: int, max_exit_error: float):
    _set_job(job_id, state="running", message="Validating CSV and training cascade...")
    try:
        df, ingestion = load_dataset(io.BytesIO(payload), target=target,
                                     age_unit=age_unit, drop_invalid=drop_invalid)
        model, report = train_carsel(df, TrainingSettings(
            target=target, max_layers=max_layers, max_exit_error=max_exit_error))
        report["ingestion"] = vars(ingestion)
        save_artifacts(model, report, model_path())
        archive_run(model, report, model_path())
        _cached_model.cache_clear()
        _set_job(job_id, state="completed", message="Model trained and saved.",
                 metrics=report["holdout_metrics"],
                 rows_used=ingestion.accepted_rows,
                 rows_rejected=ingestion.rejected_rows)
    except (DatasetError, ValueError) as exc:
        _set_job(job_id, state="failed", message=str(exc))
    except Exception:
        # Do not leak uploaded records or stack traces to an unauthenticated user.
        _set_job(job_id, state="failed", message="Unexpected training error. Check local server logs.")
        logging.getLogger(__name__).exception("CaRSEL training failed")


@app.post("/api/train", status_code=202)
async def train(
    background: BackgroundTasks,
    dataset: UploadFile = File(...),
    age_unit: str = Form("years"),
    target: str = Form("cardio"),
    drop_invalid: bool = Form(False),
    max_layers: int = Form(3),
    max_exit_error: float = Form(0.12),
    admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
):
    if not _training_enabled():
        raise HTTPException(403, "Training via HTTP is disabled. Use the CLI or enable explicitly.")
    expected = os.environ.get("CARSEL_ADMIN_TOKEN", "")
    if not expected or not admin_token or not secrets.compare_digest(admin_token, expected):
        raise HTTPException(401, "Missing or invalid training token")
    if not (dataset.filename or "").lower().endswith(".csv"):
        raise HTTPException(400, "Only .csv files are supported")
    if age_unit not in {"days", "years"}:
        raise HTTPException(400, "age_unit must be days or years")
    if not target.isidentifier():
        raise HTTPException(400, "Invalid outcome column name")
    try:
        TrainingSettings(target=target, max_layers=max_layers,
                         max_exit_error=max_exit_error).validate()
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    payload = await dataset.read(MAX_UPLOAD_BYTES + 1)
    await dataset.close()
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "CSV exceeds the 25 MB upload limit")
    if not payload:
        raise HTTPException(400, "CSV is empty")
    job_id = uuid.uuid4().hex
    _set_job(job_id, state="queued", message="Training job queued")
    background.add_task(_run_training, job_id, payload, age_unit, target,
                        drop_invalid, max_layers, max_exit_error)
    return {"job_id": job_id, "state": "queued"}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str, admin_token: str | None = Header(default=None, alias="X-Admin-Token")):
    expected = os.environ.get("CARSEL_ADMIN_TOKEN", "")
    if not _training_enabled() or not expected or not admin_token or not secrets.compare_digest(admin_token, expected):
        raise HTTPException(401, "Training token required")
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Unknown job ID (jobs are held in process memory)")
        return {"job_id": job_id, **job}


@app.get("/api/models")
def models():
    """Registered locally trained versions. Never invent experiments."""
    try:
        return {"active": model_path().is_file(), "runs": list_runs(model_path())}
    except Exception:
        raise HTTPException(500, "Could not read local registry")


@app.get("/api/experiments/{run_id}")
def experiment(run_id: str):
    if not (1 <= len(run_id) <= 60) or not all(c.isalnum() or c in '-_' for c in run_id):
        raise HTTPException(400, "Invalid run identifier")
    try:
        result = load_run_report(model_path(), run_id)
    except (OSError, ValueError):
        raise HTTPException(500, "Cannot access experiment report")
    if result is None:
        raise HTTPException(404, "Unknown experiment")
    return result
