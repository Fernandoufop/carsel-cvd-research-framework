"""Integration tests use artificial records only; no clinical evidence."""
import io

from fastapi.testclient import TestClient

from carsel import api
from carsel.schema import FEATURES
from carsel.training import TrainingSettings, train_carsel


def test_frontend_has_english_sections():
    response = TestClient(api.app).get("/")
    assert response.status_code == 200
    assert '<html lang="en">' in response.text
    for page in ["Dashboard", "Single Prediction", "Batch Prediction",
                 "Model Training", "Training Results", "Model Registry", "API &amp; Docs", "Settings"]:
        assert page in response.text
    for name in FEATURES:
        assert f'name="{name}"' in response.text


def test_batch_endpoint_requires_fitted_model(monkeypatch, tmp_path):
    monkeypatch.setenv('CARSEL_MODEL_PATH', str(tmp_path / 'missing.joblib'))
    client = TestClient(api.app)
    record = 'age,height,weight,sex,ap_hi,ap_lo,cholesterol,glucose,smoking,alcohol,physical_activity\n55,170,75,1,135,85,1,1,0,0,1\n'
    response = client.post('/api/predict/csv', files={'dataset': ('patients.csv', record, 'text/csv')})
    assert response.status_code == 503


def test_batch_endpoint_fitted_and_invalid_rows(monkeypatch, training_frame):
    model, _ = train_carsel(training_frame, TrainingSettings(max_layers=1, min_residual=20))
    monkeypatch.setattr(api, '_model_or_none', lambda: model)
    client = TestClient(api.app)
    records = training_frame[FEATURES].iloc[:3].to_csv(index=False)
    response = client.post('/api/predict/csv', files={'dataset': ('patients.csv', records, 'text/csv')})
    assert response.status_code == 200
    assert response.headers['X-Rows-Processed'] == '3'
    assert len(response.text.splitlines()) == 4
    assert 'routing_trace' in response.text.splitlines()[0]
    assert 'cardio' not in response.text.splitlines()[0]
    # Reject a mislabeled categorical code without exposing the individual record.
    invalid = records.replace(',1,1,', ',8,1,', 1)
    invalid_response = client.post('/api/predict/csv', files={'dataset': ('patients.csv', invalid, 'text/csv')})
    assert invalid_response.status_code == 422
    assert 'Invalid feature' in invalid_response.json()['detail']
