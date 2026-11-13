from fastapi.testclient import TestClient
from carsel.api import app


def test_health_schema_and_no_model(monkeypatch, tmp_path):
    monkeypatch.setenv("CARSEL_MODEL_PATH", str(tmp_path / "missing.joblib"))
    client = TestClient(app)
    assert client.get("/").status_code == 200
    assert "CaRSEL-CVD" in client.get("/").text
    assert client.get("/assets/app.js").status_code == 200
    assert client.get("/api/health").status_code == 200
    assert len(client.get("/api/schema").json()["features"]) == 11
    assert client.get("/api/model").json()["trained"] is False
    response = client.post("/api/predict", json={
        "age": 55, "height": 175, "weight": 76, "sex": 1,
        "ap_hi": 132, "ap_lo": 82, "cholesterol": 1, "glucose": 1,
        "smoking": 0, "alcohol": 0, "physical_activity": 1,
    })
    assert response.status_code == 503


def test_training_endpoint_disabled_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("CARSEL_ENABLE_TRAINING", raising=False)
    response = TestClient(app).post("/api/train", files={"dataset": ("d.csv", b"a,b\n1,2", "text/csv")})
    assert response.status_code == 403
