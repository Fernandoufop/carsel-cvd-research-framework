import io
import pytest

from carsel.data import DatasetError, load_dataset
from carsel.schema import PatientFeatures


def test_aliases_and_age_conversion(training_frame):
    source = training_frame.copy()
    source["age"] = source["age"] * 365.25
    source = source.rename(columns={
        "sex": "gender", "glucose": "gluc", "smoking": "smoke",
        "alcohol": "alco", "physical_activity": "active",
    })
    frame, summary = load_dataset(io.BytesIO(source.to_csv(index=False).encode()), age_unit="days")
    assert frame["age"].mean() == pytest.approx(training_frame["age"].mean())
    assert summary.aliases_applied["gender"] == "sex"
    assert summary.rejected_rows == 0


def test_invalid_pressure_rejected(training_frame):
    bad = training_frame.copy()
    bad.loc[0, "ap_lo"] = bad.loc[0, "ap_hi"] + 10
    source = io.BytesIO(bad.to_csv(index=False).encode())
    with pytest.raises(DatasetError, match="invalid rows"):
        load_dataset(source)
    frame, summary = load_dataset(io.BytesIO(bad.to_csv(index=False).encode()), drop_invalid=True)
    assert summary.rejected_rows == 1
    assert len(frame) == len(bad) - 1


def test_patient_rejects_inverted_bp():
    with pytest.raises(Exception, match="ap_hi"):
        PatientFeatures(age=55, height=170, weight=75, sex=1, ap_hi=100, ap_lo=120,
                        cholesterol=1, glucose=1, smoking=0, alcohol=0, physical_activity=1)
