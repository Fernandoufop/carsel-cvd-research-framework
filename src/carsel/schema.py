"""Schema for the *eleven* predictor fields shown by the researcher."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

FEATURES = [
    "age", "height", "weight", "sex", "ap_hi", "ap_lo",
    "cholesterol", "glucose", "smoking", "alcohol", "physical_activity",
]
NUMERIC = ["age", "height", "weight", "ap_hi", "ap_lo"]
CATEGORICAL = ["sex", "cholesterol", "glucose", "smoking", "alcohol", "physical_activity"]

# Optional aliases of the public Cardiovascular Disease dataset. Mapping is
# documented, not inferred from data. To avoid collisions, dual names are rejected.
ALIASES = {
    "gender": "sex", "gluc": "glucose", "smoke": "smoking",
    "alco": "alcohol", "active": "physical_activity",
}

class PatientFeatures(BaseModel):
    """Adult tabular research record; ages are in years for API calls."""

    model_config = ConfigDict(extra="forbid")
    age: float = Field(ge=18, le=110, description="Age in years")
    height: float = Field(ge=100, le=230, description="Height in centimeters")
    weight: float = Field(ge=25, le=350, description="Weight in kilograms")
    sex: Literal[1, 2] = Field(description="Coded sex: 1/2; must match training data")
    ap_hi: float = Field(ge=60, le=300, description="Systolic blood pressure mmHg")
    ap_lo: float = Field(ge=30, le=200, description="Diastolic blood pressure mmHg")
    cholesterol: Literal[1, 2, 3]
    glucose: Literal[1, 2, 3]
    smoking: Literal[0, 1]
    alcohol: Literal[0, 1]
    physical_activity: Literal[0, 1]

    @model_validator(mode="after")
    def pressure_is_ordered(self):
        if self.ap_hi <= self.ap_lo:
            raise ValueError("ap_hi must be strictly greater than ap_lo")
        return self


def public_schema():
    return {
        "features": FEATURES,
        "label": "cardio",
        "age_unit_api": "years",
        "units": {
            "age": "years", "height": "cm", "weight": "kg",
            "ap_hi": "mmHg", "ap_lo": "mmHg",
        },
        "codings": {
            "sex": {"1": "code 1 (often female)", "2": "code 2 (often male)"},
            "cholesterol": {"1": "normal", "2": "above normal", "3": "well above normal"},
            "glucose": {"1": "normal", "2": "above normal", "3": "well above normal"},
            "smoking": {"0": "no", "1": "yes"},
            "alcohol": {"0": "no", "1": "yes"},
            "physical_activity": {"0": "no", "1": "yes"},
        },
        "aliases_csv": ALIASES,
        "note": "Confirm the feature encodings and outcome labels against your original dataset.",
    }
