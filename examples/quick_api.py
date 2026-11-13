"""API usage after a researcher has already trained a local model."""
from __future__ import annotations

import json
import urllib.request

# Does not include a true outcome label: the server must perform label-free inference.
record = {
    "age": 55, "height": 170, "weight": 75, "sex": 1,
    "ap_hi": 135, "ap_lo": 85, "cholesterol": 1, "glucose": 1,
    "smoking": 0, "alcohol": 0, "physical_activity": 1,
}
request = urllib.request.Request(
    "http://127.0.0.1:8000/api/predict",
    data=json.dumps(record).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
try:
    with urllib.request.urlopen(request, timeout=30) as response:
        print(json.dumps(json.load(response), indent=2))
except Exception as exc:
    print("The backend must be running with a trained model:", exc)
