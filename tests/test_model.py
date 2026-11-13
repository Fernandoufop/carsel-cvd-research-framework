import numpy as np

from carsel.schema import FEATURES
from carsel.training import TrainingSettings, train_carsel


def test_training_inference_is_label_free(training_frame):
    model, report = train_carsel(training_frame, TrainingSettings(max_layers=2, min_residual=20))
    X = training_frame[FEATURES].iloc[:8].copy()
    before = model.predict_with_trace(X)
    # No ground-truth y is accepted by the prediction API; extra dataframe
    # columns are unused, so label modifications cannot change inference.
    X["cardio"] = training_frame["cardio"].iloc[:8].values
    after = model.predict_with_trace(X)
    X["cardio"] = 1 - X["cardio"]
    assert model.predict_with_trace(X) == before == after
    assert report["dataset"]["untouched_test_rows"] > 0
    assert sum(report["holdout_routing_counts"].values()) == report["dataset"]["untouched_test_rows"]
    probabilities = model.predict_proba(X)
    assert probabilities.shape == (len(X), 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.isfinite(probabilities).all()
