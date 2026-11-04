import numpy as np
import pandas as pd

from credit_risk.pipeline import run_pipeline
from credit_risk.training import RandomForestTrainer


def test_run_pipeline(tmp_path):
    rng = np.random.default_rng(0)
    data = pd.DataFrame({"x": rng.normal(size=100)})
    data["default"] = (data["x"] > 0).astype(int)
    csv = tmp_path / "credit.csv"
    data.to_csv(csv, index=False)
    _, score = run_pipeline(RandomForestTrainer(), csv)
    assert 0.0 <= score <= 1.0
