import pandas as pd
import pytest

from credit_risk.prediction import predict
from credit_risk.training import (
    LogisticRegressionTrainer,
    ModelTrainer,
    RandomForestTrainer,
)


@pytest.mark.parametrize(
    "trainer_cls", [RandomForestTrainer, LogisticRegressionTrainer]
)
def test_trainers_are_interchangeable(trainer_cls):
    features = pd.DataFrame({"x": [0, 1, 2, 3, 4, 5]})
    target = pd.Series([0, 0, 0, 1, 1, 1])
    trainer = trainer_cls()
    assert isinstance(trainer, ModelTrainer)
    model = trainer.train(features, target)
    assert len(predict(model, features)) == 6
