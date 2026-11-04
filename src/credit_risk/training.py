from abc import ABC, abstractmethod

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class ModelTrainer(ABC):
    """Abstraction : tout entraineur de modele expose train()."""

    @abstractmethod
    def train(self, features, target):
        """Entraine et retourne un modele."""


class RandomForestTrainer(ModelTrainer):
    def train(self, features, target):
        model = RandomForestClassifier(random_state=42)
        model.fit(features, target)
        return model


class LogisticRegressionTrainer(ModelTrainer):
    def train(self, features, target):
        model = LogisticRegression(max_iter=1000)
        model.fit(features, target)
        return model
