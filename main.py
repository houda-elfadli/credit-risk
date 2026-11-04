from pathlib import Path

import joblib

from credit_risk.pipeline import run_pipeline
from credit_risk.training import RandomForestTrainer

DATA_PATH = "data/raw/credit.csv"
MODEL_PATH = Path("models/model.pkl")


def main():
    model, score = run_pipeline(RandomForestTrainer(), DATA_PATH)
    print(f"Accuracy (jeu de test) : {score:.3f}")
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Modele sauvegarde dans {MODEL_PATH}")


if __name__ == "__main__":
    main()
