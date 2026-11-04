from sklearn.model_selection import train_test_split

from credit_risk.data import load_data
from credit_risk.evaluation import evaluate_model
from credit_risk.preprocessing import clean_data, split_features_target
from credit_risk.training import ModelTrainer


def run_pipeline(trainer: ModelTrainer, csv_path, target_column="default"):
    """Pipeline complet. Depend de l'abstraction ModelTrainer (DIP)."""
    data = clean_data(load_data(csv_path))
    features, target = split_features_target(data, target_column)
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    model = trainer.train(x_train, y_train)
    score = evaluate_model(model, x_test, y_test)
    return model, score
