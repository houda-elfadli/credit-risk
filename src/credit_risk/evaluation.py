def evaluate_model(model, features, target):
    """Retourne le score (accuracy) du modele."""
    return model.score(features, target)
