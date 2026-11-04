def clean_data(data):
    """Supprime les observations contenant des valeurs manquantes."""
    return data.dropna()


def split_features_target(data, target_column="default"):
    """Separe les variables explicatives (X) de la cible (y)."""
    features = data.drop(target_column, axis=1)
    target = data[target_column]
    return features, target
