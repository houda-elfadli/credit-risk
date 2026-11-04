import pandas as pd

from credit_risk.preprocessing import clean_data, split_features_target


def test_clean_data_removes_missing_values():
    data = pd.DataFrame({"a": [1, None, 3], "default": [0, 1, 0]})
    assert len(clean_data(data)) == 2


def test_split_features_target():
    data = pd.DataFrame({"a": [1, 2], "default": [0, 1]})
    features, target = split_features_target(data)
    assert list(features.columns) == ["a"]
    assert list(target) == [0, 1]
