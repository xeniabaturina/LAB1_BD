import numpy as np
from src.preprocess import load_and_preprocess_data


def test_data_normalization(generated_fixture_data):
    train_path, test_path, scaler_path = generated_fixture_data
    x_train_scaled, y_train, x_test_scaled, ids = load_and_preprocess_data(train_path, test_path, scaler_path)

    assert np.isclose(x_train_scaled.mean(),
                      0, atol=0.1).all(), "Feature scaling mean for train data is not approximately 0"
    assert np.isclose(x_train_scaled.std(),
                      1, atol=0.1).all(), "Feature scaling std deviation for train data is not approximately 1"
    assert np.isclose(x_test_scaled.mean(),
                      0, atol=0.1).all(), "Feature scaling mean for tests data is not approximately 0"
    assert np.isclose(x_test_scaled.std(),
                      1, atol=0.1).all(), "Feature scaling std deviation for tests data is not approximately 1"
