import pytest
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))


@pytest.fixture(scope="module")
def generated_fixture_data(tmp_path_factory):
    n_rows = 20
    np.random.seed(0)
    data = {
        'ID': np.arange(n_rows),
        'crim': np.random.uniform(0, 100, n_rows),
        'zn': np.random.uniform(0, 100, n_rows),
        'indus': np.random.uniform(0, 25, n_rows),
        'chas': np.random.randint(0, 2, n_rows),
        'nox': np.random.uniform(0.4, 0.7, n_rows),
        'rm': np.random.uniform(3, 8, n_rows),
        'age': np.random.uniform(0, 100, n_rows),
        'dis': np.random.uniform(1, 12, n_rows),
        'rad': np.random.randint(1, 25, n_rows),
        'tax': np.random.uniform(200, 800, n_rows),
        'ptratio': np.random.uniform(12, 22, n_rows),
        'black': np.random.uniform(0, 400, n_rows),
        'lstat': np.random.uniform(2, 37, n_rows),
        'medv': np.random.uniform(5, 50, n_rows),
    }
    train_df = pd.DataFrame(data)
    test_df = train_df.drop(columns=['medv'])

    tmp_dir = tmp_path_factory.mktemp("data")
    test_path = tmp_dir / "train_fixture.csv"
    train_path = tmp_dir / "test_fixture.csv"
    scaler_path = tmp_dir / "scaler_fixture.pkl"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    return train_path, test_path, scaler_path
