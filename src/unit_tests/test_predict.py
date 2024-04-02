from src.train import train
from src.predict import predict
import pandas as pd


def test_predict_saves_predictions(generated_fixture_data):
    train_path, test_path, scaler_path = generated_fixture_data
    model_path = scaler_path.parent / "temp_model.pth"
    predictions_path = scaler_path.parent / "predictions.csv"

    train(train_path, model_path, scaler_path, num_epochs=1)
    predict(test_path, model_path, scaler_path, predictions_path)

    assert predictions_path.exists(), "Predictions file was not created."

    predictions_df = pd.read_csv(predictions_path)
    assert not predictions_df.empty, "Predictions DataFrame is empty."
