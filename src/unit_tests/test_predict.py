import pandas as pd

from src.predict import Predictor
from src.train import Trainer


def test_predict_saves_predictions(generated_fixture_data):
    train_path, test_path, scaler_path = generated_fixture_data
    model_path = scaler_path.parent / "temp_model.pth"
    predictions_path = scaler_path.parent / "predictions.csv"

    trainer = Trainer(train_path, model_path, scaler_path, 1)
    trainer.train()

    predictor = Predictor(test_path, model_path, scaler_path, predictions_path)
    predictor.predict()

    assert predictions_path.exists(), "Predictions file was not created."

    predictions_df = pd.read_csv(predictions_path)
    assert not predictions_df.empty, "Predictions DataFrame is empty."
