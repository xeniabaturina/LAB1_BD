import torch
import pandas as pd
import configparser

from .preprocess import DataProcessor
from .utils import ModelHandler


class Predictor:
    def __init__(self, test_path, model_path, scaler_path, predictions_path):
        self.test_path = test_path
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.predictions_path = predictions_path

    def predict(self):
        processor = DataProcessor(scaler_path=self.scaler_path)
        _, _, x_test, ids = processor.load_and_preprocess_data(None, self.test_path)
        handler = ModelHandler
        model = handler.load_model(self.model_path, input_size=x_test.shape[1])

        x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
        with torch.no_grad():
            test_predictions = model(x_test_tensor)

        submission_df = pd.DataFrame({
            'ID': ids,
            'medv': test_predictions.numpy().flatten()
        })

        submission_df.to_csv(self.predictions_path, index=False)
        print(f'Submission saved to {self.predictions_path}')


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('config.ini')

    path_to_test = config['data']['test_data_path']
    path_to_model = config['model']['model_path']
    path_to_scaler = config['preprocessing']['scaler_path']
    path_to_predictions = 'submission_my_model.csv'

    predictor = Predictor(path_to_test, path_to_model, path_to_scaler, path_to_predictions)
    predictor.predict()
