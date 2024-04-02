import torch.optim as optim
import torch.nn as nn
import configparser

from src.preprocess import DataProcessor
from src.utils import RegressionModel, ModelHandler


class Trainer:
    def __init__(self, train_path, model_path, scaler_path, num_epochs=600, lr=0.001):
        self.train_path = train_path
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.num_epochs = num_epochs
        self.lr = lr

    def train(self):
        processor = DataProcessor(scaler_path=self.scaler_path)
        x_train, y_train, _, _ = processor.load_and_preprocess_data(self.train_path, None)
        train_loader = processor.create_dataloaders(x_train, y_train)

        model = RegressionModel(x_train.shape[1])
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        for epoch in range(self.num_epochs):
            for inputs, targets in train_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {loss.item():.4f}')

        handler = ModelHandler
        handler.save_model(model, self.model_path)


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('config.ini')

    path_to_train = config['data']['train_data_path']
    path_to_model = config['model']['model_path']
    path_to_scaler = config['preprocessing']['scaler_path']
    epochs_number = int(config['hyperparameters']['num_epochs'])
    learning_rate = float(config['hyperparameters']['learning_rate'])

    trainer = Trainer(path_to_train, path_to_model, path_to_scaler, epochs_number, learning_rate)
    trainer.train()
