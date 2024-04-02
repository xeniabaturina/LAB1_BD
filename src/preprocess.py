import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import torch
import pickle
import os


class DataProcessor:
    def __init__(self, scaler_path):
        self.scaler_path = scaler_path
        self.scaler = None

    def load_and_preprocess_data(self, train_path=None, test_path=None):
        x_train_scaled = y_train = x_test_scaled = ids = None

        if train_path is not None:
            train_df = pd.read_csv(train_path)
            x_train = train_df.drop(['medv', 'ID'], axis=1).values
            y_train = train_df['medv'].values

            self.scaler = StandardScaler()
            x_train_scaled = self.scaler.fit_transform(x_train)

            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
        else:
            if os.path.exists(self.scaler_path):
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
            else:
                raise FileNotFoundError("Scaler file not found. Ensure training is run first.")

        if test_path is not None:
            test_df = pd.read_csv(test_path)
            x_test = test_df.drop('ID', axis=1).values
            ids = test_df['ID']

            x_test_scaled = self.scaler.transform(x_test)

        return x_train_scaled, y_train, x_test_scaled, ids

    @staticmethod
    def create_dataloaders(x_train, y_train, batch_size=64):
        x_tensor = torch.tensor(x_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        dataset = TensorDataset(x_tensor, y_tensor)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return train_loader
