import torch
import torch.nn as nn


class RegressionModel(nn.Module):
    def __init__(self, input_size):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 20)
        self.fc3 = nn.Linear(20, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out


class ModelHandler:
    @staticmethod
    def save_model(model, filepath):
        torch.save(model.state_dict(), filepath)

    @staticmethod
    def load_model(filepath, input_size):
        model = RegressionModel(input_size)
        model.load_state_dict(torch.load(filepath))
        model.eval()
        return model
