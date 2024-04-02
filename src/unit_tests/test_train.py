import os

from src.train import Trainer
from src.utils import ModelHandler


def test_train_model_saves_file(generated_fixture_data):
    train_path, _, scaler_path = generated_fixture_data
    model_path = scaler_path.parent / "temp_model.pth"

    trainer = Trainer(train_path, model_path, scaler_path, num_epochs=1)
    trainer.train()

    assert os.path.exists(model_path), "Model file was not created after training."

    handler = ModelHandler()
    model = handler.load_model(model_path, input_size=14)

    assert model is not None, "Loaded model is None."
