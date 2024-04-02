import os
from src.train import train
from src.utils import load_model


def test_train_model_saves_file(generated_fixture_data):
    train_path, _, scaler_path = generated_fixture_data
    model_path = scaler_path.parent / "temp_model.pth"

    train(train_path, model_path, scaler_path, num_epochs=1)

    assert os.path.exists(model_path), "Model file was not created after training."

    model = load_model(model_path, input_size=14)
    assert model is not None, "Loaded model is None."
