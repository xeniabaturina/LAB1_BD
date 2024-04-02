from src.utils import save_model, load_model, RegressionModel
import torch


def test_save_and_load_model(tmp_path):
    model = RegressionModel(input_size=13)  # Assuming 13 features as input
    model_path = tmp_path / "model.pth"

    save_model(model, model_path)

    assert model_path.exists(), "Model file was not saved."

    loaded_model = load_model(model_path, input_size=13)
    assert isinstance(loaded_model, torch.nn.Module), "Failed to load model correctly."
