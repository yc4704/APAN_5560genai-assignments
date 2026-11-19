from functools import lru_cache
from io import BytesIO
from pathlib import Path

import torch
from torchvision.utils import save_image

from helper_lib.energy import EnergyModel, generate_samples


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "cifar10_energy.pth"


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError("Train and export the Energy notebook to create models/cifar10_energy.pth.")
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model = EnergyModel()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.requires_grad_(False)
    return model, checkpoint["step_size"], checkpoint["noise"]


def generate_energy_images(num_samples=16, steps=60):
    if not 1 <= num_samples <= 16 or not 1 <= steps <= 200:
        raise ValueError("num_samples must be 1-16 and steps must be 1-200.")
    model, step_size, noise = load_model()
    initial_noise = torch.rand(num_samples, 3, 32, 32) * 2 - 1
    images = generate_samples(model, initial_noise, steps, step_size, noise)
    images = ((images + 1) / 2).clamp(0, 1)
    buffer = BytesIO()
    save_image(images, buffer, nrow=min(4, num_samples), format="PNG")
    return buffer.getvalue()
