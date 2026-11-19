from functools import lru_cache
from io import BytesIO
from pathlib import Path

import torch
from torchvision.utils import save_image

from helper_lib.diffusion import DiffusionModel, UNet, offset_cosine_diffusion_schedule


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "cifar10_diffusion.pth"


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError("Train and export the Diffusion notebook to create models/cifar10_diffusion.pth.")
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    network = UNet(checkpoint["image_size"], checkpoint["num_channels"])
    model = DiffusionModel(network, offset_cosine_diffusion_schedule)
    model.ema_network.load_state_dict(checkpoint["ema_model_state_dict"])
    model.set_normalizer(checkpoint["normalizer_mean"], checkpoint["normalizer_std"])
    model.eval()
    model.requires_grad_(False)
    return model


def generate_diffusion_images(num_samples=16, steps=20):
    if not 1 <= num_samples <= 16 or not 1 <= steps <= 200:
        raise ValueError("num_samples must be 1-16 and steps must be 1-200.")
    images = load_model().generate(num_samples, steps)
    buffer = BytesIO()
    save_image(images, buffer, nrow=min(4, num_samples), format="PNG")
    return buffer.getvalue()
