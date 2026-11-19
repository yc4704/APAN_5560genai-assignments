"""Verify the published inference weights through the actual FastAPI routes."""

import io
import unittest

import torch
from fastapi.testclient import TestClient
from PIL import Image

from app import diffusion_service, energy_service
from app.main import app


class TrainedAssignment4APITests(unittest.TestCase):
    def test_included_epoch_10_models_return_png_at_default_steps(self):
        torch.set_num_threads(2)
        with TestClient(app) as client:
            for route, service in (
                ("/generate-energy", energy_service),
                ("/generate-diffusion", diffusion_service),
            ):
                with self.subTest(route=route):
                    checkpoint = torch.load(service.MODEL_PATH, map_location="cpu", weights_only=True)
                    self.assertEqual(checkpoint["epoch"], 10)
                    service.load_model.cache_clear()
                    response = client.get(route, params={"num_samples": 1})
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.headers["content-type"], "image/png")
                    with Image.open(io.BytesIO(response.content)) as image:
                        image.load()
                        self.assertEqual(image.format, "PNG")
                        self.assertEqual(image.mode, "RGB")
                        self.assertEqual(image.size, (32, 32))
                    service.load_model.cache_clear()


if __name__ == "__main__":
    unittest.main()
