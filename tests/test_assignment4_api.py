import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from fastapi.testclient import TestClient
from PIL import Image

from app import diffusion_service, energy_service
from app.main import app
from helper_lib.diffusion import DiffusionModel, UNet, offset_cosine_diffusion_schedule
from helper_lib.energy import EBM, EnergyModel


torch.set_num_threads(2)


class Assignment4APITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        root = Path(cls.directory.name)
        cls.energy_path = root / "energy.pth"
        cls.diffusion_path = root / "diffusion.pth"
        energy = EnergyModel()
        ebm = EBM(energy, 0.1, 2, 10.0, 0.005, "cpu")
        ebm.train_step(torch.rand(2, 3, 32, 32) * 2 - 1, torch.optim.Adam(energy.parameters()))
        torch.save({"model_state_dict": energy.state_dict(), "step_size": 10.0,
                    "noise": 0.005}, cls.energy_path)
        diffusion = DiffusionModel(UNet(32, 3), offset_cosine_diffusion_schedule)
        diffusion.set_normalizer([0.4, 0.5, 0.6], [0.2, 0.3, 0.4])
        diffusion.train_step(torch.rand(2, 3, 32, 32),
                             torch.optim.Adam(diffusion.network.parameters()), torch.nn.MSELoss())
        torch.save({"ema_model_state_dict": diffusion.ema_network.state_dict(),
                    "normalizer_mean": diffusion.normalizer_mean,
                    "normalizer_std": diffusion.normalizer_std,
                    "image_size": 32, "num_channels": 3}, cls.diffusion_path)
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        cls.client.close()
        cls.directory.cleanup()

    def tearDown(self):
        energy_service.load_model.cache_clear()
        diffusion_service.load_model.cache_clear()

    def test_new_routes_return_real_png_after_checkpoint_loading(self):
        for route, service, path in (
            ("/generate-energy", energy_service, self.energy_path),
            ("/generate-diffusion", diffusion_service, self.diffusion_path),
        ):
            service.load_model.cache_clear()
            with patch.object(service, "MODEL_PATH", path):
                response = self.client.get(route, params={"num_samples": 1, "steps": 2})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["content-type"], "image/png")
            image = Image.open(io.BytesIO(response.content))
            image.load()
            self.assertEqual(image.size, (32, 32))
            self.assertEqual(image.mode, "RGB")

    def test_missing_checkpoints_return_503(self):
        for route, service in (("/generate-energy", energy_service),
                               ("/generate-diffusion", diffusion_service)):
            service.load_model.cache_clear()
            with patch.object(service, "MODEL_PATH", Path(self.directory.name) / "absent.pth"):
                response = self.client.get(route, params={"num_samples": 1, "steps": 1})
            self.assertEqual(response.status_code, 503)
            self.assertIn("Train and export", response.json()["detail"])

    def test_new_routes_reject_invalid_queries(self):
        for route in ("/generate-energy", "/generate-diffusion"):
            for query in ({"num_samples": 0}, {"num_samples": 17},
                          {"steps": 0}, {"steps": 201}, {"steps": "abc"}):
                response = self.client.get(route, params=query)
                self.assertEqual(response.status_code, 422)

    def test_existing_health_text_and_gan_routes(self):
        self.assertEqual(self.client.get("/health").json(), {"status": "ok"})
        response = self.client.post("/generate", json={"start_word": "The", "length": 5})
        self.assertEqual(response.status_code, 200)
        self.assertIn("generated_text", response.json())
        response = self.client.get("/generate-digits", params={"num_samples": 1})
        self.assertEqual(response.status_code, 200)
        image = Image.open(io.BytesIO(response.content))
        image.load()
        self.assertEqual(image.size, (28, 28))


if __name__ == "__main__":
    unittest.main()
