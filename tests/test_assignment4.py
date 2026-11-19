import importlib.util
import io
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image


torch.set_num_threads(2)


class Assignment4Tests(unittest.TestCase):
    def test_modules_available(self):
        for name in ("helper_lib.energy", "helper_lib.diffusion"):
            self.assertIsNotNone(importlib.util.find_spec(name), name)

    def test_sampling_lowers_energy_without_updating_weights(self):
        from helper_lib.energy import generate_samples

        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3072, 1))
        with torch.no_grad():
            model[1].weight.fill_(0.01)
            model[1].bias.zero_()
        model.train()
        inputs = torch.full((2, 3, 32, 32), 0.5)
        weights = model[1].weight.detach().clone()
        initial_energy = model(inputs).mean().item()
        with torch.no_grad():
            samples = generate_samples(model, inputs, 2, 1.0, 0.0)
        self.assertLess(model(samples).mean().item(), initial_energy)
        self.assertTrue(torch.equal(weights, model[1].weight))
        self.assertIsNone(model[1].weight.grad)
        self.assertTrue(model.training)
        self.assertFalse(samples.requires_grad)

    def test_energy_training_and_variable_batch_buffer(self):
        from helper_lib.energy import EBM, EnergyModel

        model = EnergyModel()
        ebm = EBM(model, 0.1, 2, 1.0, 0.005, "cpu")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        before = model.conv1.weight.detach().clone()
        for size in (3, 1):
            values = ebm.train_step(torch.rand(size, 3, 32, 32) * 2 - 1, optimizer)
            self.assertTrue(all(math.isfinite(v) for v in values.values()))
        self.assertFalse(torch.equal(before, model.conv1.weight))
        self.assertEqual(model(torch.zeros(2, 3, 32, 32)).shape, (2, 1))
        self.assertTrue(all(not x.requires_grad for x in ebm.buffer.examples))
        with patch("numpy.random.binomial", return_value=1):
            samples = ebm.buffer.sample_new_exmps(1, 2, 1.0, 0.005)
        self.assertEqual(samples.shape, (1, 3, 32, 32))

    def test_schedule_is_per_image_and_has_finite_endpoints(self):
        from helper_lib.diffusion import offset_cosine_diffusion_schedule

        times = torch.tensor([0.0, 0.5, 1.0]).view(3, 1, 1, 1)
        noise, signal = offset_cosine_diffusion_schedule(times)
        self.assertTrue(torch.allclose(noise.square() + signal.square(), torch.ones_like(times)))
        self.assertAlmostEqual(signal[0].item(), 0.95, places=5)
        self.assertAlmostEqual(signal[-1].item(), 0.02, places=5)
        one_noise, one_signal = offset_cosine_diffusion_schedule(times[1:2])
        self.assertTrue(torch.equal(signal[1:2], one_signal))
        self.assertTrue(torch.equal(noise[1:2], one_noise))

    def test_diffusion_trains_and_updates_ema(self):
        from helper_lib.diffusion import DiffusionModel, UNet, offset_cosine_diffusion_schedule

        model = DiffusionModel(UNet(32, 3), offset_cosine_diffusion_schedule)
        model.set_normalizer([0.4, 0.5, 0.6], [0.2, 0.3, 0.4])
        before = model.ema_network.final.weight.detach().clone()
        optimizer = torch.optim.Adam(model.network.parameters(), lr=0.001)
        loss = model.train_step(torch.rand(2, 3, 32, 32), optimizer, torch.nn.MSELoss())
        self.assertTrue(math.isfinite(loss))
        self.assertFalse(torch.equal(before, model.ema_network.final.weight))
        expected = before * 0.8 + model.network.final.weight.detach() * 0.2
        self.assertTrue(torch.allclose(model.ema_network.final.weight, expected))
        self.assertTrue(all(p.grad is None for p in model.ema_network.parameters()))
        samples = model.generate(2, 2)
        self.assertEqual(samples.shape, (2, 3, 32, 32))
        self.assertTrue(torch.isfinite(samples).all())
        self.assertGreaterEqual(samples.min().item(), 0)
        self.assertLessEqual(samples.max().item(), 1)

    def test_services_load_checkpoints_and_return_rgb_png(self):
        from helper_lib.energy import EnergyModel
        from helper_lib.diffusion import DiffusionModel, UNet, offset_cosine_diffusion_schedule
        from app import energy_service, diffusion_service

        with tempfile.TemporaryDirectory() as directory:
            energy_path = Path(directory) / "energy.pth"
            diffusion_path = Path(directory) / "diffusion.pth"
            torch.save({"model_state_dict": EnergyModel().state_dict(), "image_size": 32,
                        "num_channels": 3, "step_size": 10.0, "noise": 0.005}, energy_path)
            model = DiffusionModel(UNet(32, 3), offset_cosine_diffusion_schedule)
            model.set_normalizer([0.4, 0.5, 0.6], [0.2, 0.3, 0.4])
            torch.save({"ema_model_state_dict": model.ema_network.state_dict(),
                        "normalizer_mean": model.normalizer_mean,
                        "normalizer_std": model.normalizer_std,
                        "image_size": 32, "num_channels": 3}, diffusion_path)
            for service, path, name in ((energy_service, energy_path, "generate_energy_images"),
                                         (diffusion_service, diffusion_path, "generate_diffusion_images")):
                service.load_model.cache_clear()
                with patch.object(service, "MODEL_PATH", path):
                    result = getattr(service, name)(num_samples=2, steps=2)
                    image = Image.open(io.BytesIO(result))
                    image.load()
                    self.assertEqual(image.format, "PNG")
                    self.assertEqual(image.mode, "RGB")
                service.load_model.cache_clear()

    def test_missing_weights_do_not_generate_random_model_outputs(self):
        from app import energy_service, diffusion_service

        for service in (energy_service, diffusion_service):
            service.load_model.cache_clear()
            with patch.object(service, "MODEL_PATH", Path("missing_assignment4_weights.pth")):
                with self.assertRaises(FileNotFoundError):
                    service.load_model()


if __name__ == "__main__":
    unittest.main()
