import random

import numpy as np
import torch
import torch.nn as nn


def swish(x):
    return x * torch.sigmoid(x)


class EnergyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 2 * 2, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = swish(self.conv1(x))
        x = swish(self.conv2(x))
        x = swish(self.conv3(x))
        x = swish(self.conv4(x))
        x = self.flatten(x)
        x = swish(self.fc1(x))
        return self.fc2(x)


def generate_samples(nn_energy_model, inp_imgs, steps, step_size, noise_std):
    if steps < 1:
        raise ValueError("steps must be positive.")
    was_training = nn_energy_model.training
    nn_energy_model.eval()
    inp_imgs = inp_imgs.detach()
    try:
        for _ in range(steps):
            with torch.no_grad():
                noise = torch.randn_like(inp_imgs) * noise_std
                inp_imgs = (inp_imgs + noise).clamp(-1.0, 1.0)
            with torch.enable_grad():
                inp_imgs.requires_grad_(True)
                energy = nn_energy_model(inp_imgs)
                grads, = torch.autograd.grad(
                    energy, inp_imgs, grad_outputs=torch.ones_like(energy)
                )
            with torch.no_grad():
                grads = grads.clamp(-0.03, 0.03)
                inp_imgs = (inp_imgs - step_size * grads).clamp(-1.0, 1.0)
    finally:
        nn_energy_model.train(was_training)
    return inp_imgs.detach()


class Buffer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.examples = [
            torch.rand((1, 3, 32, 32), device=device) * 2 - 1
            for _ in range(128)
        ]

    def sample_new_exmps(self, batch_size, steps, step_size, noise):
        n_new = np.random.binomial(batch_size, 0.05)
        new_rand_imgs = torch.rand((n_new, 3, 32, 32), device=self.device) * 2 - 1
        old_examples = random.choices(self.examples, k=batch_size - n_new)
        if old_examples:
            old_imgs = torch.cat(old_examples, dim=0)
            inp_imgs = torch.cat([new_rand_imgs, old_imgs], dim=0)
        else:
            inp_imgs = new_rand_imgs
        new_imgs = generate_samples(self.model, inp_imgs, steps, step_size, noise)
        self.examples = [img.clone() for img in new_imgs.split(1)] + self.examples
        self.examples = self.examples[:8192]
        return new_imgs


class Metric:
    def __init__(self):
        self.reset()

    def update(self, val, count=1):
        self.total += val.item() * count
        self.count += count

    def result(self):
        return self.total / self.count if self.count > 0 else 0.0

    def reset(self):
        self.total = 0.0
        self.count = 0


class EBM(nn.Module):
    def __init__(self, model, alpha, steps, step_size, noise, device):
        super().__init__()
        self.device = device
        self.model = model.to(device)
        self.buffer = Buffer(self.model, device)
        self.alpha = alpha
        self.steps = steps
        self.step_size = step_size
        self.noise = noise
        self.loss_metric = Metric()
        self.reg_loss_metric = Metric()
        self.cdiv_loss_metric = Metric()
        self.real_out_metric = Metric()
        self.fake_out_metric = Metric()

    def metrics(self):
        return {
            "loss": self.loss_metric.result(),
            "reg": self.reg_loss_metric.result(),
            "cdiv": self.cdiv_loss_metric.result(),
            "real": self.real_out_metric.result(),
            "fake": self.fake_out_metric.result(),
        }

    def reset_metrics(self):
        for metric in (self.loss_metric, self.reg_loss_metric, self.cdiv_loss_metric,
                       self.real_out_metric, self.fake_out_metric):
            metric.reset()

    def train_step(self, real_imgs, optimizer):
        self.model.train()
        real_imgs = real_imgs.to(self.device)
        real_imgs = (real_imgs + torch.randn_like(real_imgs) * self.noise).clamp(-1, 1)
        batch_size = real_imgs.size(0)
        fake_imgs = self.buffer.sample_new_exmps(
            batch_size, self.steps, self.step_size, self.noise
        )
        inp_imgs = torch.cat([real_imgs, fake_imgs], dim=0).detach()
        out_scores = self.model(inp_imgs)
        real_out, fake_out = torch.split(out_scores, [batch_size, batch_size], dim=0)
        cdiv_loss = real_out.mean() - fake_out.mean()
        reg_loss = self.alpha * (real_out.square().mean() + fake_out.square().mean())
        loss = cdiv_loss + reg_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
        optimizer.step()
        self.loss_metric.update(loss, batch_size)
        self.reg_loss_metric.update(reg_loss, batch_size)
        self.cdiv_loss_metric.update(cdiv_loss, batch_size)
        self.real_out_metric.update(real_out.mean(), batch_size)
        self.fake_out_metric.update(fake_out.mean(), batch_size)
        return self.metrics()

    def test_step(self, real_imgs):
        self.model.eval()
        real_imgs = real_imgs.to(self.device)
        batch_size = real_imgs.size(0)
        fake_imgs = torch.rand_like(real_imgs) * 2 - 1
        with torch.no_grad():
            out_scores = self.model(torch.cat([real_imgs, fake_imgs], dim=0))
            real_out, fake_out = out_scores.split(batch_size)
            cdiv = real_out.mean() - fake_out.mean()
        self.cdiv_loss_metric.update(cdiv, batch_size)
        self.real_out_metric.update(real_out.mean(), batch_size)
        self.fake_out_metric.update(fake_out.mean(), batch_size)
        return {
            "cdiv": self.cdiv_loss_metric.result(),
            "real": self.real_out_metric.result(),
            "fake": self.fake_out_metric.result(),
        }
