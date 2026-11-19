import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalEmbedding(nn.Module):
    def __init__(self, num_frequencies=16):
        super().__init__()
        self.num_frequencies = num_frequencies
        frequencies = torch.exp(
            torch.linspace(math.log(1.0), math.log(1000.0), num_frequencies)
        )
        self.register_buffer(
            "angular_speeds", 2.0 * math.pi * frequencies.view(1, 1, 1, -1)
        )

    def forward(self, x):
        x = x.expand(-1, 1, 1, self.num_frequencies)
        sin_part = torch.sin(self.angular_speeds * x)
        cos_part = torch.cos(self.angular_speeds * x)
        return torch.cat([sin_part, cos_part], dim=-1)


def offset_cosine_diffusion_schedule(diffusion_times, min_signal_rate=0.02,
                                   max_signal_rate=0.95):
    start_angle = math.acos(max_signal_rate)
    end_angle = math.acos(min_signal_rate)
    diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)
    return torch.sin(diffusion_angles), torch.cos(diffusion_angles)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        if in_channels != out_channels:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.proj = nn.Identity()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, x):
        residual = self.proj(x)
        x = self.swish(self.conv1(x))
        x = self.conv2(x)
        return x + residual


class DownBlock(nn.Module):
    def __init__(self, width, block_depth, in_channels):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(block_depth):
            self.blocks.append(ResidualBlock(in_channels, width))
            in_channels = width
        self.pool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x, skips):
        for block in self.blocks:
            x = block(x)
            skips.append(x)
        return self.pool(x)


class UpBlock(nn.Module):
    def __init__(self, width, block_depth, in_channels):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(block_depth):
            self.blocks.append(ResidualBlock(in_channels + width, width))
            in_channels = width

    def forward(self, x, skips):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        for block in self.blocks:
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = block(x)
        return x


class UNet(nn.Module):
    def __init__(self, image_size, num_channels, embedding_dim=32):
        super().__init__()
        if image_size % 8 != 0 or embedding_dim % 2 != 0:
            raise ValueError("image_size must be divisible by 8 and embedding_dim must be even.")
        self.initial = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.num_channels = num_channels
        self.image_size = image_size
        self.embedding_dim = embedding_dim
        self.embedding = SinusoidalEmbedding(num_frequencies=embedding_dim // 2)
        self.down1 = DownBlock(32, in_channels=32 + embedding_dim, block_depth=2)
        self.down2 = DownBlock(64, in_channels=32, block_depth=2)
        self.down3 = DownBlock(96, in_channels=64, block_depth=2)
        self.mid1 = ResidualBlock(96, 128)
        self.mid2 = ResidualBlock(128, 128)
        self.up1 = UpBlock(96, in_channels=128, block_depth=2)
        self.up2 = UpBlock(64, in_channels=96, block_depth=2)
        self.up3 = UpBlock(32, in_channels=64, block_depth=2)
        self.final = nn.Conv2d(32, num_channels, kernel_size=1)
        nn.init.zeros_(self.final.weight)

    def forward(self, noisy_images, noise_variances):
        skips = []
        x = self.initial(noisy_images)
        noise_emb = self.embedding(noise_variances)
        noise_emb = F.interpolate(
            noise_emb.permute(0, 3, 1, 2),
            size=(self.image_size, self.image_size), mode="nearest"
        )
        x = torch.cat([x, noise_emb], dim=1)
        x = self.down1(x, skips)
        x = self.down2(x, skips)
        x = self.down3(x, skips)
        x = self.mid1(x)
        x = self.mid2(x)
        x = self.up1(x, skips)
        x = self.up2(x, skips)
        x = self.up3(x, skips)
        return self.final(x)


class DiffusionModel(nn.Module):
    def __init__(self, model, schedule_fn):
        super().__init__()
        self.network = model
        self.ema_network = copy.deepcopy(model)
        self.ema_network.requires_grad_(False)
        self.ema_network.eval()
        self.ema_decay = 0.8
        self.schedule_fn = schedule_fn
        self.register_buffer("normalizer_mean", torch.zeros(1, model.num_channels, 1, 1))
        self.register_buffer("normalizer_std", torch.ones(1, model.num_channels, 1, 1))

    def set_normalizer(self, mean, std):
        device = self.normalizer_mean.device
        mean = torch.as_tensor(mean, dtype=torch.float32, device=device).reshape(1, -1, 1, 1)
        std = torch.as_tensor(std, dtype=torch.float32, device=device).reshape(1, -1, 1, 1)
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("Normalization requires finite means and positive standard deviations.")
        self.normalizer_mean.copy_(mean)
        self.normalizer_std.copy_(std)

    def denormalize(self, x):
        return (x * self.normalizer_std + self.normalizer_mean).clamp(0.0, 1.0)

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        if training:
            network = self.network
            network.train()
        else:
            network = self.ema_network
            network.eval()
        pred_noises = network(noisy_images, noise_rates.square())
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        if diffusion_steps < 1:
            raise ValueError("diffusion_steps must be positive.")
        step_size = 1.0 / diffusion_steps
        current_images = initial_noise
        for step in range(diffusion_steps):
            t = torch.full(
                (initial_noise.size(0), 1, 1, 1), 1 - step * step_size,
                device=initial_noise.device
            )
            noise_rates, signal_rates = self.schedule_fn(t)
            pred_noises, pred_images = self.denoise(
                current_images, noise_rates, signal_rates, training=False
            )
            next_noise_rates, next_signal_rates = self.schedule_fn((t - step_size).clamp(0, 1))
            current_images = next_signal_rates * pred_images + next_noise_rates * pred_noises
        return pred_images

    def generate(self, num_images, diffusion_steps, initial_noise=None):
        if initial_noise is None:
            initial_noise = torch.randn(
                num_images, self.network.num_channels,
                self.network.image_size, self.network.image_size,
                device=self.normalizer_mean.device
            )
        with torch.no_grad():
            return self.denormalize(self.reverse_diffusion(initial_noise, diffusion_steps))

    def train_step(self, images, optimizer, loss_fn):
        images = (images - self.normalizer_mean) / self.normalizer_std
        noises = torch.randn_like(images)
        diffusion_times = torch.rand((images.size(0), 1, 1, 1), device=images.device)
        noise_rates, signal_rates = self.schedule_fn(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises
        pred_noises, _ = self.denoise(noisy_images, noise_rates, signal_rates, training=True)
        loss = loss_fn(pred_noises, noises)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            for ema_param, param in zip(self.ema_network.parameters(), self.network.parameters()):
                ema_param.copy_(self.ema_decay * ema_param + (1 - self.ema_decay) * param)
        return loss.item()

    def test_step(self, images, loss_fn):
        with torch.no_grad():
            images = (images - self.normalizer_mean) / self.normalizer_std
            noises = torch.randn_like(images)
            diffusion_times = torch.rand((images.size(0), 1, 1, 1), device=images.device)
            noise_rates, signal_rates = self.schedule_fn(diffusion_times)
            noisy_images = signal_rates * images + noise_rates * noises
            pred_noises, _ = self.denoise(noisy_images, noise_rates, signal_rates, training=False)
            loss = loss_fn(pred_noises, noises)
        return loss.item()
