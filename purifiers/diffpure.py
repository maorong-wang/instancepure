from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from purifiers.base import BasePurifier
from purifiers.guided_diffusion_imagenet import load_imagenet_guided_diffusion


def _extract_into_tensor(arr_or_func, timesteps, broadcast_shape):
    if callable(arr_or_func):
        values = arr_or_func(timesteps).float()
    else:
        values = arr_or_func.to(device=timesteps.device)[timesteps].float()
    while len(values.shape) < len(broadcast_shape):
        values = values[..., None]
    return values.expand(broadcast_shape)


class _ReverseVPSDE(nn.Module):
    def __init__(self, model, beta_min=0.1, beta_max=20.0, num_steps=1000, image_shape=(3, 256, 256)):
        super().__init__()
        self.model = model
        self.beta_0 = float(beta_min)
        self.beta_1 = float(beta_max)
        self.num_steps = int(num_steps)
        self.image_shape = image_shape
        self.discrete_betas = torch.linspace(beta_min / num_steps, beta_max / num_steps, num_steps)
        self.alphas = 1.0 - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_one_minus_alphas_cumprod_neg_recip_cont = (
            lambda t: -1.0
            / torch.sqrt(torch.clamp_min(1.0 - self.alphas_cumprod_cont(t), 1e-12))
        )
        self.noise_type = "diagonal"
        self.sde_type = "ito"

    def alphas_cumprod_cont(self, t):
        return torch.exp(-0.5 * (self.beta_1 - self.beta_0) * t.square() - self.beta_0 * t)

    def _scale_timesteps(self, t):
        return (t.float() * self.num_steps).long().clamp_(0, self.num_steps - 1)

    def vpsde_fn(self, t, x):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t[:, None] * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def reverse_drift(self, t, x):
        drift, diffusion = self.vpsde_fn(t, x)
        x_image = x.view(-1, *self.image_shape)
        discrete_steps = self._scale_timesteps(t)
        model_output = self.model(x_image, discrete_steps)
        model_output, _ = torch.split(model_output, self.image_shape[0], dim=1)
        score = _extract_into_tensor(
            self.sqrt_one_minus_alphas_cumprod_neg_recip_cont,
            t,
            x.shape,
        ) * model_output.view(x.shape[0], -1)
        return drift - diffusion[:, None].square() * score

    def f(self, t, x):
        t = t.expand(x.shape[0])
        return -self.reverse_drift(1 - t, x)

    def g(self, t, x):
        t = t.expand(x.shape[0])
        _, diffusion = self.vpsde_fn(1 - t, x)
        return diffusion[:, None].expand_as(x)


@dataclass
class DiffPureConfig:
    diffusion_type: str = "sde"
    sampling_method: str = "ddpm"
    sample_step: int = 1
    timestep: int = 150
    rand_t: bool = False
    t_delta: int = 15
    use_brownian: bool = False
    pretrained_root: str = "pretrained"
    checkpoint_path: Optional[str] = None
    use_fp16: bool = True


class DiffPurePurifier(BasePurifier):
    name = "diffpure"
    supports_sdedit_attack = True

    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = torch.device(device)
        self.model, self.diffusion, self.betas, self.checkpoint_path = load_imagenet_guided_diffusion(
            device=self.device,
            pretrained_root=config.pretrained_root,
            checkpoint_path=config.checkpoint_path,
            use_fp16=config.use_fp16,
            timestep_respacing="1000",
        )
        self.reverse_vpsde = _ReverseVPSDE(self.model).to(self.device)

    def _sample_timestep(self, timestep):
        timestep = max(1, min(int(timestep), 1000))
        if not self.config.rand_t:
            return timestep
        low = max(1, timestep - int(self.config.t_delta))
        high = min(1000, max(low, timestep + int(self.config.t_delta)))
        sampled = torch.randint(low, high + 1, (1,), device=self.device).item()
        return int(sampled)

    def _q_sample(self, x, timestep):
        return self._q_sample_with_noise(x, timestep, noise=None)

    def _q_sample_with_noise(self, x, timestep, noise=None):
        if timestep <= 0:
            return x
        t_tensor = torch.full((x.shape[0],), int(timestep) - 1, dtype=torch.long, device=x.device)
        return self.diffusion.q_sample(x, t_tensor, noise=noise)

    def _reverse_ddpm(self, sample, timestep):
        if timestep <= 0:
            return sample
        for step in reversed(range(int(timestep))):
            t_tensor = torch.full((sample.shape[0],), step, dtype=torch.long, device=sample.device)
            if self.config.sampling_method == "ddim":
                output = self.diffusion.ddim_sample(self.model, sample, t_tensor)
            else:
                output = self.diffusion.p_sample(self.model, sample, t_tensor)
            sample = output["sample"]
        return sample

    def _reverse_sde(self, sample, timestep):
        try:
            import torchsde
        except ImportError as exc:
            raise ImportError(
                "DiffPure with --diffpure_diffusion_type sde requires torchsde. "
                "Install torchsde or switch to --diffpure_diffusion_type ddpm."
            ) from exc

        if timestep <= 0:
            return sample

        batch_size = sample.shape[0]
        flat_sample = sample.view(batch_size, -1)
        state_size = flat_sample.shape[1]
        t0 = 1.0 - float(timestep) / 1000.0
        t1 = 1.0 - 1.0e-5
        times = torch.linspace(t0, t1, 2, device=sample.device, dtype=torch.float32)
        kwargs = {}
        if self.config.use_brownian:
            kwargs["bm"] = torchsde.BrownianInterval(
                t0=t0,
                t1=t1,
                size=(batch_size, state_size),
                device=sample.device,
            )
        result = torchsde.sdeint_adjoint(
            self.reverse_vpsde,
            flat_sample,
            times,
            method="euler",
            **kwargs,
        )
        return result[-1].view_as(sample)

    def _run_diffpure(self, x, timestep):
        purified = x
        for _ in range(max(int(self.config.sample_step), 1)):
            current_t = self._sample_timestep(timestep)
            sample = self._q_sample(purified, current_t)
            if self.config.diffusion_type == "sde":
                purified = self._reverse_sde(sample, current_t)
            else:
                purified = self._reverse_ddpm(sample, current_t)
        return purified

    def _build_forward_reference(self, x, timestep, noise):
        if timestep <= 0:
            return [x]
        references = []
        for step in reversed(range(max(int(timestep) - 1, 0))):
            t_tensor = torch.full((x.shape[0],), step, dtype=torch.long, device=x.device)
            references.append(self.diffusion.q_sample(x, t_tensor, noise=noise))
        references.append(x)
        return references

    def _reverse_ddpm_trace(self, sample, timestep):
        if timestep <= 0:
            return sample, [sample]
        mid_x = []
        current = sample
        for step in reversed(range(int(timestep))):
            t_tensor = torch.full((current.shape[0],), step, dtype=torch.long, device=current.device)
            if self.config.sampling_method == "ddim":
                output = self.diffusion.ddim_sample(self.model, current, t_tensor)
            else:
                output = self.diffusion.p_sample(self.model, current, t_tensor)
            current = output["sample"]
            mid_x.append(current)
        return current, mid_x

    def _reverse_sde_trace(self, sample, timestep):
        try:
            import torchsde
        except ImportError as exc:
            raise ImportError(
                "DiffPure trace mode requires torchsde when --diffpure_diffusion_type sde is active. "
                "Install torchsde or switch to --diffpure_diffusion_type ddpm."
            ) from exc

        if timestep <= 0:
            return sample, [sample]

        batch_size = sample.shape[0]
        flat_sample = sample.view(batch_size, -1)
        state_size = flat_sample.shape[1]
        t0 = 1.0 - float(timestep) / 1000.0
        t1 = 1.0 - 1.0e-5
        times = torch.linspace(t0, t1, int(timestep) + 1, device=sample.device, dtype=torch.float32)
        kwargs = {}
        if self.config.use_brownian:
            kwargs["bm"] = torchsde.BrownianInterval(
                t0=t0,
                t1=t1,
                size=(batch_size, state_size),
                device=sample.device,
            )
        result = torchsde.sdeint_adjoint(
            self.reverse_vpsde,
            flat_sample,
            times,
            method="euler",
            **kwargs,
        )
        mid_x = [result[index].view_as(sample) for index in range(1, result.shape[0])]
        return mid_x[-1], mid_x

    def attack_trace(self, x, timestep=None, to_01=True):
        original_size = x.shape[-2:]
        diffusion_input = x
        if original_size != (256, 256):
            diffusion_input = F.interpolate(
                diffusion_input,
                size=(256, 256),
                mode="bilinear",
                align_corners=False,
            )
        diffusion_input = diffusion_input.clamp(0.0, 1.0)
        diffusion_input = diffusion_input * 2.0 - 1.0

        current = diffusion_input
        last_mid_x = []
        last_ori_x = []
        for _ in range(max(int(self.config.sample_step), 1)):
            current_t = self._sample_timestep(self.config.timestep if timestep is None else timestep)
            noise = torch.randn_like(current)
            sample = self._q_sample_with_noise(current, current_t, noise=noise)
            last_ori_x = self._build_forward_reference(current, current_t, noise)
            if self.config.diffusion_type == "sde":
                current, last_mid_x = self._reverse_sde_trace(sample, current_t)
            else:
                current, last_mid_x = self._reverse_ddpm_trace(sample, current_t)

        purified = current
        if to_01:
            purified = (purified + 1.0) * 0.5
        if original_size != (256, 256):
            purified = F.interpolate(
                purified,
                size=original_size,
                mode="bilinear",
                align_corners=False,
            )
        return purified, last_mid_x, last_ori_x

    def sdedit(self, x, timestep=None, to_01=True):
        original_size = x.shape[-2:]
        diffusion_input = x
        if original_size != (256, 256):
            diffusion_input = F.interpolate(
                diffusion_input,
                size=(256, 256),
                mode="bilinear",
                align_corners=False,
            )
        diffusion_input = diffusion_input.clamp(0.0, 1.0)
        diffusion_input = diffusion_input * 2.0 - 1.0
        purified = self._run_diffpure(
            diffusion_input,
            timestep=self.config.timestep if timestep is None else timestep,
        )
        if to_01:
            purified = (purified + 1.0) * 0.5
        if original_size != (256, 256):
            purified = F.interpolate(
                purified,
                size=original_size,
                mode="bilinear",
                align_corners=False,
            )
        return purified

    def purify(self, x):
        return self.sdedit(x, timestep=self.config.timestep, to_01=True)
