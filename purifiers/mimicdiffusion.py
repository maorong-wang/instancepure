from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from purifiers.base import BasePurifier
from purifiers.guided_diffusion_imagenet import load_imagenet_guided_diffusion


def _build_diffusion_schedule(max_timesteps, num_denoising_steps):
    timestep_values = [int(value) for value in str(max_timesteps).split(",") if value.strip()]
    denoising_values = [int(value) for value in str(num_denoising_steps).split(",") if value.strip()]
    if not timestep_values or not denoising_values:
        raise ValueError("MimicDiffusion requires non-empty max timestep and denoising-step schedules.")
    if len(timestep_values) != len(denoising_values):
        raise ValueError("MimicDiffusion max-timestep and denoising-step schedules must have the same length.")

    max_timestep_list = []
    diffusion_steps = []
    for timestep, denoise_steps in zip(timestep_values, denoising_values):
        if timestep <= 0 or denoise_steps <= 0:
            raise ValueError("MimicDiffusion schedule values must be positive.")
        stride = max(timestep // denoise_steps, 1)
        sequence = [step - 1 for step in range(stride, timestep + 1, stride)]
        if not sequence:
            sequence = [timestep - 1]
        max_timestep_list.append(timestep - 1)
        diffusion_steps.append(sequence)
    return max_timestep_list, diffusion_steps


def _upsample_projection(x, scale_factor):
    return F.interpolate(x, scale_factor=scale_factor, mode="bilinear", align_corners=False)


@dataclass
class MimicDiffusionConfig:
    max_timesteps: str = "1000"
    num_denoising_steps: str = "100"
    sampling_method: str = "ddpm"
    rho_scale: float = 3000.0
    guidance_start_step: int = 20
    guidance_end_step: int = 90
    projection_scale: int = 4
    pretrained_root: str = "pretrained"
    checkpoint_path: Optional[str] = None
    use_fp16: bool = True


class MimicDiffusionPurifier(BasePurifier):
    name = "mimicdiffusion"
    supports_sdedit_attack = False

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
        self.max_timestep_list, self.diffusion_steps = _build_diffusion_schedule(
            config.max_timesteps,
            config.num_denoising_steps,
        )
        self.eta = 0.0 if config.sampling_method == "ddim" else 1.0

    def _compute_alpha(self, timestep):
        beta = torch.cat([torch.zeros(1, device=self.betas.device), self.betas], dim=0)
        return (1.0 - beta).cumprod(dim=0).index_select(0, timestep + 1).view(-1, 1, 1, 1)

    def _get_noised_x(self, x, timestep):
        noise = torch.randn_like(x)
        if isinstance(timestep, int):
            timestep = torch.full((x.shape[0],), timestep, dtype=torch.long, device=x.device)
        alpha = (1.0 - self.betas).cumprod(dim=0).index_select(0, timestep).view(-1, 1, 1, 1)
        return x * alpha.sqrt() + noise * (1.0 - alpha).sqrt()

    def _denoising_process(self, sample, sequence, reference):
        sample_count = sample.shape[0]
        next_sequence = [-1] + list(sequence[:-1])
        xt = sample
        guidance_start = int(self.config.guidance_start_step)
        guidance_end = int(self.config.guidance_end_step)
        rho_scale = float(self.config.rho_scale)
        for count, (step, next_step) in enumerate(zip(reversed(sequence), reversed(next_sequence))):
            t = torch.full((sample_count,), step, dtype=torch.long, device=xt.device)
            next_t = torch.full((sample_count,), next_step, dtype=torch.long, device=xt.device)
            at = self._compute_alpha(t)
            at_next = self._compute_alpha(next_t)
            c1 = self.eta * torch.sqrt(((1.0 - at / at_next) * (1.0 - at_next) / (1.0 - at)).clamp_min(0.0))
            c2 = torch.sqrt(((1.0 - at_next) - c1.square()).clamp_min(0.0))

            use_guidance = guidance_start < count < guidance_end
            if use_guidance:
                with torch.enable_grad():
                    xt = xt.detach().requires_grad_(True)
                    et = self.model(xt, t)
                    et, _ = torch.split(et, 3, dim=1)
                    x0_t = (xt - et * (1.0 - at).sqrt()) / at.sqrt()
                    norm_measure = torch.norm(x0_t - reference, p=1, dim=1).mean()
                    projection_measure = torch.norm(
                        _upsample_projection(x0_t, self.config.projection_scale)
                        - _upsample_projection(reference, self.config.projection_scale),
                        p=1,
                        dim=1,
                    ).mean()
                    guidance_gradient = torch.autograd.grad(
                        norm_measure + projection_measure,
                        xt,
                    )[0].detach()
                rho = rho_scale * at.sqrt()
                guidance = rho * guidance_gradient
                et = et.detach()
                x0_t = x0_t.detach()
            else:
                xt = xt.detach()
                et = self.model(xt, t)
                et, _ = torch.split(et, 3, dim=1)
                x0_t = (xt - et * (1.0 - at).sqrt()) / at.sqrt()
                guidance = 0.0

            xt = at_next.sqrt() * x0_t + c1 * torch.randn_like(xt) + c2 * et - guidance
        return xt

    def purify(self, x):
        original_size = x.shape[-2:]
        diffusion_input = x.clamp(0.0, 1.0)
        if original_size != (256, 256):
            diffusion_input = F.interpolate(
                diffusion_input,
                size=(256, 256),
                mode="bilinear",
                align_corners=False,
            )
        diffusion_input = diffusion_input * 2.0 - 1.0

        purified = diffusion_input
        for max_timestep, sequence in zip(self.max_timestep_list, self.diffusion_steps):
            noised = self._get_noised_x(purified, max_timestep)
            purified = self._denoising_process(noised, sequence, reference=purified)

        purified = (purified + 1.0) * 0.5
        if original_size != (256, 256):
            purified = F.interpolate(
                purified,
                size=original_size,
                mode="bilinear",
                align_corners=False,
            )
        return purified
