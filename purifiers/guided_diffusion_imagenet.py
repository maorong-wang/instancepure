from pathlib import Path

import torch

from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults


DEFAULT_GUIDED_DIFFUSION_PRETRAINED_ROOT = Path("/home_fmg/maorong/python/DiffPure/pretrained")
DEFAULT_GUIDED_DIFFUSION_CHECKPOINT_RELATIVE = Path("guided_diffusion/256x256_diffusion_uncond.pt")


def resolve_guided_diffusion_checkpoint(pretrained_root=None, checkpoint_path=None):
    if checkpoint_path:
        resolved_path = Path(checkpoint_path).expanduser()
    else:
        resolved_root = Path(pretrained_root or DEFAULT_GUIDED_DIFFUSION_PRETRAINED_ROOT).expanduser()
        resolved_path = resolved_root / DEFAULT_GUIDED_DIFFUSION_CHECKPOINT_RELATIVE
    if not resolved_path.is_file():
        raise FileNotFoundError(
            "Could not find the ImageNet guided-diffusion checkpoint at "
            f"{resolved_path}. Pass --guided_diffusion_checkpoint_path or "
            "--guided_diffusion_pretrained_root."
        )
    return resolved_path


def build_imagenet_guided_diffusion_config(use_fp16=True, timestep_respacing="1000"):
    config = model_and_diffusion_defaults()
    config.update(
        {
            "attention_resolutions": "32,16,8",
            "class_cond": False,
            "diffusion_steps": 1000,
            "image_size": 256,
            "learn_sigma": True,
            "noise_schedule": "linear",
            "num_channels": 256,
            "num_head_channels": 64,
            "num_res_blocks": 2,
            "resblock_updown": True,
            "rescale_timesteps": True,
            "timestep_respacing": timestep_respacing,
            "use_fp16": bool(use_fp16),
            "use_scale_shift_norm": True,
        }
    )
    return config


def load_imagenet_guided_diffusion(
    device,
    pretrained_root=None,
    checkpoint_path=None,
    use_fp16=True,
    timestep_respacing="1000",
):
    device = torch.device(device)
    model_config = build_imagenet_guided_diffusion_config(
        use_fp16=use_fp16,
        timestep_respacing=timestep_respacing,
    )
    model, diffusion = create_model_and_diffusion(
        **{key: model_config[key] for key in model_and_diffusion_defaults().keys()}
    )
    resolved_checkpoint = resolve_guided_diffusion_checkpoint(
        pretrained_root=pretrained_root,
        checkpoint_path=checkpoint_path,
    )
    state_dict = torch.load(resolved_checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)
    model.requires_grad_(False).eval().to(device)
    if use_fp16:
        model.convert_to_fp16()
    betas = torch.from_numpy(diffusion.betas).float().to(device)
    return model, diffusion, betas, resolved_checkpoint
