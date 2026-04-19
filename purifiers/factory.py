from purifiers.base import IdentityPurifier


def build_purifier(args, device):
    purifier_name = getattr(args, "purifier_name", "instantpure").lower()
    if purifier_name == "none":
        return IdentityPurifier()

    if purifier_name == "instantpure":
        from purifiers.instantpure import InstantPureConfig, InstantPurePurifier

        config = InstantPureConfig(
            model_name=getattr(args, "model", "LCM"),
            load_origin_lora=getattr(args, "load_origin_lora", False),
            lora_input_dir=getattr(args, "lora_input_dir", None),
            num_inference_step=getattr(args, "num_inference_step", 1),
            strength=getattr(args, "strength", 0.1),
            seed=getattr(args, "seed", 3407),
            guidance_scale=getattr(args, "guidance_scale", 1.0),
            control_scale=getattr(args, "control_scale", 0.8),
            diffusion_respace=getattr(args, "diffusion_respace", "ddim50"),
            diffusion_timestep=getattr(args, "diffusion_timestep", 150),
        )
        return InstantPurePurifier(config, device=device)

    if purifier_name in {"instancepure", "puriflow"}:
        raise NotImplementedError(
            f"Purifier '{purifier_name}' is not wired in this repository yet. "
            "The refactor adds the purifier layer and registry, but the external backend is not present locally."
        )

    raise ValueError(f"Unknown purifier '{purifier_name}'.")
