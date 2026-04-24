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

    if purifier_name == "diffpure":
        from purifiers.diffpure import DiffPureConfig, DiffPurePurifier

        config = DiffPureConfig(
            diffusion_type=getattr(args, "diffpure_diffusion_type", "sde"),
            sampling_method=getattr(args, "diffpure_sampling_method", "ddpm"),
            sample_step=getattr(args, "diffpure_sample_step", 1),
            timestep=getattr(args, "diffpure_t", 150),
            rand_t=getattr(args, "diffpure_rand_t", False),
            t_delta=getattr(args, "diffpure_t_delta", 15),
            use_brownian=getattr(args, "diffpure_use_brownian", False),
            pretrained_root=getattr(
                args,
                "guided_diffusion_pretrained_root",
                "/home_fmg/maorong/python/DiffPure/pretrained",
            ),
            checkpoint_path=getattr(args, "guided_diffusion_checkpoint_path", None),
            use_fp16=getattr(args, "guided_diffusion_use_fp16", False),
        )
        return DiffPurePurifier(config, device=device)

    if purifier_name == "mimicdiffusion":
        from purifiers.mimicdiffusion import MimicDiffusionConfig, MimicDiffusionPurifier

        config = MimicDiffusionConfig(
            max_timesteps=getattr(args, "mimicdiffusion_max_timesteps", "1000"),
            num_denoising_steps=getattr(args, "mimicdiffusion_num_denoising_steps", "100"),
            sampling_method=getattr(args, "mimicdiffusion_sampling_method", "ddpm"),
            rho_scale=getattr(args, "mimicdiffusion_rho_scale", 3000.0),
            guidance_start_step=getattr(args, "mimicdiffusion_guidance_start_step", 20),
            guidance_end_step=getattr(args, "mimicdiffusion_guidance_end_step", 90),
            projection_scale=getattr(args, "mimicdiffusion_projection_scale", 4),
            pretrained_root=getattr(
                args,
                "guided_diffusion_pretrained_root",
                "/home_fmg/maorong/python/DiffPure/pretrained",
            ),
            checkpoint_path=getattr(args, "guided_diffusion_checkpoint_path", None),
            use_fp16=getattr(args, "guided_diffusion_use_fp16", False),
        )
        return MimicDiffusionPurifier(config, device=device)

    if purifier_name in {"instancepure", "puriflow"}:
        raise NotImplementedError(
            f"Purifier '{purifier_name}' is not wired in this repository yet. "
            "The refactor adds the purifier layer and registry, but the external backend is not present locally."
        )

    raise ValueError(f"Unknown purifier '{purifier_name}'.")
