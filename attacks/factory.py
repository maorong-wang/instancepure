from attacks.purifier_attacks import (
    BPDAEOTPGDAttack,
    BPDAEOTPGDConfig,
    PurifierAAConfig,
    PurifierAutoAttack,
    PurifierPGDAttack,
    PurifierPGDConfig,
)
from attacks.diffattack import DiffAttackAttack, DiffAttackConfig
from attacks.diffhammer import DiffHammerAttack, DiffHammerConfig
from attacks.base import IdentityAttack
from attacks.sdedit import SDEditDiffusionPGDAttack
from attacks.standard import StandardModelAttack


def gen_pgd_confs(eps, alpha, iter, input_range=(0, 1)):
    scale = float(input_range[1] - input_range[0]) / 255.0
    return {
        "eps": eps * scale,
        "alpha": alpha * scale,
        "iter": iter,
        "input_range": input_range,
    }


def _resolve_diffattack_config(args, pgd_conf):
    config = DiffAttackConfig(
        eps=pgd_conf["eps"],
        n_iter=getattr(args, "diffattack_n_iter", getattr(args, "atk_iter", 100)),
        n_restarts=getattr(args, "diffattack_n_restarts", 5),
        eot_iter=getattr(args, "diffattack_eot_iter", 5),
        rho=getattr(args, "diffattack_rho", 0.75),
        version=getattr(args, "diffattack_version", "rand"),
        attacks_to_run=getattr(args, "diffattack_attacks_to_run", ""),
        t_interval=getattr(args, "diffattack_t_interval", 10),
        use_trace_loss=getattr(args, "diffattack_use_trace_loss", True),
        trace_lambda=getattr(args, "diffattack_trace_lambda", 1.0),
        timestep=getattr(args, "diffpure_t", getattr(args, "diffusion_timestep", 150)),
        seed=getattr(args, "seed", 0),
    )
    preset = str(getattr(args, "diffattack_preset", "default")).lower()
    if preset == "fast":
        config.version = "rand"
        config.attacks_to_run = ""
        config.n_iter = min(int(config.n_iter), 30) if int(config.n_iter) > 0 else 30
        config.n_restarts = 1
        config.eot_iter = 1
        config.use_trace_loss = False
    return config


def _resolve_diffhammer_config(args, pgd_conf):
    config = DiffHammerConfig(
        eps=pgd_conf["eps"],
        method=getattr(args, "diffhammer_method", "apgd"),
        n_iters=getattr(args, "diffhammer_n_iters", "50,50,50"),
        loss_names=getattr(args, "diffhammer_loss_names", "CW,CE,DLR"),
        n_restart=getattr(args, "diffhammer_n_restart", 3),
        n_eval=getattr(args, "diffhammer_n_eval", 10),
        n_eot=getattr(args, "diffhammer_n_eot", 5),
        grad_mode=getattr(args, "diffhammer_grad_mode", "bpda"),
        pgd_cmd=getattr(args, "diffhammer_pgd_cmd", ""),
        pgd_step_size=getattr(args, "diffhammer_pgd_step_size", 0.0),
        em=getattr(args, "diffhammer_em", True),
        em_alpha=getattr(args, "diffhammer_em_alpha", 0.5),
        em_lam=getattr(args, "diffhammer_em_lam", 5.0),
        em_steps=getattr(args, "diffhammer_em_steps", 5),
        seed=getattr(args, "seed", 0),
    )
    preset = str(getattr(args, "diffhammer_preset", "default")).lower()
    if preset == "fast":
        config.method = "apgd"
        config.n_iters = "30,30,30"
        config.loss_names = "CW,CE,DLR"
        config.n_restart = 3
        config.n_eval = 1
        config.n_eot = 1
        config.grad_mode = "bpda"
        config.em = False
    return config


def _resolve_purifier_pgd_config(args, pgd_conf):
    explicit_step_size = float(getattr(args, "purifier_pgd_step_size", 0.0))
    scale = float(pgd_conf["input_range"][1] - pgd_conf["input_range"][0]) / 255.0
    step_size = pgd_conf["alpha"] if explicit_step_size <= 0 else explicit_step_size * scale
    return PurifierPGDConfig(
        eps=pgd_conf["eps"],
        n_iter=getattr(args, "atk_iter", 40),
        step_size=step_size,
        random_start=getattr(args, "purifier_pgd_random_start", False),
        seed=getattr(args, "seed", 0),
    )


def _resolve_purifier_aa_config(args, pgd_conf):
    return PurifierAAConfig(
        eps=pgd_conf["eps"],
        aa_version=getattr(args, "purifier_aa_version", "rand"),
        n_iter=getattr(args, "purifier_aa_n_iter", 100),
        seed=getattr(args, "seed", 0),
    )


def _resolve_bpda_eot_pgd_config(args, pgd_conf):
    explicit_step_size = float(getattr(args, "bpda_pgd_step_size", 0.0))
    scale = float(pgd_conf["input_range"][1] - pgd_conf["input_range"][0]) / 255.0
    step_size = pgd_conf["alpha"] if explicit_step_size <= 0 else explicit_step_size * scale
    return BPDAEOTPGDConfig(
        eps=pgd_conf["eps"],
        n_iter=getattr(args, "atk_iter", 40),
        step_size=step_size,
        eot_iter=getattr(args, "bpda_eot_iter", 5),
        random_start=getattr(args, "bpda_pgd_random_start", False),
        seed=getattr(args, "seed", 0),
    )


def build_attack(args, raw_classifier, purified_classifier, purifier, pgd_conf, device):
    attack_name = str(getattr(args, "attack_method", "Linf_pgd")).lower()
    if attack_name in {"", "none"}:
        return IdentityAttack()

    if attack_name == "diffattack":
        config = _resolve_diffattack_config(args, pgd_conf)
        return DiffAttackAttack(
            purifier=purifier,
            classifier=raw_classifier,
            device=device,
            config=config,
        )

    if attack_name == "diffhammer":
        config = _resolve_diffhammer_config(args, pgd_conf)
        return DiffHammerAttack(
            purifier=purifier,
            classifier=raw_classifier,
            device=device,
            config=config,
        )

    if attack_name == "purifier_pgd":
        config = _resolve_purifier_pgd_config(args, pgd_conf)
        return PurifierPGDAttack(
            model=raw_classifier,
            device=device,
            config=config,
        )

    if attack_name == "purifier_aa":
        config = _resolve_purifier_aa_config(args, pgd_conf)
        return PurifierAutoAttack(
            model=raw_classifier,
            device=device,
            config=config,
        )

    if attack_name == "bpda_eot_pgd":
        config = _resolve_bpda_eot_pgd_config(args, pgd_conf)
        return BPDAEOTPGDAttack(
            purifier=purifier,
            classifier=raw_classifier,
            device=device,
            config=config,
        )

    if attack_name == "diff_pgd" or getattr(args, "attack_version", "v1") == "v2":
        return SDEditDiffusionPGDAttack(
            purifier=purifier,
            classifier=raw_classifier,
            pgd_conf=pgd_conf,
            device=device,
            timestep=getattr(args, "diffusion_timestep", 150),
        )

    attack_target = getattr(args, "attack_target", "victim")
    attack_model = purified_classifier if attack_target == "purified" else raw_classifier
    return StandardModelAttack(
        model=attack_model,
        attack_name=attack_name,
        device=device,
        pgd_conf=pgd_conf,
        stadv_num_iterations=getattr(args, "stadv_num_iterations", 100),
        stadv_eot_iter=getattr(args, "stadv_eot_iter", 5),
    )
