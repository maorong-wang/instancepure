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


def build_attack(args, raw_classifier, purified_classifier, purifier, pgd_conf, device):
    attack_name = str(getattr(args, "attack_method", "Linf_pgd")).lower()
    if attack_name in {"", "none"}:
        return IdentityAttack()

    if attack_name in {"diffhammer", "diffattack"}:
        raise NotImplementedError(
            f"Attack '{attack_name}' is not wired in this repository yet. "
            "The refactor adds the attack layer and registry, but the external backend is not present locally."
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
        stadv_eot_iter=getattr(args, "stadv_eot_iter", 20),
    )
