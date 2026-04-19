import sys
from pathlib import Path

import torch

from attacks.base import BaseAttack

try:
    import foolbox as fb
except ImportError:
    fb = None

try:
    from autoattack import AutoAttack
except ImportError:
    AutoAttack = None


def load_diffpure_stadv_attack():
    diffpure_root = Path(__file__).resolve().parent.parent.parent / "DiffPure"
    if not diffpure_root.exists():
        raise ImportError(f"DiffPure was not found at {diffpure_root}.")
    diffpure_root_str = str(diffpure_root)
    if diffpure_root_str not in sys.path:
        sys.path.insert(0, diffpure_root_str)
    from stadv_eot.attacks import StAdvAttack

    return StAdvAttack


class StandardModelAttack(BaseAttack):
    def __init__(self, model, attack_name, device, pgd_conf, stadv_num_iterations=100, stadv_eot_iter=20):
        self.model = model
        self.attack_name = str(attack_name).lower()
        self.device = device
        self.pgd_conf = pgd_conf
        self.stadv_num_iterations = stadv_num_iterations
        self.stadv_eot_iter = stadv_eot_iter

    def _run_foolbox_pgd(self, x, criterion, attack_cls, eps, abs_stepsize):
        if fb is None:
            raise ImportError("foolbox is not installed in the active environment.")
        fmodel = fb.PyTorchModel(self.model, bounds=(0, 1), device=self.device)
        attack = attack_cls(
            steps=self.pgd_conf["iter"],
            random_start=False,
            abs_stepsize=abs_stepsize,
        )
        _, clipped_advs, _ = attack(fmodel, x, criterion, epsilons=[eps])
        return clipped_advs[0] if isinstance(clipped_advs, (list, tuple)) else clipped_advs[0]

    def run(self, x, y):
        if self.attack_name == "linf_pgd":
            return self._run_foolbox_pgd(
                x,
                y,
                fb.attacks.LinfPGD,
                eps=self.pgd_conf["eps"],
                abs_stepsize=self.pgd_conf["alpha"],
            ).to(self.device)

        if self.attack_name == "l2_pgd":
            return self._run_foolbox_pgd(
                x,
                y,
                fb.attacks.L2PGD,
                eps=0.5,
                abs_stepsize=0.1,
            ).to(self.device)

        if self.attack_name == "stadv":
            stadv_attack_cls = load_diffpure_stadv_attack()
            adversary = stadv_attack_cls(
                self.model,
                bound=self.pgd_conf["eps"],
                num_iterations=self.stadv_num_iterations,
                eot_iter=self.stadv_eot_iter,
            )
            return adversary(x, y)

        if self.attack_name == "autoattack":
            if AutoAttack is None:
                raise ImportError("autoattack is not installed in the active environment.")
            adversary = AutoAttack(
                self.model,
                norm="Linf",
                eps=self.pgd_conf["eps"],
                version="standard",
                device=self.device,
            )
            return adversary.run_standard_evaluation(x, y, bs=x.size(0))

        if self.attack_name == "target_linf_pgd":
            label_offset = torch.randint(low=1, high=1000, size=y.shape, device=self.device)
            random_target = torch.remainder(y + label_offset, 1000)
            return self._run_foolbox_pgd(
                x,
                fb.criteria.TargetedMisclassification(random_target),
                fb.attacks.LinfPGD,
                eps=self.pgd_conf["eps"],
                abs_stepsize=self.pgd_conf["alpha"],
            ).to(self.device)

        raise NotImplementedError(f"Unknown standard attack '{self.attack_name}'.")
