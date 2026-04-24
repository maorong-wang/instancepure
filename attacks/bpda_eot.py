from dataclasses import dataclass

import torch
import torch.nn as nn

from attacks.adaptive_utils import (
    bpda_gradient,
    build_loss_fn,
    project_linf,
    purifier_forward,
    random_linf_start,
)
from attacks.base import BaseAttack

try:
    from autoattack import AutoAttack
except ImportError:
    AutoAttack = None


@dataclass
class BPDAEOTPGDConfig:
    eps: float
    n_iter: int = 40
    step_size: float = 0.0
    eot_iter: int = 1
    random_start: bool = False
    seed: int = 0


@dataclass
class BPDAEOTAAConfig:
    eps: float
    aa_version: str = "rand"
    n_iter: int = 40
    eot_iter: int = 1
    seed: int = 0
    attacks_to_run: str = ""


class BPDAEOTPGDAttack(BaseAttack):
    name = "bpda_eot_pgd"

    def __init__(self, purifier, classifier, device, config):
        self.purifier = purifier
        self.classifier = classifier
        self.device = torch.device(device)
        self.config = config
        self.runtime_config = config
        self.loss_fn = build_loss_fn("ce")
        self.eot_iter = int(config.eot_iter)
        self.eval_eot_iter = 1

    def _step_gradient(self, x_adv, y, iteration):
        grad = torch.zeros_like(x_adv)
        for eot_index in range(max(self.eot_iter, 1)):
            seed = int(self.config.seed) + iteration * 10_000 + eot_index
            current_grad, _, _, _ = bpda_gradient(
                self.purifier,
                self.classifier,
                x_adv,
                y,
                self.loss_fn,
                seed=seed,
            )
            grad += current_grad
        return grad / float(max(self.eot_iter, 1))

    def run(self, x, y):
        if int(self.config.n_iter) <= 0:
            return x.detach().clone()

        if self.config.random_start:
            x_adv = random_linf_start(x, float(self.config.eps)).detach()
        else:
            x_adv = x.detach().clone()

        for iteration in range(max(int(self.config.n_iter), 1)):
            grad = self._step_gradient(x_adv, y, iteration=iteration)
            x_adv = project_linf(
                x_adv + float(self.config.step_size) * grad.sign(),
                x,
                float(self.config.eps),
            ).detach()

        return x_adv.detach()


class _BPDAEOTSurrogate(nn.Module):
    def __init__(self, purifier, classifier, base_seed):
        super().__init__()
        self.purifier = purifier
        self.classifier = classifier
        self.base_seed = int(base_seed)
        self.forward_calls = 0

    def reset_seed_counter(self):
        self.forward_calls = 0

    def _next_seed(self):
        seed = self.base_seed + self.forward_calls
        self.forward_calls += 1
        return seed

    def forward(self, x):
        seed = self._next_seed()
        purified = purifier_forward(self.purifier, x.detach(), seed=seed).detach()
        bpda_input = x + (purified - x).detach()
        return self.classifier(bpda_input)


class BPDAEOTAutoAttack(BaseAttack):
    name = "bpda_eot_aa"

    def __init__(self, purifier, classifier, device, config):
        self.purifier = purifier
        self.classifier = classifier
        self.device = torch.device(device)
        self.config = config
        self.runtime_config = config
        self.eot_iter = int(config.eot_iter)
        self.eval_eot_iter = 1
        self.version = str(config.aa_version).lower()
        self.attacks_to_run = self._resolve_attacks_to_run(self.version)
        self.runtime_config.attacks_to_run = ",".join(self.attacks_to_run)
        self.surrogate = _BPDAEOTSurrogate(
            purifier=self.purifier,
            classifier=self.classifier,
            base_seed=self.config.seed,
        ).to(self.device)

    @staticmethod
    def _resolve_attacks_to_run(version):
        if version == "rand":
            return ["apgd-ce", "apgd-dlr"]
        if version == "full":
            return ["apgd-ce", "apgd-dlr", "fab", "square"]
        if version == "apgdt":
            return ["apgd-t"]
        raise ValueError(
            f"Unsupported BPDA+EOT AutoAttack version '{version}'. "
            "Use one of: rand, full, apgdt."
        )

    def _build_adversary(self):
        if AutoAttack is None:
            raise ImportError("autoattack is not installed in the active environment.")

        self.surrogate.reset_seed_counter()
        adversary = AutoAttack(
            self.surrogate,
            norm="Linf",
            eps=self.config.eps,
            version="custom",
            device=self.device,
            verbose=False,
        )
        adversary.attacks_to_run = list(self.attacks_to_run)

        if self.version in {"rand", "full"}:
            adversary.apgd.n_iter = int(self.config.n_iter)
            adversary.apgd.eot_iter = self.eot_iter
        if self.version == "rand":
            adversary.apgd.n_restarts = 1
        elif self.version == "apgdt":
            adversary.apgd_targeted.n_iter = int(self.config.n_iter)
            adversary.apgd_targeted.n_restarts = 1
            adversary.apgd_targeted.n_target_classes = 9
            adversary.apgd_targeted.eot_iter = self.eot_iter

        return adversary

    def run(self, x, y):
        adversary = self._build_adversary()
        return adversary.run_standard_evaluation(x, y, bs=x.size(0)).detach()
