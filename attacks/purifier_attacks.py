from dataclasses import dataclass

import torch
import torch.nn.functional as F

from attacks.adaptive_utils import (
    bpda_gradient,
    build_loss_fn,
    project_linf,
    random_linf_start,
)
from attacks.base import BaseAttack

try:
    from autoattack import AutoAttack
except ImportError:
    AutoAttack = None


@dataclass
class PurifierPGDConfig:
    eps: float
    n_iter: int = 40
    step_size: float = 0.0
    random_start: bool = False
    seed: int = 0


@dataclass
class PurifierAAConfig:
    eps: float
    aa_version: str = "rand"
    n_iter: int = 100
    seed: int = 0
    attacks_to_run: str = ""


@dataclass
class BPDAEOTPGDConfig:
    eps: float
    n_iter: int = 40
    step_size: float = 0.0
    eot_iter: int = 10
    random_start: bool = False
    seed: int = 0


class PurifierPGDAttack(BaseAttack):
    name = "purifier_pgd"

    def __init__(self, model, device, config):
        self.model = model
        self.device = torch.device(device)
        self.config = config
        self.runtime_config = config

    def _step_gradient(self, x_adv, y):
        logits = self.model(x_adv)
        loss = F.cross_entropy(logits, y, reduction="sum")
        grad = torch.autograd.grad(loss, x_adv, allow_unused=True)[0]
        if grad is None:
            grad = torch.zeros_like(x_adv)
        return grad

    def run(self, x, y):
        if int(self.config.n_iter) <= 0:
            return x.detach().clone()

        if self.config.random_start:
            x_adv = random_linf_start(x, float(self.config.eps)).detach()
        else:
            x_adv = x.detach().clone()

        for _ in range(max(int(self.config.n_iter), 1)):
            x_adv = x_adv.detach().requires_grad_(True)
            grad = self._step_gradient(x_adv, y)
            x_adv = project_linf(
                x_adv + float(self.config.step_size) * grad.sign(),
                x,
                float(self.config.eps),
            ).detach()

        return x_adv.detach()


class PurifierAutoAttack(BaseAttack):
    name = "purifier_aa"

    def __init__(self, model, device, config):
        self.device = torch.device(device)
        self.config = config
        self.runtime_config = config
        self.version = str(config.aa_version).lower()
        self.attacks_to_run = self._resolve_attacks_to_run(self.version)
        self.runtime_config.attacks_to_run = ",".join(self.attacks_to_run)
        self.model = model.to(self.device)

    @staticmethod
    def _resolve_attacks_to_run(version):
        if version == "rand":
            return ["apgd-ce", "apgd-dlr"]
        if version == "full":
            return ["apgd-ce", "apgd-dlr", "fab", "square"]
        if version == "apgdt":
            return ["apgd-t"]
        raise ValueError(
            f"Unsupported purifier AutoAttack version '{version}'. "
            "Use one of: rand, full, apgdt."
        )

    def _build_adversary(self):
        if AutoAttack is None:
            raise ImportError("autoattack is not installed in the active environment.")

        adversary = AutoAttack(
            self.model,
            norm="Linf",
            eps=self.config.eps,
            version="custom",
            device=self.device,
            verbose=False,
        )
        adversary.attacks_to_run = list(self.attacks_to_run)

        if self.version in {"rand", "full"}:
            adversary.apgd.n_iter = int(self.config.n_iter)
        if self.version == "rand":
            adversary.apgd.n_restarts = 1
        elif self.version == "apgdt":
            adversary.apgd_targeted.n_iter = int(self.config.n_iter)
            adversary.apgd_targeted.n_restarts = 1
            adversary.apgd_targeted.n_target_classes = 9

        return adversary

    def run(self, x, y):
        adversary = self._build_adversary()
        return adversary.run_standard_evaluation(x, y, bs=x.size(0)).detach()


class BPDAEOTPGDAttack(BaseAttack):
    name = "bpda_eot_pgd"

    def __init__(self, purifier, classifier, device, config):
        self.purifier = purifier
        self.classifier = classifier
        self.device = torch.device(device)
        self.config = config
        self.runtime_config = config
        self.loss_fn = build_loss_fn("ce")
        self.eot_iter = max(int(config.eot_iter), 1)
        self.eval_eot_iter = 1

    def _step_gradient(self, x_adv, y, iteration):
        grad = torch.zeros_like(x_adv)
        for eot_index in range(self.eot_iter):
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
        return grad / float(self.eot_iter)

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
