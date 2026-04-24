from dataclasses import dataclass

import torch

from attacks.adaptive_utils import (
    LinfAPGDAttacker,
    bpda_gradient,
    build_loss_fn,
    forked_seed,
    purifier_forward,
    trajectory_mse,
)
from attacks.base import BaseAttack


@dataclass
class DiffAttackConfig:
    eps: float
    n_iter: int = 100
    n_restarts: int = 5
    eot_iter: int = 20
    rho: float = 0.75
    version: str = "rand"
    attacks_to_run: str = ""
    t_interval: int = 10
    use_trace_loss: bool = True
    trace_lambda: float = 1.0
    timestep: int = 150
    seed: int = 0


class DiffAttackAttack(BaseAttack):
    name = "diffattack"

    def __init__(self, purifier, classifier, device, config):
        self.purifier = purifier
        self.classifier = classifier
        self.device = torch.device(device)
        self.config = config
        self.runtime_config = config
        self.eot_iter = int(config.eot_iter)
        self.eval_eot_iter = 1
        self.attacks_to_run = self._resolve_attacks_to_run(config.version, config.attacks_to_run)
        self.trace_enabled = bool(config.use_trace_loss and hasattr(purifier, "attack_trace"))

    @staticmethod
    def _resolve_attacks_to_run(version, attacks_to_run):
        normalized_version = str(version).strip().lower()
        if normalized_version == "rand":
            return ["apgd-ce", "apgd-dlr"]
        if normalized_version == "custom":
            requested = [item.strip().lower() for item in str(attacks_to_run).split(",") if item.strip()]
            if not requested:
                raise ValueError("DiffAttack custom mode requires --diffattack_attacks_to_run.")
            unsupported = [item for item in requested if item not in {"apgd-ce", "apgd-dlr"}]
            if unsupported:
                raise NotImplementedError(
                    "This repository currently supports DiffAttack ImageNet APGD variants only: "
                    f"{', '.join(sorted(set(unsupported)))}"
                )
            return requested
        raise NotImplementedError(
            f"DiffAttack version '{version}' is not implemented here. "
            "Use the ImageNet default `rand` version or `custom` with APGD losses."
        )

    def _trace_loss(self, x_adv, seed):
        if not self.trace_enabled:
            return None, None
        x_trace = x_adv.detach().clone().requires_grad_(True)
        with forked_seed(seed, self.device):
            _, mid_x, ori_x = self.purifier.attack_trace(
                x_trace,
                timestep=self.config.timestep,
                to_01=False,
            )
        trace_loss = trajectory_mse(mid_x, ori_x, self.config.t_interval)
        if trace_loss is None:
            return None, None
        trace_grad = torch.autograd.grad(
            trace_loss.sum(),
            x_trace,
            allow_unused=True,
        )[0]
        if trace_grad is None:
            trace_grad = torch.zeros_like(x_adv)
        return trace_loss.detach(), trace_grad.detach()

    def _gradient_and_loss(self, x_adv, y, loss_name, attack_index, iteration):
        loss_fn = build_loss_fn(loss_name)
        grad = torch.zeros_like(x_adv)
        logits_acc = None
        loss_acc = torch.zeros(x_adv.shape[0], device=x_adv.device)

        for eot_index in range(max(self.eot_iter, 1)):
            seed = int(self.config.seed) + attack_index * 1_000_000 + iteration * 10_000 + eot_index
            cls_grad, logits, cls_loss, _ = bpda_gradient(
                self.purifier,
                self.classifier,
                x_adv,
                y,
                loss_fn,
                seed=seed,
            )
            trace_loss, trace_grad = self._trace_loss(x_adv, seed=seed)
            total_grad = cls_grad
            total_loss = cls_loss
            if trace_loss is not None and trace_grad is not None:
                total_grad = total_grad + float(self.config.trace_lambda) * trace_grad
                total_loss = total_loss + float(self.config.trace_lambda) * trace_loss
            grad += total_grad
            if logits_acc is None:
                logits_acc = torch.zeros_like(logits)
            logits_acc += logits
            loss_acc += total_loss

        scale = float(max(self.eot_iter, 1))
        return grad / scale, logits_acc / scale, loss_acc / scale

    def _predict(self, x, seed):
        with torch.no_grad():
            purified = purifier_forward(self.purifier, x, seed=seed)
            return self.classifier(purified)

    def run(self, x, y):
        remaining = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
        x_adv = x.detach().clone()

        for attack_offset, attack_name in enumerate(self.attacks_to_run):
            if not remaining.any():
                break

            active_index = remaining.nonzero(as_tuple=False).squeeze(1)
            x_active = x[active_index].clone()
            y_active = y[active_index].clone()
            loss_name = attack_name.split("-", 1)[1]
            optimizer = LinfAPGDAttacker(
                eps=self.config.eps,
                n_iter=self.config.n_iter,
                rho=self.config.rho,
            )

            candidate = None
            candidate_success = torch.zeros_like(y_active, dtype=torch.bool)
            for restart in range(max(int(self.config.n_restarts), 1)):
                candidate = optimizer.run(
                    x_active,
                    y_active,
                    lambda adv, iteration, offset=attack_offset, rs=restart: self._gradient_and_loss(
                        adv,
                        y_active,
                        loss_name=loss_name,
                        attack_index=offset * 100 + rs,
                        iteration=iteration,
                    ),
                )
                logits = self._predict(
                    candidate,
                    seed=int(self.config.seed) + attack_offset * 10_000 + restart,
                )
                candidate_success = logits.argmax(dim=1) != y_active
                if candidate_success.all():
                    break

            if candidate is None:
                continue
            x_adv[active_index[candidate_success]] = candidate[candidate_success]
            remaining[active_index[candidate_success]] = False

        return x_adv.detach()
