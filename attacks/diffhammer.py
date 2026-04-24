from dataclasses import dataclass

import torch
import torch.nn.functional as F

from attacks.adaptive_utils import (
    LinfAPGDAttacker,
    bpda_gradient,
    build_loss_fn,
    match_length,
    parse_csv_list,
    project_linf,
    purifier_forward,
    random_linf_start,
)
from attacks.base import BaseAttack


@dataclass
class DiffHammerConfig:
    eps: float
    method: str = "apgd"
    n_iters: str = "50,50,50"
    loss_names: str = "CW,CE,DLR"
    n_restart: int = 3
    n_eval: int = 10
    n_eot: int = 1
    grad_mode: str = "bpda"
    pgd_cmd: str = ""
    pgd_step_size: float = 0.0
    em: bool = True
    em_alpha: float = 0.5
    em_lam: float = 5.0
    em_steps: int = 5
    seed: int = 0


class DiffHammerAttack(BaseAttack):
    name = "diffhammer"

    def __init__(self, purifier, classifier, device, config):
        self.purifier = purifier
        self.classifier = classifier
        self.device = torch.device(device)
        self.config = config
        self.runtime_config = config
        self.loss_names = match_length(parse_csv_list(config.loss_names, str), int(config.n_restart))
        self.n_iters = match_length(parse_csv_list(config.n_iters, int), int(config.n_restart))
        self.eot_iter = int(config.n_eot)
        self.eval_eot_iter = int(config.n_eval)
        self.requested_grad_mode = str(config.grad_mode).lower()
        self.effective_grad_mode = "bpda"
        if self.requested_grad_mode == "full":
            self.effective_grad_mode = "bpda"
        self.blur_kernel = self._build_blur_kernel()

    def _build_blur_kernel(self):
        kernel = torch.tensor(
            [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
            device=self.device,
        )
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, 3, 3)

    def _seed_for(self, restart, iteration, offset):
        return int(self.config.seed) + restart * 1_000_000 + iteration * 10_000 + offset

    def _unit(self, x):
        return x.sign()

    def _blur(self, grad):
        kernel = self.blur_kernel.expand(grad.shape[1], 1, -1, -1)
        return F.conv2d(grad, kernel, padding=1, groups=grad.shape[1])

    def _predict(self, x, seed):
        with torch.no_grad():
            purified = purifier_forward(self.purifier, x, seed=seed)
            return self.classifier(purified)

    def _evaluate_state(self, x_adv, y, loss_fn, restart, iteration):
        losses = []
        successes = []
        grads = []
        seeds = []
        for eval_index in range(max(int(self.config.n_eval), 1)):
            seed = self._seed_for(restart, iteration, 500 + eval_index)
            seeds.append(seed)
            if self.config.em:
                grad, logits, loss_indiv, _ = bpda_gradient(
                    self.purifier,
                    self.classifier,
                    x_adv,
                    y,
                    loss_fn,
                    seed=seed,
                )
                grads.append(grad.reshape(grad.shape[0], -1))
            else:
                logits = self._predict(x_adv, seed=seed)
                loss_indiv = loss_fn(logits, y)
            losses.append(loss_indiv.detach())
            successes.append((logits.argmax(dim=1) != y).detach())
        result = {
            "loss": torch.stack(losses, dim=0),
            "success": torch.stack(successes, dim=0),
            "seeds": seeds,
        }
        if grads:
            result["grads"] = torch.stack(grads, dim=0)
        return result

    def _seeds_select(self, grads, loss):
        eval_grads_a = grads.unsqueeze(0)
        eval_grads_b = grads.unsqueeze(1)
        delta = (self._unit(eval_grads_a) * eval_grads_b).sum(dim=-1)
        expected_loss = F.elu(loss.view(-1, 1, loss.shape[1]) + float(self.config.eps) * delta).sum(dim=0)
        topk = min(max(int(self.config.n_eot), 1), expected_loss.shape[0])
        return expected_loss.topk(k=topk, dim=0)[1]

    def _compute_em(self, eval_dict):
        loss = eval_dict["loss"]
        grads = eval_dict["grads"]
        seeds = torch.tensor(eval_dict["seeds"], device=loss.device)
        batch_index = torch.arange(loss.shape[1], device=loss.device)
        idxs = self._seeds_select(grads, loss)
        r = grads[idxs[0], batch_index].unsqueeze(0)
        for _ in range(max(int(self.config.em_steps), 1)):
            expected_loss = loss + (self._unit(r) * grads).sum(dim=-1)
            w = torch.sigmoid(float(self.config.em_lam) * expected_loss).unsqueeze(-1)
            r = (w * grads).sum(dim=0, keepdim=True) / w.sum(dim=0, keepdim=True).clamp_min(1.0e-12)

        selected_grads = grads[idxs, batch_index]
        sub_weight = F.cosine_similarity(grads.unsqueeze(1), selected_grads.unsqueeze(0), dim=-1)
        sub_weight = F.softmax(sub_weight, dim=1).permute(1, 0, 2).unsqueeze(-1)
        sub_weight = float(max(int(self.config.n_eot), 1)) * sub_weight * w.unsqueeze(0)
        em_grad = (sub_weight * grads.unsqueeze(0)).sum(dim=1) / sub_weight.sum(dim=1).clamp_min(1.0e-12)
        attack_seeds = [seeds[idxs[index]].tolist() for index in range(idxs.shape[0])]
        return em_grad, attack_seeds

    def _update_best(self, best_adv, best_success, best_loss, x_adv, eval_dict):
        success_worst = eval_dict["success"].any(dim=0)
        loss_avg = eval_dict["loss"].mean(dim=0)
        improved = (success_worst & ~best_success) | ((success_worst == best_success) & (loss_avg > best_loss))
        best_adv[improved] = x_adv[improved]
        best_success[improved] = success_worst[improved]
        best_loss[improved] = loss_avg[improved]

    def _attack_gradient(self, x_adv, y, loss_fn, restart, iteration, attack_seeds, em_grad):
        grad = torch.zeros_like(x_adv)
        for eot_index in range(len(attack_seeds)):
            seed = attack_seeds[eot_index]
            external_grad = None
            if em_grad is not None:
                external_grad = em_grad[eot_index].view_as(x_adv)
            current_grad, _, _, _ = bpda_gradient(
                self.purifier,
                self.classifier,
                x_adv,
                y,
                loss_fn,
                seed=seed,
                external_grad=external_grad,
            )
            grad += current_grad
        return grad / float(max(len(attack_seeds), 1))

    def _run_apgd_restart(self, x, y, loss_fn, restart, best_adv, best_success, best_loss):
        eval_dict = self._evaluate_state(x, y, loss_fn, restart=restart, iteration=-1)
        self._update_best(best_adv, best_success, best_loss, x, eval_dict)
        em_grad = None
        attack_seeds = [self._seed_for(restart, 0, index) for index in range(max(int(self.config.n_eot), 1))]
        if self.config.em:
            em_grad, attack_seeds = self._compute_em(eval_dict)

        past_grad = None

        def gradient_fn(x_adv, iteration):
            nonlocal em_grad, attack_seeds, past_grad
            if not self.config.em:
                attack_seeds = [self._seed_for(restart, iteration, index) for index in range(max(int(self.config.n_eot), 1))]
            grad = self._attack_gradient(
                x_adv,
                y,
                loss_fn,
                restart=restart,
                iteration=iteration,
                attack_seeds=attack_seeds,
                em_grad=em_grad,
            )
            if self.config.em and past_grad is not None:
                mix = float(iteration + 1) ** (-float(self.config.em_alpha))
                grad = mix * grad + (1.0 - mix) * past_grad
            eval_now = self._evaluate_state(x_adv, y, loss_fn, restart=restart, iteration=iteration)
            self._update_best(best_adv, best_success, best_loss, x_adv, eval_now)
            if self.config.em:
                em_grad, attack_seeds = self._compute_em(eval_now)
                past_grad = grad.detach().clone()
            loss_for_apgd = eval_now["loss"].max(dim=0)[0]
            logits_for_apgd = torch.stack(
                [self._predict(x_adv, seed=seed) for seed in attack_seeds[:1]],
                dim=0,
            ).mean(dim=0)
            return grad, logits_for_apgd.detach(), loss_for_apgd.detach()

        optimizer = LinfAPGDAttacker(
            eps=self.config.eps,
            n_iter=self.n_iters[restart],
        )
        candidate = optimizer.run(x, y, gradient_fn)
        final_eval = self._evaluate_state(candidate, y, loss_fn, restart=restart, iteration=self.n_iters[restart] + 1)
        self._update_best(best_adv, best_success, best_loss, candidate, final_eval)

    def _run_pgd_restart(self, x, y, loss_fn, restart, best_adv, best_success, best_loss):
        x_adv = random_linf_start(x, float(self.config.eps)).detach()
        grad_last = torch.zeros_like(x_adv)
        step_size = float(self.config.pgd_step_size) if float(self.config.pgd_step_size) > 0 else 1.2 * float(self.config.eps) / max(self.n_iters[restart], 1)
        eval_dict = self._evaluate_state(x_adv, y, loss_fn, restart=restart, iteration=-1)
        self._update_best(best_adv, best_success, best_loss, x_adv, eval_dict)
        em_grad = None
        attack_seeds = [self._seed_for(restart, 0, index) for index in range(max(int(self.config.n_eot), 1))]
        if self.config.em:
            em_grad, attack_seeds = self._compute_em(eval_dict)

        for iteration in range(max(self.n_iters[restart], 1)):
            if not self.config.em:
                attack_seeds = [self._seed_for(restart, iteration, index) for index in range(max(int(self.config.n_eot), 1))]
            grad = self._attack_gradient(
                x_adv,
                y,
                loss_fn,
                restart=restart,
                iteration=iteration,
                attack_seeds=attack_seeds,
                em_grad=em_grad,
            )
            if "T" in str(self.config.pgd_cmd).upper():
                grad = self._blur(grad)
            if "M" in str(self.config.pgd_cmd).upper():
                grad = grad_last + grad / grad.abs().sum(dim=(1, 2, 3), keepdim=True).clamp_min(1.0e-12)
            grad_last = grad.detach()
            x_adv = project_linf(x_adv + step_size * grad.sign(), x, float(self.config.eps)).detach()

            eval_now = self._evaluate_state(x_adv, y, loss_fn, restart=restart, iteration=iteration)
            self._update_best(best_adv, best_success, best_loss, x_adv, eval_now)
            if self.config.em:
                em_grad, attack_seeds = self._compute_em(eval_now)

    def run(self, x, y):
        best_adv = x.detach().clone()
        best_success = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        best_loss = torch.full((x.shape[0],), float("-inf"), device=x.device)

        for restart in range(max(int(self.config.n_restart), 1)):
            loss_fn = build_loss_fn(self.loss_names[restart])
            if str(self.config.method).lower() == "pgd":
                self._run_pgd_restart(x, y, loss_fn, restart, best_adv, best_success, best_loss)
            else:
                self._run_apgd_restart(x, y, loss_fn, restart, best_adv, best_success, best_loss)
            if best_success.all():
                break

        return best_adv.detach()
