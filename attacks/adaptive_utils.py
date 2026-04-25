import inspect
from contextlib import contextmanager

import torch


def _normalize_seed_spec(seed):
    if seed is None:
        return None
    if isinstance(seed, torch.Tensor):
        values = seed.detach().cpu().view(-1).tolist()
        if len(values) == 1:
            return int(values[0])
        return [int(value) for value in values]
    if isinstance(seed, (list, tuple)):
        if len(seed) == 1:
            return _normalize_seed_spec(seed[0])
        return [int(value) for value in seed]
    return int(seed)


def parse_csv_list(raw_value, cast=str):
    if isinstance(raw_value, (list, tuple)):
        return [cast(item) for item in raw_value]
    values = [item.strip() for item in str(raw_value).split(",") if item.strip()]
    return [cast(item) for item in values]


def match_length(values, target_length):
    values = list(values)
    if not values:
        raise ValueError("Expected at least one value.")
    if len(values) >= target_length:
        return values[:target_length]
    repeats = []
    while len(repeats) < target_length:
        repeats.extend(values)
    return repeats[:target_length]


@contextmanager
def forked_seed(seed, device):
    if seed is None:
        yield
        return

    device = torch.device(device)
    cuda_devices = []
    if device.type == "cuda" and torch.cuda.is_available():
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]

    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        yield


def random_linf_start(x, eps):
    return torch.clamp(x + torch.empty_like(x).uniform_(-eps, eps), 0.0, 1.0)


def project_linf(x_adv, x_orig, eps):
    return torch.clamp(torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps), 0.0, 1.0)


def cw_loss(logits, y):
    one_hot = torch.nn.functional.one_hot(y, num_classes=logits.shape[1]).to(logits.dtype)
    other = (logits - 1.0e5 * one_hot).max(dim=1)[0]
    correct = (logits * one_hot).sum(dim=1)
    return other - correct


def dlr_loss(logits, y):
    logits_sorted, ind_sorted = logits.sort(dim=1)
    correct_is_top = (ind_sorted[:, -1] == y).float()
    batch_index = torch.arange(logits.shape[0], device=logits.device)
    numerator = logits[batch_index, y] - logits_sorted[:, -2] * correct_is_top - logits_sorted[:, -1] * (1.0 - correct_is_top)
    denominator = logits_sorted[:, -1] - logits_sorted[:, -3] + 1.0e-12
    return -(numerator / denominator)


def build_loss_fn(loss_name):
    normalized = str(loss_name).strip().lower()
    if normalized == "ce":
        return torch.nn.CrossEntropyLoss(reduction="none")
    if normalized == "cw":
        return cw_loss
    if normalized == "dlr":
        return dlr_loss
    raise ValueError(f"Unsupported loss '{loss_name}'.")


def purifier_forward(purifier, x, seed=None):
    normalized_seed = _normalize_seed_spec(seed)

    def _call_purifier(module, batch, item_seed):
        if hasattr(module, "purify"):
            purify_fn = module.purify
            try:
                parameters = inspect.signature(purify_fn).parameters
            except (TypeError, ValueError):
                parameters = {}
            if "seed" in parameters:
                return purify_fn(batch, seed=item_seed)
            return purify_fn(batch)
        return module(batch)

    if isinstance(normalized_seed, list):
        if len(normalized_seed) != x.shape[0]:
            raise ValueError(
                f"Per-example seed list length {len(normalized_seed)} does not match batch size {x.shape[0]}."
            )
        outputs = []
        for index, item_seed in enumerate(normalized_seed):
            with forked_seed(item_seed, x.device):
                outputs.append(_call_purifier(purifier, x[index : index + 1], item_seed))
        return torch.cat(outputs, dim=0)

    with forked_seed(normalized_seed, x.device):
        return _call_purifier(purifier, x, normalized_seed)


def bpda_gradient(purifier, classifier, x, y, loss_fn, seed=None, external_grad=None):
    x_base = x.detach().clone()
    if external_grad is None:
        x_base.requires_grad_(True)
    purified = purifier_forward(purifier, x_base.detach(), seed=seed).detach()
    bpda_input = x_base + (purified - x_base).detach()
    logits = classifier(bpda_input)
    loss_indiv = loss_fn(logits, y)
    if external_grad is None:
        grad = torch.autograd.grad(loss_indiv.sum(), x_base)[0]
    else:
        grad = external_grad.detach().clone()
    return grad.detach(), logits.detach(), loss_indiv.detach(), purified


def trajectory_mse(mid_x, ori_x, step_interval):
    if not mid_x or not ori_x:
        return None
    total_steps = min(len(mid_x), len(ori_x))
    if total_steps <= 0:
        return None
    interval = max(int(step_interval), 1)
    selected_indices = list(range(0, total_steps, interval))
    if not selected_indices:
        selected_indices = [total_steps - 1]

    per_step_losses = []
    for index in selected_indices:
        mid = mid_x[index].reshape(mid_x[index].shape[0], -1)
        ori = ori_x[index].reshape(ori_x[index].shape[0], -1)
        per_step_losses.append((mid - ori).pow(2).mean(dim=1))
    return torch.stack(per_step_losses, dim=0).mean(dim=0)


def build_apgd_checkpoints(n_iter):
    raw_points = [0.0, 0.22, 0.41, 0.57, 0.70, 0.80, 0.87, 0.93]
    return sorted({int(point * n_iter) for point in raw_points})


class LinfAPGDAttacker:
    def __init__(self, eps, n_iter, rho=0.75):
        self.eps = float(eps)
        self.n_iter = int(n_iter)
        self.rho = float(rho)
        self.checkpoints = build_apgd_checkpoints(self.n_iter)

    def run(self, x, y, gradient_fn, init_fn=None):
        if self.n_iter <= 0:
            return x.detach().clone()

        x_adv = init_fn(x, self.eps) if init_fn is not None else random_linf_start(x, self.eps)
        x_adv = x_adv.detach()
        x_best = x_adv.clone()
        x_best_adv = x_adv.clone()
        x_adv_last = x_adv.clone()
        step_size = 2.0 * self.eps * torch.ones_like(x[:, :1, :1, :1])
        step_size_last = step_size.clone()
        grad, logits, loss_indiv = gradient_fn(x_adv, 0)
        grad_best = grad.clone()
        loss_best = loss_indiv.clone()
        loss_best_hist = [loss_best.clone()]
        success = logits.argmax(dim=1) != y
        success_seen = success.clone()
        x_best_adv[success] = x_adv[success]

        for iteration in range(self.n_iter):
            grad_delta = x_adv - x_adv_last
            x_adv_last = x_adv.clone()
            momentum = 0.75 if iteration > 0 else 1.0
            z = project_linf(x_adv + step_size * grad.sign(), x, self.eps)
            x_adv = project_linf(x_adv + momentum * (z - x_adv) + (1.0 - momentum) * grad_delta, x, self.eps).detach()

            grad, logits, loss_indiv = gradient_fn(x_adv, iteration + 1)
            success = logits.argmax(dim=1) != y
            success_seen = success_seen | success
            x_best_adv[success] = x_adv[success]

            improved = loss_indiv > loss_best
            x_best[improved] = x_adv[improved]
            grad_best[improved] = grad[improved]
            loss_best[improved] = loss_indiv[improved]
            loss_best_hist.append(loss_best.clone())

            checkpoint_step = iteration + 1
            if checkpoint_step in self.checkpoints and checkpoint_step > 0:
                checkpoint_index = self.checkpoints.index(checkpoint_step)
                prev_checkpoint = self.checkpoints[checkpoint_index - 1] if checkpoint_index > 0 else 0
                history = torch.stack(loss_best_hist, dim=0)
                current_window = history[prev_checkpoint + 1 : checkpoint_step + 1]
                previous_window = history[prev_checkpoint:checkpoint_step]
                mask1 = (current_window > previous_window).sum(dim=0) < self.rho * (checkpoint_step - prev_checkpoint)
                mask2 = (step_size == step_size_last).view(-1) & (history[checkpoint_step] == history[prev_checkpoint])
                mask = mask1 | mask2
                step_size_last = step_size.clone()
                step_size = torch.where(mask.view(-1, 1, 1, 1), step_size / 2.0, step_size)
                x_adv[mask] = x_best[mask].clone()
                grad[mask] = grad_best[mask].clone()

        x_best_adv[~success_seen] = x_best[~success_seen]
        return x_best_adv.detach()
