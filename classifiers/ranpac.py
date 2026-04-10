import os

from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from classifiers.mean_sparse import (
    DEFAULT_MEANSPARSE_STAT_EPS,
    apply_mean_centered_soft_threshold,
    format_cache_value,
    is_meansparse_enabled,
    strip_meansparse_tag,
)
from classifiers.stability_ridge import (
    DEFAULT_STABILITY_RIDGE_STAT_EPS,
    build_stability_ridge_tag,
    compute_stability_ridge_prior,
    is_stability_ridge_enabled,
    solve_ridge_system,
)

RIDGE_CANDIDATES = [10.0 ** power for power in range(-8, 14)]
DEFAULT_RANPAC_FEATURE_STAT_EPS = 1e-6
RANPAC_FEATURE_MODE_GELU = "gelu"
RANPAC_FEATURE_MODE_GELU_ZSCORE_L2_CENTERED = "gelu_zscore_l2_centered"
RANPAC_FEATURE_MODES = (
    RANPAC_FEATURE_MODE_GELU,
    RANPAC_FEATURE_MODE_GELU_ZSCORE_L2_CENTERED,
)
RANPAC_CACHE_VERSION = 17


def build_ranpac_feature_mode_tag(
    feature_mode,
    feature_stat_eps=DEFAULT_RANPAC_FEATURE_STAT_EPS,
    separator="_",
):
    if feature_mode == RANPAC_FEATURE_MODE_GELU:
        return ""

    parts = [f"rfm{format_cache_value(feature_mode)}"]
    if float(feature_stat_eps) != DEFAULT_RANPAC_FEATURE_STAT_EPS:
        parts.append(f"rfeps{format_cache_value(feature_stat_eps)}")
    return separator + separator.join(parts)


def _validate_ranpac_feature_mode(feature_mode):
    if feature_mode not in RANPAC_FEATURE_MODES:
        raise ValueError(
            f"Unsupported RanPAC feature mode '{feature_mode}'. "
            f"Expected one of {', '.join(RANPAC_FEATURE_MODES)}."
        )


def _project_ranpac_backbone_features(features, w_rand):
    return F.gelu(features @ w_rand).float()


def _transform_ranpac_projected_features(
    projected,
    feature_mode,
    projected_feature_mean=None,
    projected_feature_std=None,
    feature_stat_eps=DEFAULT_RANPAC_FEATURE_STAT_EPS,
):
    projected = projected.float()
    if feature_mode == RANPAC_FEATURE_MODE_GELU:
        return projected

    if projected_feature_mean is None or projected_feature_std is None:
        raise ValueError(f"RanPAC feature mode '{feature_mode}' requires projected train mean/std.")

    mean = projected_feature_mean.to(device=projected.device, dtype=projected.dtype).view(1, -1)
    std = projected_feature_std.to(device=projected.device, dtype=projected.dtype).clamp_min(float(feature_stat_eps)).view(1, -1)
    standardized = (projected - mean) / std
    # In one dimension, the L2 norm collapses to abs(.). Use squared magnitude so
    # the distance feature remains non-negative but is not the original abs z-score.
    return standardized.square()


def _center_ranpac_statistics(g_matrix, q_matrix, feature_sum, target_sum, sample_count, feature_center):
    if feature_center is None or sample_count <= 0:
        return g_matrix, q_matrix

    feature_center = feature_center.to(dtype=torch.float32)
    feature_sum = feature_sum.to(dtype=torch.float32)
    target_sum = target_sum.to(dtype=torch.float32)
    center_outer = torch.outer(feature_center, feature_center)
    centered_g = (
        g_matrix
        - torch.outer(feature_center, feature_sum)
        - torch.outer(feature_sum, feature_center)
        + float(sample_count) * center_outer
    )
    centered_q = q_matrix - torch.outer(feature_center, target_sum)
    return centered_g, centered_q


class RanPACLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        rp_dim,
        weight,
        w_rand,
        soft_threshold_mean,
        soft_threshold_std,
        soft_threshold_alpha,
        soft_threshold_beta,
        soft_threshold_stat_eps,
        feature_mode=RANPAC_FEATURE_MODE_GELU,
        feature_stat_eps=DEFAULT_RANPAC_FEATURE_STAT_EPS,
        projected_feature_mean=None,
        projected_feature_std=None,
        projected_feature_center=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rp_dim = rp_dim
        self.register_buffer("weight", weight)
        self.register_buffer("w_rand", w_rand)
        self.register_buffer("soft_threshold_mean", soft_threshold_mean)
        self.register_buffer("soft_threshold_std", soft_threshold_std)
        self.register_buffer("projected_feature_mean", projected_feature_mean)
        self.register_buffer("projected_feature_std", projected_feature_std)
        self.register_buffer("projected_feature_center", projected_feature_center)
        self.soft_threshold_alpha = float(soft_threshold_alpha)
        self.soft_threshold_beta = float(soft_threshold_beta)
        self.soft_threshold_stat_eps = float(soft_threshold_stat_eps)
        self.feature_mode = feature_mode
        self.feature_stat_eps = float(feature_stat_eps)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = apply_mean_centered_soft_threshold(
            x,
            self.soft_threshold_mean,
            self.soft_threshold_std,
            alpha=self.soft_threshold_alpha,
            beta=self.soft_threshold_beta,
            stat_eps=self.soft_threshold_stat_eps,
        )
        projected = _project_ranpac_backbone_features(x, self.w_rand)
        features = _transform_ranpac_projected_features(
            projected,
            self.feature_mode,
            projected_feature_mean=self.projected_feature_mean,
            projected_feature_std=self.projected_feature_std,
            feature_stat_eps=self.feature_stat_eps,
        )
        if self.projected_feature_center is not None:
            features = features - self.projected_feature_center.to(device=features.device, dtype=features.dtype).view(1, -1)
        return features @ self.weight.t()


class ResidualRanPACLinear(nn.Module):
    def __init__(self, original_linear, ranpac_linear, ranpac_lambda, ranpac_temp, baseline_logit_mean):
        super().__init__()
        self.original_linear = original_linear
        self.ranpac_linear = ranpac_linear
        self.ranpac_lambda = float(ranpac_lambda)
        self.ranpac_temp = float(ranpac_temp)
        self.register_buffer("baseline_logit_mean", baseline_logit_mean.reshape(()))

    def forward(self, x):
        baseline_logits = self.original_linear(x)
        baseline_logits = baseline_logits - self.baseline_logit_mean.to(dtype=baseline_logits.dtype)
        ranpac_logits = self.ranpac_linear(x) / self.ranpac_temp
        return (1 - self.ranpac_lambda) * baseline_logits + self.ranpac_lambda * ranpac_logits


def _get_module_by_name(model, module_name):
    module = model
    for attr in module_name.split("."):
        module = getattr(module, attr)
    return module


def _set_module_by_name(model, module_name, new_module):
    parts = module_name.split(".")
    parent = model
    for attr in parts[:-1]:
        parent = getattr(parent, attr)
    setattr(parent, parts[-1], new_module)


def _find_last_linear(model):
    linear_modules = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    if not linear_modules:
        raise ValueError("RanPAC requires a model with a final nn.Linear layer.")
    return linear_modules[-1]


def _resolve_imagenet_train_dir(dataset_root):
    candidates = []
    if dataset_root:
        candidates.extend(
            [
                dataset_root,
                os.path.join(dataset_root, "imagenet"),
                os.path.join(dataset_root, "image_net"),
            ]
        )

    env_root = os.environ.get("IMAGENET_LOC_ENV")
    if env_root:
        candidates.append(env_root)

    candidates.extend(["./image_net", "./dataset/imagenet"])

    for root in candidates:
        if not root:
            continue
        if os.path.basename(root.rstrip("/")) == "train" and os.path.isdir(root):
            return root

        train_dir = os.path.join(root, "train")
        if os.path.isdir(train_dir):
            return train_dir

    raise FileNotFoundError(
        "Could not locate the ImageNet train directory. Set IMAGENET_LOC_ENV or pass --ranpac_dataset_root."
    )


def _build_imagenet_train_loaders(dataset_root, batch_size, num_workers, seed, train_transform=None):
    train_dir = _resolve_imagenet_train_dir(dataset_root)
    transform = train_transform
    if transform is None:
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )
    dataset = torchvision.datasets.ImageFolder(train_dir, transform=transform)
    train_cutoff = int(len(dataset) * 0.8)
    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    train_subset = Subset(dataset, indices[:train_cutoff])
    val_subset = Subset(dataset, indices[train_cutoff:])

    common_kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=num_workers,
    )
    return (
        DataLoader(train_subset, **common_kwargs),
        DataLoader(val_subset, **common_kwargs),
    )


def _build_train_loaders(classifier_name, dataset_root, batch_size, num_workers, seed, train_transform=None):
    if "imagenet" in classifier_name:
        return _build_imagenet_train_loaders(
            dataset_root,
            batch_size,
            num_workers,
            seed,
            train_transform=train_transform,
        )
    raise NotImplementedError(f"RanPAC train loader is not implemented for {classifier_name}.")


def _sample_linf_noisy_inputs(inputs, eps):
    noise = torch.empty_like(inputs).uniform_(-eps, eps)
    return torch.clamp(inputs + noise, 0.0, 1.0)


def _build_supervised_targets(logits, labels, out_features, hardneg_topk, hardneg_gamma):
    targets = F.one_hot(labels.cpu(), num_classes=out_features).float()
    if hardneg_topk <= 0 or hardneg_gamma <= 0:
        return targets

    topk = min(int(hardneg_topk), out_features - 1)
    if topk <= 0:
        return targets

    logits_cpu = logits.detach().float().cpu()
    logits_cpu[torch.arange(logits_cpu.size(0)), labels.cpu()] = float("-inf")
    confusing_classes = logits_cpu.topk(topk, dim=1).indices
    suppress_values = targets.new_full(confusing_classes.shape, -float(hardneg_gamma) / float(topk))
    targets.scatter_add_(1, confusing_classes, suppress_values)
    return targets


def _accumulate_statistics(
    model,
    linear_layer,
    loader,
    w_rand,
    out_features,
    device,
    description,
    adapt_noise_eps,
    adapt_noise_num,
    adapt_alpha,
    hardneg_topk,
    hardneg_gamma,
):
    feature_buffer = []

    def hook(_, inputs):
        feature_buffer.append(inputs[0].detach())

    handle = linear_layer.register_forward_pre_hook(hook)
    g_matrix = torch.zeros(w_rand.size(1), w_rand.size(1), dtype=torch.float32)
    q_matrix = torch.zeros(w_rand.size(1), out_features, dtype=torch.float32)
    target_norm = 0.0
    sample_count = 0
    del adapt_alpha
    use_noisy_adaptation = adapt_noise_num > 0 and adapt_noise_eps > 0
    noisy_sample_weight = 1.0 / adapt_noise_num if use_noisy_adaptation else 0.0

    try:
        with torch.no_grad():
            for inputs, targets in tqdm(loader, desc=description):
                inputs = inputs.to(device)
                targets = targets.to(device)
                if use_noisy_adaptation:
                    for _ in range(adapt_noise_num):
                        noisy_inputs = _sample_linf_noisy_inputs(inputs, adapt_noise_eps)
                        feature_buffer.clear()
                        noisy_logits = model(noisy_inputs)
                        noisy_features = feature_buffer.pop().view(inputs.size(0), -1).cpu()
                        noisy_projected = _project_ranpac_backbone_features(noisy_features, w_rand)
                        noisy_targets = _build_supervised_targets(
                            noisy_logits,
                            targets,
                            out_features,
                            hardneg_topk=hardneg_topk,
                            hardneg_gamma=hardneg_gamma,
                        )
                        target_norm_value = noisy_targets.square().sum().item()

                        g_matrix += noisy_sample_weight * (noisy_projected.t() @ noisy_projected)
                        q_matrix += noisy_sample_weight * (noisy_projected.t() @ noisy_targets)
                        target_norm += noisy_sample_weight * target_norm_value
                        sample_count += noisy_sample_weight * noisy_targets.size(0)
                else:
                    feature_buffer.clear()
                    logits = model(inputs)
                    features = feature_buffer.pop().view(inputs.size(0), -1).cpu()
                    projected = _project_ranpac_backbone_features(features, w_rand)
                    supervised_targets = _build_supervised_targets(
                        logits,
                        targets,
                        out_features,
                        hardneg_topk=hardneg_topk,
                        hardneg_gamma=hardneg_gamma,
                    )
                    target_norm_value = supervised_targets.square().sum().item()

                    g_matrix += projected.t() @ projected
                    q_matrix += projected.t() @ supervised_targets
                    target_norm += target_norm_value
                    sample_count += supervised_targets.size(0)
    finally:
        handle.remove()

    return g_matrix, q_matrix, target_norm, sample_count


def _collect_clean_train_base_statistics(
    model,
    linear_layer,
    loader,
    w_rand,
    device,
    description,
    collect_feature_stats,
    collect_projected_feature_stats,
):
    feature_buffer = []

    def hook(_, inputs):
        feature_buffer.append(inputs[0].detach())

    handle = linear_layer.register_forward_pre_hook(hook)
    logit_sum = 0.0
    logit_count = 0
    feature_sum = None
    feature_sum_sq = None
    feature_sample_count = 0
    projected_sum = None
    projected_sum_sq = None
    projected_sample_count = 0

    try:
        model = model.eval().to(device)
        with torch.no_grad():
            for inputs, _ in tqdm(loader, desc=description):
                inputs = inputs.to(device)
                feature_buffer.clear()
                clean_logits = model(inputs)
                clean_features = feature_buffer.pop().view(inputs.size(0), -1).float().cpu()
                logit_sum += clean_logits.float().sum().item()
                logit_count += clean_logits.numel()

                if collect_feature_stats:
                    if feature_sum is None:
                        feature_sum = torch.zeros(clean_features.size(1), dtype=torch.float64)
                        feature_sum_sq = torch.zeros(clean_features.size(1), dtype=torch.float64)
                    feature_sum += clean_features.sum(dim=0, dtype=torch.float64)
                    feature_sum_sq += clean_features.square().sum(dim=0, dtype=torch.float64)
                    feature_sample_count += clean_features.size(0)

                if collect_projected_feature_stats:
                    clean_projected = _project_ranpac_backbone_features(clean_features, w_rand)
                    if projected_sum is None:
                        projected_sum = torch.zeros(clean_projected.size(1), dtype=torch.float64)
                        projected_sum_sq = torch.zeros(clean_projected.size(1), dtype=torch.float64)
                    projected_sum += clean_projected.sum(dim=0, dtype=torch.float64)
                    projected_sum_sq += clean_projected.square().sum(dim=0, dtype=torch.float64)
                    projected_sample_count += clean_projected.size(0)
    finally:
        handle.remove()

    if logit_count == 0:
        raise ValueError("RanPAC train statistics loader is empty.")

    baseline_logit_mean = torch.tensor(logit_sum / float(logit_count), dtype=torch.float32)
    feature_mean = None
    feature_std = None
    if collect_feature_stats:
        if feature_sample_count == 0:
            raise ValueError("RanPAC soft-threshold statistics loader is empty.")
        feature_mean = feature_sum / float(feature_sample_count)
        feature_var = feature_sum_sq / float(feature_sample_count) - feature_mean.square()
        feature_mean = feature_mean.float()
        feature_std = feature_var.clamp_min(0.0).sqrt().float()

    projected_feature_mean = None
    projected_feature_std = None
    if collect_projected_feature_stats:
        if projected_sample_count == 0:
            raise ValueError("RanPAC projected feature statistics loader is empty.")
        projected_feature_mean = projected_sum / float(projected_sample_count)
        projected_feature_var = projected_sum_sq / float(projected_sample_count) - projected_feature_mean.square()
        projected_feature_mean = projected_feature_mean.float()
        projected_feature_std = projected_feature_var.clamp_min(0.0).sqrt().float()

    return {
        "baseline_logit_mean": baseline_logit_mean,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "projected_feature_mean": projected_feature_mean,
        "projected_feature_std": projected_feature_std,
    }


def _accumulate_mode_statistics(
    model,
    linear_layer,
    loader,
    w_rand,
    out_features,
    device,
    description,
    adapt_noise_eps,
    adapt_noise_num,
    adapt_alpha,
    hardneg_topk,
    hardneg_gamma,
    feature_mode,
    projected_feature_mean,
    projected_feature_std,
    feature_stat_eps,
):
    feature_buffer = []

    def hook(_, inputs):
        feature_buffer.append(inputs[0].detach())

    handle = linear_layer.register_forward_pre_hook(hook)
    feature_dim = w_rand.size(1)
    g_matrix = torch.zeros(feature_dim, feature_dim, dtype=torch.float32)
    q_matrix = torch.zeros(feature_dim, out_features, dtype=torch.float32)
    feature_sum = torch.zeros(feature_dim, dtype=torch.float64)
    feature_sum_sq = torch.zeros(feature_dim, dtype=torch.float64)
    feature_abs_sum = torch.zeros(feature_dim, dtype=torch.float64)
    target_sum = torch.zeros(out_features, dtype=torch.float64)
    target_norm = 0.0
    sample_count = 0.0
    del adapt_alpha
    use_noisy_adaptation = adapt_noise_num > 0 and adapt_noise_eps > 0
    noisy_sample_weight = 1.0 / adapt_noise_num if use_noisy_adaptation else 0.0

    def accumulate_batch(batch_features, batch_targets, weight):
        nonlocal target_norm, sample_count
        projected = _project_ranpac_backbone_features(batch_features, w_rand)
        transformed = _transform_ranpac_projected_features(
            projected,
            feature_mode,
            projected_feature_mean=projected_feature_mean,
            projected_feature_std=projected_feature_std,
            feature_stat_eps=feature_stat_eps,
        )
        g_matrix.add_(weight * (transformed.t() @ transformed))
        q_matrix.add_(weight * (transformed.t() @ batch_targets))
        feature_sum.add_(weight * transformed.sum(dim=0, dtype=torch.float64))
        feature_sum_sq.add_(weight * transformed.square().sum(dim=0, dtype=torch.float64))
        feature_abs_sum.add_(weight * transformed.abs().sum(dim=0, dtype=torch.float64))
        target_sum.add_(weight * batch_targets.sum(dim=0, dtype=torch.float64))
        target_norm += weight * batch_targets.square().sum().item()
        sample_count += weight * batch_targets.size(0)

    try:
        model = model.eval().to(device)
        with torch.no_grad():
            for inputs, targets in tqdm(loader, desc=description):
                inputs = inputs.to(device)
                targets = targets.to(device)
                if use_noisy_adaptation:
                    for _ in range(adapt_noise_num):
                        noisy_inputs = _sample_linf_noisy_inputs(inputs, adapt_noise_eps)
                        feature_buffer.clear()
                        noisy_logits = model(noisy_inputs)
                        noisy_features = feature_buffer.pop().view(inputs.size(0), -1).float().cpu()
                        noisy_targets = _build_supervised_targets(
                            noisy_logits,
                            targets,
                            out_features,
                            hardneg_topk=hardneg_topk,
                            hardneg_gamma=hardneg_gamma,
                        )
                        accumulate_batch(noisy_features, noisy_targets, noisy_sample_weight)
                else:
                    feature_buffer.clear()
                    logits = model(inputs)
                    features = feature_buffer.pop().view(inputs.size(0), -1).float().cpu()
                    supervised_targets = _build_supervised_targets(
                        logits,
                        targets,
                        out_features,
                        hardneg_topk=hardneg_topk,
                        hardneg_gamma=hardneg_gamma,
                    )
                    accumulate_batch(features, supervised_targets, 1.0)
    finally:
        handle.remove()

    return {
        "g_matrix": g_matrix,
        "q_matrix": q_matrix,
        "feature_sum": feature_sum,
        "feature_sum_sq": feature_sum_sq,
        "feature_abs_sum": feature_abs_sum,
        "target_sum": target_sum,
        "target_norm": target_norm,
        "sample_count": sample_count,
    }


def _select_ridge_by_regression_loss(
    g_train,
    q_train,
    g_val,
    q_val,
    val_target_norm,
    num_classes,
    val_sample_count,
    device,
    diagonal_prior=None,
):
    g_train = g_train.to(device)
    q_train = q_train.to(device)
    g_val = g_val.to(device)
    q_val = q_val.to(device)

    best_ridge = RIDGE_CANDIDATES[0]
    best_loss = None
    denominator = max(val_sample_count * num_classes, 1)

    with torch.no_grad():
        for ridge in RIDGE_CANDIDATES:
            weights = solve_ridge_system(g_train, q_train, ridge, diagonal_prior=diagonal_prior).t()
            if val_sample_count == 0:
                continue

            quadratic = torch.trace(weights @ g_val @ weights.t())
            cross_term = torch.trace(weights @ q_val)
            loss = (quadratic - 2.0 * cross_term + val_target_norm) / denominator
            loss_value = loss.item()
            if best_loss is None or loss_value < best_loss:
                best_loss = loss_value
                best_ridge = ridge

    return best_ridge, best_loss


def _collect_train_statistics(
    model,
    linear_layer,
    loader,
    w_rand,
    out_features,
    device,
    description,
    adapt_noise_eps,
    adapt_noise_num,
    adapt_alpha,
    hardneg_topk,
    hardneg_gamma,
    collect_feature_stats,
    accumulate_ridge_stats,
    collect_projected_stability_stats,
):
    feature_buffer = []

    def hook(_, inputs):
        feature_buffer.append(inputs[0].detach())

    handle = linear_layer.register_forward_pre_hook(hook)
    logit_sum = 0.0
    logit_count = 0
    feature_sum = None
    feature_sum_sq = None
    feature_sample_count = 0
    projected_sum = None
    projected_sum_sq = None
    projected_abs_sum = None
    projected_sample_count = 0.0
    g_matrix = None
    q_matrix = None
    target_norm = 0.0
    sample_count = 0.0
    del adapt_alpha
    use_noisy_adaptation = adapt_noise_num > 0 and adapt_noise_eps > 0
    noisy_sample_weight = 1.0 / adapt_noise_num if use_noisy_adaptation else 0.0

    try:
        model = model.eval().to(device)
        with torch.no_grad():
            for inputs, targets in tqdm(loader, desc=description):
                inputs = inputs.to(device)
                targets = targets.to(device)
                feature_buffer.clear()
                clean_logits = model(inputs)
                clean_features = feature_buffer.pop().view(inputs.size(0), -1).float().cpu()
                logit_sum += clean_logits.float().sum().item()
                logit_count += clean_logits.numel()

                if collect_feature_stats:
                    if feature_sum is None:
                        feature_sum = torch.zeros(clean_features.size(1), dtype=torch.float64)
                        feature_sum_sq = torch.zeros(clean_features.size(1), dtype=torch.float64)
                    feature_sum += clean_features.sum(dim=0, dtype=torch.float64)
                    feature_sum_sq += clean_features.square().sum(dim=0, dtype=torch.float64)
                    feature_sample_count += clean_features.size(0)

                if not accumulate_ridge_stats and not collect_projected_stability_stats:
                    continue

                if g_matrix is None:
                    g_matrix = torch.zeros(w_rand.size(1), w_rand.size(1), dtype=torch.float32)
                    q_matrix = torch.zeros(w_rand.size(1), out_features, dtype=torch.float32)

                if collect_projected_stability_stats and projected_sum is None:
                    projected_sum = torch.zeros(w_rand.size(1), dtype=torch.float64)
                    projected_sum_sq = torch.zeros(w_rand.size(1), dtype=torch.float64)
                    projected_abs_sum = torch.zeros(w_rand.size(1), dtype=torch.float64)

                if use_noisy_adaptation:
                    for _ in range(adapt_noise_num):
                        noisy_inputs = _sample_linf_noisy_inputs(inputs, adapt_noise_eps)
                        feature_buffer.clear()
                        noisy_logits = model(noisy_inputs)
                        noisy_features = feature_buffer.pop().view(inputs.size(0), -1).cpu()
                        noisy_projected = _project_ranpac_backbone_features(noisy_features, w_rand)
                        if collect_projected_stability_stats:
                            projected_sum += noisy_sample_weight * noisy_projected.sum(dim=0, dtype=torch.float64)
                            projected_sum_sq += noisy_sample_weight * noisy_projected.square().sum(dim=0, dtype=torch.float64)
                            projected_abs_sum += noisy_sample_weight * noisy_projected.abs().sum(dim=0, dtype=torch.float64)
                            projected_sample_count += noisy_sample_weight * noisy_projected.size(0)
                        if not accumulate_ridge_stats:
                            continue
                        noisy_targets = _build_supervised_targets(
                            noisy_logits,
                            targets,
                            out_features,
                            hardneg_topk=hardneg_topk,
                            hardneg_gamma=hardneg_gamma,
                        )
                        target_norm_value = noisy_targets.square().sum().item()
                        g_matrix += noisy_sample_weight * (noisy_projected.t() @ noisy_projected)
                        q_matrix += noisy_sample_weight * (noisy_projected.t() @ noisy_targets)
                        target_norm += noisy_sample_weight * target_norm_value
                        sample_count += noisy_sample_weight * noisy_targets.size(0)
                else:
                    projected = _project_ranpac_backbone_features(clean_features, w_rand)
                    if collect_projected_stability_stats:
                        projected_sum += projected.sum(dim=0, dtype=torch.float64)
                        projected_sum_sq += projected.square().sum(dim=0, dtype=torch.float64)
                        projected_abs_sum += projected.abs().sum(dim=0, dtype=torch.float64)
                        projected_sample_count += projected.size(0)
                    if not accumulate_ridge_stats:
                        continue
                    supervised_targets = _build_supervised_targets(
                        clean_logits,
                        targets,
                        out_features,
                        hardneg_topk=hardneg_topk,
                        hardneg_gamma=hardneg_gamma,
                    )
                    target_norm_value = supervised_targets.square().sum().item()
                    g_matrix += projected.t() @ projected
                    q_matrix += projected.t() @ supervised_targets
                    target_norm += target_norm_value
                    sample_count += supervised_targets.size(0)
    finally:
        handle.remove()

    if logit_count == 0:
        raise ValueError("RanPAC train statistics loader is empty.")

    baseline_logit_mean = torch.tensor(logit_sum / float(logit_count), dtype=torch.float32)
    feature_mean = None
    feature_std = None
    if collect_feature_stats:
        if feature_sample_count == 0:
            raise ValueError("RanPAC soft-threshold statistics loader is empty.")
        feature_mean = feature_sum / float(feature_sample_count)
        feature_var = feature_sum_sq / float(feature_sample_count) - feature_mean.square()
        feature_mean = feature_mean.float()
        feature_std = feature_var.clamp_min(0.0).sqrt().float()

    return {
        "baseline_logit_mean": baseline_logit_mean,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "g_matrix": g_matrix,
        "q_matrix": q_matrix,
        "target_norm": target_norm,
        "sample_count": sample_count,
        "projected_sum": projected_sum,
        "projected_sum_sq": projected_sum_sq,
        "projected_abs_sum": projected_abs_sum,
        "projected_sample_count": projected_sample_count,
    }


def _fit_ranpac_state(
    model,
    classifier_name,
    dataset_root,
    cache_path,
    rp_dim,
    batch_size,
    num_workers,
    seed,
    device,
    train_loader=None,
    val_loader=None,
    train_transform=None,
    adapt_noise_eps=0.0,
    adapt_noise_num=0,
    adapt_alpha=1.0,
    hardneg_topk=9,
    hardneg_gamma=1.0,
    soft_threshold_alpha=0.0,
    soft_threshold_beta=8.0,
    soft_threshold_stat_eps=DEFAULT_MEANSPARSE_STAT_EPS,
    feature_mode=RANPAC_FEATURE_MODE_GELU,
    feature_stat_eps=DEFAULT_RANPAC_FEATURE_STAT_EPS,
    stability_ridge_gamma=0.0,
    stability_ridge_stat_eps=DEFAULT_STABILITY_RIDGE_STAT_EPS,
):
    layer_name, linear_layer = _find_last_linear(model)
    in_features = linear_layer.in_features
    out_features = linear_layer.out_features
    if train_loader is None:
        train_loader, val_loader = _build_train_loaders(
            classifier_name,
            dataset_root,
            batch_size,
            num_workers,
            seed,
            train_transform=train_transform,
        )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    w_rand = torch.randn(in_features, rp_dim, generator=generator, dtype=torch.float32)

    model = model.eval().to(device)
    print(f"Fitting RanPAC head for {cache_path}...")
    projected_feature_mean = None
    projected_feature_std = None
    projected_feature_center = None
    if feature_mode == RANPAC_FEATURE_MODE_GELU:
        # Keep ridge fitting on the original continuous backbone features and only
        # collect clean feature stats for inference-time thresholding in the same pass.
        train_stats = _collect_train_statistics(
            model,
            linear_layer,
            train_loader,
            w_rand,
            out_features,
            device,
            description="RanPAC train baseline, ridge, and feature stats",
            adapt_noise_eps=adapt_noise_eps,
            adapt_noise_num=adapt_noise_num,
            adapt_alpha=adapt_alpha,
            hardneg_topk=hardneg_topk,
            hardneg_gamma=hardneg_gamma,
            collect_feature_stats=True,
            accumulate_ridge_stats=True,
            collect_projected_stability_stats=is_stability_ridge_enabled(stability_ridge_gamma),
        )
        baseline_logit_mean = train_stats["baseline_logit_mean"]
        soft_threshold_mean = train_stats["feature_mean"]
        soft_threshold_std = train_stats["feature_std"]
        g_train = train_stats["g_matrix"]
        q_train = train_stats["q_matrix"]
        stability_diagonal_prior = compute_stability_ridge_prior(
            train_stats["projected_sum"],
            train_stats["projected_sum_sq"],
            train_stats["projected_abs_sum"],
            train_stats["projected_sample_count"],
            stability_ridge_gamma,
            stability_ridge_stat_eps,
        )
        if val_loader is not None:
            g_val, q_val, val_target_norm, val_sample_count = _accumulate_statistics(
                model,
                linear_layer,
                val_loader,
                w_rand,
                out_features,
                device,
                description="RanPAC val stats",
                adapt_noise_eps=adapt_noise_eps,
                adapt_noise_num=adapt_noise_num,
                adapt_alpha=adapt_alpha,
                hardneg_topk=hardneg_topk,
                hardneg_gamma=hardneg_gamma,
            )
        else:
            g_val = torch.zeros_like(g_train)
            q_val = torch.zeros_like(q_train)
            val_target_norm = 0.0
            val_sample_count = 0
    else:
        base_stats = _collect_clean_train_base_statistics(
            model,
            linear_layer,
            train_loader,
            w_rand,
            device,
            description="RanPAC clean train baseline and projected stats",
            collect_feature_stats=True,
            collect_projected_feature_stats=True,
        )
        baseline_logit_mean = base_stats["baseline_logit_mean"]
        soft_threshold_mean = base_stats["feature_mean"]
        soft_threshold_std = base_stats["feature_std"]
        projected_feature_mean = base_stats["projected_feature_mean"]
        projected_feature_std = base_stats["projected_feature_std"]

        train_stats = _accumulate_mode_statistics(
            model,
            linear_layer,
            train_loader,
            w_rand,
            out_features,
            device,
            description="RanPAC train transformed stats",
            adapt_noise_eps=adapt_noise_eps,
            adapt_noise_num=adapt_noise_num,
            adapt_alpha=adapt_alpha,
            hardneg_topk=hardneg_topk,
            hardneg_gamma=hardneg_gamma,
            feature_mode=feature_mode,
            projected_feature_mean=projected_feature_mean,
            projected_feature_std=projected_feature_std,
            feature_stat_eps=feature_stat_eps,
        )
        if train_stats["sample_count"] <= 0:
            raise ValueError("RanPAC transformed train statistics loader is empty.")

        projected_feature_center = (
            train_stats["feature_sum"] / float(train_stats["sample_count"])
        ).float()
        g_train, q_train = _center_ranpac_statistics(
            train_stats["g_matrix"],
            train_stats["q_matrix"],
            train_stats["feature_sum"],
            train_stats["target_sum"],
            train_stats["sample_count"],
            projected_feature_center,
        )
        stability_diagonal_prior = compute_stability_ridge_prior(
            train_stats["feature_sum"],
            train_stats["feature_sum_sq"],
            train_stats["feature_abs_sum"],
            train_stats["sample_count"],
            stability_ridge_gamma,
            stability_ridge_stat_eps,
        )

        if val_loader is not None:
            val_stats = _accumulate_mode_statistics(
                model,
                linear_layer,
                val_loader,
                w_rand,
                out_features,
                device,
                description="RanPAC val transformed stats",
                adapt_noise_eps=adapt_noise_eps,
                adapt_noise_num=adapt_noise_num,
                adapt_alpha=adapt_alpha,
                hardneg_topk=hardneg_topk,
                hardneg_gamma=hardneg_gamma,
                feature_mode=feature_mode,
                projected_feature_mean=projected_feature_mean,
                projected_feature_std=projected_feature_std,
                feature_stat_eps=feature_stat_eps,
            )
            g_val, q_val = _center_ranpac_statistics(
                val_stats["g_matrix"],
                val_stats["q_matrix"],
                val_stats["feature_sum"],
                val_stats["target_sum"],
                val_stats["sample_count"],
                projected_feature_center,
            )
            val_target_norm = val_stats["target_norm"]
            val_sample_count = val_stats["sample_count"]
        else:
            g_val = torch.zeros_like(g_train)
            q_val = torch.zeros_like(q_train)
            val_target_norm = 0.0
            val_sample_count = 0

    regression_ridge, mse_loss = _select_ridge_by_regression_loss(
        g_train,
        q_train,
        g_val,
        q_val,
        val_target_norm,
        out_features,
        val_sample_count,
        device,
        diagonal_prior=stability_diagonal_prior,
    )

    print(f"RanPAC optimal ridge (regression): {regression_ridge}")
    if mse_loss is not None:
        print(f"RanPAC best mse loss: {mse_loss:.6f}")

    g_full = (g_train + g_val).to(device)
    q_full = (q_train + q_val).to(device)
    weight = solve_ridge_system(g_full, q_full, regression_ridge, diagonal_prior=stability_diagonal_prior).t().cpu()

    state = {
        "version": RANPAC_CACHE_VERSION,
        "layer_name": layer_name,
        "in_features": in_features,
        "out_features": out_features,
        "rp_dim": rp_dim,
        "split_seed": seed,
        "ridge_candidates": RIDGE_CANDIDATES,
        "selection_method": "regression",
        "target_type": "hard_negative_supervised" if hardneg_topk > 0 and hardneg_gamma > 0 else "ground_truth",
        "adaptation_input_source": "noisy_only" if adapt_noise_num > 0 and adapt_noise_eps > 0 else "clean_only",
        "adapt_noise_eps": adapt_noise_eps,
        "adapt_noise_num": adapt_noise_num,
        "adapt_alpha": adapt_alpha,
        "hardneg_topk": hardneg_topk,
        "hardneg_gamma": hardneg_gamma,
        "baseline_logit_mean_source": "train_clean_global_scalar",
        "baseline_logit_mean": baseline_logit_mean,
        "soft_threshold_enabled": is_meansparse_enabled(soft_threshold_alpha),
        "soft_threshold_mean_source": "train_clean_feature_channel",
        "soft_threshold_mean": soft_threshold_mean,
        "soft_threshold_std": soft_threshold_std,
        "soft_threshold_stats_collected": True,
        "soft_threshold_alpha": soft_threshold_alpha,
        "soft_threshold_beta": soft_threshold_beta,
        "soft_threshold_stat_eps": soft_threshold_stat_eps,
        "feature_mode": feature_mode,
        "feature_stat_eps": feature_stat_eps,
        "projected_feature_mean": projected_feature_mean,
        "projected_feature_std": projected_feature_std,
        "projected_feature_center": projected_feature_center,
        "projected_feature_mean_source": "train_clean_projected_channel"
        if feature_mode != RANPAC_FEATURE_MODE_GELU
        else None,
        "projected_feature_center_source": "train_adaptation_feature_channel"
        if projected_feature_center is not None
        else None,
        "stability_ridge_enabled": is_stability_ridge_enabled(stability_ridge_gamma),
        "stability_ridge_gamma": stability_ridge_gamma,
        "stability_ridge_stat_eps": stability_ridge_stat_eps,
        "stability_ridge_source": "train_transformed_feature_snr"
        if feature_mode != RANPAC_FEATURE_MODE_GELU
        else "train_projected_feature_snr",
        "ridge": regression_ridge,
        "weight": weight,
        "w_rand": w_rand,
        "selection_metrics": {
            "mse_loss": mse_loss,
        },
    }
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(state, cache_path)
    return state


def apply_ranpac_head(
    model,
    classifier_name,
    dataset_root=None,
    rp_dim=5000,
    batch_size=256,
    num_workers=4,
    seed=0,
    device=None,
    cache_dir="pretrained/ranpac",
    selection_method="regression",
    train_loader=None,
    val_loader=None,
    train_transform=None,
    adapt_noise_eps=0.0,
    adapt_noise_num=0,
    adapt_alpha=1.0,
    ranpac_lambda=1.0,
    ranpac_temp=1.0,
    hardneg_topk=9,
    hardneg_gamma=1.0,
    soft_threshold_alpha=0.0,
    soft_threshold_beta=8.0,
    soft_threshold_stat_eps=DEFAULT_MEANSPARSE_STAT_EPS,
    feature_mode=RANPAC_FEATURE_MODE_GELU,
    feature_stat_eps=DEFAULT_RANPAC_FEATURE_STAT_EPS,
    stability_ridge_gamma=0.0,
    stability_ridge_stat_eps=DEFAULT_STABILITY_RIDGE_STAT_EPS,
):
    if train_loader is None and "imagenet" not in classifier_name:
        raise NotImplementedError("RanPAC head replacement is currently implemented for ImageNet classifiers only.")
    _validate_ranpac_feature_mode(feature_mode)
    if selection_method != "regression":
        raise ValueError(f"Unsupported RanPAC selection method '{selection_method}'.")
    if adapt_noise_eps < 0:
        raise ValueError("RanPAC adapt_noise_eps must be non-negative.")
    if adapt_noise_num < 0:
        raise ValueError("RanPAC adapt_noise_num must be non-negative.")
    if adapt_alpha < 0:
        raise ValueError("RanPAC adapt_alpha must be non-negative.")
    if ranpac_lambda < 0:
        raise ValueError("RanPAC ranpac_lambda must be in [0, inf).")
    if ranpac_temp <= 0:
        raise ValueError("RanPAC ranpac_temp must be positive.")
    if hardneg_topk < 0:
        raise ValueError("RanPAC hardneg_topk must be non-negative.")
    if hardneg_gamma < 0:
        raise ValueError("RanPAC hardneg_gamma must be non-negative.")
    if soft_threshold_alpha < 0:
        raise ValueError("RanPAC soft_threshold_alpha must be non-negative.")
    if is_meansparse_enabled(soft_threshold_alpha) and soft_threshold_beta <= 0:
        raise ValueError("RanPAC soft_threshold_beta must be positive when soft thresholding is enabled.")
    if soft_threshold_stat_eps <= 0:
        raise ValueError("RanPAC soft_threshold_stat_eps must be positive.")
    if feature_stat_eps <= 0:
        raise ValueError("RanPAC feature_stat_eps must be positive.")
    if stability_ridge_gamma < 0:
        raise ValueError("RanPAC stability_ridge_gamma must be non-negative.")
    if stability_ridge_stat_eps <= 0:
        raise ValueError("RanPAC stability_ridge_stat_eps must be positive.")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    noise_tag = (
        f"_neps{format_cache_value(adapt_noise_eps)}"
        f"_nnum{adapt_noise_num}"
        f"_na{format_cache_value(adapt_alpha)}"
    )
    hardneg_tag = (
        f"_htk{hardneg_topk}"
        f"_hg{format_cache_value(hardneg_gamma)}"
    )
    feature_mode_tag = build_ranpac_feature_mode_tag(
        feature_mode=feature_mode,
        feature_stat_eps=feature_stat_eps,
        separator="_",
    )
    stability_tag = build_stability_ridge_tag(
        gamma=stability_ridge_gamma,
        stat_eps=stability_ridge_stat_eps,
        separator="_",
    )
    cache_classifier_name = strip_meansparse_tag(classifier_name)
    cache_base = (
        f"{cache_classifier_name.replace('/', '_')}_rp{rp_dim}_seed{seed}"
        f"{noise_tag}{hardneg_tag}{feature_mode_tag}{stability_tag}"
    )
    cache_name = (
        f"{cache_base}_ranpac_v{RANPAC_CACHE_VERSION}.pt"
    )
    cache_path = os.path.join(cache_dir, cache_name)

    layer_name, linear_layer = _find_last_linear(model)
    if os.path.exists(cache_path):
        state = torch.load(cache_path, map_location="cpu")
    else:
        state = None

    if state is None or state.get("version") != RANPAC_CACHE_VERSION or "weight" not in state:
        state = _fit_ranpac_state(
            model,
            classifier_name=classifier_name,
            dataset_root=dataset_root,
            cache_path=cache_path,
            rp_dim=rp_dim,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            train_transform=train_transform,
            adapt_noise_eps=adapt_noise_eps,
            adapt_noise_num=adapt_noise_num,
            adapt_alpha=adapt_alpha,
            hardneg_topk=hardneg_topk,
            hardneg_gamma=hardneg_gamma,
            soft_threshold_alpha=soft_threshold_alpha,
            soft_threshold_beta=soft_threshold_beta,
            soft_threshold_stat_eps=soft_threshold_stat_eps,
            feature_mode=feature_mode,
            feature_stat_eps=feature_stat_eps,
            stability_ridge_gamma=stability_ridge_gamma,
            stability_ridge_stat_eps=stability_ridge_stat_eps,
        )

    if state["layer_name"] != layer_name:
        raise ValueError(f"Cached RanPAC head expects layer {state['layer_name']} but found {layer_name}.")
    if state["in_features"] != linear_layer.in_features or state["out_features"] != linear_layer.out_features:
        raise ValueError("Cached RanPAC head does not match the pretrained classifier dimensions.")

    ranpac_branch = RanPACLinear(
        in_features=state["in_features"],
        out_features=state["out_features"],
        rp_dim=state["rp_dim"],
        weight=state["weight"],
        w_rand=state["w_rand"],
        soft_threshold_mean=state["soft_threshold_mean"],
        soft_threshold_std=state["soft_threshold_std"],
        soft_threshold_alpha=soft_threshold_alpha,
        soft_threshold_beta=soft_threshold_beta,
        soft_threshold_stat_eps=soft_threshold_stat_eps,
        feature_mode=state.get("feature_mode", RANPAC_FEATURE_MODE_GELU),
        feature_stat_eps=state.get("feature_stat_eps", DEFAULT_RANPAC_FEATURE_STAT_EPS),
        projected_feature_mean=state.get("projected_feature_mean"),
        projected_feature_std=state.get("projected_feature_std"),
        projected_feature_center=state.get("projected_feature_center"),
    )
    ranpac_head = ResidualRanPACLinear(
        original_linear=linear_layer,
        ranpac_linear=ranpac_branch,
        ranpac_lambda=ranpac_lambda,
        ranpac_temp=ranpac_temp,
        baseline_logit_mean=state["baseline_logit_mean"],
    )
    _set_module_by_name(model, layer_name, ranpac_head)
    return model
