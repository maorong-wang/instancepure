import copy
import os
import random
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from classifiers.mean_sparse import (
    DEFAULT_MEANSPARSE_MODE,
    DEFAULT_MEANSPARSE_STAT_EPS,
    MEANSPARSE_MODES,
    apply_mean_centered_soft_threshold,
    build_meansparse_tag,
    format_cache_value,
    is_meansparse_enabled,
    strip_meansparse_tag,
    validate_meansparse_mode,
)
from classifiers.ranpac import RIDGE_CANDIDATES
from classifiers.stability_ridge import (
    DEFAULT_STABILITY_RIDGE_STAT_EPS,
    build_stability_ridge_tag,
    compute_stability_ridge_prior,
    is_stability_ridge_enabled,
    solve_ridge_system,
)
from dataset import get_dataset


HIRA_CACHE_VERSION = 35


def is_hira_subspace_enabled(subspace_rank, subspace_shrink):
    return int(subspace_rank) > 0 and float(subspace_shrink) < 1.0


def _resolve_hira_subspace_sample_cap(subspace_rank):
    if int(subspace_rank) <= 0:
        return 0
    return max(512, 8 * int(subspace_rank))


def build_hira_subspace_tag(subspace_rank, subspace_shrink, separator="-"):
    if int(subspace_rank) <= 0:
        return ""
    parts = [f"subr{int(subspace_rank)}"]
    if float(subspace_shrink) != 1.0:
        parts.append(f"subs{format_cache_value(subspace_shrink)}")
    return separator + separator.join(parts)


class HiRAHalfPrecisionWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.use_autocast = True

    def forward(self, x):
        autocast_context = nullcontext()
        if self.use_autocast and x.device.type == "cuda":
            autocast_context = torch.autocast(device_type="cuda", dtype=torch.float16)
        with autocast_context:
            return self.base_model(x)


class HiRAAdapter(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        expansion_dim,
        soft_threshold_alpha=0.0,
        soft_threshold_beta=8.0,
        soft_threshold_stat_eps=DEFAULT_MEANSPARSE_STAT_EPS,
        soft_threshold_mode=DEFAULT_MEANSPARSE_MODE,
        subspace_rank=0,
        subspace_shrink=1.0,
    ):
        super().__init__()
        self.register_buffer("b_rand", torch.randn(in_features, expansion_dim, dtype=torch.float16))
        self.a_weight = nn.Parameter(torch.empty(out_features, expansion_dim, dtype=torch.float16))
        self.register_buffer("soft_threshold_mean", torch.zeros(expansion_dim, dtype=torch.float32))
        self.register_buffer("soft_threshold_std", torch.ones(expansion_dim, dtype=torch.float32))
        self.register_buffer("clean_subspace_mean", torch.zeros(expansion_dim, dtype=torch.float32))
        self.register_buffer("clean_subspace_basis", torch.zeros(expansion_dim, int(subspace_rank), dtype=torch.float32))
        self.register_buffer("clean_subspace_coeff_mean", torch.zeros(int(subspace_rank), dtype=torch.float32))
        self.register_buffer("clean_subspace_coeff_std", torch.ones(int(subspace_rank), dtype=torch.float32))
        self.register_buffer("clean_subspace_valid_rank", torch.zeros((), dtype=torch.int64))
        self.soft_threshold_alpha = float(soft_threshold_alpha)
        self.soft_threshold_beta = float(soft_threshold_beta)
        self.soft_threshold_stat_eps = float(soft_threshold_stat_eps)
        self.soft_threshold_mode = validate_meansparse_mode(soft_threshold_mode)
        self.subspace_rank = int(subspace_rank)
        self.subspace_shrink = float(subspace_shrink)
        self.force_fp32 = False
        self._fp32_cache_device = None
        self._b_rand_fp32 = None
        self._a_weight_fp32 = None
        nn.init.kaiming_uniform_(self.a_weight, a=0.0, nonlinearity="relu")

    def clear_fp32_cache(self):
        self._fp32_cache_device = None
        self._b_rand_fp32 = None
        self._a_weight_fp32 = None

    def _ensure_fp32_cache(self, device):
        cache_device = str(device)
        if self._fp32_cache_device == cache_device:
            return
        self._b_rand_fp32 = self.b_rand.detach().float().to(device=device)
        self._a_weight_fp32 = self.a_weight.detach().float().to(device=device)
        self._fp32_cache_device = cache_device

    def apply_clean_subspace_calibration(self, projected):
        if (
            self.subspace_rank <= 0
            or self.subspace_shrink >= 1.0
            or int(self.clean_subspace_valid_rank.item()) <= 0
        ):
            return projected

        projected_float = projected.float()
        mean = self.clean_subspace_mean.to(device=projected.device, dtype=torch.float32).view(1, -1)
        basis = self.clean_subspace_basis[:, : int(self.clean_subspace_valid_rank.item())].to(
            device=projected.device,
            dtype=torch.float32,
        )
        delta = projected_float - mean
        coeff = delta @ basis
        coeff_std = self.clean_subspace_coeff_std[: int(self.clean_subspace_valid_rank.item())].to(
            device=projected.device,
            dtype=torch.float32,
        ).clamp_min(float(self.soft_threshold_stat_eps)).view(1, -1)
        coeff_scale = float(self.soft_threshold_alpha) * coeff_std
        coeff_scale = coeff_scale.clamp_min(float(self.soft_threshold_stat_eps))
        coeff_tilde = coeff_scale * torch.tanh(coeff / coeff_scale)

        parallel = coeff @ basis.t()
        parallel_tilde = coeff_tilde @ basis.t()
        orthogonal = delta - parallel
        calibrated = mean + parallel_tilde + self.subspace_shrink * orthogonal
        return calibrated.to(dtype=projected.dtype)

    def project_hidden(self, x, apply_soft_threshold):
        token_features = x.reshape(-1, x.shape[-1])
        if self.force_fp32:
            token_features = token_features.float()
            if token_features.device.type == "cuda":
                self._ensure_fp32_cache(token_features.device)
                projected = token_features @ self._b_rand_fp32
            else:
                projected = token_features @ self.b_rand.float()
        elif token_features.device.type == "cuda":
            token_features = token_features.to(dtype=self.b_rand.dtype)
            projected = token_features @ self.b_rand
        else:
            token_features = token_features.float()
            projected = token_features @ self.b_rand.float()
        projected = F.gelu(projected)
        if not self.training:
            projected = self.apply_clean_subspace_calibration(projected)
        return projected

    def forward(self, x):
        token_shape = x.shape[:-1]
        projected = self.project_hidden(x, apply_soft_threshold=True)
        if projected.device.type == "cuda" and not self.force_fp32:
            output = projected @ self.a_weight.t()
        else:
            if projected.device.type == "cuda":
                self._ensure_fp32_cache(projected.device)
                output = projected.float() @ self._a_weight_fp32.t()
            else:
                output = projected.float() @ self.a_weight.float().t()
        return output.to(x.dtype).reshape(*token_shape, self.a_weight.shape[0])


class HiRAMlpWrapper(nn.Module):
    def __init__(
        self,
        base_mlp,
        expansion_dim,
        soft_threshold_alpha=0.0,
        soft_threshold_beta=8.0,
        soft_threshold_stat_eps=DEFAULT_MEANSPARSE_STAT_EPS,
        soft_threshold_mode=DEFAULT_MEANSPARSE_MODE,
        subspace_rank=0,
        subspace_shrink=1.0,
    ):
        super().__init__()
        self.base_mlp = base_mlp
        mlp_input_dim = _infer_mlp_input_dim(base_mlp)
        self.mlp_adapter = HiRAAdapter(
            mlp_input_dim,
            mlp_input_dim,
            expansion_dim,
            soft_threshold_alpha=soft_threshold_alpha,
            soft_threshold_beta=soft_threshold_beta,
            soft_threshold_stat_eps=soft_threshold_stat_eps,
            soft_threshold_mode=soft_threshold_mode,
            subspace_rank=subspace_rank,
            subspace_shrink=subspace_shrink,
        )

    def forward(self, x, *args, **kwargs):
        return self.base_mlp(self.mlp_adapter(x), *args, **kwargs)


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


def _list_mlp_modules(model):
    return [
        (name, module)
        for name, module in model.named_modules()
        if name.endswith(".mlp")
    ]


def _infer_mlp_input_dim(module):
    for attr_path in ("fc1.in_features", "fc1.in_channels"):
        try:
            candidate = _get_module_by_name(module, attr_path)
        except AttributeError:
            continue
        if isinstance(candidate, int):
            return candidate
    raise ValueError("Unable to infer MLP input dimension for HiRA attachment.")


def _resolve_target_mlp_modules(model, num_adapter_blocks):
    mlp_modules = _list_mlp_modules(model)
    if num_adapter_blocks <= 0:
        raise ValueError("HiRA requires --hira_num_blocks to be a positive integer.")
    if len(mlp_modules) < num_adapter_blocks:
        raise ValueError(
            f"HiRA requested {num_adapter_blocks} adapter blocks, but the backbone only exposes {len(mlp_modules)} MLP blocks."
        )
    return mlp_modules[-num_adapter_blocks:]


def _attach_hira_modules(
    model,
    expansion_dim,
    num_adapter_blocks,
    soft_threshold_alpha=0.0,
    soft_threshold_beta=8.0,
    soft_threshold_stat_eps=DEFAULT_MEANSPARSE_STAT_EPS,
    soft_threshold_mode=DEFAULT_MEANSPARSE_MODE,
    subspace_rank=0,
    subspace_shrink=1.0,
):
    target_mlp_names = []
    for module_name, module in _resolve_target_mlp_modules(model, num_adapter_blocks):
        if isinstance(module, HiRAMlpWrapper):
            module.mlp_adapter.soft_threshold_alpha = float(soft_threshold_alpha)
            module.mlp_adapter.soft_threshold_beta = float(soft_threshold_beta)
            module.mlp_adapter.soft_threshold_stat_eps = float(soft_threshold_stat_eps)
            module.mlp_adapter.soft_threshold_mode = validate_meansparse_mode(soft_threshold_mode)
            module.mlp_adapter.subspace_rank = int(subspace_rank)
            module.mlp_adapter.subspace_shrink = float(subspace_shrink)
            if module.mlp_adapter.clean_subspace_basis.shape[1] != int(subspace_rank):
                expansion_dim_current = module.mlp_adapter.clean_subspace_mean.numel()
                module.mlp_adapter.clean_subspace_basis = torch.zeros(
                    expansion_dim_current,
                    int(subspace_rank),
                    dtype=torch.float32,
                )
                module.mlp_adapter.clean_subspace_coeff_mean = torch.zeros(
                    int(subspace_rank),
                    dtype=torch.float32,
                )
                module.mlp_adapter.clean_subspace_coeff_std = torch.ones(
                    int(subspace_rank),
                    dtype=torch.float32,
                )
                module.mlp_adapter.clean_subspace_valid_rank.zero_()
            target_mlp_names.append(module_name)
            continue
        _set_module_by_name(
            model,
            module_name,
            HiRAMlpWrapper(
                module,
                expansion_dim,
                soft_threshold_alpha=soft_threshold_alpha,
                soft_threshold_beta=soft_threshold_beta,
                soft_threshold_stat_eps=soft_threshold_stat_eps,
                soft_threshold_mode=soft_threshold_mode,
                subspace_rank=subspace_rank,
                subspace_shrink=subspace_shrink,
            ),
        )
        target_mlp_names.append(module_name)
    return target_mlp_names


def _freeze_model(model):
    for parameter in model.parameters():
        parameter.requires_grad = False


def _cached_hira_state_dict(model, mlp_module_names):
    cache_prefixes = tuple(f"{module_name}.mlp_adapter." for module_name in mlp_module_names)
    return {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if name.startswith(cache_prefixes)
    }


def _enable_half_precision_forward(model):
    if isinstance(model, HiRAHalfPrecisionWrapper):
        return model
    return HiRAHalfPrecisionWrapper(model)


def set_hira_half_precision(model, enabled):
    for module in model.modules():
        if isinstance(module, HiRAHalfPrecisionWrapper):
            module.use_autocast = enabled
        elif isinstance(module, HiRAAdapter):
            module.force_fp32 = not enabled
            if enabled:
                module.clear_fp32_cache()
    return model


def _prepare_hira_model_for_eval(model):
    if isinstance(model, HiRAHalfPrecisionWrapper):
        model.use_autocast = False
        model = model.base_model
    model = model.float()
    for module in model.modules():
        if isinstance(module, HiRAAdapter):
            module.force_fp32 = False
            module.clear_fp32_cache()
    return model.eval()


def _format_block_tag(num_adapter_blocks, prefix):
    if num_adapter_blocks == 2:
        return ""
    return f"{prefix}{num_adapter_blocks}"


def _sample_linf_noisy_inputs(inputs, eps):
    noise = torch.empty_like(inputs).uniform_(-eps, eps)
    return torch.clamp(inputs + noise, 0.0, 1.0)


def build_hira_variant_name(
    classifier_name,
    expansion_dim,
    epochs,
    lr,
    weight_decay,
    max_train_samples,
    seed,
    num_adapter_blocks,
    adapt_noise_eps=0.0,
    adapt_noise_num=0,
    adapt_alpha=1.0,
    soft_threshold_alpha=0.0,
    soft_threshold_beta=8.0,
    soft_threshold_stat_eps=DEFAULT_MEANSPARSE_STAT_EPS,
    soft_threshold_mode=DEFAULT_MEANSPARSE_MODE,
    subspace_rank=0,
    subspace_shrink=1.0,
    stability_ridge_gamma=0.0,
    stability_ridge_stat_eps=DEFAULT_STABILITY_RIDGE_STAT_EPS,
):
    del epochs, lr, weight_decay

    sample_tag = "full" if max_train_samples is None or max_train_samples < 0 else str(max_train_samples)
    block_tag = _format_block_tag(num_adapter_blocks, "-blk")
    noise_tag = (
        f"-neps{format_cache_value(adapt_noise_eps)}"
        f"-nnum{adapt_noise_num}"
        f"-na{format_cache_value(adapt_alpha)}"
    )
    meansparse_tag = build_meansparse_tag(
        alpha=soft_threshold_alpha,
        beta=soft_threshold_beta,
        stat_eps=soft_threshold_stat_eps,
        separator="-",
        mode=soft_threshold_mode,
    )
    subspace_tag = build_hira_subspace_tag(
        subspace_rank=subspace_rank,
        subspace_shrink=subspace_shrink,
        separator="-",
    )
    stability_tag = build_stability_ridge_tag(
        gamma=stability_ridge_gamma,
        stat_eps=stability_ridge_stat_eps,
        separator="-",
    )
    return (
        f"{classifier_name}-hira-v{HIRA_CACHE_VERSION}-pre-mlp-closedform-gelu"
        f"{block_tag}"
        f"-exp{expansion_dim}"
        f"-ns{sample_tag}"
        f"-seed{seed}"
        f"{noise_tag}"
        f"{meansparse_tag}"
        f"{subspace_tag}"
        f"{stability_tag}"
    )


def _build_cache_name(
    classifier_name,
    expansion_dim,
    epochs,
    lr,
    weight_decay,
    max_train_samples,
    seed,
    num_adapter_blocks,
    adapt_noise_eps,
    adapt_noise_num,
    adapt_alpha,
    soft_threshold_alpha,
    soft_threshold_beta,
    soft_threshold_stat_eps,
    soft_threshold_mode,
    subspace_rank,
    subspace_shrink,
    stability_ridge_gamma,
    stability_ridge_stat_eps,
):
    del soft_threshold_alpha, soft_threshold_beta, soft_threshold_stat_eps, soft_threshold_mode, subspace_shrink
    return build_hira_variant_name(
        classifier_name=strip_meansparse_tag(classifier_name),
        expansion_dim=expansion_dim,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        max_train_samples=max_train_samples,
        seed=seed,
        num_adapter_blocks=num_adapter_blocks,
        adapt_noise_eps=adapt_noise_eps,
        adapt_noise_num=adapt_noise_num,
        adapt_alpha=adapt_alpha,
        soft_threshold_alpha=0.0,
        soft_threshold_beta=8.0,
        soft_threshold_stat_eps=DEFAULT_MEANSPARSE_STAT_EPS,
        soft_threshold_mode=DEFAULT_MEANSPARSE_MODE,
        subspace_rank=subspace_rank,
        subspace_shrink=1.0,
        stability_ridge_gamma=stability_ridge_gamma,
        stability_ridge_stat_eps=stability_ridge_stat_eps,
    ).replace("/", "_") + ".pt"


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_train_loaders(dataset_root, batch_size, num_workers, seed, max_train_samples, train_transform=None):
    if dataset_root:
        os.environ["IMAGENET_LOC_ENV"] = dataset_root

    dataset = get_dataset("imagenet", split="train", adv=False)
    if hasattr(dataset, "transform") and train_transform is not None:
        dataset.transform = train_transform
    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    if max_train_samples is not None and max_train_samples > 0 and max_train_samples < len(indices):
        indices = indices[:max_train_samples]

    if len(indices) == 1:
        train_indices = indices
        val_indices = []
    else:
        train_cutoff = int(len(indices) * 0.8)
        train_cutoff = min(max(train_cutoff, 1), len(indices) - 1)
        train_indices = indices[:train_cutoff]
        val_indices = indices[train_cutoff:]

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

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


def _init_hira_statistics(
    model,
    mlp_module_names,
    stats_device,
    collect_feature_stats=False,
    collect_projected_stability_stats=False,
    collect_clean_subspace_stats=False,
    clean_subspace_rank=0,
):
    statistics = {}
    for module_name in mlp_module_names:
        wrapper = _get_module_by_name(model, module_name)
        rp_dim = wrapper.mlp_adapter.b_rand.size(1)
        out_dim = wrapper.mlp_adapter.a_weight.size(0)
        entry = {
            "g_matrix": torch.zeros(rp_dim, rp_dim, dtype=torch.float32, device=stats_device),
            "q_matrix": torch.zeros(rp_dim, out_dim, dtype=torch.float32, device=stats_device),
            "target_norm": torch.zeros((), dtype=torch.float32, device=stats_device),
            "sample_count": 0.0,
        }
        if collect_feature_stats:
            entry["feature_sum"] = torch.zeros(rp_dim, dtype=torch.float64, device=stats_device)
            entry["feature_sum_sq"] = torch.zeros(rp_dim, dtype=torch.float64, device=stats_device)
            entry["feature_sample_count"] = 0
        if collect_clean_subspace_stats:
            subspace_sample_cap = _resolve_hira_subspace_sample_cap(clean_subspace_rank)
            entry["clean_subspace_sample_cap"] = subspace_sample_cap
            entry["clean_subspace_rank"] = int(clean_subspace_rank)
            entry["clean_subspace_samples"] = torch.empty(
                subspace_sample_cap,
                rp_dim,
                dtype=torch.float16,
                device="cpu",
            )
            entry["clean_subspace_sample_count"] = 0
        if collect_projected_stability_stats:
            entry["projected_sum"] = torch.zeros(rp_dim, dtype=torch.float64, device=stats_device)
            entry["projected_sum_sq"] = torch.zeros(rp_dim, dtype=torch.float64, device=stats_device)
            entry["projected_abs_sum"] = torch.zeros(rp_dim, dtype=torch.float64, device=stats_device)
            entry["projected_sample_count"] = 0.0
        statistics[module_name] = entry
    return statistics


def _accumulate_feature_statistics(entry, source_tokens):
    if "feature_sum" not in entry:
        return

    source_tokens = source_tokens.to(dtype=torch.float64)
    entry["feature_sum"] += source_tokens.sum(dim=0)
    entry["feature_sum_sq"] += source_tokens.square().sum(dim=0)
    entry["feature_sample_count"] += source_tokens.size(0)


def _accumulate_clean_subspace_samples(entry, source_tokens):
    sample_cap = entry.get("clean_subspace_sample_cap", 0)
    sample_count = entry.get("clean_subspace_sample_count", 0)
    if sample_cap <= 0 or sample_count >= sample_cap:
        return

    take_count = min(sample_cap - sample_count, source_tokens.size(0))
    entry["clean_subspace_samples"][sample_count:sample_count + take_count].copy_(
        source_tokens[:take_count].to(device="cpu", dtype=torch.float16)
    )
    entry["clean_subspace_sample_count"] += take_count


def _build_clean_subspace_basis(entry):
    requested_rank = int(entry.get("clean_subspace_rank", 0))
    subspace_rank = min(
        requested_rank,
        int(entry.get("clean_subspace_sample_count", 0)),
    )
    feature_mean = entry["feature_sum"] / float(entry["feature_sample_count"])
    rp_dim = feature_mean.numel()
    empty_mean = torch.zeros(requested_rank, dtype=torch.float32)
    empty_std = torch.ones(requested_rank, dtype=torch.float32)
    if requested_rank <= 0:
        return feature_mean.float().cpu(), torch.empty(rp_dim, 0, dtype=torch.float32), empty_mean, empty_std, 0
    if subspace_rank <= 0:
        return feature_mean.float().cpu(), torch.zeros(rp_dim, requested_rank, dtype=torch.float32), empty_mean, empty_std, 0

    sampled = entry["clean_subspace_samples"][: entry["clean_subspace_sample_count"]].float()
    centered = sampled - feature_mean.float().cpu().view(1, -1)
    if centered.size(0) <= 1:
        return feature_mean.float().cpu(), torch.zeros(rp_dim, requested_rank, dtype=torch.float32), empty_mean, empty_std, 0

    gram = centered @ centered.t()
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    positive = eigenvalues > 1e-8
    if int(positive.sum().item()) == 0:
        return feature_mean.float().cpu(), torch.zeros(rp_dim, requested_rank, dtype=torch.float32), empty_mean, empty_std, 0

    valid_rank = min(subspace_rank, int(positive.sum().item()))
    top_values = eigenvalues[positive][-valid_rank:]
    top_vectors = eigenvectors[:, positive][:, -valid_rank:]
    basis = centered.t() @ top_vectors
    basis = basis / top_values.clamp_min(1e-8).sqrt().unsqueeze(0)
    basis = torch.linalg.qr(basis, mode="reduced").Q.float().cpu()
    coeff = centered @ basis[:, :valid_rank]
    coeff_mean = coeff.mean(dim=0).float().cpu()
    coeff_std = coeff.var(dim=0, unbiased=False).clamp_min(0.0).sqrt().float().cpu()
    if valid_rank < requested_rank:
        padded_basis = torch.zeros(rp_dim, requested_rank, dtype=torch.float32)
        padded_basis[:, :valid_rank] = basis
        basis = padded_basis
        padded_coeff_mean = torch.zeros(requested_rank, dtype=torch.float32)
        padded_coeff_std = torch.ones(requested_rank, dtype=torch.float32)
        padded_coeff_mean[:valid_rank] = coeff_mean
        padded_coeff_std[:valid_rank] = coeff_std
        coeff_mean = padded_coeff_mean
        coeff_std = padded_coeff_std
    return feature_mean.float().cpu(), basis, coeff_mean, coeff_std, valid_rank


def _project_hira_tokens(source_tokens, b_rand):
    projected_inputs = source_tokens.to(dtype=b_rand.dtype)
    return F.gelu(projected_inputs @ b_rand).float()


def _accumulate_projected_statistics(
    entry,
    projected,
    target_tokens,
    weight,
    collect_projected_stability_stats=False,
):
    if weight <= 0:
        return

    if collect_projected_stability_stats:
        entry["projected_sum"] += weight * projected.sum(dim=0, dtype=torch.float64)
        entry["projected_sum_sq"] += weight * projected.square().sum(dim=0, dtype=torch.float64)
        entry["projected_abs_sum"] += weight * projected.abs().sum(dim=0, dtype=torch.float64)
        entry["projected_sample_count"] += weight * projected.size(0)
    entry["g_matrix"] += weight * (projected.t() @ projected)
    entry["q_matrix"] += weight * (projected.t() @ target_tokens)
    entry["target_norm"] += weight * target_tokens.square().sum()
    entry["sample_count"] += weight * target_tokens.size(0)


def _select_ridge_by_regression_loss(
    g_train,
    q_train,
    g_val,
    q_val,
    val_target_norm,
    out_features,
    val_sample_count,
    device,
    diagonal_prior=None,
):
    if val_sample_count == 0:
        return RIDGE_CANDIDATES[0], None

    g_train = g_train.to(device)
    q_train = q_train.to(device)
    g_val = g_val.to(device)
    q_val = q_val.to(device)
    best_ridge = RIDGE_CANDIDATES[0]
    best_loss = None
    denominator = max(val_sample_count * out_features, 1)

    with torch.no_grad():
        for ridge in RIDGE_CANDIDATES:
            weight = solve_ridge_system(g_train, q_train, ridge, diagonal_prior=diagonal_prior).t()
            quadratic = torch.trace(weight @ g_val @ weight.t())
            cross_term = torch.trace(weight @ q_val)
            loss = (quadratic - 2.0 * cross_term + val_target_norm) / denominator
            loss_value = loss.item()
            if best_loss is None or loss_value < best_loss:
                best_loss = loss_value
                best_ridge = ridge

    return best_ridge, best_loss


def _accumulate_hira_statistics(
    model,
    teacher_model,
    loader,
    mlp_module_names,
    device,
    description,
    adapt_noise_eps,
    adapt_noise_num,
    adapt_alpha,
    collect_feature_stats=False,
    collect_projected_stability_stats=False,
    collect_clean_subspace_stats=False,
    clean_subspace_rank=0,
):
    stats_device = device if device.type == "cuda" else torch.device("cpu")
    statistics = _init_hira_statistics(
        model,
        mlp_module_names,
        stats_device,
        collect_feature_stats=collect_feature_stats,
        collect_projected_stability_stats=collect_projected_stability_stats,
        collect_clean_subspace_stats=collect_clean_subspace_stats,
        clean_subspace_rank=clean_subspace_rank,
    )
    b_rand_by_module = {
        module_name: _get_module_by_name(model, module_name).mlp_adapter.b_rand.to(
            device=stats_device,
            dtype=torch.float16 if stats_device.type == "cuda" else torch.float32,
        )
        for module_name in mlp_module_names
    }
    batch_cache = {}
    handles = []
    del adapt_alpha
    use_noisy_adaptation = adapt_noise_num > 0 and adapt_noise_eps > 0
    noisy_sample_weight = 1.0 / adapt_noise_num if use_noisy_adaptation else 0.0

    def _run_teacher_forward(batch_inputs):
        autocast_context = nullcontext()
        if device.type == "cuda":
            autocast_context = torch.autocast(device_type="cuda", dtype=torch.float16)
        with autocast_context:
            _ = teacher_model(batch_inputs)

    for module_name in mlp_module_names:
        teacher_module = _get_module_by_name(teacher_model, module_name)

        def hook(_, inputs, module_name=module_name):
            batch_cache[module_name] = inputs[0].detach()

        handles.append(teacher_module.register_forward_pre_hook(hook))

    teacher_model = teacher_model.to(device).eval()
    try:
        with torch.no_grad():
            for inputs, _ in tqdm(loader, desc=description):
                inputs = inputs.to(device, non_blocking=True)
                if collect_feature_stats or not use_noisy_adaptation:
                    batch_cache.clear()
                    _run_teacher_forward(inputs)
                    for module_name in mlp_module_names:
                        clean_inputs = batch_cache[module_name].reshape(-1, batch_cache[module_name].shape[-1]).to(
                            stats_device,
                            dtype=torch.float32,
                        )
                        clean_projected = _project_hira_tokens(clean_inputs, b_rand_by_module[module_name])
                        if collect_feature_stats:
                            _accumulate_feature_statistics(statistics[module_name], clean_projected)
                        if collect_clean_subspace_stats:
                            _accumulate_clean_subspace_samples(statistics[module_name], clean_projected)
                        if not use_noisy_adaptation:
                            _accumulate_projected_statistics(
                                statistics[module_name],
                                clean_projected,
                                clean_inputs,
                                weight=1.0,
                                collect_projected_stability_stats=collect_projected_stability_stats,
                            )

                if use_noisy_adaptation:
                    for _ in range(adapt_noise_num):
                        noisy_inputs = _sample_linf_noisy_inputs(inputs, adapt_noise_eps)
                        batch_cache.clear()
                        _run_teacher_forward(noisy_inputs)

                        for module_name in mlp_module_names:
                            noisy_inputs_tokens = batch_cache[module_name].reshape(
                                -1,
                                batch_cache[module_name].shape[-1],
                            ).to(
                                stats_device,
                                dtype=torch.float32,
                            )
                            noisy_projected = _project_hira_tokens(
                                noisy_inputs_tokens,
                                b_rand_by_module[module_name],
                            )
                            _accumulate_projected_statistics(
                                statistics[module_name],
                                noisy_projected,
                                noisy_inputs_tokens,
                                weight=noisy_sample_weight,
                                collect_projected_stability_stats=collect_projected_stability_stats,
                            )
    finally:
        for handle in handles:
            handle.remove()

    if collect_feature_stats:
        for module_name in mlp_module_names:
            entry = statistics[module_name]
            if entry["feature_sample_count"] == 0:
                raise ValueError(f"HiRA soft-threshold statistics loader is empty for {module_name}.")
            feature_mean = entry["feature_sum"] / float(entry["feature_sample_count"])
            feature_var = entry["feature_sum_sq"] / float(entry["feature_sample_count"]) - feature_mean.square()
            entry["soft_threshold_mean"] = feature_mean.float().cpu()
            entry["soft_threshold_std"] = feature_var.clamp_min(0.0).sqrt().float().cpu()
            if collect_clean_subspace_stats:
                (
                    clean_subspace_mean,
                    clean_subspace_basis,
                    clean_subspace_coeff_mean,
                    clean_subspace_coeff_std,
                    clean_subspace_valid_rank,
                ) = _build_clean_subspace_basis(entry)
                entry["clean_subspace_mean"] = clean_subspace_mean
                entry["clean_subspace_basis"] = clean_subspace_basis
                entry["clean_subspace_coeff_mean"] = clean_subspace_coeff_mean
                entry["clean_subspace_coeff_std"] = clean_subspace_coeff_std
                entry["clean_subspace_valid_rank"] = clean_subspace_valid_rank

    return statistics


def _fit_hira_weights_closed_form(
    model,
    teacher_model,
    train_loader,
    val_loader,
    mlp_module_names,
    device,
    adapt_noise_eps,
    adapt_noise_num,
    adapt_alpha,
    subspace_rank,
    stability_ridge_gamma,
    stability_ridge_stat_eps,
):
    # Fit the pre-MLP adapter on the original continuous MLP inputs and collect
    # clean GELU(A(x)) statistics for inference-time sparsification.
    train_stats = _accumulate_hira_statistics(
        model,
        teacher_model,
        train_loader,
        mlp_module_names,
        device,
        description="HiRA train stats",
        adapt_noise_eps=adapt_noise_eps,
        adapt_noise_num=adapt_noise_num,
        adapt_alpha=adapt_alpha,
        collect_feature_stats=True,
        collect_clean_subspace_stats=int(subspace_rank) > 0,
        clean_subspace_rank=subspace_rank,
        collect_projected_stability_stats=is_stability_ridge_enabled(stability_ridge_gamma),
    )
    val_stats = _accumulate_hira_statistics(
        model,
        teacher_model,
        val_loader,
        mlp_module_names,
        device,
        description="HiRA val stats",
        adapt_noise_eps=adapt_noise_eps,
        adapt_noise_num=adapt_noise_num,
        adapt_alpha=adapt_alpha,
    )

    fit_summary = {}
    for module_name in mlp_module_names:
        wrapper = _get_module_by_name(model, module_name)
        train_entry = train_stats[module_name]
        val_entry = val_stats[module_name]
        out_features = wrapper.mlp_adapter.a_weight.size(0)
        diagonal_prior = compute_stability_ridge_prior(
            train_entry.get("projected_sum"),
            train_entry.get("projected_sum_sq"),
            train_entry.get("projected_abs_sum"),
            train_entry.get("projected_sample_count", 0.0),
            stability_ridge_gamma,
            stability_ridge_stat_eps,
        )

        with torch.no_grad():
            wrapper.mlp_adapter.soft_threshold_mean.copy_(train_entry["soft_threshold_mean"])
            wrapper.mlp_adapter.soft_threshold_std.copy_(train_entry["soft_threshold_std"])
            if int(subspace_rank) > 0:
                wrapper.mlp_adapter.clean_subspace_mean.copy_(train_entry["clean_subspace_mean"])
                wrapper.mlp_adapter.clean_subspace_basis = train_entry["clean_subspace_basis"].to(
                    dtype=torch.float32,
                )
                wrapper.mlp_adapter.clean_subspace_coeff_mean.copy_(train_entry["clean_subspace_coeff_mean"])
                wrapper.mlp_adapter.clean_subspace_coeff_std.copy_(train_entry["clean_subspace_coeff_std"])
                wrapper.mlp_adapter.clean_subspace_valid_rank.fill_(int(train_entry["clean_subspace_valid_rank"]))

        ridge, regression_loss = _select_ridge_by_regression_loss(
            train_entry["g_matrix"],
            train_entry["q_matrix"],
            val_entry["g_matrix"],
            val_entry["q_matrix"],
            val_entry["target_norm"],
            out_features,
            val_entry["sample_count"],
            device,
            diagonal_prior=diagonal_prior,
        )

        g_full = (train_entry["g_matrix"] + val_entry["g_matrix"]).to(device)
        q_full = (train_entry["q_matrix"] + val_entry["q_matrix"]).to(device)
        weight = solve_ridge_system(g_full, q_full, ridge, diagonal_prior=diagonal_prior).t()
        weight = weight.to(dtype=wrapper.mlp_adapter.a_weight.dtype).cpu()
        with torch.no_grad():
            wrapper.mlp_adapter.a_weight.copy_(weight)

        fit_summary[module_name] = {
            "ridge": ridge,
            "regression_loss": regression_loss,
            "train_tokens": train_entry["sample_count"],
            "val_tokens": val_entry["sample_count"],
        }
        print(f"HiRA optimal ridge for {module_name}: {ridge}")
        if regression_loss is not None:
            print(f"HiRA regression loss for {module_name}: {regression_loss:.6f}")

    return fit_summary


def apply_hira_adaptation(
    model,
    classifier_name,
    dataset_root=None,
    expansion_dim=4096,
    num_adapter_blocks=2,
    batch_size=32,
    num_workers=4,
    epochs=1,
    lr=1e-4,
    weight_decay=1e-4,
    grad_clip_norm=1.0,
    seed=0,
    device=None,
    cache_dir="pretrained/hira",
    max_train_samples=-1,
    force_retrain=False,
    train_transform=None,
    adapt_noise_eps=0.0,
    adapt_noise_num=0,
    adapt_alpha=1.0,
    soft_threshold_alpha=0.0,
    soft_threshold_beta=8.0,
    soft_threshold_stat_eps=DEFAULT_MEANSPARSE_STAT_EPS,
    soft_threshold_mode=DEFAULT_MEANSPARSE_MODE,
    subspace_rank=0,
    subspace_shrink=1.0,
    stability_ridge_gamma=0.0,
    stability_ridge_stat_eps=DEFAULT_STABILITY_RIDGE_STAT_EPS,
):
    if adapt_noise_eps < 0:
        raise ValueError("HiRA adapt_noise_eps must be non-negative.")
    if adapt_noise_num < 0:
        raise ValueError("HiRA adapt_noise_num must be non-negative.")
    if adapt_alpha < 0:
        raise ValueError("HiRA adapt_alpha must be non-negative.")
    if soft_threshold_alpha < 0:
        raise ValueError("HiRA soft_threshold_alpha must be non-negative.")
    if is_meansparse_enabled(soft_threshold_alpha) and soft_threshold_beta <= 0:
        raise ValueError("HiRA soft_threshold_beta must be positive when soft thresholding is enabled.")
    if soft_threshold_stat_eps <= 0:
        raise ValueError("HiRA soft_threshold_stat_eps must be positive.")
    validate_meansparse_mode(soft_threshold_mode)
    if subspace_rank < 0:
        raise ValueError("HiRA subspace_rank must be non-negative.")
    if not 0.0 <= subspace_shrink <= 1.0:
        raise ValueError("HiRA subspace_shrink must lie in [0, 1].")
    if stability_ridge_gamma < 0:
        raise ValueError("HiRA stability_ridge_gamma must be non-negative.")
    if stability_ridge_stat_eps <= 0:
        raise ValueError("HiRA stability_ridge_stat_eps must be positive.")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    teacher_model = copy.deepcopy(model).eval()
    _seed_everything(seed)
    mlp_module_names = _attach_hira_modules(
        model,
        expansion_dim,
        num_adapter_blocks,
        soft_threshold_alpha=soft_threshold_alpha,
        soft_threshold_beta=soft_threshold_beta,
        soft_threshold_stat_eps=soft_threshold_stat_eps,
        soft_threshold_mode=soft_threshold_mode,
        subspace_rank=subspace_rank,
        subspace_shrink=subspace_shrink,
    )
    _freeze_model(model)

    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(
        cache_dir,
        _build_cache_name(
            classifier_name=classifier_name,
            expansion_dim=expansion_dim,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            max_train_samples=max_train_samples,
            seed=seed,
            num_adapter_blocks=num_adapter_blocks,
            adapt_noise_eps=adapt_noise_eps,
            adapt_noise_num=adapt_noise_num,
            adapt_alpha=adapt_alpha,
            soft_threshold_alpha=soft_threshold_alpha,
            soft_threshold_beta=soft_threshold_beta,
            soft_threshold_stat_eps=soft_threshold_stat_eps,
            soft_threshold_mode=soft_threshold_mode,
            subspace_rank=subspace_rank,
            subspace_shrink=subspace_shrink,
            stability_ridge_gamma=stability_ridge_gamma,
            stability_ridge_stat_eps=stability_ridge_stat_eps,
        ),
    )

    if os.path.exists(cache_path) and not force_retrain:
        state = torch.load(cache_path, map_location="cpu")
        if state.get("version") == HIRA_CACHE_VERSION and "hira_state" in state:
            model.load_state_dict(state["hira_state"], strict=False)
            return _prepare_hira_model_for_eval(model)

    print(f"Fitting HiRA replacement layers for {cache_path}...")
    train_loader, val_loader = _build_train_loaders(
        dataset_root,
        batch_size,
        num_workers,
        seed,
        max_train_samples,
        train_transform=train_transform,
    )
    fit_summary = _fit_hira_weights_closed_form(
        model,
        teacher_model,
        train_loader,
        val_loader,
        mlp_module_names,
        device,
        adapt_noise_eps=adapt_noise_eps,
        adapt_noise_num=adapt_noise_num,
        adapt_alpha=adapt_alpha,
        subspace_rank=subspace_rank,
        stability_ridge_gamma=stability_ridge_gamma,
        stability_ridge_stat_eps=stability_ridge_stat_eps,
    )

    state = {
        "version": HIRA_CACHE_VERSION,
        "classifier_name": classifier_name,
        "expansion_dim": expansion_dim,
        "max_train_samples": max_train_samples,
        "seed": seed,
        "num_adapter_blocks": num_adapter_blocks,
        "adapt_noise_eps": adapt_noise_eps,
        "adapt_noise_num": adapt_noise_num,
        "adapt_alpha": adapt_alpha,
        "soft_threshold_enabled": is_meansparse_enabled(soft_threshold_alpha),
        "soft_threshold_alpha": soft_threshold_alpha,
        "soft_threshold_beta": soft_threshold_beta,
        "soft_threshold_stat_eps": soft_threshold_stat_eps,
        "soft_threshold_mode": soft_threshold_mode,
        "subspace_enabled": is_hira_subspace_enabled(subspace_rank, subspace_shrink),
        "subspace_rank": subspace_rank,
        "subspace_shrink": subspace_shrink,
        "subspace_sample_cap": _resolve_hira_subspace_sample_cap(subspace_rank),
        "stability_ridge_enabled": is_stability_ridge_enabled(stability_ridge_gamma),
        "stability_ridge_gamma": stability_ridge_gamma,
        "stability_ridge_stat_eps": stability_ridge_stat_eps,
        "stability_ridge_source": "train_projected_feature_snr",
        "mlp_module_names": mlp_module_names,
        "fit_method": "ridge_regression",
        "ridge_candidates": RIDGE_CANDIDATES,
        "soft_label_source": "noisy_mlp_input" if adapt_noise_num > 0 and adapt_noise_eps > 0 else "original_mlp_input",
        "noisy_adaptation_target": "noisy_mlp_input" if adapt_noise_num > 0 and adapt_noise_eps > 0 else "original_mlp_input",
        "adaptation_input_source": "noisy_only" if adapt_noise_num > 0 and adapt_noise_eps > 0 else "clean_only",
        "activation": "gelu",
        "soft_threshold_mean_source": "unused_channelwise_projected_threshold",
        "soft_threshold_stage": "disabled",
        "soft_threshold_train_usage": "disabled",
        "soft_threshold_eval_usage": "disabled",
        "subspace_mean_source": "train_clean_gelu_a_feature_channel",
        "subspace_basis_source": "train_clean_gelu_a_feature_top_pcs",
        "subspace_coeff_mean_source": "unused_pc_tanh_calibration",
        "subspace_coeff_std_source": "train_clean_pc_coeff_std",
        "subspace_stage": "after_gelu_before_b",
        "subspace_train_usage": "disabled",
        "subspace_eval_usage": "enabled" if is_hira_subspace_enabled(subspace_rank, subspace_shrink) else "disabled",
        "frozen_b": True,
        "ridge_fit_summary": fit_summary,
        "cache_adapter_dtype": "float16",
        "fit_forward_dtype": "float16" if device.type == "cuda" else "float32",
        "fit_backbone_autocast": device.type == "cuda",
        "eval_forward_dtype": "float32",
        "eval_backbone_autocast": False,
        "value_only": False,
        "token_projection": False,
        "mlp_projection": True,
        "mlp_residual": False,
        "pre_mlp_attachment": True,
        "post_fc2_attachment": False,
        "fc2_replacement": False,
        "insert_before_last_blocks": 0,
        "insert_on_last_blocks_mlp": num_adapter_blocks,
        "hira_state": _cached_hira_state_dict(model, mlp_module_names),
    }
    torch.save(state, cache_path)
    return _prepare_hira_model_for_eval(model)
