import copy
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from classifiers.ranpac import RIDGE_CANDIDATES
from dataset import get_dataset


class HiRAAdapter(nn.Module):
    def __init__(self, in_features, out_features, expansion_dim):
        super().__init__()
        self.register_buffer("b_rand", torch.randn(in_features, expansion_dim, dtype=torch.float32))
        self.a_weight = nn.Parameter(torch.empty(out_features, expansion_dim))
        nn.init.kaiming_uniform_(self.a_weight, a=0.0, nonlinearity="relu")

    def project(self, x):
        token_features = x.reshape(-1, x.shape[-1]).float()
        return F.gelu(token_features @ self.b_rand.float())

    def forward(self, x):
        token_shape = x.shape[:-1]
        projected = self.project(x)
        output = projected @ self.a_weight.float().t()
        return output.to(x.dtype).reshape(*token_shape, self.a_weight.shape[0])


class HiRAMlpWrapper(nn.Module):
    def __init__(self, base_mlp, expansion_dim):
        super().__init__()
        self.base_mlp = base_mlp
        mlp_output_dim = _infer_mlp_output_dim(base_mlp)
        self.mlp_adapter = HiRAAdapter(mlp_output_dim, mlp_output_dim, expansion_dim)

    def post_fc2(self, x):
        x = self.base_mlp.fc1(x)
        if hasattr(self.base_mlp, "act"):
            x = self.base_mlp.act(x)
        if hasattr(self.base_mlp, "drop1"):
            x = self.base_mlp.drop1(x)
        elif hasattr(self.base_mlp, "drop"):
            x = self.base_mlp.drop(x)
        if hasattr(self.base_mlp, "norm"):
            x = self.base_mlp.norm(x)
        return self.base_mlp.fc2(x)

    def forward(self, x, *args, **kwargs):
        return self.mlp_adapter(self.post_fc2(x))


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


def _infer_mlp_output_dim(module):
    for attr_path in ("fc2.out_features", "fc2.out_channels"):
        try:
            candidate = _get_module_by_name(module, attr_path)
        except AttributeError:
            continue
        if isinstance(candidate, int):
            return candidate
    raise ValueError("Unable to infer MLP output dimension for HiRA attachment.")


def _resolve_target_mlp_modules(model, num_adapter_blocks):
    mlp_modules = _list_mlp_modules(model)
    if num_adapter_blocks <= 0:
        raise ValueError("HiRA requires --hira_num_blocks to be a positive integer.")
    if len(mlp_modules) < num_adapter_blocks:
        raise ValueError(
            f"HiRA requested {num_adapter_blocks} adapter blocks, but the backbone only exposes {len(mlp_modules)} MLP blocks."
        )
    return mlp_modules[-num_adapter_blocks:]


def _attach_hira_modules(model, expansion_dim, num_adapter_blocks):
    target_mlp_names = []
    for module_name, module in _resolve_target_mlp_modules(model, num_adapter_blocks):
        if isinstance(module, HiRAMlpWrapper):
            target_mlp_names.append(module_name)
            continue
        _set_module_by_name(model, module_name, HiRAMlpWrapper(module, expansion_dim))
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


def _format_block_tag(num_adapter_blocks, prefix):
    if num_adapter_blocks == 2:
        return ""
    return f"{prefix}{num_adapter_blocks}"


def _build_cache_name(classifier_name, expansion_dim, epochs, lr, weight_decay, max_train_samples, seed, num_adapter_blocks):
    sample_tag = "full" if max_train_samples is None or max_train_samples < 0 else str(max_train_samples)
    block_tag = _format_block_tag(num_adapter_blocks, "_blk")
    return (
        f"{classifier_name.replace('/', '_')}_hira_v22_post_fc2_closedform_gelu{block_tag}"
        f"_exp{expansion_dim}"
        f"_ns{sample_tag}"
        f"_seed{seed}.pt"
    )


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_train_loaders(dataset_root, batch_size, num_workers, seed, max_train_samples):
    if dataset_root:
        os.environ["IMAGENET_LOC_ENV"] = dataset_root

    dataset = get_dataset("imagenet", split="train", adv=False)
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


def _init_hira_statistics(model, mlp_module_names, stats_device):
    statistics = {}
    for module_name in mlp_module_names:
        wrapper = _get_module_by_name(model, module_name)
        rp_dim = wrapper.mlp_adapter.b_rand.size(1)
        out_dim = wrapper.mlp_adapter.a_weight.size(0)
        statistics[module_name] = {
            "g_matrix": torch.zeros(rp_dim, rp_dim, dtype=torch.float32, device=stats_device),
            "q_matrix": torch.zeros(rp_dim, out_dim, dtype=torch.float32, device=stats_device),
            "target_norm": torch.zeros((), dtype=torch.float32, device=stats_device),
            "sample_count": 0,
        }
    return statistics


def _select_ridge_by_regression_loss(g_train, q_train, g_val, q_val, val_target_norm, out_features, val_sample_count, device):
    if val_sample_count == 0:
        return RIDGE_CANDIDATES[0], None

    g_train = g_train.to(device)
    q_train = q_train.to(device)
    g_val = g_val.to(device)
    q_val = q_val.to(device)
    eye = torch.eye(g_train.size(0), device=device, dtype=g_train.dtype)

    best_ridge = RIDGE_CANDIDATES[0]
    best_loss = None
    denominator = max(val_sample_count * out_features, 1)

    with torch.no_grad():
        for ridge in RIDGE_CANDIDATES:
            weight = torch.linalg.solve(g_train + ridge * eye, q_train).t()
            quadratic = torch.trace(weight @ g_val @ weight.t())
            cross_term = torch.trace(weight @ q_val)
            loss = (quadratic - 2.0 * cross_term + val_target_norm) / denominator
            loss_value = loss.item()
            if best_loss is None or loss_value < best_loss:
                best_loss = loss_value
                best_ridge = ridge

    return best_ridge, best_loss


def _accumulate_hira_statistics(model, teacher_model, loader, mlp_module_names, device, description):
    stats_device = device if device.type == "cuda" else torch.device("cpu")
    statistics = _init_hira_statistics(model, mlp_module_names, stats_device)
    b_rand_by_module = {
        module_name: _get_module_by_name(model, module_name).mlp_adapter.b_rand.to(device=stats_device, dtype=torch.float32)
        for module_name in mlp_module_names
    }
    batch_cache = {}
    handles = []

    for module_name in mlp_module_names:
        teacher_module = _get_module_by_name(teacher_model, module_name)
        teacher_fc2 = teacher_module.fc2

        def hook(_, inputs, output, module_name=module_name):
            batch_cache[module_name] = output.detach()

        handles.append(teacher_fc2.register_forward_hook(hook))

    teacher_model = teacher_model.to(device).eval()
    try:
        with torch.no_grad():
            for inputs, _ in tqdm(loader, desc=description):
                batch_cache.clear()
                _ = teacher_model(inputs.to(device, non_blocking=True))
                for module_name in mlp_module_names:
                    mlp_outputs = batch_cache[module_name]
                    token_features = mlp_outputs.reshape(-1, mlp_outputs.shape[-1]).to(stats_device, dtype=torch.float32)
                    projected = F.gelu(token_features @ b_rand_by_module[module_name])
                    targets = token_features

                    statistics[module_name]["g_matrix"] += projected.t() @ projected
                    statistics[module_name]["q_matrix"] += projected.t() @ targets
                    statistics[module_name]["target_norm"] += targets.square().sum()
                    statistics[module_name]["sample_count"] += targets.size(0)

                    del token_features, projected, targets
    finally:
        for handle in handles:
            handle.remove()

    return statistics


def _fit_hira_weights_closed_form(model, teacher_model, train_loader, val_loader, mlp_module_names, device):
    train_stats = _accumulate_hira_statistics(
        model,
        teacher_model,
        train_loader,
        mlp_module_names,
        device,
        description="HiRA train stats",
    )
    val_stats = _accumulate_hira_statistics(
        model,
        teacher_model,
        val_loader,
        mlp_module_names,
        device,
        description="HiRA val stats",
    )

    fit_summary = {}
    for module_name in mlp_module_names:
        wrapper = _get_module_by_name(model, module_name)
        train_entry = train_stats[module_name]
        val_entry = val_stats[module_name]
        out_features = wrapper.mlp_adapter.a_weight.size(0)

        ridge, regression_loss = _select_ridge_by_regression_loss(
            train_entry["g_matrix"],
            train_entry["q_matrix"],
            val_entry["g_matrix"],
            val_entry["q_matrix"],
            val_entry["target_norm"],
            out_features,
            val_entry["sample_count"],
            device,
        )

        g_full = (train_entry["g_matrix"] + val_entry["g_matrix"]).to(device)
        q_full = (train_entry["q_matrix"] + val_entry["q_matrix"]).to(device)
        eye = torch.eye(g_full.size(0), device=device, dtype=g_full.dtype)
        weight = torch.linalg.solve(g_full + ridge * eye, q_full).t().cpu()
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
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    teacher_model = copy.deepcopy(model).eval()
    _seed_everything(seed)
    mlp_module_names = _attach_hira_modules(model, expansion_dim, num_adapter_blocks)
    _freeze_model(model)

    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(
        cache_dir,
        _build_cache_name(
            classifier_name,
            expansion_dim,
            epochs,
            lr,
            weight_decay,
            max_train_samples,
            seed,
            num_adapter_blocks,
        ),
    )

    if os.path.exists(cache_path) and not force_retrain:
        state = torch.load(cache_path, map_location="cpu")
        model.load_state_dict(state["hira_state"], strict=False)
        return model.eval()

    print(f"Fitting HiRA replacement layers for {cache_path}...")
    train_loader, val_loader = _build_train_loaders(dataset_root, batch_size, num_workers, seed, max_train_samples)
    fit_summary = _fit_hira_weights_closed_form(
        model,
        teacher_model,
        train_loader,
        val_loader,
        mlp_module_names,
        device,
    )

    state = {
        "version": 22,
        "classifier_name": classifier_name,
        "expansion_dim": expansion_dim,
        "max_train_samples": max_train_samples,
        "seed": seed,
        "num_adapter_blocks": num_adapter_blocks,
        "mlp_module_names": mlp_module_names,
        "fit_method": "ridge_regression",
        "ridge_candidates": RIDGE_CANDIDATES,
        "soft_label_source": "original_mlp_output",
        "activation": "gelu",
        "frozen_b": True,
        "ridge_fit_summary": fit_summary,
        "forward_dtype": "float32",
        "value_only": False,
        "token_projection": False,
        "mlp_projection": True,
        "mlp_residual": False,
        "post_fc2_attachment": True,
        "fc2_replacement": False,
        "insert_before_last_blocks": 0,
        "insert_on_last_blocks_mlp": num_adapter_blocks,
        "hira_state": _cached_hira_state_dict(model, mlp_module_names),
    }
    torch.save(state, cache_path)
    return model.eval()
