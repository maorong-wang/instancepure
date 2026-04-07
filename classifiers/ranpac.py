import os

from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

RIDGE_CANDIDATES = [10.0 ** power for power in range(-8, 14)]
RANPAC_CACHE_VERSION = 8


def _format_cache_value(value):
    text = str(value)
    for old, new in (("/", "_"), (" ", ""), (".", "p"), ("-", "m")):
        text = text.replace(old, new)
    return text


class RanPACLinear(nn.Module):
    def __init__(self, in_features, out_features, rp_dim, weight, w_rand):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rp_dim = rp_dim
        self.register_buffer("weight", weight)
        self.register_buffer("w_rand", w_rand)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        projected = F.gelu(x @ self.w_rand)
        return projected @ self.weight.t()


class ResidualRanPACLinear(nn.Module):
    def __init__(self, original_linear, ranpac_linear, ranpac_lambda):
        super().__init__()
        self.original_linear = original_linear
        self.ranpac_linear = ranpac_linear
        self.ranpac_lambda = float(ranpac_lambda)

    def forward(self, x):
        return self.original_linear(x) + self.ranpac_lambda * self.ranpac_linear(x)


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
                onehot = F.one_hot(targets.cpu(), num_classes=out_features).float()
                target_norm_value = onehot.square().sum().item()

                if use_noisy_adaptation:
                    for _ in range(adapt_noise_num):
                        noisy_inputs = _sample_linf_noisy_inputs(inputs, adapt_noise_eps)
                        feature_buffer.clear()
                        _ = model(noisy_inputs)
                        noisy_features = feature_buffer.pop().view(inputs.size(0), -1).cpu()
                        noisy_projected = F.gelu(noisy_features @ w_rand)

                        g_matrix += noisy_sample_weight * (noisy_projected.t() @ noisy_projected)
                        q_matrix += noisy_sample_weight * (noisy_projected.t() @ onehot)
                        target_norm += noisy_sample_weight * target_norm_value
                        sample_count += noisy_sample_weight * onehot.size(0)
                else:
                    feature_buffer.clear()
                    _ = model(inputs)
                    features = feature_buffer.pop().view(inputs.size(0), -1).cpu()
                    projected = F.gelu(features @ w_rand)

                    g_matrix += projected.t() @ projected
                    q_matrix += projected.t() @ onehot
                    target_norm += target_norm_value
                    sample_count += onehot.size(0)
    finally:
        handle.remove()

    return g_matrix, q_matrix, target_norm, sample_count


def _select_ridge_by_regression_loss(g_train, q_train, g_val, q_val, val_target_norm, num_classes, val_sample_count, device):
    g_train = g_train.to(device)
    q_train = q_train.to(device)
    g_val = g_val.to(device)
    q_val = q_val.to(device)
    eye = torch.eye(g_train.size(0), device=device, dtype=g_train.dtype)

    best_ridge = RIDGE_CANDIDATES[0]
    best_loss = None
    denominator = max(val_sample_count * num_classes, 1)

    with torch.no_grad():
        for ridge in RIDGE_CANDIDATES:
            weights = torch.linalg.solve(g_train + ridge * eye, q_train).t()
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
    g_train, q_train, _, _ = _accumulate_statistics(
        model,
        linear_layer,
        train_loader,
        w_rand,
        out_features,
        device,
        description="RanPAC train stats",
        adapt_noise_eps=adapt_noise_eps,
        adapt_noise_num=adapt_noise_num,
        adapt_alpha=adapt_alpha,
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
        )
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
    )

    print(f"RanPAC optimal ridge (regression): {regression_ridge}")
    if mse_loss is not None:
        print(f"RanPAC best mse loss: {mse_loss:.6f}")

    g_full = (g_train + g_val).to(device)
    q_full = (q_train + q_val).to(device)
    eye = torch.eye(g_full.size(0), device=device, dtype=g_full.dtype)
    weight = torch.linalg.solve(g_full + regression_ridge * eye, q_full).t().cpu()

    state = {
        "version": RANPAC_CACHE_VERSION,
        "layer_name": layer_name,
        "in_features": in_features,
        "out_features": out_features,
        "rp_dim": rp_dim,
        "split_seed": seed,
        "ridge_candidates": RIDGE_CANDIDATES,
        "selection_method": "regression",
        "target_type": "ground_truth",
        "adaptation_input_source": "noisy_only" if adapt_noise_num > 0 and adapt_noise_eps > 0 else "clean_only",
        "adapt_noise_eps": adapt_noise_eps,
        "adapt_noise_num": adapt_noise_num,
        "adapt_alpha": adapt_alpha,
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
):
    if train_loader is None and "imagenet" not in classifier_name:
        raise NotImplementedError("RanPAC head replacement is currently implemented for ImageNet classifiers only.")
    if selection_method != "regression":
        raise ValueError(f"Unsupported RanPAC selection method '{selection_method}'.")
    if adapt_noise_eps < 0:
        raise ValueError("RanPAC adapt_noise_eps must be non-negative.")
    if adapt_noise_num < 0:
        raise ValueError("RanPAC adapt_noise_num must be non-negative.")
    if adapt_alpha < 0:
        raise ValueError("RanPAC adapt_alpha must be non-negative.")
    if ranpac_lambda < 0:
        raise ValueError("RanPAC ranpac_lambda must be non-negative.")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    noise_tag = (
        f"_neps{_format_cache_value(adapt_noise_eps)}"
        f"_nnum{adapt_noise_num}"
        f"_na{_format_cache_value(adapt_alpha)}"
    )
    cache_name = (
        f"{classifier_name.replace('/', '_')}_rp{rp_dim}_seed{seed}"
        f"{noise_tag}_ranpac_v{RANPAC_CACHE_VERSION}.pt"
    )
    cache_path = os.path.join(cache_dir, cache_name)

    layer_name, linear_layer = _find_last_linear(model)
    if os.path.exists(cache_path):
        state = torch.load(cache_path, map_location="cpu")
        if state.get("version") != RANPAC_CACHE_VERSION or "weight" not in state:
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
            )
    else:
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
    )
    ranpac_head = ResidualRanPACLinear(
        original_linear=linear_layer,
        ranpac_linear=ranpac_branch,
        ranpac_lambda=ranpac_lambda,
    )
    _set_module_by_name(model, layer_name, ranpac_head)
    return model
