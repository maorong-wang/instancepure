import os

from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

RIDGE_CANDIDATES = [10.0 ** power for power in range(-8, 14)]


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


def _build_imagenet_train_loaders(dataset_root, batch_size, num_workers, seed):
    train_dir = _resolve_imagenet_train_dir(dataset_root)
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


def _build_train_loaders(classifier_name, dataset_root, batch_size, num_workers, seed):
    if "imagenet" in classifier_name:
        return _build_imagenet_train_loaders(dataset_root, batch_size, num_workers, seed)
    raise NotImplementedError(f"RanPAC train loader is not implemented for {classifier_name}.")


def _accumulate_statistics(model, linear_layer, loader, w_rand, out_features, device, description):
    feature_buffer = []

    def hook(_, inputs):
        feature_buffer.append(inputs[0].detach())

    handle = linear_layer.register_forward_pre_hook(hook)
    g_matrix = torch.zeros(w_rand.size(1), w_rand.size(1), dtype=torch.float32)
    q_matrix = torch.zeros(w_rand.size(1), out_features, dtype=torch.float32)
    target_norm = 0.0
    sample_count = 0

    try:
        with torch.no_grad():
            for inputs, targets in tqdm(loader, desc=description):
                inputs = inputs.to(device)
                targets = targets.to(device)
                feature_buffer.clear()
                _ = model(inputs)
                features = feature_buffer.pop().view(inputs.size(0), -1).cpu()
                projected = F.gelu(features @ w_rand)
                onehot = F.one_hot(targets.cpu(), num_classes=out_features).float()

                g_matrix += projected.t() @ projected
                q_matrix += projected.t() @ onehot
                target_norm += onehot.square().sum().item()
                sample_count += onehot.size(0)
    finally:
        handle.remove()

    return g_matrix, q_matrix, target_norm, sample_count


def _compute_candidate_weights(g_train, q_train, g_val, q_val, val_target_norm, num_classes, val_sample_count, device):
    g_train = g_train.to(device)
    q_train = q_train.to(device)
    g_val = g_val.to(device)
    q_val = q_val.to(device)
    eye = torch.eye(g_train.size(0), device=device, dtype=g_train.dtype)

    best_ridge = RIDGE_CANDIDATES[0]
    best_loss = None
    denominator = max(val_sample_count * num_classes, 1)
    weights_by_ridge = {}

    with torch.no_grad():
        for ridge in RIDGE_CANDIDATES:
            weights = torch.linalg.solve(g_train + ridge * eye, q_train).t()
            weights_by_ridge[ridge] = weights.cpu()
            if val_sample_count == 0:
                continue

            quadratic = torch.trace(weights @ g_val @ weights.t())
            cross_term = torch.trace(weights @ q_val)
            loss = (quadratic - 2.0 * cross_term + val_target_norm) / denominator
            loss_value = loss.item()
            if best_loss is None or loss_value < best_loss:
                best_loss = loss_value
                best_ridge = ridge

    return weights_by_ridge, best_ridge, best_loss


def _optimise_ridge_by_validation_accuracy(
    model,
    linear_layer,
    loader,
    w_rand,
    weights_by_ridge,
    device,
    description,
):
    candidate_ridges = list(weights_by_ridge.keys())
    if loader is None or not candidate_ridges:
        return RIDGE_CANDIDATES[0], None

    feature_buffer = []

    def hook(_, inputs):
        feature_buffer.append(inputs[0].detach())

    handle = linear_layer.register_forward_pre_hook(hook)
    correct_by_ridge = {ridge: 0 for ridge in candidate_ridges}
    sample_count = 0

    try:
        with torch.no_grad():
            for inputs, targets in tqdm(loader, desc=description):
                inputs = inputs.to(device)
                targets = targets.cpu()
                feature_buffer.clear()
                _ = model(inputs)
                features = feature_buffer.pop().view(inputs.size(0), -1).cpu()
                projected = F.gelu(features @ w_rand)

                for ridge in candidate_ridges:
                    weight = weights_by_ridge[ridge]
                    predictions = (projected @ weight.t()).argmax(dim=1)
                    correct_by_ridge[ridge] += (predictions == targets).sum().item()
                sample_count += targets.size(0)
    finally:
        handle.remove()

    best_ridge = max(candidate_ridges, key=lambda ridge: (correct_by_ridge[ridge], -ridge))
    best_acc = correct_by_ridge[best_ridge] / max(sample_count, 1)
    return best_ridge, best_acc


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
):
    layer_name, linear_layer = _find_last_linear(model)
    in_features = linear_layer.in_features
    out_features = linear_layer.out_features
    if train_loader is None:
        train_loader, val_loader = _build_train_loaders(classifier_name, dataset_root, batch_size, num_workers, seed)

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
        )
    else:
        g_val = torch.zeros_like(g_train)
        q_val = torch.zeros_like(q_train)
        val_target_norm = 0.0
        val_sample_count = 0

    weights_by_ridge, regression_ridge, regression_loss = _compute_candidate_weights(
        g_train,
        q_train,
        g_val,
        q_val,
        val_target_norm,
        out_features,
        val_sample_count,
        device,
    )
    val_acc_ridge, val_acc = _optimise_ridge_by_validation_accuracy(
        model,
        linear_layer,
        val_loader,
        w_rand,
        weights_by_ridge,
        device,
        description="RanPAC val accuracy",
    )
    if val_sample_count == 0:
        regression_ridge = val_acc_ridge
        regression_loss = None

    print(f"RanPAC optimal ridge (regression): {regression_ridge}")
    if regression_loss is not None:
        print(f"RanPAC best regression loss: {regression_loss:.6f}")
    print(f"RanPAC optimal ridge (val_acc): {val_acc_ridge}")
    if val_acc is not None:
        print(f"RanPAC best val accuracy: {val_acc:.6f}")

    g_full = (g_train + g_val).to(device)
    q_full = (q_train + q_val).to(device)
    eye = torch.eye(g_full.size(0), device=device, dtype=g_full.dtype)

    selected_heads = {}
    for selection_method, ridge in (("regression", regression_ridge), ("val_acc", val_acc_ridge)):
        weight = torch.linalg.solve(g_full + ridge * eye, q_full).t().cpu()
        selected_heads[selection_method] = {
            "ridge": ridge,
            "weight": weight,
        }

    state = {
        "version": 2,
        "layer_name": layer_name,
        "in_features": in_features,
        "out_features": out_features,
        "rp_dim": rp_dim,
        "split_seed": seed,
        "ridge_candidates": RIDGE_CANDIDATES,
        "w_rand": w_rand,
        "selection_metrics": {
            "regression_loss": regression_loss,
            "val_acc": val_acc,
        },
        "heads": selected_heads,
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
):
    if train_loader is None and "imagenet" not in classifier_name:
        raise NotImplementedError("RanPAC head replacement is currently implemented for ImageNet classifiers only.")
    if selection_method not in {"regression", "val_acc"}:
        raise ValueError(f"Unsupported RanPAC selection method '{selection_method}'.")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    cache_name = f"{classifier_name.replace('/', '_')}_rp{rp_dim}_seed{seed}_ranpac_v2.pt"
    cache_path = os.path.join(cache_dir, cache_name)

    layer_name, linear_layer = _find_last_linear(model)
    if os.path.exists(cache_path):
        state = torch.load(cache_path, map_location="cpu")
        if "heads" not in state:
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
        )

    if state["layer_name"] != layer_name:
        raise ValueError(f"Cached RanPAC head expects layer {state['layer_name']} but found {layer_name}.")
    if state["in_features"] != linear_layer.in_features or state["out_features"] != linear_layer.out_features:
        raise ValueError("Cached RanPAC head does not match the pretrained classifier dimensions.")
    if selection_method not in state["heads"]:
        raise ValueError(
            f"Cached RanPAC head does not contain selection method '{selection_method}'. "
            f"Available: {sorted(state['heads'])}"
        )

    selected_head = state["heads"][selection_method]

    ranpac_head = RanPACLinear(
        in_features=state["in_features"],
        out_features=state["out_features"],
        rp_dim=state["rp_dim"],
        weight=selected_head["weight"],
        w_rand=state["w_rand"],
    )
    _set_module_by_name(model, layer_name, ranpac_head)
    return model
