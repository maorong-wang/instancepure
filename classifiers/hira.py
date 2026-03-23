import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from dataset import get_dataset


def _format_cache_value(value):
    text = str(value)
    for old, new in (("/", "_"), (" ", ""), (".", "p"), ("-", "m")):
        text = text.replace(old, new)
    return text


class HiRAAdapter(nn.Module):
    def __init__(self, in_features, out_features, expansion_dim):
        super().__init__()
        self.expand = nn.Linear(in_features, expansion_dim, bias=False)
        self.reduce = nn.Linear(expansion_dim, out_features, bias=False)
        nn.init.kaiming_uniform_(self.expand.weight, a=0.0, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.reduce.weight, a=0.0, nonlinearity="relu")

    def forward(self, x):
        return self.reduce(F.relu(self.expand(x)))


class HiRAQKVWrapper(nn.Module):
    def __init__(self, base_linear, expansion_dim):
        super().__init__()
        if base_linear.out_features % 3 != 0:
            raise ValueError("HiRA requires qkv out_features to be divisible by 3.")

        chunk_dim = base_linear.out_features // 3
        self.base_linear = base_linear
        self.q_adapter = HiRAAdapter(base_linear.in_features, chunk_dim, expansion_dim)
        self.k_adapter = HiRAAdapter(base_linear.in_features, chunk_dim, expansion_dim)
        self.v_adapter = HiRAAdapter(base_linear.in_features, chunk_dim, expansion_dim)

    def forward(self, x):
        qkv = self.base_linear(x)
        q_base, k_base, v_base = qkv.chunk(3, dim=-1)
        q = q_base + self.q_adapter(x)
        k = k_base + self.k_adapter(x)
        v = v_base + self.v_adapter(x)
        return torch.cat((q, k, v), dim=-1)


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


def _list_qkv_modules(model):
    return [
        name
        for name, module in model.named_modules()
        if name.endswith("attn.qkv") and isinstance(module, nn.Linear)
    ]


def _attach_hira_modules(model, expansion_dim):
    qkv_module_names = _list_qkv_modules(model)
    if len(qkv_module_names) < 2:
        raise ValueError("HiRA requires a transformer backbone with at least two attention qkv layers.")

    target_module_names = qkv_module_names[-2:]
    for module_name in target_module_names:
        module = _get_module_by_name(model, module_name)
        if isinstance(module, HiRAQKVWrapper):
            continue
        _set_module_by_name(model, module_name, HiRAQKVWrapper(module, expansion_dim))
    return target_module_names


def _find_last_linear_module_names(model):
    return [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]


def _find_classifier_module_names(model):
    module_names = []
    for candidate in ("model.head", "model.fc", "model.classifier", "head", "fc", "classifier"):
        try:
            module = _get_module_by_name(model, candidate)
        except AttributeError:
            continue
        if any(True for _ in module.parameters(recurse=True)):
            module_names.append(candidate)

    if not module_names:
        linear_names = _find_last_linear_module_names(model)
        if linear_names:
            module_names.append(linear_names[-1])

    return list(dict.fromkeys(module_names))


def _freeze_model(model):
    for parameter in model.parameters():
        parameter.requires_grad = False


def _mark_trainable(model, qkv_module_names):
    _freeze_model(model)

    for qkv_module_name in qkv_module_names[-2:]:
        wrapper = _get_module_by_name(model, qkv_module_name)
        for adapter_name in ("q_adapter", "k_adapter", "v_adapter"):
            for parameter in getattr(wrapper, adapter_name).parameters():
                parameter.requires_grad = True

    for module_name in _find_classifier_module_names(model):
        module = _get_module_by_name(model, module_name)
        for parameter in module.parameters():
            parameter.requires_grad = True


def _trainable_state_dict(model):
    trainable_keys = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    state_dict = model.state_dict()
    return {name: tensor.detach().cpu() for name, tensor in state_dict.items() if name in trainable_keys}


def _build_cache_name(classifier_name, expansion_dim, epochs, lr, weight_decay, max_train_samples, seed):
    sample_tag = "full" if max_train_samples is None or max_train_samples < 0 else str(max_train_samples)
    return (
        f"{classifier_name.replace('/', '_')}_hira_v2"
        f"_exp{expansion_dim}"
        f"_ep{epochs}"
        f"_lr{_format_cache_value(lr)}"
        f"_wd{_format_cache_value(weight_decay)}"
        f"_ns{sample_tag}"
        f"_seed{seed}.pt"
    )


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_train_loader(dataset_root, batch_size, num_workers, seed, max_train_samples):
    if dataset_root:
        os.environ["IMAGENET_LOC_ENV"] = dataset_root

    dataset = get_dataset("imagenet", split="train", adv=False)
    if max_train_samples is not None and max_train_samples > 0 and max_train_samples < len(dataset):
        generator = torch.Generator()
        generator.manual_seed(seed)
        indices = torch.randperm(len(dataset), generator=generator)[:max_train_samples].tolist()
        dataset = Subset(dataset, indices)

    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
        num_workers=num_workers,
    )


def _fit_hira_weights(model, loader, device, epochs, lr, weight_decay):
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("HiRA did not mark any parameters as trainable.")

    optimizer = torch.optim.AdamW(trainable_parameters, lr=lr, weight_decay=weight_decay)
    scaler = GradScaler(enabled=device.type == "cuda")
    use_amp = device.type == "cuda"
    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        sample_count = 0

        progress = tqdm(loader, desc=f"HiRA fine-tune {epoch + 1}/{epochs}")
        for inputs, targets in progress:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                logits = model(inputs)
                loss = F.cross_entropy(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            running_correct += (logits.detach().argmax(dim=1) == targets).sum().item()
            sample_count += batch_size
            progress.set_postfix(
                loss=f"{running_loss / max(sample_count, 1):.4f}",
                acc=f"{running_correct / max(sample_count, 1):.4f}",
            )

    model.eval()
    return model


def apply_hira_adaptation(
    model,
    classifier_name,
    dataset_root=None,
    expansion_dim=4096,
    batch_size=32,
    num_workers=4,
    epochs=1,
    lr=1e-4,
    weight_decay=1e-4,
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

    qkv_module_names = _attach_hira_modules(model, expansion_dim)
    _mark_trainable(model, qkv_module_names)

    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(
        cache_dir,
        _build_cache_name(classifier_name, expansion_dim, epochs, lr, weight_decay, max_train_samples, seed),
    )

    if os.path.exists(cache_path) and not force_retrain:
        state = torch.load(cache_path, map_location="cpu")
        model.load_state_dict(state["trainable_state"], strict=False)
        return model.eval()

    _seed_everything(seed)
    loader = _build_train_loader(dataset_root, batch_size, num_workers, seed, max_train_samples)
    _fit_hira_weights(model, loader, device, epochs, lr, weight_decay)

    state = {
        "classifier_name": classifier_name,
        "expansion_dim": expansion_dim,
        "epochs": epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "max_train_samples": max_train_samples,
        "seed": seed,
        "qkv_module_names": qkv_module_names,
        "trainable_state": _trainable_state_dict(model),
    }
    torch.save(state, cache_path)
    return model.eval()
