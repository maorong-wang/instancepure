import copy
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from dataset import get_dataset


def _format_cache_value(value):
    text = str(value)
    for old, new in ("/", "_"), (" ", ""), (".", "p"), ("-", "m"):
        text = text.replace(old, new)
    return text


HIRA_ACTIVATION_POOL = ("gelu")


def _apply_random_activation(x):
    if x.is_cuda:
        activation_index = int(torch.randint(len(HIRA_ACTIVATION_POOL), (), device=x.device).item())
    else:
        activation_index = int(torch.randint(len(HIRA_ACTIVATION_POOL), ()).item())
    activation_name = HIRA_ACTIVATION_POOL[activation_index]
    if activation_name == "relu":
        return F.relu(x)
    if activation_name == "gelu":
        return F.gelu(x)
    if activation_name == "silu":
        return F.silu(x)
    return F.leaky_relu(x, negative_slope=0.01)


class HiRAAdapter(nn.Module):
    def __init__(self, in_features, out_features, expansion_dim):
        super().__init__()
        self.register_buffer("b_rand", torch.randn(in_features, expansion_dim, dtype=torch.float32))
        self.a_weight = nn.Parameter(torch.empty(out_features, expansion_dim))
        nn.init.kaiming_uniform_(self.a_weight, a=0.0, nonlinearity="relu")

    def forward(self, x):
        token_shape = x.shape[:-1]
        token_features = x.reshape(-1, x.shape[-1]).float()
        projected = token_features @ self.b_rand
        projected = _apply_random_activation(projected / math.sqrt(self.b_rand.shape[0]))
        output = projected @ self.a_weight.float().t()
        return output.to(x.dtype).reshape(*token_shape, self.a_weight.shape[0])


class HiRAMlpWrapper(nn.Module):
    def __init__(self, base_mlp, embed_dim, expansion_dim):
        super().__init__()
        self.base_mlp = base_mlp
        self.mlp_adapter = HiRAAdapter(embed_dim, embed_dim, expansion_dim)

    def forward(self, x, *args, **kwargs):
        return self.base_mlp(x, *args, **kwargs) + self.mlp_adapter(x)


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


def _infer_mlp_embed_dim(module):
    for attr_path in ("fc1.in_features", "fc2.out_features"):
        try:
            candidate = _get_module_by_name(module, attr_path)
        except AttributeError:
            continue
        if isinstance(candidate, int):
            return candidate
    raise ValueError("Unable to infer MLP embed dimension for HiRA attachment.")


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
        _set_module_by_name(model, module_name, HiRAMlpWrapper(module, _infer_mlp_embed_dim(module), expansion_dim))
        target_mlp_names.append(module_name)
    return target_mlp_names


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


def _mark_trainable(model, mlp_module_names):
    _freeze_model(model)

    for mlp_module_name in mlp_module_names:
        wrapper = _get_module_by_name(model, mlp_module_name)
        for parameter in wrapper.mlp_adapter.parameters():
            parameter.requires_grad = True

    for module_name in _find_classifier_module_names(model):
        module = _get_module_by_name(model, module_name)
        for parameter in module.parameters():
            parameter.requires_grad = True


def _cached_hira_state_dict(model):
    trainable_keys = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    state_dict = model.state_dict()
    return {
        name: tensor.detach().cpu()
        for name, tensor in state_dict.items()
        if name in trainable_keys or name.endswith(".b_rand")
    }


def _format_block_tag(num_adapter_blocks, prefix):
    if num_adapter_blocks == 2:
        return ""
    return f"{prefix}{num_adapter_blocks}"


def _build_cache_name(classifier_name, expansion_dim, epochs, lr, weight_decay, max_train_samples, seed, num_adapter_blocks):
    sample_tag = "full" if max_train_samples is None or max_train_samples < 0 else str(max_train_samples)
    block_tag = _format_block_tag(num_adapter_blocks, "_blk")
    return (
        f"{classifier_name.replace('/', '_')}_hira_v10_mlp_residual_randact{block_tag}"
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


def _kl_distillation_loss(student_logits, teacher_logits, temperature=1.0):
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)


def _fit_hira_weights(model, teacher_model, loader, device, epochs, lr, weight_decay, grad_clip_norm=1.0):
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("HiRA did not mark any parameters as trainable.")

    optimizer = torch.optim.AdamW(trainable_parameters, lr=lr, weight_decay=weight_decay)
    model = model.to(device).eval()
    teacher_model = teacher_model.to(device).eval()
    for parameter in teacher_model.parameters():
        parameter.requires_grad = False

    for epoch in range(epochs):
        running_loss = 0.0
        sample_count = 0

        progress = tqdm(loader, desc=f"HiRA fine-tune {epoch + 1}/{epochs}")
        for inputs, _ in progress:
            inputs = inputs.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs).float()
            with torch.no_grad():
                teacher_logits = teacher_model(inputs).float()
            loss = _kl_distillation_loss(logits, teacher_logits)

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite HiRA loss detected. "
                    "The MLP HiRA projection is diverging; try a smaller --hira-lr or --hira-expansion-dim."
                )

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, grad_clip_norm)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(
                    "Non-finite HiRA gradient norm detected. "
                    "The MLP HiRA projection is numerically unstable."
                )
            optimizer.step()

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            sample_count += batch_size
            progress.set_postfix(loss=f"{running_loss / max(sample_count, 1):.4f}")

    model.eval()
    return model


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
    _mark_trainable(model, mlp_module_names)

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

    loader = _build_train_loader(dataset_root, batch_size, num_workers, seed, max_train_samples)
    _fit_hira_weights(model, teacher_model, loader, device, epochs, lr, weight_decay, grad_clip_norm=grad_clip_norm)

    state = {
        "classifier_name": classifier_name,
        "expansion_dim": expansion_dim,
        "epochs": epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "grad_clip_norm": grad_clip_norm,
        "max_train_samples": max_train_samples,
        "seed": seed,
        "num_adapter_blocks": num_adapter_blocks,
        "mlp_module_names": mlp_module_names,
        "loss_type": "teacher_kl",
        "headwise_randomization": False,
        "frozen_b": True,
        "normalized_random_projection": True,
        "forward_dtype": "float32",
        "value_only": False,
        "token_projection": False,
        "mlp_projection": True,
        "mlp_residual": True,
        "random_activation_pool": list(HIRA_ACTIVATION_POOL),
        "insert_before_last_blocks": 0,
        "insert_on_last_blocks_mlp": num_adapter_blocks,
        "hira_state": _cached_hira_state_dict(model),
    }
    torch.save(state, cache_path)
    return model.eval()
