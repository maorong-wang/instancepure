import os

import torch
import torch.nn as nn
import torchvision

from classifiers.hira import apply_hira_adaptation, build_hira_variant_name
from classifiers.mean_sparse import DEFAULT_MEANSPARSE_STAT_EPS
from classifiers.ranpac import apply_ranpac_head


IMAGENET_MODEL = (
    "resnet50",
    "resnet152",
    "wrn50",
    "vit_base",
    "vit_small",
    "vit_tiny",
    "swin_s",
    "swin_b",
    "convnext_b",
)

_IMAGENET_ALIASES = {
    "resnet50": "resnet50",
    "imagenet-resnet50": "resnet50",
    "resnet152": "resnet152",
    "imagenet-resnet152": "resnet152",
    "wrn50": "wrn50",
    "wideresnet-50-2": "wrn50",
    "imagenet-wideresnet-50-2": "wrn50",
    "vit": "vit_base",
    "vit_base": "vit_base",
    "vit-base": "vit_base",
    "imagenet-vit-base": "vit_base",
    "vitb": "vit_base",
    "vit_small": "vit_small",
    "vit-small": "vit_small",
    "imagenet-vit-small": "vit_small",
    "vits": "vit_small",
    "vit_tiny": "vit_tiny",
    "vit-tiny": "vit_tiny",
    "imagenet-vit-tiny": "vit_tiny",
    "vitt": "vit_tiny",
    "swin_s": "swin_s",
    "swin-s": "swin_s",
    "swin_small": "swin_s",
    "swin-small": "swin_s",
    "imagenet-swin-s": "swin_s",
    "imagenet-swin-small": "swin_s",
    "swin_b": "swin_b",
    "swin-base": "swin_b",
    "swin_base": "swin_b",
    "imagenet-swin-base": "swin_b",
    "convnext_b": "convnext_b",
    "convnext-base": "convnext_b",
    "imagenet-convnext-base": "convnext_b",
}

_DEFAULT_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_DEFAULT_IMAGENET_STD = (0.229, 0.224, 0.225)


class NormalizedImageClassifier(nn.Module):
    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        return self.model(x)


def _resolve_imagenet_arch(arch):
    key = arch.lower()
    if key not in _IMAGENET_ALIASES:
        raise NotImplementedError(f"Unknown ImageNet classifier '{arch}'. Available: {', '.join(IMAGENET_MODEL)}")
    return _IMAGENET_ALIASES[key]


def _load_timm_model(model_name, display_name):
    try:
        import timm
    except ImportError as exc:
        raise ImportError(
            f"timm is required for {display_name}. Install timm in the active environment."
        ) from exc

    model = timm.create_model(model_name, pretrained=True).eval()
    cfg = getattr(model, "default_cfg", {})
    mean = cfg.get("mean", _DEFAULT_IMAGENET_MEAN)
    std = cfg.get("std", _DEFAULT_IMAGENET_STD)
    return model, mean, std


def _build_imagenet_classifier(arch):
    if arch == "resnet50":
        model = torchvision.models.resnet50(weights="DEFAULT").eval()
        return model, _DEFAULT_IMAGENET_MEAN, _DEFAULT_IMAGENET_STD

    if arch == "resnet152":
        model = torchvision.models.resnet152(weights="DEFAULT").eval()
        return model, _DEFAULT_IMAGENET_MEAN, _DEFAULT_IMAGENET_STD

    if arch == "wrn50":
        model = torchvision.models.wide_resnet50_2(weights="DEFAULT").eval()
        return model, _DEFAULT_IMAGENET_MEAN, _DEFAULT_IMAGENET_STD

    if arch == "convnext_b":
        model = torchvision.models.convnext_base(weights="DEFAULT").eval()
        return model, _DEFAULT_IMAGENET_MEAN, _DEFAULT_IMAGENET_STD

    if arch == "vit_base":
        return _load_timm_model("vit_base_patch16_224", "ImageNet ViT-base")

    if arch == "vit_small":
        return _load_timm_model("vit_small_patch16_224", "ImageNet ViT-small")

    if arch == "vit_tiny":
        return _load_timm_model("vit_tiny_patch16_224", "ImageNet ViT-tiny")

    if arch == "swin_s":
        return _load_timm_model("swin_small_patch4_window7_224", "ImageNet Swin-S")

    if arch == "swin_b":
        return _load_timm_model("swin_base_patch4_window7_224", "ImageNet Swin-B")

    raise NotImplementedError(f"Unknown ImageNet classifier '{arch}'.")


def _default_ranpac_root(dataset):
    if dataset == "imagenet":
        return os.environ.get("IMAGENET_LOC_ENV", "./image_net")
    return None


def _default_hira_root(dataset):
    return _default_ranpac_root(dataset)


def _supports_hira(arch):
    return arch.startswith("vit_") or arch.startswith("swin_")


def get_archs(
    arch,
    dataset="imagenet",
    use_hira=False,
    hira_expansion_dim=4096,
    hira_num_blocks=2,
    hira_batch_size=32,
    hira_num_workers=4,
    hira_epochs=1,
    hira_lr=1e-4,
    hira_weight_decay=1e-4,
    hira_seed=0,
    hira_cache_dir="pretrained/hira",
    hira_dataset_root=None,
    hira_max_train_samples=-1,
    hira_force_retrain=False,
    adapt_noise_eps=0.0,
    adapt_noise_num=0,
    adapt_alpha=1.0,
    soft_threshold_alpha=0.0,
    soft_threshold_beta=8.0,
    soft_threshold_stat_eps=DEFAULT_MEANSPARSE_STAT_EPS,
    use_ranpac=False,
    ranpac_rp_dim=5000,
    ranpac_batch_size=256,
    ranpac_num_workers=4,
    ranpac_seed=0,
    ranpac_selection_method="regression",
    ranpac_lambda=1.0,
    ranpac_temp=1.0,
    ranpac_hardneg_topk=9,
    ranpac_hardneg_gamma=1.0,
    ranpac_cache_dir="pretrained/ranpac",
    ranpac_dataset_root=None,
    device=None,
):
    if dataset != "imagenet":
        raise NotImplementedError(f"Unsupported dataset '{dataset}'.")

    canonical_arch = _resolve_imagenet_arch(arch)
    model, mean, std = _build_imagenet_classifier(canonical_arch)
    classifier = NormalizedImageClassifier(model, mean=mean, std=std)
    classifier_name = f"{dataset}-{canonical_arch.replace('_', '-')}"

    if use_hira:
        if not _supports_hira(canonical_arch):
            raise ValueError(f"HiRA is only supported for ViT-family backbones, but got '{canonical_arch}'.")
        classifier = apply_hira_adaptation(
            classifier,
            classifier_name=classifier_name,
            dataset_root=hira_dataset_root or _default_hira_root(dataset),
            expansion_dim=hira_expansion_dim,
            num_adapter_blocks=hira_num_blocks,
            batch_size=hira_batch_size,
            num_workers=hira_num_workers,
            epochs=hira_epochs,
            lr=hira_lr,
            weight_decay=hira_weight_decay,
            seed=hira_seed,
            device=device,
            cache_dir=hira_cache_dir,
            max_train_samples=hira_max_train_samples,
            force_retrain=hira_force_retrain,
            adapt_noise_eps=adapt_noise_eps,
            adapt_noise_num=adapt_noise_num,
            adapt_alpha=adapt_alpha,
            soft_threshold_alpha=soft_threshold_alpha,
            soft_threshold_beta=soft_threshold_beta,
            soft_threshold_stat_eps=soft_threshold_stat_eps,
        )
        classifier_name = build_hira_variant_name(
            classifier_name,
            expansion_dim=hira_expansion_dim,
            epochs=hira_epochs,
            lr=hira_lr,
            weight_decay=hira_weight_decay,
            max_train_samples=hira_max_train_samples,
            seed=hira_seed,
            num_adapter_blocks=hira_num_blocks,
            adapt_noise_eps=adapt_noise_eps,
            adapt_noise_num=adapt_noise_num,
            adapt_alpha=adapt_alpha,
            soft_threshold_alpha=soft_threshold_alpha,
            soft_threshold_beta=soft_threshold_beta,
            soft_threshold_stat_eps=soft_threshold_stat_eps,
        )

    if use_ranpac:
        classifier = apply_ranpac_head(
            classifier,
            classifier_name=classifier_name,
            dataset_root=ranpac_dataset_root or _default_ranpac_root(dataset),
            rp_dim=ranpac_rp_dim,
            batch_size=ranpac_batch_size,
            num_workers=ranpac_num_workers,
            seed=ranpac_seed,
            selection_method=ranpac_selection_method,
            device=device,
            cache_dir=ranpac_cache_dir,
            adapt_noise_eps=adapt_noise_eps,
            adapt_noise_num=adapt_noise_num,
            adapt_alpha=adapt_alpha,
            soft_threshold_alpha=soft_threshold_alpha,
            soft_threshold_beta=soft_threshold_beta,
            soft_threshold_stat_eps=soft_threshold_stat_eps,
            ranpac_lambda=ranpac_lambda,
            ranpac_temp=ranpac_temp,
            hardneg_topk=ranpac_hardneg_topk,
            hardneg_gamma=ranpac_hardneg_gamma,
        )

    return classifier
