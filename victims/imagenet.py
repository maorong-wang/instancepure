from dataclasses import dataclass

import torch
import torch.nn as nn


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
    "resnet50": ("resnet50", "resnet50", "ImageNet ResNet-50"),
    "imagenet-resnet50": ("resnet50", "resnet50", "ImageNet ResNet-50"),
    "resnet152": ("resnet152", "resnet152", "ImageNet ResNet-152"),
    "imagenet-resnet152": ("resnet152", "resnet152", "ImageNet ResNet-152"),
    "wrn50": ("wrn50", "wide_resnet50_2", "ImageNet Wide-ResNet-50-2"),
    "wideresnet-50-2": ("wrn50", "wide_resnet50_2", "ImageNet Wide-ResNet-50-2"),
    "imagenet-wideresnet-50-2": ("wrn50", "wide_resnet50_2", "ImageNet Wide-ResNet-50-2"),
    "vit": ("vit_base", "vit_base_patch16_224", "ImageNet ViT-base"),
    "vit_base": ("vit_base", "vit_base_patch16_224", "ImageNet ViT-base"),
    "vit-base": ("vit_base", "vit_base_patch16_224", "ImageNet ViT-base"),
    "imagenet-vit-base": ("vit_base", "vit_base_patch16_224", "ImageNet ViT-base"),
    "vitb": ("vit_base", "vit_base_patch16_224", "ImageNet ViT-base"),
    "vit_small": ("vit_small", "vit_small_patch16_224", "ImageNet ViT-small"),
    "vit-small": ("vit_small", "vit_small_patch16_224", "ImageNet ViT-small"),
    "imagenet-vit-small": ("vit_small", "vit_small_patch16_224", "ImageNet ViT-small"),
    "vits": ("vit_small", "vit_small_patch16_224", "ImageNet ViT-small"),
    "vit_tiny": ("vit_tiny", "vit_tiny_patch16_224", "ImageNet ViT-tiny"),
    "vit-tiny": ("vit_tiny", "vit_tiny_patch16_224", "ImageNet ViT-tiny"),
    "imagenet-vit-tiny": ("vit_tiny", "vit_tiny_patch16_224", "ImageNet ViT-tiny"),
    "vitt": ("vit_tiny", "vit_tiny_patch16_224", "ImageNet ViT-tiny"),
    "swin_s": ("swin_s", "swin_small_patch4_window7_224", "ImageNet Swin-S"),
    "swin-s": ("swin_s", "swin_small_patch4_window7_224", "ImageNet Swin-S"),
    "swin_small": ("swin_s", "swin_small_patch4_window7_224", "ImageNet Swin-S"),
    "swin-small": ("swin_s", "swin_small_patch4_window7_224", "ImageNet Swin-S"),
    "imagenet-swin-s": ("swin_s", "swin_small_patch4_window7_224", "ImageNet Swin-S"),
    "imagenet-swin-small": ("swin_s", "swin_small_patch4_window7_224", "ImageNet Swin-S"),
    "swin_b": ("swin_b", "swin_base_patch4_window7_224", "ImageNet Swin-B"),
    "swin-base": ("swin_b", "swin_base_patch4_window7_224", "ImageNet Swin-B"),
    "swin_base": ("swin_b", "swin_base_patch4_window7_224", "ImageNet Swin-B"),
    "imagenet-swin-base": ("swin_b", "swin_base_patch4_window7_224", "ImageNet Swin-B"),
    "convnext_b": ("convnext_b", "convnext_base", "ImageNet ConvNeXt-Base"),
    "convnext-base": ("convnext_b", "convnext_base", "ImageNet ConvNeXt-Base"),
    "imagenet-convnext-base": ("convnext_b", "convnext_base", "ImageNet ConvNeXt-Base"),
}

_DEFAULT_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_DEFAULT_IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ImageNetVictimSpec:
    requested_name: str
    canonical_name: str
    timm_model_name: str
    display_name: str

    @property
    def classifier_name(self):
        return f"imagenet-{self.canonical_name.replace('_', '-')}"


class NormalizedImageClassifier(nn.Module):
    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        return self.model(x)


def resolve_imagenet_victim(arch):
    key = arch.lower()
    if key in _IMAGENET_ALIASES:
        canonical_name, timm_model_name, display_name = _IMAGENET_ALIASES[key]
        return ImageNetVictimSpec(
            requested_name=arch,
            canonical_name=canonical_name,
            timm_model_name=timm_model_name,
            display_name=display_name,
        )

    return ImageNetVictimSpec(
        requested_name=arch,
        canonical_name=arch.replace("/", "-").replace("_", "-"),
        timm_model_name=arch,
        display_name=f"ImageNet {arch}",
    )


def build_imagenet_victim(arch, pretrained=True):
    try:
        import timm
    except ImportError as exc:
        raise ImportError("timm is required to build ImageNet victims in the active environment.") from exc

    spec = resolve_imagenet_victim(arch)
    try:
        model = timm.create_model(spec.timm_model_name, pretrained=pretrained).eval()
    except Exception as exc:
        raise NotImplementedError(
            f"Unknown ImageNet classifier '{arch}'. Available aliases: {', '.join(IMAGENET_MODEL)} "
            f"or any valid timm model name."
        ) from exc

    cfg = getattr(model, "default_cfg", {})
    mean = cfg.get("mean", _DEFAULT_IMAGENET_MEAN)
    std = cfg.get("std", _DEFAULT_IMAGENET_STD)
    classifier = NormalizedImageClassifier(model, mean=mean, std=std)
    return classifier, spec.classifier_name, spec


def supports_hira(spec_or_name):
    model_name = getattr(spec_or_name, "timm_model_name", str(spec_or_name))
    return model_name.startswith("vit") or model_name.startswith("swin")
