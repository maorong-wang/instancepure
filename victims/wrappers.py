import os
from dataclasses import dataclass
from typing import Optional

from classifiers.hira import apply_hira_adaptation, build_hira_variant_name
from classifiers.mean_sparse import DEFAULT_MEANSPARSE_STAT_EPS, format_cache_value
from classifiers.ranpac import apply_ranpac_head
from classifiers.stability_ridge import DEFAULT_STABILITY_RIDGE_STAT_EPS, build_stability_ridge_tag


@dataclass
class VictimWrapperConfig:
    dataset: str = "imagenet"
    use_hira: bool = False
    hira_expansion_dim: int = 4096
    hira_num_blocks: int = 2
    hira_batch_size: int = 32
    hira_num_workers: int = 4
    hira_epochs: int = 1
    hira_lr: float = 1e-4
    hira_weight_decay: float = 1e-4
    hira_seed: int = 0
    hira_cache_dir: str = "pretrained/hira"
    hira_dataset_root: Optional[str] = None
    hira_max_train_samples: int = -1
    hira_force_retrain: bool = False
    adapt_noise_eps: float = 0.0
    adapt_noise_num: int = 0
    adapt_alpha: float = 1.0
    soft_threshold_alpha: float = 0.0
    soft_threshold_beta: float = 8.0
    soft_threshold_stat_eps: float = DEFAULT_MEANSPARSE_STAT_EPS
    soft_threshold_mode: str = "away_from_mean"
    stability_ridge_gamma: float = 0.0
    stability_ridge_stat_eps: float = DEFAULT_STABILITY_RIDGE_STAT_EPS
    use_ranpac: bool = False
    ranpac_rp_dim: int = 5000
    ranpac_batch_size: int = 256
    ranpac_num_workers: int = 4
    ranpac_seed: int = 0
    ranpac_selection_method: str = "regression"
    ranpac_lambda: float = 1.0
    ranpac_temp: float = 1.0
    ranpac_hardneg_topk: int = 9
    ranpac_hardneg_gamma: float = 1.0
    ranpac_cache_dir: str = "pretrained/ranpac"
    ranpac_dataset_root: Optional[str] = None


def _default_dataset_root(dataset):
    if dataset == "imagenet":
        return os.environ.get("IMAGENET_LOC_ENV", "./image_net")
    return None


def _build_adapt_noise_tag(config):
    if config.adapt_noise_eps <= 0 or config.adapt_noise_num <= 0 or config.adapt_alpha <= 0:
        return ""
    return (
        f"-neps{format_cache_value(config.adapt_noise_eps)}"
        f"-nnum{config.adapt_noise_num}"
        f"-na{format_cache_value(config.adapt_alpha)}"
    )


def build_wrapper_config_from_namespace(args, dataset="imagenet"):
    use_hira = getattr(args, "use_hira_adapter", getattr(args, "use_hira", False))
    use_ranpac = getattr(args, "use_ranpac_head", getattr(args, "use_ranpac", False))
    ranpac_batch_size = getattr(args, "ranpac_batch_size", getattr(args, "ranpac_fit_batch_size", 256))
    return VictimWrapperConfig(
        dataset=dataset,
        use_hira=use_hira,
        hira_expansion_dim=getattr(args, "hira_expansion_dim", 4096),
        hira_num_blocks=getattr(args, "hira_num_blocks", 2),
        hira_batch_size=getattr(args, "hira_batch_size", 32),
        hira_num_workers=getattr(args, "hira_num_workers", 4),
        hira_epochs=getattr(args, "hira_epochs", 1),
        hira_lr=getattr(args, "hira_lr", 1e-4),
        hira_weight_decay=getattr(args, "hira_weight_decay", 1e-4),
        hira_seed=getattr(args, "hira_seed", 0),
        hira_cache_dir=getattr(args, "hira_cache_dir", "pretrained/hira"),
        hira_dataset_root=getattr(args, "hira_dataset_root", None),
        hira_max_train_samples=getattr(args, "hira_max_train_samples", -1),
        hira_force_retrain=getattr(args, "hira_force_retrain", False),
        adapt_noise_eps=getattr(args, "adapt_noise_eps", 0.0),
        adapt_noise_num=getattr(args, "adapt_noise_num", 0),
        adapt_alpha=getattr(args, "adapt_alpha", 1.0),
        soft_threshold_alpha=getattr(args, "soft_threshold_alpha", 0.0),
        soft_threshold_beta=getattr(args, "soft_threshold_beta", 8.0),
        soft_threshold_stat_eps=getattr(args, "soft_threshold_stat_eps", DEFAULT_MEANSPARSE_STAT_EPS),
        soft_threshold_mode=getattr(args, "soft_threshold_mode", "away_from_mean"),
        stability_ridge_gamma=getattr(args, "stability_ridge_gamma", 0.0),
        stability_ridge_stat_eps=getattr(args, "stability_ridge_stat_eps", DEFAULT_STABILITY_RIDGE_STAT_EPS),
        use_ranpac=use_ranpac,
        ranpac_rp_dim=getattr(args, "ranpac_rp_dim", 5000),
        ranpac_batch_size=ranpac_batch_size,
        ranpac_num_workers=getattr(args, "ranpac_num_workers", 4),
        ranpac_seed=getattr(args, "ranpac_seed", 0),
        ranpac_selection_method=getattr(args, "ranpac_selection_method", "regression"),
        ranpac_lambda=getattr(args, "ranpac_lambda", 1.0),
        ranpac_temp=getattr(args, "ranpac_temp", 1.0),
        ranpac_hardneg_topk=getattr(args, "ranpac_hardneg_topk", 9),
        ranpac_hardneg_gamma=getattr(args, "ranpac_hardneg_gamma", 1.0),
        ranpac_cache_dir=getattr(args, "ranpac_cache_dir", "pretrained/ranpac"),
        ranpac_dataset_root=getattr(args, "ranpac_dataset_root", None),
    )


def apply_victim_wrappers(classifier, classifier_name, supports_hira_arch, config, device=None):
    wrapped_classifier_name = classifier_name

    if config.use_hira:
        if not supports_hira_arch:
            raise ValueError("HiRA is only supported for ViT-family backbones.")
        classifier = apply_hira_adaptation(
            classifier,
            classifier_name=classifier_name,
            dataset_root=config.hira_dataset_root or _default_dataset_root(config.dataset),
            expansion_dim=config.hira_expansion_dim,
            num_adapter_blocks=config.hira_num_blocks,
            batch_size=config.hira_batch_size,
            num_workers=config.hira_num_workers,
            epochs=config.hira_epochs,
            lr=config.hira_lr,
            weight_decay=config.hira_weight_decay,
            seed=config.hira_seed,
            device=device,
            cache_dir=config.hira_cache_dir,
            max_train_samples=config.hira_max_train_samples,
            force_retrain=config.hira_force_retrain,
            adapt_noise_eps=config.adapt_noise_eps,
            adapt_noise_num=config.adapt_noise_num,
            adapt_alpha=config.adapt_alpha,
            soft_threshold_alpha=config.soft_threshold_alpha,
            soft_threshold_beta=config.soft_threshold_beta,
            soft_threshold_stat_eps=config.soft_threshold_stat_eps,
            soft_threshold_mode=config.soft_threshold_mode,
            stability_ridge_gamma=config.stability_ridge_gamma,
            stability_ridge_stat_eps=config.stability_ridge_stat_eps,
        )
        wrapped_classifier_name = build_hira_variant_name(
            classifier_name,
            expansion_dim=config.hira_expansion_dim,
            epochs=config.hira_epochs,
            lr=config.hira_lr,
            weight_decay=config.hira_weight_decay,
            max_train_samples=config.hira_max_train_samples,
            seed=config.hira_seed,
            num_adapter_blocks=config.hira_num_blocks,
            adapt_noise_eps=config.adapt_noise_eps,
            adapt_noise_num=config.adapt_noise_num,
            adapt_alpha=config.adapt_alpha,
            soft_threshold_alpha=config.soft_threshold_alpha,
            soft_threshold_beta=config.soft_threshold_beta,
            soft_threshold_stat_eps=config.soft_threshold_stat_eps,
            soft_threshold_mode=config.soft_threshold_mode,
            stability_ridge_gamma=config.stability_ridge_gamma,
            stability_ridge_stat_eps=config.stability_ridge_stat_eps,
        )

    if config.use_ranpac:
        classifier = apply_ranpac_head(
            classifier,
            classifier_name=wrapped_classifier_name,
            dataset_root=config.ranpac_dataset_root or _default_dataset_root(config.dataset),
            rp_dim=config.ranpac_rp_dim,
            batch_size=config.ranpac_batch_size,
            num_workers=config.ranpac_num_workers,
            seed=config.ranpac_seed,
            selection_method=config.ranpac_selection_method,
            device=device,
            cache_dir=config.ranpac_cache_dir,
            adapt_noise_eps=config.adapt_noise_eps,
            adapt_noise_num=config.adapt_noise_num,
            adapt_alpha=config.adapt_alpha,
            stability_ridge_gamma=config.stability_ridge_gamma,
            stability_ridge_stat_eps=config.stability_ridge_stat_eps,
            ranpac_lambda=config.ranpac_lambda,
            ranpac_temp=config.ranpac_temp,
            hardneg_topk=config.ranpac_hardneg_topk,
            hardneg_gamma=config.ranpac_hardneg_gamma,
        )
        wrapped_classifier_name = (
            f"{wrapped_classifier_name}-ranpac-{config.ranpac_selection_method}-bbias"
        )
        if config.ranpac_lambda != 1.0:
            wrapped_classifier_name = f"{wrapped_classifier_name}-lam{format_cache_value(config.ranpac_lambda)}"
        if config.ranpac_temp != 1.0:
            wrapped_classifier_name = f"{wrapped_classifier_name}-temp{format_cache_value(config.ranpac_temp)}"
        if config.ranpac_hardneg_topk != 9 or config.ranpac_hardneg_gamma != 1.0:
            wrapped_classifier_name = (
                f"{wrapped_classifier_name}-htk{config.ranpac_hardneg_topk}"
                f"-hg{format_cache_value(config.ranpac_hardneg_gamma)}"
            )
        if not config.use_hira:
            adapt_noise_tag = _build_adapt_noise_tag(config)
            if adapt_noise_tag:
                wrapped_classifier_name = f"{wrapped_classifier_name}{adapt_noise_tag}"
            stability_tag = build_stability_ridge_tag(
                gamma=config.stability_ridge_gamma,
                stat_eps=config.stability_ridge_stat_eps,
                separator="-",
            )
            if stability_tag:
                wrapped_classifier_name = f"{wrapped_classifier_name}{stability_tag}"

    return classifier, wrapped_classifier_name
