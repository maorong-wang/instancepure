from classifiers.mean_sparse import DEFAULT_MEANSPARSE_STAT_EPS
from classifiers.stability_ridge import DEFAULT_STABILITY_RIDGE_STAT_EPS
from victims import (
    IMAGENET_MODEL,
    NormalizedImageClassifier,
    VictimWrapperConfig,
    apply_victim_wrappers,
    build_imagenet_victim,
    supports_hira,
)


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
    soft_threshold_mode="away_from_mean",
    stability_ridge_gamma=0.0,
    stability_ridge_stat_eps=DEFAULT_STABILITY_RIDGE_STAT_EPS,
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

    classifier, classifier_name, victim_spec = build_imagenet_victim(arch)
    wrapper_config = VictimWrapperConfig(
        dataset=dataset,
        use_hira=use_hira,
        hira_expansion_dim=hira_expansion_dim,
        hira_num_blocks=hira_num_blocks,
        hira_batch_size=hira_batch_size,
        hira_num_workers=hira_num_workers,
        hira_epochs=hira_epochs,
        hira_lr=hira_lr,
        hira_weight_decay=hira_weight_decay,
        hira_seed=hira_seed,
        hira_cache_dir=hira_cache_dir,
        hira_dataset_root=hira_dataset_root,
        hira_max_train_samples=hira_max_train_samples,
        hira_force_retrain=hira_force_retrain,
        adapt_noise_eps=adapt_noise_eps,
        adapt_noise_num=adapt_noise_num,
        adapt_alpha=adapt_alpha,
        soft_threshold_alpha=soft_threshold_alpha,
        soft_threshold_beta=soft_threshold_beta,
        soft_threshold_stat_eps=soft_threshold_stat_eps,
        soft_threshold_mode=soft_threshold_mode,
        stability_ridge_gamma=stability_ridge_gamma,
        stability_ridge_stat_eps=stability_ridge_stat_eps,
        use_ranpac=use_ranpac,
        ranpac_rp_dim=ranpac_rp_dim,
        ranpac_batch_size=ranpac_batch_size,
        ranpac_num_workers=ranpac_num_workers,
        ranpac_seed=ranpac_seed,
        ranpac_selection_method=ranpac_selection_method,
        ranpac_lambda=ranpac_lambda,
        ranpac_temp=ranpac_temp,
        ranpac_hardneg_topk=ranpac_hardneg_topk,
        ranpac_hardneg_gamma=ranpac_hardneg_gamma,
        ranpac_cache_dir=ranpac_cache_dir,
        ranpac_dataset_root=ranpac_dataset_root,
    )
    classifier, _ = apply_victim_wrappers(
        classifier,
        classifier_name=classifier_name,
        supports_hira_arch=supports_hira(victim_spec),
        config=wrapper_config,
        device=device,
    )
    return classifier


__all__ = [
    "DEFAULT_MEANSPARSE_STAT_EPS",
    "DEFAULT_STABILITY_RIDGE_STAT_EPS",
    "IMAGENET_MODEL",
    "NormalizedImageClassifier",
    "get_archs",
]
