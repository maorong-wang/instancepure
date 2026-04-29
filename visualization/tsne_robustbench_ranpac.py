#!/usr/bin/env python3
"""
t-SNE visualization for RobustBench ImageNet models with and without HiRA+RanPAC.

This script intentionally lives outside the main evaluation path and does not modify
existing repository code. It compares three feature spaces:
  1. original: penultimate features from the RobustBench model
  2. HiRA: penultimate features from the HiRA+RanPAC-wrapped model
  3. RanPAC projected: random projected features after HiRA+RanPAC wrapping

For each variant, it generates white-box PGD adversarial examples, collects clean
and adversarial features for a balanced 20-class / 50-image-per-class ImageNet
subset by default, and saves separate clean/PGD t-SNE plots plus metadata.
"""

import argparse
import csv
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from robustbench import load_model
    from robustbench.data import get_preprocessing
    from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel
except ImportError:
    fallback_root = REPO_ROOT.parent / "adversarial-attacks-pytorch"
    if str(fallback_root) not in sys.path:
        sys.path.insert(0, str(fallback_root))
    from robustbench import load_model
    from robustbench.data import get_preprocessing
    from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel

from classifiers.hira import apply_hira_adaptation, build_hira_variant_name
from classifiers.mean_sparse import DEFAULT_MEANSPARSE_STAT_EPS
from classifiers.ranpac import RanPACLinear, ResidualRanPACLinear, apply_ranpac_head
from classifiers.stability_ridge import DEFAULT_STABILITY_RIDGE_STAT_EPS
from dataset import get_dataset as instantpure_get_dataset


VARIANT_ORIGINAL = "original"
VARIANT_HIRA_RANPAC = "hira_ranpac_regression"
DATASET = BenchmarkDataset.imagenet.value


@dataclass
class FeatureBundle:
    clean_features: np.ndarray
    adv_features: np.ndarray
    labels: np.ndarray
    clean_predictions: np.ndarray
    adv_predictions: np.ndarray


@dataclass
class EmbeddingBundle:
    embedding: np.ndarray
    labels: np.ndarray
    domains: np.ndarray


class PenultimateFeatureExtractor:
    name = "penultimate"

    def __init__(self, model):
        self.module_name, self.module = find_last_linear(model)

    def __call__(self, model, inputs):
        captured_features = []

        def hook(_, hook_inputs):
            features = hook_inputs[0].detach().view(inputs.size(0), -1).float().cpu()
            captured_features.append(features)

        handle = self.module.register_forward_pre_hook(hook)
        try:
            with torch.no_grad():
                logits = model(inputs)
        finally:
            handle.remove()

        if not captured_features:
            raise RuntimeError(f"No penultimate features were captured from {self.module_name}.")
        return captured_features[-1], logits.detach().float().cpu()


class RanPACProjectedFeatureExtractor:
    name = "ranpac_projected"

    def __init__(self, model):
        modules = [(name, module) for name, module in model.named_modules() if isinstance(module, RanPACLinear)]
        if not modules:
            raise ValueError("No RanPACLinear module found. Did apply_ranpac_head run successfully?")
        if len(modules) > 1:
            print(f"Found {len(modules)} RanPACLinear modules; using the last one: {modules[-1][0]}")
        self.module_name, self.module = modules[-1]

    def __call__(self, model, inputs):
        captured_features = []

        def hook(module, hook_inputs):
            raw_features = hook_inputs[0].detach().view(inputs.size(0), -1).float()
            w_rand = module.w_rand.to(device=raw_features.device, dtype=raw_features.dtype)
            projected = F.gelu(raw_features @ w_rand).detach().float().cpu()
            captured_features.append(projected)

        handle = self.module.register_forward_pre_hook(hook)
        try:
            with torch.no_grad():
                logits = model(inputs)
        finally:
            handle.remove()

        if not captured_features:
            raise RuntimeError(f"No RanPAC projected features were captured from {self.module_name}.")
        return captured_features[-1], logits.detach().float().cpu()


class HiraPenultimateFeatureExtractor:
    name = "hira_penultimate"

    def __init__(self, model):
        modules = [(name, module) for name, module in model.named_modules() if isinstance(module, ResidualRanPACLinear)]
        if not modules:
            raise ValueError("No ResidualRanPACLinear module found. Did apply_ranpac_head run successfully?")
        if len(modules) > 1:
            print(f"Found {len(modules)} ResidualRanPACLinear modules; using the last one: {modules[-1][0]}")
        self.module_name, self.module = modules[-1]

    def __call__(self, model, inputs):
        captured_features = []

        def hook(_, hook_inputs):
            features = hook_inputs[0].detach().view(inputs.size(0), -1).float().cpu()
            captured_features.append(features)

        handle = self.module.register_forward_pre_hook(hook)
        try:
            with torch.no_grad():
                logits = model(inputs)
        finally:
            handle.remove()

        if not captured_features:
            raise RuntimeError(f"No HiRA penultimate features were captured from {self.module_name}.")
        return captured_features[-1], logits.detach().float().cpu()


def str2bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).lower()
    if normalized in {"yes", "true", "t", "y", "1"}:
        return True
    if normalized in {"no", "false", "f", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_float_or_fraction(value):
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        return float(numerator) / float(denominator)
    return float(text)


def parse_class_ids(value):
    if value is None or str(value).strip() == "":
        return None
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def sanitize_name(value):
    text = str(value)
    for old, new in (("/", "_"), (" ", ""), (",", "-"), (".", "p"), (":", "_")):
        text = text.replace(old, new)
    return text


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device):
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str) and device.isdigit():
        return torch.device(f"cuda:{device}")
    return torch.device(device)


def find_last_linear(model):
    linear_modules = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    if not linear_modules:
        raise ValueError("No nn.Linear layer found for penultimate feature extraction.")
    return linear_modules[-1]


def freeze_model(model):
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model.eval()


def resolve_model_preprocessing(model_name, threat_model):
    return get_preprocessing(BenchmarkDataset(DATASET), ThreatModel(threat_model), model_name, None)


def build_imagenet_dataset(data_dir, transform):
    os.environ["IMAGENET_LOC_ENV"] = str(data_dir)
    dataset = instantpure_get_dataset("imagenet", split="test", adv=False)
    if hasattr(dataset, "transform") and transform is not None:
        dataset.transform = transform
    return dataset


def get_dataset_targets(dataset):
    if hasattr(dataset, "targets"):
        return list(dataset.targets)
    if hasattr(dataset, "samples"):
        return [target for _, target in dataset.samples]
    raise ValueError("The ImageNet dataset must expose targets or samples for balanced class selection.")


def select_balanced_indices(dataset, num_classes, samples_per_class, seed, class_ids=None):
    targets = get_dataset_targets(dataset)
    by_class = {}
    for index, target in enumerate(targets):
        by_class.setdefault(int(target), []).append(index)

    eligible_classes = sorted(
        class_id for class_id, indices in by_class.items() if len(indices) >= samples_per_class
    )
    if class_ids is None:
        if len(eligible_classes) < num_classes:
            raise ValueError(
                f"Only {len(eligible_classes)} classes have at least {samples_per_class} samples; "
                f"requested {num_classes}."
            )
        rng = random.Random(seed)
        selected_classes = sorted(rng.sample(eligible_classes, num_classes))
    else:
        selected_classes = class_ids
        missing = [class_id for class_id in selected_classes if class_id not in by_class]
        too_small = [
            class_id for class_id in selected_classes
            if class_id in by_class and len(by_class[class_id]) < samples_per_class
        ]
        if missing:
            raise ValueError(f"Requested class IDs are absent from the dataset: {missing}")
        if too_small:
            raise ValueError(
                f"Requested class IDs have fewer than {samples_per_class} samples: {too_small}"
            )

    selected_indices = []
    rng = random.Random(seed)
    for class_id in selected_classes:
        candidate_indices = list(by_class[class_id])
        rng.shuffle(candidate_indices)
        selected_indices.extend(candidate_indices[:samples_per_class])

    return selected_indices, selected_classes


def build_eval_loader(dataset, indices, batch_size, num_workers):
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def load_robustbench_model(model_name, threat_model, model_dir, device):
    model = load_model(
        model_name=model_name,
        model_dir=model_dir,
        dataset=DATASET,
        threat_model=threat_model,
    )
    return model.to(device).eval()


def build_hira_ranpac_model(args, model_preprocessing, device):
    model = load_robustbench_model(args.model_name, args.threat_model, args.model_dir, device)
    classifier_name = f"{DATASET}_{args.threat_model}_{args.model_name}"
    hira_dataset_root = args.hira_dataset_root or args.data_dir
    ranpac_dataset_root = args.ranpac_dataset_root or args.data_dir

    model = apply_hira_adaptation(
        model,
        classifier_name=classifier_name,
        dataset_root=hira_dataset_root,
        expansion_dim=args.hira_expansion_dim,
        num_adapter_blocks=args.hira_num_blocks,
        batch_size=args.hira_batch_size,
        num_workers=args.hira_num_workers,
        epochs=args.hira_epochs,
        lr=args.hira_lr,
        weight_decay=args.hira_weight_decay,
        seed=args.hira_seed,
        device=device,
        cache_dir=args.hira_cache_dir,
        max_train_samples=args.hira_max_train_samples,
        force_retrain=args.hira_force_retrain,
        train_transform=model_preprocessing,
        adapt_noise_eps=args.adapt_noise_eps,
        adapt_noise_num=args.adapt_noise_num,
        adapt_alpha=args.adapt_alpha,
        soft_threshold_alpha=args.soft_threshold_alpha,
        soft_threshold_beta=args.soft_threshold_beta,
        soft_threshold_stat_eps=args.soft_threshold_stat_eps,
        soft_threshold_mode=args.soft_threshold_mode,
        stability_ridge_gamma=args.stability_ridge_gamma,
        stability_ridge_stat_eps=args.stability_ridge_stat_eps,
    ).to(device).eval()
    classifier_name = build_hira_variant_name(
        classifier_name,
        expansion_dim=args.hira_expansion_dim,
        epochs=args.hira_epochs,
        lr=args.hira_lr,
        weight_decay=args.hira_weight_decay,
        max_train_samples=args.hira_max_train_samples,
        seed=args.hira_seed,
        num_adapter_blocks=args.hira_num_blocks,
        adapt_noise_eps=args.adapt_noise_eps,
        adapt_noise_num=args.adapt_noise_num,
        adapt_alpha=args.adapt_alpha,
        soft_threshold_alpha=args.soft_threshold_alpha,
        soft_threshold_beta=args.soft_threshold_beta,
        soft_threshold_stat_eps=args.soft_threshold_stat_eps,
        soft_threshold_mode=args.soft_threshold_mode,
        stability_ridge_gamma=args.stability_ridge_gamma,
        stability_ridge_stat_eps=args.stability_ridge_stat_eps,
    )

    model = apply_ranpac_head(
        model,
        classifier_name=classifier_name,
        dataset_root=ranpac_dataset_root,
        rp_dim=args.ranpac_rp_dim,
        batch_size=args.ranpac_fit_batch_size,
        num_workers=args.ranpac_num_workers,
        seed=args.ranpac_seed,
        selection_method="regression",
        device=device,
        cache_dir=args.ranpac_cache_dir,
        train_transform=model_preprocessing,
        adapt_noise_eps=args.adapt_noise_eps,
        adapt_noise_num=args.adapt_noise_num,
        adapt_alpha=args.adapt_alpha,
        stability_ridge_gamma=args.stability_ridge_gamma,
        stability_ridge_stat_eps=args.stability_ridge_stat_eps,
        ranpac_lambda=args.ranpac_lambda,
        ranpac_temp=args.ranpac_temp,
        hardneg_topk=args.ranpac_hardneg_topk,
        hardneg_gamma=args.ranpac_hardneg_gamma,
    ).to(device).eval()
    return model


def mask_logits_to_classes(logits, class_ids):
    if class_ids is None:
        return logits

    class_ids = torch.as_tensor(class_ids, device=logits.device, dtype=torch.long)
    mask = torch.zeros(logits.size(1), device=logits.device, dtype=torch.bool)
    mask[class_ids] = True
    return logits.masked_fill(~mask.view(1, -1), float("-inf"))


def pgd_linf_attack(model, inputs, targets, eps, steps, step_size, random_start, attack_class_ids=None):
    model.eval()
    x_orig = inputs.detach()
    if random_start:
        delta = torch.empty_like(x_orig).uniform_(-eps, eps)
        delta = torch.clamp(x_orig + delta, 0.0, 1.0) - x_orig
    else:
        delta = torch.zeros_like(x_orig)

    for _ in range(steps):
        adv_inputs = torch.clamp(x_orig + delta, 0.0, 1.0).detach().requires_grad_(True)
        logits = model(adv_inputs)
        logits = mask_logits_to_classes(logits, attack_class_ids)
        loss = F.cross_entropy(logits, targets, reduction="sum")
        grad = torch.autograd.grad(loss, adv_inputs, only_inputs=True)[0]
        delta = (delta + step_size * grad.sign()).detach()
        delta = torch.clamp(delta, -eps, eps)
        delta = torch.clamp(x_orig + delta, 0.0, 1.0) - x_orig

    return torch.clamp(x_orig + delta, 0.0, 1.0).detach()


def collect_features_for_variant(model, loader, extractor, device, args, variant_name):
    clean_feature_batches = []
    adv_feature_batches = []
    label_batches = []
    clean_prediction_batches = []
    adv_prediction_batches = []

    pgd_step_size = args.pgd_step_size
    if pgd_step_size is None:
        pgd_step_size = 2.0 * args.eps / max(args.pgd_steps, 1)

    progress = tqdm(loader, desc=f"{variant_name}: clean/PGD feature collection", dynamic_ncols=True)
    for inputs, labels in progress:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        clean_features, clean_logits = extractor(model, inputs)
        adv_inputs = pgd_linf_attack(
            model,
            inputs,
            labels,
            eps=args.eps,
            steps=args.pgd_steps,
            step_size=pgd_step_size,
            random_start=args.pgd_random_start,
            attack_class_ids=getattr(args, "attack_class_ids", None),
        )
        adv_features, adv_logits = extractor(model, adv_inputs)

        clean_feature_batches.append(clean_features)
        adv_feature_batches.append(adv_features)
        label_batches.append(labels.detach().cpu())
        clean_prediction_batches.append(clean_logits.argmax(dim=1))
        adv_prediction_batches.append(adv_logits.argmax(dim=1))

        seen = sum(batch.numel() for batch in label_batches)
        clean_correct = sum(
            (pred == target).sum().item()
            for pred, target in zip(clean_prediction_batches, label_batches)
        )
        adv_correct = sum(
            (pred == target).sum().item()
            for pred, target in zip(adv_prediction_batches, label_batches)
        )
        progress.set_postfix(clean_acc=f"{clean_correct / seen:.3f}", pgd_acc=f"{adv_correct / seen:.3f}")

    return FeatureBundle(
        clean_features=torch.cat(clean_feature_batches, dim=0).numpy(),
        adv_features=torch.cat(adv_feature_batches, dim=0).numpy(),
        labels=torch.cat(label_batches, dim=0).numpy(),
        clean_predictions=torch.cat(clean_prediction_batches, dim=0).numpy(),
        adv_predictions=torch.cat(adv_prediction_batches, dim=0).numpy(),
    )


def normalize_feature_matrix(features, normalize_features):
    features = np.asarray(features, dtype=np.float32)
    if normalize_features:
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        features = features / np.maximum(norms, 1e-12)
    return features


def preprocess_for_tsne(features, pca_dim, normalize_features, seed):
    try:
        from sklearn.decomposition import PCA
    except ImportError as exc:
        raise ImportError("t-SNE visualization requires scikit-learn. Install it with `pip install scikit-learn`.") from exc

    features = normalize_feature_matrix(features, normalize_features)

    n_components = min(int(pca_dim), features.shape[0] - 1, features.shape[1])
    if n_components <= 0 or n_components >= features.shape[1]:
        return features

    pca = PCA(n_components=n_components, random_state=seed)
    return pca.fit_transform(features)


def run_tsne(features, args, seed_offset=0):
    try:
        from sklearn.manifold import TSNE
    except ImportError as exc:
        raise ImportError("t-SNE visualization requires scikit-learn. Install it with `pip install scikit-learn`.") from exc

    kwargs = dict(
        n_components=2,
        perplexity=args.tsne_perplexity,
        metric=args.tsne_metric,
        init=args.tsne_init,
        random_state=args.seed + seed_offset,
        verbose=1 if args.tsne_verbose else 0,
    )
    try:
        return TSNE(max_iter=args.tsne_iterations, **kwargs).fit_transform(features)
    except TypeError:
        return TSNE(n_iter=args.tsne_iterations, **kwargs).fit_transform(features)


def compute_embedding(bundle, args, seed_offset=0):
    features = np.concatenate([bundle.clean_features, bundle.adv_features], axis=0)
    labels = np.concatenate([bundle.labels, bundle.labels], axis=0)
    domains = np.array(["clean"] * len(bundle.labels) + ["pgd"] * len(bundle.labels))
    tsne_inputs = preprocess_for_tsne(
        features,
        pca_dim=args.pca_dim,
        normalize_features=args.normalize_features,
        seed=args.seed + seed_offset,
    )
    embedding = run_tsne(tsne_inputs, args, seed_offset=seed_offset)
    return EmbeddingBundle(embedding=embedding, labels=labels, domains=domains)


def compute_domain_embedding(features, labels, domain, args, seed_offset=0):
    tsne_inputs = preprocess_for_tsne(
        features,
        pca_dim=args.pca_dim,
        normalize_features=args.normalize_features,
        seed=args.seed + seed_offset,
    )
    embedding = run_tsne(tsne_inputs, args, seed_offset=seed_offset)
    return EmbeddingBundle(
        embedding=embedding,
        labels=np.asarray(labels, dtype=np.int64),
        domains=np.array([domain] * len(labels)),
    )


def compute_clean_pgd_embeddings(bundle, args, seed_offset=0):
    return (
        compute_domain_embedding(bundle.clean_features, bundle.labels, "clean", args, seed_offset=seed_offset),
        compute_domain_embedding(bundle.adv_features, bundle.labels, "pgd", args, seed_offset=seed_offset + 100),
    )


def preprocess_for_umap(features, pca_dim, normalize_features, seed):
    try:
        from sklearn.decomposition import PCA
    except ImportError as exc:
        raise ImportError("UMAP pre-reduction requires scikit-learn. Install it with `pip install scikit-learn`.") from exc

    features = normalize_feature_matrix(features, normalize_features)
    if pca_dim <= 0:
        return features

    n_components = min(int(pca_dim), features.shape[0] - 1, features.shape[1])
    if n_components <= 0 or n_components >= features.shape[1]:
        return features

    pca = PCA(n_components=n_components, random_state=seed)
    return pca.fit_transform(features)


def compute_umap_embedding(bundle, args, seed_offset=0):
    try:
        import umap
    except ImportError as exc:
        raise ImportError("UMAP visualization requires umap-learn. Install it with `pip install umap-learn`.") from exc

    features = np.concatenate([bundle.clean_features, bundle.adv_features], axis=0)
    labels = np.concatenate([bundle.labels, bundle.labels], axis=0)
    domains = np.array(["clean"] * len(bundle.labels) + ["pgd"] * len(bundle.labels))
    features = preprocess_for_umap(
        features,
        pca_dim=args.umap_pca_dim,
        normalize_features=args.normalize_features,
        seed=args.seed + seed_offset,
    )
    n_neighbors = min(int(args.umap_n_neighbors), max(features.shape[0] - 1, 2))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        random_state=args.seed + seed_offset,
        low_memory=True,
    )
    embedding = reducer.fit_transform(features)
    return EmbeddingBundle(embedding=embedding, labels=labels, domains=domains)


def get_class_names(dataset, class_ids):
    classes = getattr(dataset, "classes", None)
    names = {}
    for class_id in class_ids:
        if classes is not None and 0 <= class_id < len(classes):
            names[class_id] = classes[class_id]
        else:
            names[class_id] = str(class_id)
    return names


def plot_single_embedding(embedding_bundle, class_ids, class_names, title, output_path, draw_pair_lines=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    color_map = plt.get_cmap("tab20", len(class_ids))
    class_to_color = {class_id: color_map(index) for index, class_id in enumerate(class_ids)}
    clean_mask = embedding_bundle.domains == "clean"
    adv_mask = embedding_bundle.domains == "pgd"

    fig, ax = plt.subplots(figsize=(10, 8))
    if draw_pair_lines:
        half = embedding_bundle.embedding.shape[0] // 2
        for index in range(half):
            class_id = int(embedding_bundle.labels[index])
            ax.plot(
                [embedding_bundle.embedding[index, 0], embedding_bundle.embedding[index + half, 0]],
                [embedding_bundle.embedding[index, 1], embedding_bundle.embedding[index + half, 1]],
                color=class_to_color[class_id],
                alpha=0.08,
                linewidth=0.4,
            )

    for class_id in class_ids:
        class_mask = embedding_bundle.labels == class_id
        ax.scatter(
            embedding_bundle.embedding[clean_mask & class_mask, 0],
            embedding_bundle.embedding[clean_mask & class_mask, 1],
            s=16,
            marker="o",
            alpha=0.72,
            color=class_to_color[class_id],
            linewidths=0.0,
        )
        ax.scatter(
            embedding_bundle.embedding[adv_mask & class_mask, 0],
            embedding_bundle.embedding[adv_mask & class_mask, 1],
            s=22,
            marker="x",
            alpha=0.86,
            color=class_to_color[class_id],
            linewidths=0.8,
        )

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    domain_handles = [
        Line2D([0], [0], marker="o", color="black", label="clean", linestyle="None", markersize=6),
        Line2D([0], [0], marker="x", color="black", label="PGD", linestyle="None", markersize=6),
    ]
    class_handles = [
        Line2D([0], [0], marker="o", color=class_to_color[class_id], label=f"{class_id}: {class_names[class_id]}", linestyle="None", markersize=5)
        for class_id in class_ids
    ]
    first_legend = ax.legend(handles=domain_handles, loc="upper right", frameon=True)
    ax.add_artist(first_legend)
    ax.legend(handles=class_handles, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_domain_embedding(embedding_bundle, class_ids, class_names, title, output_path, marker="o"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    color_map = plt.get_cmap("tab20", len(class_ids))
    class_to_color = {class_id: color_map(index) for index, class_id in enumerate(class_ids)}

    fig, ax = plt.subplots(figsize=(10, 8))
    for class_id in class_ids:
        class_mask = embedding_bundle.labels == class_id
        ax.scatter(
            embedding_bundle.embedding[class_mask, 0],
            embedding_bundle.embedding[class_mask, 1],
            s=18 if marker == "o" else 24,
            marker=marker,
            alpha=0.78,
            color=class_to_color[class_id],
            linewidths=0.8 if marker == "x" else 0.0,
        )

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    class_handles = [
        Line2D([0], [0], marker="o", color=class_to_color[class_id], label=f"{class_id}: {class_names[class_id]}", linestyle="None", markersize=5)
        for class_id in class_ids
    ]
    ax.legend(handles=class_handles, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_side_by_side(original_embedding, ranpac_embedding, class_ids, class_names, output_path, draw_pair_lines=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    color_map = plt.get_cmap("tab20", len(class_ids))
    class_to_color = {class_id: color_map(index) for index, class_id in enumerate(class_ids)}
    panels = [
        (original_embedding, "Original penultimate features"),
        (ranpac_embedding, "HiRA+RanPAC projected features"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    for ax, (embedding_bundle, title) in zip(axes, panels):
        clean_mask = embedding_bundle.domains == "clean"
        adv_mask = embedding_bundle.domains == "pgd"
        if draw_pair_lines:
            half = embedding_bundle.embedding.shape[0] // 2
            for index in range(half):
                class_id = int(embedding_bundle.labels[index])
                ax.plot(
                    [embedding_bundle.embedding[index, 0], embedding_bundle.embedding[index + half, 0]],
                    [embedding_bundle.embedding[index, 1], embedding_bundle.embedding[index + half, 1]],
                    color=class_to_color[class_id],
                    alpha=0.06,
                    linewidth=0.35,
                )
        for class_id in class_ids:
            class_mask = embedding_bundle.labels == class_id
            ax.scatter(
                embedding_bundle.embedding[clean_mask & class_mask, 0],
                embedding_bundle.embedding[clean_mask & class_mask, 1],
                s=14,
                marker="o",
                alpha=0.70,
                color=class_to_color[class_id],
                linewidths=0.0,
            )
            ax.scatter(
                embedding_bundle.embedding[adv_mask & class_mask, 0],
                embedding_bundle.embedding[adv_mask & class_mask, 1],
                s=20,
                marker="x",
                alpha=0.85,
                color=class_to_color[class_id],
                linewidths=0.8,
            )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    domain_handles = [
        Line2D([0], [0], marker="o", color="black", label="clean", linestyle="None", markersize=6),
        Line2D([0], [0], marker="x", color="black", label="PGD", linestyle="None", markersize=6),
    ]
    class_handles = [
        Line2D([0], [0], marker="o", color=class_to_color[class_id], label=f"{class_id}: {class_names[class_id]}", linestyle="None", markersize=5)
        for class_id in class_ids
    ]
    axes[0].legend(handles=domain_handles, loc="upper right", frameon=True)
    axes[1].legend(handles=class_handles, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_embedding_csv(path, variant_name, embedding_bundle, selected_indices, class_names, clean_predictions, adv_predictions, x_name="x", y_name="y"):
    num_samples = len(selected_indices)
    rows = []
    predictions = np.concatenate([clean_predictions, adv_predictions], axis=0)
    repeated_indices = list(selected_indices) + list(selected_indices)
    for row_index in range(embedding_bundle.embedding.shape[0]):
        label = int(embedding_bundle.labels[row_index])
        rows.append(
            {
                "variant": variant_name,
                "sample_index": repeated_indices[row_index],
                "domain": embedding_bundle.domains[row_index],
                "class_id": label,
                "class_name": class_names[label],
                "prediction": int(predictions[row_index]),
                "correct": int(predictions[row_index] == label),
                x_name: float(embedding_bundle.embedding[row_index, 0]),
                y_name: float(embedding_bundle.embedding[row_index, 1]),
            }
        )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_domain_embedding_csv(path, variant_name, embedding_bundle, selected_indices, class_names, predictions, x_name="x", y_name="y"):
    rows = []
    for row_index in range(embedding_bundle.embedding.shape[0]):
        label = int(embedding_bundle.labels[row_index])
        rows.append(
            {
                "variant": variant_name,
                "sample_index": selected_indices[row_index],
                "domain": embedding_bundle.domains[row_index],
                "class_id": label,
                "class_name": class_names[label],
                "prediction": int(predictions[row_index]),
                "correct": int(predictions[row_index] == label),
                x_name: float(embedding_bundle.embedding[row_index, 0]),
                y_name: float(embedding_bundle.embedding[row_index, 1]),
            }
        )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def compute_silhouette_score(embedding_bundle):
    try:
        from sklearn.metrics import silhouette_score
    except ImportError as exc:
        raise ImportError("Silhouette scoring requires scikit-learn. Install it with `pip install scikit-learn`.") from exc

    labels = np.asarray(embedding_bundle.labels, dtype=np.int64)
    embedding = np.asarray(embedding_bundle.embedding, dtype=np.float32)
    unique_labels = np.unique(labels)
    if embedding.shape[0] < 3 or unique_labels.shape[0] < 2 or unique_labels.shape[0] >= embedding.shape[0]:
        return float("nan")
    return float(silhouette_score(embedding, labels, metric="euclidean"))


def build_silhouette_rows(named_embeddings):
    rows = []
    for name, feature_space, embedding_bundle in named_embeddings:
        domains = np.unique(embedding_bundle.domains.astype(str))
        rows.append(
            {
                "embedding": name,
                "feature_space": feature_space,
                "domain": "+".join(domains.tolist()),
                "num_points": int(embedding_bundle.embedding.shape[0]),
                "num_classes": int(np.unique(embedding_bundle.labels).shape[0]),
                "silhouette_score": compute_silhouette_score(embedding_bundle),
            }
        )
    return rows


def write_silhouette_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_outputs(
    args,
    run_dir,
    dataset,
    selected_indices,
    class_ids,
    original_bundle,
    hira_bundle,
    ranpac_bundle,
    original_embedding,
    hira_embedding,
    ranpac_embedding,
    original_clean_embedding,
    original_pgd_embedding,
    hira_clean_embedding,
    hira_pgd_embedding,
    ranpac_clean_embedding,
    ranpac_pgd_embedding,
    original_umap_embedding,
    ranpac_umap_embedding,
):
    class_names = get_class_names(dataset, class_ids)
    run_dir.mkdir(parents=True, exist_ok=True)

    plot_side_by_side(
        original_embedding,
        ranpac_embedding,
        class_ids=class_ids,
        class_names=class_names,
        output_path=run_dir / "tsne_original_vs_hira_ranpac.png",
        draw_pair_lines=args.draw_pair_lines,
    )
    plot_single_embedding(
        original_embedding,
        class_ids=class_ids,
        class_names=class_names,
        title="Original penultimate features",
        output_path=run_dir / "tsne_original_penultimate.png",
        draw_pair_lines=args.draw_pair_lines,
    )
    plot_single_embedding(
        hira_embedding,
        class_ids=class_ids,
        class_names=class_names,
        title="HiRA penultimate features",
        output_path=run_dir / "tsne_hira_penultimate.png",
        draw_pair_lines=args.draw_pair_lines,
    )
    plot_single_embedding(
        ranpac_embedding,
        class_ids=class_ids,
        class_names=class_names,
        title="HiRA+RanPAC projected features",
        output_path=run_dir / "tsne_hira_ranpac_projected.png",
        draw_pair_lines=args.draw_pair_lines,
    )
    for embedding_bundle, title, filename, marker in (
        (original_clean_embedding, "Clean: original penultimate features", "tsne_clean_original_penultimate.png", "o"),
        (hira_clean_embedding, "Clean: HiRA penultimate features", "tsne_clean_hira_penultimate.png", "o"),
        (ranpac_clean_embedding, "Clean: RanPAC projected features", "tsne_clean_ranpac_projected.png", "o"),
        (original_pgd_embedding, "PGD: original penultimate features", "tsne_pgd_original_penultimate.png", "o"),
        (hira_pgd_embedding, "PGD: HiRA penultimate features", "tsne_pgd_hira_penultimate.png", "o"),
        (ranpac_pgd_embedding, "PGD: RanPAC projected features", "tsne_pgd_ranpac_projected.png", "o"),
    ):
        plot_domain_embedding(
            embedding_bundle,
            class_ids=class_ids,
            class_names=class_names,
            title=title,
            output_path=run_dir / filename,
            marker="o",
        )
    plot_side_by_side(
        original_umap_embedding,
        ranpac_umap_embedding,
        class_ids=class_ids,
        class_names=class_names,
        output_path=run_dir / "umap_original_vs_hira_ranpac.png",
        draw_pair_lines=args.draw_pair_lines,
    )
    plot_single_embedding(
        original_umap_embedding,
        class_ids=class_ids,
        class_names=class_names,
        title="Original penultimate features (UMAP)",
        output_path=run_dir / "umap_original_penultimate.png",
        draw_pair_lines=args.draw_pair_lines,
    )
    plot_single_embedding(
        ranpac_umap_embedding,
        class_ids=class_ids,
        class_names=class_names,
        title="HiRA+RanPAC projected features (UMAP)",
        output_path=run_dir / "umap_hira_ranpac_projected.png",
        draw_pair_lines=args.draw_pair_lines,
    )

    write_embedding_csv(
        run_dir / "tsne_original_embeddings.csv",
        VARIANT_ORIGINAL,
        original_embedding,
        selected_indices,
        class_names,
        original_bundle.clean_predictions,
        original_bundle.adv_predictions,
        x_name="tsne_x",
        y_name="tsne_y",
    )
    write_embedding_csv(
        run_dir / "tsne_hira_ranpac_embeddings.csv",
        VARIANT_HIRA_RANPAC,
        ranpac_embedding,
        selected_indices,
        class_names,
        ranpac_bundle.clean_predictions,
        ranpac_bundle.adv_predictions,
        x_name="tsne_x",
        y_name="tsne_y",
    )
    write_embedding_csv(
        run_dir / "tsne_hira_embeddings.csv",
        "hira",
        hira_embedding,
        selected_indices,
        class_names,
        hira_bundle.clean_predictions,
        hira_bundle.adv_predictions,
        x_name="tsne_x",
        y_name="tsne_y",
    )
    write_embedding_csv(
        run_dir / "umap_original_embeddings.csv",
        VARIANT_ORIGINAL,
        original_umap_embedding,
        selected_indices,
        class_names,
        original_bundle.clean_predictions,
        original_bundle.adv_predictions,
        x_name="umap_x",
        y_name="umap_y",
    )
    write_embedding_csv(
        run_dir / "umap_hira_ranpac_embeddings.csv",
        VARIANT_HIRA_RANPAC,
        ranpac_umap_embedding,
        selected_indices,
        class_names,
        ranpac_bundle.clean_predictions,
        ranpac_bundle.adv_predictions,
        x_name="umap_x",
        y_name="umap_y",
    )

    for filename, variant_name, embedding_bundle, predictions in (
        ("tsne_clean_original_penultimate_embeddings.csv", VARIANT_ORIGINAL, original_clean_embedding, original_bundle.clean_predictions),
        ("tsne_clean_hira_penultimate_embeddings.csv", "hira", hira_clean_embedding, hira_bundle.clean_predictions),
        ("tsne_clean_ranpac_projected_embeddings.csv", VARIANT_HIRA_RANPAC, ranpac_clean_embedding, ranpac_bundle.clean_predictions),
        ("tsne_pgd_original_penultimate_embeddings.csv", VARIANT_ORIGINAL, original_pgd_embedding, original_bundle.adv_predictions),
        ("tsne_pgd_hira_penultimate_embeddings.csv", "hira", hira_pgd_embedding, hira_bundle.adv_predictions),
        ("tsne_pgd_ranpac_projected_embeddings.csv", VARIANT_HIRA_RANPAC, ranpac_pgd_embedding, ranpac_bundle.adv_predictions),
    ):
        write_domain_embedding_csv(
            run_dir / filename,
            variant_name,
            embedding_bundle,
            selected_indices,
            class_names,
            predictions,
            x_name="tsne_x",
            y_name="tsne_y",
        )

    np.savez_compressed(
        run_dir / "tsne_embeddings.npz",
        original_embedding=original_embedding.embedding,
        original_labels=original_embedding.labels,
        original_domains=original_embedding.domains,
        hira_embedding=hira_embedding.embedding,
        hira_labels=hira_embedding.labels,
        hira_domains=hira_embedding.domains,
        hira_ranpac_embedding=ranpac_embedding.embedding,
        hira_ranpac_labels=ranpac_embedding.labels,
        hira_ranpac_domains=ranpac_embedding.domains,
        original_clean_embedding=original_clean_embedding.embedding,
        original_pgd_embedding=original_pgd_embedding.embedding,
        hira_clean_embedding=hira_clean_embedding.embedding,
        hira_pgd_embedding=hira_pgd_embedding.embedding,
        ranpac_clean_embedding=ranpac_clean_embedding.embedding,
        ranpac_pgd_embedding=ranpac_pgd_embedding.embedding,
        selected_indices=np.array(selected_indices, dtype=np.int64),
        selected_class_ids=np.array(class_ids, dtype=np.int64),
    )
    np.savez_compressed(
        run_dir / "umap_embeddings.npz",
        original_embedding=original_umap_embedding.embedding,
        original_labels=original_umap_embedding.labels,
        original_domains=original_umap_embedding.domains,
        hira_ranpac_embedding=ranpac_umap_embedding.embedding,
        hira_ranpac_labels=ranpac_umap_embedding.labels,
        hira_ranpac_domains=ranpac_umap_embedding.domains,
        selected_indices=np.array(selected_indices, dtype=np.int64),
        selected_class_ids=np.array(class_ids, dtype=np.int64),
    )
    if args.save_features:
        np.savez_compressed(
            run_dir / "raw_features.npz",
            original_clean_features=original_bundle.clean_features,
            original_adv_features=original_bundle.adv_features,
            hira_clean_features=hira_bundle.clean_features,
            hira_adv_features=hira_bundle.adv_features,
            hira_ranpac_clean_features=ranpac_bundle.clean_features,
            hira_ranpac_adv_features=ranpac_bundle.adv_features,
            labels=original_bundle.labels,
            selected_indices=np.array(selected_indices, dtype=np.int64),
            selected_class_ids=np.array(class_ids, dtype=np.int64),
        )

    silhouette_rows = build_silhouette_rows(
        [
            ("tsne_original_penultimate", "original_penultimate", original_embedding),
            ("tsne_hira_penultimate", "hira_penultimate", hira_embedding),
            ("tsne_ranpac_projected", "ranpac_projected", ranpac_embedding),
            ("tsne_clean_original_penultimate", "original_penultimate", original_clean_embedding),
            ("tsne_clean_hira_penultimate", "hira_penultimate", hira_clean_embedding),
            ("tsne_clean_ranpac_projected", "ranpac_projected", ranpac_clean_embedding),
            ("tsne_pgd_original_penultimate", "original_penultimate", original_pgd_embedding),
            ("tsne_pgd_hira_penultimate", "hira_penultimate", hira_pgd_embedding),
            ("tsne_pgd_ranpac_projected", "ranpac_projected", ranpac_pgd_embedding),
        ]
    )
    write_silhouette_csv(run_dir / "tsne_silhouette_scores.csv", silhouette_rows)
    silhouette_scores = {row["embedding"]: row["silhouette_score"] for row in silhouette_rows}

    num_samples = len(selected_indices)
    summary = {
        "model_name": args.model_name,
        "dataset": DATASET,
        "threat_model": args.threat_model,
        "selected_class_ids": class_ids,
        "selected_class_names": {str(class_id): class_names[class_id] for class_id in class_ids},
        "samples_per_class": args.samples_per_class,
        "num_samples": num_samples,
        "pgd_eps": args.eps,
        "pgd_steps": args.pgd_steps,
        "pgd_step_size": args.pgd_step_size if args.pgd_step_size is not None else 2.0 * args.eps / max(args.pgd_steps, 1),
        "pgd_random_start": args.pgd_random_start,
        "pgd_mask_logits_to_visualization_classes": args.mask_pgd_logits,
        "pgd_mask_class_ids": class_ids if args.mask_pgd_logits else None,
        "original_clean_acc": float((original_bundle.clean_predictions == original_bundle.labels).mean()),
        "original_pgd_acc": float((original_bundle.adv_predictions == original_bundle.labels).mean()),
        "hira_ranpac_clean_acc": float((ranpac_bundle.clean_predictions == ranpac_bundle.labels).mean()),
        "hira_ranpac_pgd_acc": float((ranpac_bundle.adv_predictions == ranpac_bundle.labels).mean()),
        "original_feature_space": "penultimate_input_to_last_linear",
        "hira_feature_space": "penultimate_input_to_residual_ranpac_head_after_hira",
        "hira_ranpac_feature_space": "gelu_penultimate_times_ranpac_w_rand",
        "tsne_perplexity": args.tsne_perplexity,
        "tsne_iterations": args.tsne_iterations,
        "tsne_metric": args.tsne_metric,
        "umap_components": 2,
        "umap_n_neighbors": args.umap_n_neighbors,
        "umap_min_dist": args.umap_min_dist,
        "umap_metric": args.umap_metric,
        "umap_pca_dim": args.umap_pca_dim,
        "pca_dim": args.pca_dim,
        "normalize_features": args.normalize_features,
        "hira_expansion_dim": args.hira_expansion_dim,
        "hira_num_blocks": args.hira_num_blocks,
        "ranpac_rp_dim": args.ranpac_rp_dim,
        "ranpac_lambda": args.ranpac_lambda,
        "ranpac_temp": args.ranpac_temp,
        "ranpac_hardneg_topk": args.ranpac_hardneg_topk,
        "ranpac_hardneg_gamma": args.ranpac_hardneg_gamma,
        "soft_threshold_alpha": args.soft_threshold_alpha,
        "soft_threshold_beta": args.soft_threshold_beta,
        "soft_threshold_mode": args.soft_threshold_mode,
        "tsne_silhouette_scores": silhouette_scores,
        "outputs": [
            "tsne_clean_original_penultimate.png",
            "tsne_clean_hira_penultimate.png",
            "tsne_clean_ranpac_projected.png",
            "tsne_pgd_original_penultimate.png",
            "tsne_pgd_hira_penultimate.png",
            "tsne_pgd_ranpac_projected.png",
            "tsne_original_penultimate.png",
            "tsne_hira_penultimate.png",
            "tsne_hira_ranpac_projected.png",
            "tsne_original_vs_hira_ranpac.png",
            "umap_original_penultimate.png",
            "umap_hira_ranpac_projected.png",
            "umap_original_vs_hira_ranpac.png",
            "tsne_embeddings.npz",
            "tsne_silhouette_scores.csv",
            "umap_embeddings.npz",
            "summary.json",
        ],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create t-SNE plots for RobustBench original penultimate features vs HiRA+RanPAC projected features."
    )
    parser.add_argument("--model-name", "--model_name", required=True, help="RobustBench ImageNet model name.")
    parser.add_argument("--threat-model", "--threat_model", default="Linf", choices=["Linf", "L2"], help="Threat model used to load the RobustBench model.")
    parser.add_argument("--data-dir", "--data_dir", default="./dataset/imagenet", help="ImageNet root containing train/ and val/.")
    parser.add_argument("--model-dir", "--model_dir", default="./robustbench_models", help="RobustBench checkpoint cache directory.")
    parser.add_argument("--device", default="cuda:0", help="Device, e.g. cuda:0 or cpu.")
    parser.add_argument("--seed", type=int, default=0, help="Global random seed and random class-selection seed.")
    parser.add_argument("--batch-size", "--batch_size", type=int, default=32, help="Batch size for feature collection and PGD.")
    parser.add_argument("--num-workers", "--num_workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--output-dir", "--output_dir", default="visualization/tsne_outputs", help="Directory where figures and metadata are saved.")
    parser.add_argument("--run-name", "--run_name", default="", help="Optional output subdirectory name.")

    parser.add_argument("--num-classes", "--num_classes", type=int, default=20, help="Number of ImageNet classes to visualize when --class-ids is empty.")
    parser.add_argument("--samples-per-class", "--samples_per_class", type=int, default=50, help="Number of validation images per selected class.")
    parser.add_argument("--class-ids", "--class_ids", default="", help="Optional comma-separated ImageNet class IDs. Overrides --num-classes.")

    parser.add_argument("--eps", type=parse_float_or_fraction, default=4.0 / 255.0, help="Linf PGD epsilon; accepts floats or fractions like 4/255.")
    parser.add_argument("--pgd-steps", "--pgd_steps", type=int, default=40, help="PGD steps.")
    parser.add_argument("--pgd-step-size", "--pgd_step_size", type=parse_float_or_fraction, default=None, help="Optional PGD step size. Defaults to 2 * eps / steps.")
    parser.add_argument("--pgd-random-start", "--pgd_random_start", type=str2bool, default=True, help="Use random-start Linf PGD.")
    parser.add_argument("--mask-pgd-logits", "--mask_pgd_logits", type=str2bool, default=True, help="During PGD, set logits outside the selected visualization classes to -inf.")

    parser.add_argument("--pca-dim", "--pca_dim", type=int, default=50, help="PCA dimension before t-SNE.")
    parser.add_argument("--normalize-features", "--normalize_features", type=str2bool, default=True, help="L2-normalize features before t-SNE/UMAP preprocessing.")
    parser.add_argument("--tsne-perplexity", "--tsne_perplexity", type=float, default=30.0, help="t-SNE perplexity.")
    parser.add_argument("--tsne-iterations", "--tsne_iterations", type=int, default=1000, help="t-SNE optimization iterations.")
    parser.add_argument("--tsne-metric", "--tsne_metric", default="euclidean", help="Distance metric passed to sklearn.manifold.TSNE.")
    parser.add_argument("--tsne-init", "--tsne_init", default="pca", choices=["pca", "random"], help="t-SNE initialization.")
    parser.add_argument("--tsne-verbose", "--tsne_verbose", type=str2bool, default=False, help="Print sklearn t-SNE progress.")
    parser.add_argument("--umap-n-neighbors", "--umap_n_neighbors", type=int, default=30, help="UMAP local neighborhood size.")
    parser.add_argument("--umap-min-dist", "--umap_min_dist", type=float, default=0.1, help="UMAP minimum embedding distance.")
    parser.add_argument("--umap-metric", "--umap_metric", default="euclidean", help="Distance metric passed to UMAP.")
    parser.add_argument("--umap-pca-dim", "--umap_pca_dim", type=int, default=50, help="Optional PCA pre-reduction dimension before UMAP; <=0 disables it.")
    parser.add_argument("--draw-pair-lines", "--draw_pair_lines", type=str2bool, default=False, help="Draw faint clean-to-PGD lines for each image pair.")
    parser.add_argument("--save-features", "--save_features", action="store_true", help="Also save high-dimensional raw features. This can be large.")

    parser.add_argument("--hira-expansion-dim", "--hira_expansion_dim", type=int, default=16384, help="HiRA hidden expansion dimension.")
    parser.add_argument("--hira-num-blocks", "--hira_num_blocks", type=int, default=4, help="Number of final MLP blocks receiving HiRA adapters.")
    parser.add_argument("--hira-batch-size", "--hira_batch_size", type=int, default=128, help="HiRA fitting batch size.")
    parser.add_argument("--hira-num-workers", "--hira_num_workers", type=int, default=4, help="HiRA fitting DataLoader workers.")
    parser.add_argument("--hira-epochs", "--hira_epochs", type=int, default=1, help="Legacy compatibility flag; closed-form HiRA ignores this.")
    parser.add_argument("--hira-lr", "--hira_lr", type=float, default=1e-4, help="Legacy compatibility flag; closed-form HiRA ignores this.")
    parser.add_argument("--hira-weight-decay", "--hira_weight_decay", type=float, default=1e-4, help="Legacy compatibility flag; closed-form HiRA ignores this.")
    parser.add_argument("--hira-seed", "--hira_seed", type=int, default=0, help="HiRA projection and split seed.")
    parser.add_argument("--hira-cache-dir", "--hira_cache_dir", default="pretrained/hira_robustbench", help="HiRA cache directory.")
    parser.add_argument("--hira-dataset-root", "--hira_dataset_root", default="", help="Optional ImageNet root for HiRA fitting. Defaults to --data-dir.")
    parser.add_argument("--hira-max-train-samples", "--hira_max_train_samples", type=int, default=-1, help="Optional cap on ImageNet train samples for HiRA fitting.")
    parser.add_argument("--hira-force-retrain", "--hira_force_retrain", type=str2bool, default=False, help="Ignore cached HiRA weights and refit.")

    parser.add_argument("--adapt-noise-eps", "--adapt_noise_eps", type=parse_float_or_fraction, default=0.0, help="Linf noise radius used while fitting HiRA/RanPAC adaptation statistics.")
    parser.add_argument("--adapt-noise-num", "--adapt_noise_num", type=int, default=1, help="Number of noisy samples per training image for adaptation statistics.")
    parser.add_argument("--adapt-alpha", "--adapt_alpha", type=float, default=1.0, help="Total weight assigned to noisy adaptation statistics.")
    parser.add_argument("--soft-threshold-alpha", "--soft_threshold_alpha", type=float, default=0.9, help="HiRA MeanSparse alpha; 0 disables inference sparsification.")
    parser.add_argument("--soft-threshold-beta", "--soft_threshold_beta", type=float, default=4.0, help="HiRA MeanSparse beta.")
    parser.add_argument("--soft-threshold-stat-eps", "--soft_threshold_stat_eps", type=float, default=DEFAULT_MEANSPARSE_STAT_EPS, help="HiRA MeanSparse statistic epsilon.")
    parser.add_argument("--soft-threshold-mode", "--soft_threshold_mode", choices=["near_mean", "away_from_mean"], default="away_from_mean", help="HiRA MeanSparse mode.")
    parser.add_argument("--stability-ridge-gamma", "--stability_ridge_gamma", type=float, default=0.0, help="Stability-aware diagonal ridge strength.")
    parser.add_argument("--stability-ridge-stat-eps", "--stability_ridge_stat_eps", type=float, default=DEFAULT_STABILITY_RIDGE_STAT_EPS, help="Stability-aware ridge statistic epsilon.")

    parser.add_argument("--ranpac-rp-dim", "--ranpac_rp_dim", type=int, default=10000, help="RanPAC random projection dimension.")
    parser.add_argument("--ranpac-fit-batch-size", "--ranpac_fit_batch_size", type=int, default=64, help="RanPAC fitting batch size.")
    parser.add_argument("--ranpac-num-workers", "--ranpac_num_workers", type=int, default=4, help="RanPAC fitting DataLoader workers.")
    parser.add_argument("--ranpac-seed", "--ranpac_seed", type=int, default=0, help="RanPAC random projection seed.")
    parser.add_argument("--ranpac-lambda", "--ranpac_lambda", type=float, default=0.5, help="Mixing weight for final HiRA+RanPAC logits.")
    parser.add_argument("--ranpac-temp", "--ranpac_temp", type=float, default=1.0, help="Temperature applied to RanPAC logits.")
    parser.add_argument("--ranpac-hardneg-topk", "--ranpac_hardneg_topk", type=int, default=0, help="Hard-negative classes to suppress in RanPAC targets.")
    parser.add_argument("--ranpac-hardneg-gamma", "--ranpac_hardneg_gamma", type=float, default=0.0, help="Total hard-negative suppression weight.")
    parser.add_argument("--ranpac-cache-dir", "--ranpac_cache_dir", default="pretrained/ranpac_robustbench", help="RanPAC cache directory.")
    parser.add_argument("--ranpac-dataset-root", "--ranpac_dataset_root", default="", help="Optional ImageNet root for RanPAC fitting. Defaults to --data-dir.")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    model_preprocessing = resolve_model_preprocessing(args.model_name, args.threat_model)
    dataset = build_imagenet_dataset(args.data_dir, model_preprocessing)
    class_ids_arg = parse_class_ids(args.class_ids)
    selected_indices, selected_class_ids = select_balanced_indices(
        dataset,
        num_classes=args.num_classes,
        samples_per_class=args.samples_per_class,
        seed=args.seed,
        class_ids=class_ids_arg,
    )
    loader = build_eval_loader(dataset, selected_indices, args.batch_size, args.num_workers)

    run_name = args.run_name
    if not run_name:
        eps_tag = sanitize_name(args.eps)
        run_name = f"{sanitize_name(args.model_name)}_classes{len(selected_class_ids)}_n{args.samples_per_class}_eps{eps_tag}_seed{args.seed}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving t-SNE outputs to: {run_dir}")
    print(f"Selected classes: {selected_class_ids}")
    args.attack_class_ids = selected_class_ids if args.mask_pgd_logits else None
    if args.attack_class_ids is not None:
        print("PGD logit mask enabled: attacking only among selected visualization classes.")

    print("Loading original RobustBench model...")
    original_model = freeze_model(load_robustbench_model(args.model_name, args.threat_model, args.model_dir, device))
    original_extractor = PenultimateFeatureExtractor(original_model)
    print(f"Original feature layer: {original_extractor.module_name}")
    original_bundle = collect_features_for_variant(
        original_model,
        loader,
        original_extractor,
        device,
        args,
        VARIANT_ORIGINAL,
    )
    del original_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("Loading and wrapping RobustBench model with HiRA+RanPAC...")
    ranpac_model = freeze_model(build_hira_ranpac_model(args, model_preprocessing, device))
    hira_extractor = HiraPenultimateFeatureExtractor(ranpac_model)
    ranpac_extractor = RanPACProjectedFeatureExtractor(ranpac_model)
    print(f"HiRA feature layer: {hira_extractor.module_name}")
    print(f"RanPAC projected feature layer: {ranpac_extractor.module_name}")
    hira_bundle = collect_features_for_variant(
        ranpac_model,
        loader,
        hira_extractor,
        device,
        args,
        "hira",
    )
    ranpac_bundle = collect_features_for_variant(
        ranpac_model,
        loader,
        ranpac_extractor,
        device,
        args,
        VARIANT_HIRA_RANPAC,
    )
    del ranpac_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("Running t-SNE for original penultimate features...")
    original_embedding = compute_embedding(original_bundle, args, seed_offset=0)
    print("Running t-SNE for HiRA penultimate features...")
    hira_embedding = compute_embedding(hira_bundle, args, seed_offset=1)
    print("Running t-SNE for HiRA+RanPAC projected features...")
    ranpac_embedding = compute_embedding(ranpac_bundle, args, seed_offset=2)
    print("Running separate clean/PGD t-SNE plots...")
    original_clean_embedding, original_pgd_embedding = compute_clean_pgd_embeddings(original_bundle, args, seed_offset=10)
    hira_clean_embedding, hira_pgd_embedding = compute_clean_pgd_embeddings(hira_bundle, args, seed_offset=20)
    ranpac_clean_embedding, ranpac_pgd_embedding = compute_clean_pgd_embeddings(ranpac_bundle, args, seed_offset=30)
    print("Running UMAP for original penultimate features...")
    original_umap_embedding = compute_umap_embedding(original_bundle, args, seed_offset=0)
    print("Running UMAP for HiRA+RanPAC projected features...")
    ranpac_umap_embedding = compute_umap_embedding(ranpac_bundle, args, seed_offset=1)

    summary = save_outputs(
        args,
        run_dir,
        dataset,
        selected_indices,
        selected_class_ids,
        original_bundle,
        hira_bundle,
        ranpac_bundle,
        original_embedding,
        hira_embedding,
        ranpac_embedding,
        original_clean_embedding,
        original_pgd_embedding,
        hira_clean_embedding,
        hira_pgd_embedding,
        ranpac_clean_embedding,
        ranpac_pgd_embedding,
        original_umap_embedding,
        ranpac_umap_embedding,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
