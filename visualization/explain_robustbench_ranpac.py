#!/usr/bin/env python3
"""
Explanatory plots for RobustBench ImageNet models with HiRA+RanPAC.

This script compares exactly two variants:
  1. original: the unmodified RobustBench model
  2. hira_ranpac_regression: the same model wrapped with HiRA and RanPAC

It generates white-box PGD adversarial examples for each variant at a list of
epsilon values, then saves figures for robust accuracy, margin degradation,
feature drift, mutual information between clean/adversarial features, centroid
stability, kNN neighborhood preservation, and rescue-case margin deltas.
"""

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from classifiers.mean_sparse import DEFAULT_MEANSPARSE_STAT_EPS
from classifiers.ranpac import ResidualRanPACLinear
from classifiers.stability_ridge import DEFAULT_STABILITY_RIDGE_STAT_EPS
from visualization.tsne_robustbench_ranpac import (
    DATASET,
    build_eval_loader,
    build_hira_ranpac_model,
    build_imagenet_dataset,
    find_last_linear,
    freeze_model,
    load_robustbench_model,
    mask_logits_to_classes,
    parse_float_or_fraction,
    resolve_device,
    resolve_model_preprocessing,
    sanitize_name,
    select_all_indices,
    set_seed,
    str2bool,
)


VARIANT_ORIGINAL = "original"
VARIANT_HIRA_RANPAC = "hira_ranpac_regression"


@dataclass
class EvalOutputs:
    sample_indices: np.ndarray
    labels: np.ndarray
    predictions: np.ndarray
    logits: np.ndarray
    penultimate: np.ndarray
    input_l2_perturbation: np.ndarray = None
    projected: np.ndarray = None
    baseline_logits: np.ndarray = None
    ranpac_logits: np.ndarray = None
    mixed_logits: np.ndarray = None


class OriginalProbe:
    def __init__(self, model):
        self.module_name, self.module = find_last_linear(model)

    def __call__(self, model, inputs):
        captured_features = []

        def hook(_, hook_inputs):
            captured_features.append(hook_inputs[0].detach().view(inputs.size(0), -1).float().cpu())

        handle = self.module.register_forward_pre_hook(hook)
        try:
            with torch.no_grad():
                logits = model(inputs).detach().float().cpu()
        finally:
            handle.remove()

        if not captured_features:
            raise RuntimeError(f"No penultimate features were captured from {self.module_name}.")
        return {
            "logits": logits,
            "penultimate": captured_features[-1],
        }


class HiraRanPACProbe:
    def __init__(self, model):
        heads = [(name, module) for name, module in model.named_modules() if isinstance(module, ResidualRanPACLinear)]
        if not heads:
            raise ValueError("No ResidualRanPACLinear head found. Did HiRA+RanPAC wrapping run successfully?")
        if len(heads) > 1:
            print(f"Found {len(heads)} ResidualRanPACLinear heads; using the last one: {heads[-1][0]}")
        self.module_name, self.head = heads[-1]

    def __call__(self, model, inputs):
        captured_features = []

        def hook(_, hook_inputs):
            captured_features.append(hook_inputs[0].detach().view(inputs.size(0), -1).float())

        handle = self.head.register_forward_pre_hook(hook)
        try:
            with torch.no_grad():
                logits = model(inputs).detach().float().cpu()
        finally:
            handle.remove()

        if not captured_features:
            raise RuntimeError(f"No HiRA+RanPAC head features were captured from {self.module_name}.")

        penultimate = captured_features[-1].to(inputs.device)
        with torch.no_grad():
            baseline_logits = self.head.original_linear(penultimate).detach().float().cpu()
            ranpac_logits = (self.head.ranpac_linear(penultimate) / self.head.ranpac_temp).detach().float().cpu()
            mixed_logits = (
                (1.0 - self.head.ranpac_lambda) * baseline_logits
                + self.head.ranpac_lambda * ranpac_logits
            ).detach().float().cpu()
            projected = F.gelu(
                penultimate.float() @ self.head.ranpac_linear.w_rand.to(device=penultimate.device, dtype=torch.float32)
            ).detach().float().cpu()

        return {
            "logits": logits,
            "penultimate": penultimate.detach().float().cpu(),
            "projected": projected,
            "baseline_logits": baseline_logits,
            "ranpac_logits": ranpac_logits,
            "mixed_logits": mixed_logits,
        }


def parse_eps_list(value):
    return [parse_float_or_fraction(item.strip()) for item in str(value).split(",") if item.strip()]


def build_eps_tag(eps_values):
    eps_pixels = [eps * 255.0 for eps in eps_values]
    return f"n{len(eps_values)}_min{sanitize_name(min(eps_pixels))}_max{sanitize_name(max(eps_pixels))}"


def pgd_linf_attack(model, inputs, targets, eps, steps, step_size, random_start, attack_class_ids=None):
    if eps <= 0 or steps <= 0:
        return inputs.detach()

    model.eval()
    x_orig = inputs.detach()
    if random_start:
        delta = torch.empty_like(x_orig).uniform_(-eps, eps)
        delta = torch.clamp(x_orig + delta, 0.0, 1.0) - x_orig
    else:
        delta = torch.zeros_like(x_orig)

    for _ in range(steps):
        adv_inputs = torch.clamp(x_orig + delta, 0.0, 1.0).detach().requires_grad_(True)
        logits = mask_logits_to_classes(model(adv_inputs), attack_class_ids)
        loss = F.cross_entropy(logits, targets, reduction="sum")
        grad = torch.autograd.grad(loss, adv_inputs, only_inputs=True)[0]
        delta = (delta + step_size * grad.sign()).detach()
        delta = torch.clamp(delta, -eps, eps)
        delta = torch.clamp(x_orig + delta, 0.0, 1.0) - x_orig

    return torch.clamp(x_orig + delta, 0.0, 1.0).detach()


def collect_outputs(model, probe, loader, selected_indices, device, args, variant_name, eps=None):
    arrays = {}
    label_batches = []
    index_batches = []
    prediction_batches = []
    input_l2_perturbation_batches = []
    cursor = 0
    is_adv = eps is not None
    desc = f"{variant_name}: {'PGD eps=' + format_eps(eps) if is_adv else 'clean'}"
    step_size = None if eps is None else args.pgd_step_size
    if eps is not None and step_size is None:
        step_size = 2.0 * eps / max(args.pgd_steps, 1)

    progress = tqdm(loader, desc=desc, dynamic_ncols=True)
    for inputs, labels in progress:
        batch_size = labels.size(0)
        batch_indices = selected_indices[cursor:cursor + batch_size]
        cursor += batch_size

        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if is_adv:
            eval_inputs = pgd_linf_attack(
                model,
                inputs,
                labels,
                eps=eps,
                steps=args.pgd_steps,
                step_size=step_size,
                random_start=args.pgd_random_start,
                attack_class_ids=getattr(args, "attack_class_ids", None),
            )
        else:
            eval_inputs = inputs
        input_l2_perturbation = (eval_inputs.detach() - inputs.detach()).view(batch_size, -1).norm(p=2, dim=1)

        batch_outputs = probe(model, eval_inputs)
        logits = batch_outputs["logits"]
        predictions = logits.argmax(dim=1)

        for key, value in batch_outputs.items():
            arrays.setdefault(key, []).append(value.cpu())
        label_batches.append(labels.detach().cpu())
        index_batches.append(torch.as_tensor(batch_indices, dtype=torch.long))
        prediction_batches.append(predictions.cpu())
        input_l2_perturbation_batches.append(input_l2_perturbation.detach().cpu())

        seen = sum(batch.numel() for batch in label_batches)
        correct = sum((pred == target).sum().item() for pred, target in zip(prediction_batches, label_batches))
        progress.set_postfix(acc=f"{correct / max(seen, 1):.3f}")

    stacked = {key: torch.cat(value, dim=0).numpy() for key, value in arrays.items()}
    return EvalOutputs(
        sample_indices=torch.cat(index_batches, dim=0).numpy(),
        labels=torch.cat(label_batches, dim=0).numpy(),
        predictions=torch.cat(prediction_batches, dim=0).numpy(),
        logits=stacked["logits"],
        penultimate=stacked["penultimate"],
        input_l2_perturbation=torch.cat(input_l2_perturbation_batches, dim=0).numpy(),
        projected=stacked.get("projected"),
        baseline_logits=stacked.get("baseline_logits"),
        ranpac_logits=stacked.get("ranpac_logits"),
        mixed_logits=stacked.get("mixed_logits"),
    )


def true_class_margin(logits, labels):
    logits = np.asarray(logits, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    true_logits = logits[np.arange(labels.shape[0]), labels]
    masked = logits.copy()
    masked[np.arange(labels.shape[0]), labels] = -np.inf
    return true_logits - masked.max(axis=1)


def top_logit_normalized_margin(logits, labels, eps=1e-12):
    logits = np.asarray(logits, dtype=np.float32)
    margins = true_class_margin(logits, labels)
    top_logits = logits.max(axis=1)
    return margins / np.maximum(np.abs(top_logits), float(eps))


def softmax_probability_margin(logits, labels):
    logits = np.asarray(logits, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    probabilities = exp_logits / np.maximum(exp_logits.sum(axis=1, keepdims=True), 1e-12)
    true_probs = probabilities[np.arange(labels.shape[0]), labels]
    masked = probabilities.copy()
    masked[np.arange(labels.shape[0]), labels] = -np.inf
    return true_probs - masked.max(axis=1)


def relative_l2_drift(clean_features, adv_features):
    clean_features = np.asarray(clean_features, dtype=np.float32)
    adv_features = np.asarray(adv_features, dtype=np.float32)
    numerator = np.linalg.norm(adv_features - clean_features, axis=1)
    denominator = np.maximum(np.linalg.norm(clean_features, axis=1), 1e-12)
    return numerator / denominator


def absolute_l2_drift(clean_features, adv_features):
    clean_features = np.asarray(clean_features, dtype=np.float32)
    adv_features = np.asarray(adv_features, dtype=np.float32)
    return np.linalg.norm(adv_features - clean_features, axis=1)


def linear_cka(clean_features, adv_features, max_dims=4096, seed=0, eps=1e-12):
    clean_features = np.asarray(clean_features, dtype=np.float32)
    adv_features = np.asarray(adv_features, dtype=np.float32)
    if clean_features.shape != adv_features.shape:
        raise ValueError("CKA requires clean and adversarial features with the same shape.")
    if clean_features.shape[0] <= 1 or clean_features.shape[1] == 0:
        return 0.0

    num_dims = clean_features.shape[1]
    if max_dims > 0 and num_dims > max_dims:
        rng = np.random.default_rng(seed)
        dim_indices = np.sort(rng.choice(num_dims, size=max_dims, replace=False))
        clean_features = clean_features[:, dim_indices]
        adv_features = adv_features[:, dim_indices]

    clean_features = clean_features.astype(np.float64, copy=False)
    adv_features = adv_features.astype(np.float64, copy=False)
    clean_features = clean_features - clean_features.mean(axis=0, keepdims=True)
    adv_features = adv_features - adv_features.mean(axis=0, keepdims=True)
    cross = clean_features.T @ adv_features
    clean_cov = clean_features.T @ clean_features
    adv_cov = adv_features.T @ adv_features
    numerator = np.sum(cross * cross)
    denominator = np.sqrt(np.sum(clean_cov * clean_cov) * np.sum(adv_cov * adv_cov))
    return float(numerator / max(denominator, eps))


def lipschitz_style_sensitivity(feature_drift, input_l2_perturbation):
    feature_drift = np.asarray(feature_drift, dtype=np.float32)
    input_l2_perturbation = np.asarray(input_l2_perturbation, dtype=np.float32)
    return feature_drift / np.maximum(input_l2_perturbation, 1e-12)


def average_feature_mutual_information(clean_features, adv_features, num_bins=20, max_dims=256, seed=0):
    clean_features = np.asarray(clean_features, dtype=np.float32)
    adv_features = np.asarray(adv_features, dtype=np.float32)
    if clean_features.shape != adv_features.shape:
        raise ValueError("MI requires clean and adversarial features with the same shape.")

    num_samples, num_dims = clean_features.shape
    if num_samples == 0 or num_dims == 0:
        return 0.0, 0.0
    if max_dims > 0 and num_dims > max_dims:
        rng = np.random.default_rng(seed)
        dim_indices = np.sort(rng.choice(num_dims, size=max_dims, replace=False))
    else:
        dim_indices = np.arange(num_dims)

    mi_values = []
    nmi_values = []
    for dim in dim_indices:
        clean_dim = clean_features[:, dim]
        adv_dim = adv_features[:, dim]
        if np.std(clean_dim) <= 1e-12 or np.std(adv_dim) <= 1e-12:
            continue
        hist_2d, _, _ = np.histogram2d(clean_dim, adv_dim, bins=num_bins)
        total = hist_2d.sum()
        if total <= 0:
            continue
        joint = hist_2d / total
        px = joint.sum(axis=1)
        py = joint.sum(axis=0)
        nonzero = joint > 0
        px_py = px[:, None] * py[None, :]
        mi = float(np.sum(joint[nonzero] * np.log(joint[nonzero] / np.maximum(px_py[nonzero], 1e-12))))
        hx = float(-np.sum(px[px > 0] * np.log(px[px > 0])))
        hy = float(-np.sum(py[py > 0] * np.log(py[py > 0])))
        mi_values.append(mi)
        nmi_values.append(mi / max(np.sqrt(hx * hy), 1e-12))

    if not mi_values:
        return 0.0, 0.0
    return float(np.mean(mi_values)), float(np.mean(nmi_values))


def normalize_rows(features):
    features = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, 1e-12)


def maybe_normalize(features, enabled):
    return normalize_rows(features) if enabled else np.asarray(features, dtype=np.float32)


def compute_centroids(features, labels, class_ids, normalize_features=True):
    features = maybe_normalize(features, normalize_features)
    labels = np.asarray(labels, dtype=np.int64)
    centroids = []
    centroid_labels = []
    for class_id in class_ids:
        mask = labels == class_id
        if not np.any(mask):
            continue
        centroids.append(features[mask].mean(axis=0))
        centroid_labels.append(class_id)
    if not centroids:
        raise ValueError("No class centroids could be computed.")
    centroids = np.stack(centroids, axis=0).astype(np.float32)
    if normalize_features:
        centroids = normalize_rows(centroids)
    return centroids, np.asarray(centroid_labels, dtype=np.int64)


def centroid_ratio(features, labels, centroids, centroid_labels, normalize_features=True):
    features = maybe_normalize(features, normalize_features)
    labels = np.asarray(labels, dtype=np.int64)
    distances = pairwise_squared_distances(features, centroids)
    true_positions = {int(class_id): index for index, class_id in enumerate(centroid_labels.tolist())}
    true_indices = np.asarray([true_positions[int(label)] for label in labels], dtype=np.int64)
    true_distances = distances[np.arange(labels.shape[0]), true_indices]
    wrong_distances = distances.copy()
    wrong_distances[np.arange(labels.shape[0]), true_indices] = np.inf
    nearest_wrong_distances = wrong_distances.min(axis=1)
    return np.sqrt(true_distances) / np.maximum(np.sqrt(nearest_wrong_distances), 1e-12)


def pairwise_squared_distances(left, right):
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    left_norm = np.sum(left * left, axis=1, keepdims=True)
    right_norm = np.sum(right * right, axis=1, keepdims=True).T
    return np.maximum(left_norm + right_norm - 2.0 * (left @ right.T), 0.0)


def knn_true_label_fraction(query_features, query_labels, bank_features, bank_labels, k, normalize_features=True, exclude_self=False, chunk_size=256):
    query_features = maybe_normalize(query_features, normalize_features)
    bank_features = maybe_normalize(bank_features, normalize_features)
    query_labels = np.asarray(query_labels, dtype=np.int64)
    bank_labels = np.asarray(bank_labels, dtype=np.int64)
    k = min(int(k), max(bank_features.shape[0] - (1 if exclude_self else 0), 1))
    bank_norm = np.sum(bank_features * bank_features, axis=1, keepdims=True).T
    fractions = []
    top1_matches = []

    for start in range(0, query_features.shape[0], chunk_size):
        end = min(start + chunk_size, query_features.shape[0])
        query = query_features[start:end]
        query_norm = np.sum(query * query, axis=1, keepdims=True)
        distances = np.maximum(query_norm + bank_norm - 2.0 * (query @ bank_features.T), 0.0)
        if exclude_self:
            rows = np.arange(end - start)
            distances[rows, np.arange(start, end)] = np.inf
        nearest = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
        nearest_distances = np.take_along_axis(distances, nearest, axis=1)
        nearest = np.take_along_axis(nearest, np.argsort(nearest_distances, axis=1), axis=1)
        nearest_labels = bank_labels[nearest]
        current_labels = query_labels[start:end, None]
        matches = nearest_labels == current_labels
        fractions.append(matches.mean(axis=1))
        top1_matches.append(matches[:, 0].astype(np.float32))

    return np.concatenate(fractions, axis=0), np.concatenate(top1_matches, axis=0)


def summarize_distribution(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
    }


def format_eps(eps):
    return f"{eps * 255.0:g}/255"


def eps_to_pixel(eps_values):
    return np.asarray(eps_values, dtype=np.float32) * 255.0


def build_metric_rows(eps_values, original_clean, ours_clean, original_by_eps, ours_by_eps, class_ids, args):
    original_centroids, original_centroid_labels = compute_centroids(
        original_clean.penultimate,
        original_clean.labels,
        class_ids,
        normalize_features=args.normalize_analysis_features,
    )
    ours_projected_centroids, ours_projected_centroid_labels = compute_centroids(
        ours_clean.projected,
        ours_clean.labels,
        class_ids,
        normalize_features=args.normalize_analysis_features,
    )

    rows = []
    per_sample_rows = []
    rescue_rows = []
    labels = original_clean.labels
    original_clean_margins = true_class_margin(original_clean.logits, labels)
    ours_clean_mixed_margins = true_class_margin(ours_clean.mixed_logits, labels)
    original_clean_norm_margins = top_logit_normalized_margin(original_clean.logits, labels)
    ours_clean_mixed_norm_margins = top_logit_normalized_margin(ours_clean.mixed_logits, labels)
    original_clean_prob_margins = softmax_probability_margin(original_clean.logits, labels)
    ours_clean_mixed_prob_margins = softmax_probability_margin(ours_clean.mixed_logits, labels)

    for eps in eps_values:
        original_adv = original_by_eps[eps]
        ours_adv = ours_by_eps[eps]
        original_adv_margins = true_class_margin(original_adv.logits, labels)
        ours_mixed_margins = true_class_margin(ours_adv.mixed_logits, labels)
        ours_baseline_margins = true_class_margin(ours_adv.baseline_logits, labels)
        ours_ranpac_margins = true_class_margin(ours_adv.ranpac_logits, labels)
        original_adv_norm_margins = top_logit_normalized_margin(original_adv.logits, labels)
        ours_mixed_norm_margins = top_logit_normalized_margin(ours_adv.mixed_logits, labels)
        ours_baseline_norm_margins = top_logit_normalized_margin(ours_adv.baseline_logits, labels)
        ours_ranpac_norm_margins = top_logit_normalized_margin(ours_adv.ranpac_logits, labels)
        original_adv_prob_margins = softmax_probability_margin(original_adv.logits, labels)
        ours_mixed_prob_margins = softmax_probability_margin(ours_adv.mixed_logits, labels)
        ours_baseline_prob_margins = softmax_probability_margin(ours_adv.baseline_logits, labels)
        ours_ranpac_prob_margins = softmax_probability_margin(ours_adv.ranpac_logits, labels)

        original_drift = relative_l2_drift(original_clean.penultimate, original_adv.penultimate)
        ours_penultimate_drift = relative_l2_drift(ours_clean.penultimate, ours_adv.penultimate)
        ours_projected_drift = relative_l2_drift(ours_clean.projected, ours_adv.projected)
        original_abs_drift = absolute_l2_drift(original_clean.penultimate, original_adv.penultimate)
        ours_penultimate_abs_drift = absolute_l2_drift(ours_clean.penultimate, ours_adv.penultimate)
        ours_projected_abs_drift = absolute_l2_drift(ours_clean.projected, ours_adv.projected)
        original_sensitivity = lipschitz_style_sensitivity(original_abs_drift, original_adv.input_l2_perturbation)
        ours_penultimate_sensitivity = lipschitz_style_sensitivity(ours_penultimate_abs_drift, ours_adv.input_l2_perturbation)
        ours_projected_sensitivity = lipschitz_style_sensitivity(ours_projected_abs_drift, ours_adv.input_l2_perturbation)
        original_cka = linear_cka(
            original_clean.penultimate,
            original_adv.penultimate,
            max_dims=args.cka_max_dims,
            seed=args.seed,
        )
        ours_penultimate_cka = linear_cka(
            ours_clean.penultimate,
            ours_adv.penultimate,
            max_dims=args.cka_max_dims,
            seed=args.seed,
        )
        ours_projected_cka = linear_cka(
            ours_clean.projected,
            ours_adv.projected,
            max_dims=args.cka_max_dims,
            seed=args.seed,
        )
        original_mi, original_nmi = average_feature_mutual_information(
            original_clean.penultimate,
            original_adv.penultimate,
            num_bins=args.mi_bins,
            max_dims=args.mi_max_dims,
            seed=args.seed,
        )
        ours_penultimate_mi, ours_penultimate_nmi = average_feature_mutual_information(
            ours_clean.penultimate,
            ours_adv.penultimate,
            num_bins=args.mi_bins,
            max_dims=args.mi_max_dims,
            seed=args.seed,
        )
        ours_projected_mi, ours_projected_nmi = average_feature_mutual_information(
            ours_clean.projected,
            ours_adv.projected,
            num_bins=args.mi_bins,
            max_dims=args.mi_max_dims,
            seed=args.seed,
        )

        original_centroid_ratio = centroid_ratio(
            original_adv.penultimate,
            labels,
            original_centroids,
            original_centroid_labels,
            normalize_features=args.normalize_analysis_features,
        )
        ours_projected_centroid_ratio = centroid_ratio(
            ours_adv.projected,
            labels,
            ours_projected_centroids,
            ours_projected_centroid_labels,
            normalize_features=args.normalize_analysis_features,
        )
        original_knn_frac, original_knn_top1 = knn_true_label_fraction(
            original_adv.penultimate,
            labels,
            original_clean.penultimate,
            labels,
            k=args.knn_k,
            normalize_features=args.normalize_analysis_features,
            exclude_self=eps <= 0,
        )
        ours_knn_frac, ours_knn_top1 = knn_true_label_fraction(
            ours_adv.projected,
            labels,
            ours_clean.projected,
            labels,
            k=args.knn_k,
            normalize_features=args.normalize_analysis_features,
            exclude_self=eps <= 0,
        )

        metric_row = {
            "eps": eps,
            "eps_pixel": eps * 255.0,
            "original_robust_acc": float(np.mean(original_adv.predictions == labels)),
            "hira_ranpac_robust_acc": float(np.mean(ours_adv.predictions == labels)),
            "original_margin_mean": summarize_distribution(original_adv_margins)["mean"],
            "original_margin_median": summarize_distribution(original_adv_margins)["median"],
            "hira_ranpac_mixed_margin_mean": summarize_distribution(ours_mixed_margins)["mean"],
            "hira_ranpac_mixed_margin_median": summarize_distribution(ours_mixed_margins)["median"],
            "hira_baseline_branch_margin_mean": summarize_distribution(ours_baseline_margins)["mean"],
            "ranpac_branch_margin_mean": summarize_distribution(ours_ranpac_margins)["mean"],
            "original_toplogit_norm_margin_mean": summarize_distribution(original_adv_norm_margins)["mean"],
            "original_toplogit_norm_margin_median": summarize_distribution(original_adv_norm_margins)["median"],
            "hira_ranpac_mixed_toplogit_norm_margin_mean": summarize_distribution(ours_mixed_norm_margins)["mean"],
            "hira_ranpac_mixed_toplogit_norm_margin_median": summarize_distribution(ours_mixed_norm_margins)["median"],
            "hira_baseline_branch_toplogit_norm_margin_mean": summarize_distribution(ours_baseline_norm_margins)["mean"],
            "ranpac_branch_toplogit_norm_margin_mean": summarize_distribution(ours_ranpac_norm_margins)["mean"],
            "original_prob_margin_mean": summarize_distribution(original_adv_prob_margins)["mean"],
            "original_prob_margin_median": summarize_distribution(original_adv_prob_margins)["median"],
            "hira_ranpac_mixed_prob_margin_mean": summarize_distribution(ours_mixed_prob_margins)["mean"],
            "hira_ranpac_mixed_prob_margin_median": summarize_distribution(ours_mixed_prob_margins)["median"],
            "hira_baseline_branch_prob_margin_mean": summarize_distribution(ours_baseline_prob_margins)["mean"],
            "ranpac_branch_prob_margin_mean": summarize_distribution(ours_ranpac_prob_margins)["mean"],
            "original_feature_drift_mean": summarize_distribution(original_drift)["mean"],
            "hira_penultimate_drift_mean": summarize_distribution(ours_penultimate_drift)["mean"],
            "ranpac_projected_drift_mean": summarize_distribution(ours_projected_drift)["mean"],
            "original_absolute_feature_drift_mean": summarize_distribution(original_abs_drift)["mean"],
            "hira_penultimate_absolute_drift_mean": summarize_distribution(ours_penultimate_abs_drift)["mean"],
            "ranpac_projected_absolute_drift_mean": summarize_distribution(ours_projected_abs_drift)["mean"],
            "original_lipschitz_sensitivity_mean": summarize_distribution(original_sensitivity)["mean"],
            "hira_penultimate_lipschitz_sensitivity_mean": summarize_distribution(ours_penultimate_sensitivity)["mean"],
            "ranpac_projected_lipschitz_sensitivity_mean": summarize_distribution(ours_projected_sensitivity)["mean"],
            "original_penultimate_cka_mean": original_cka,
            "hira_penultimate_cka_mean": ours_penultimate_cka,
            "ranpac_projected_cka_mean": ours_projected_cka,
            "original_penultimate_mi_mean": original_mi,
            "hira_penultimate_mi_mean": ours_penultimate_mi,
            "ranpac_projected_mi_mean": ours_projected_mi,
            "original_penultimate_nmi_mean": original_nmi,
            "hira_penultimate_nmi_mean": ours_penultimate_nmi,
            "ranpac_projected_nmi_mean": ours_projected_nmi,
            "original_centroid_ratio_mean": summarize_distribution(original_centroid_ratio)["mean"],
            "ranpac_projected_centroid_ratio_mean": summarize_distribution(ours_projected_centroid_ratio)["mean"],
            f"original_knn_true_label_frac_at_{args.knn_k}": summarize_distribution(original_knn_frac)["mean"],
            f"ranpac_projected_knn_true_label_frac_at_{args.knn_k}": summarize_distribution(ours_knn_frac)["mean"],
            f"original_knn_top1_match_at_{args.knn_k}": summarize_distribution(original_knn_top1)["mean"],
            f"ranpac_projected_knn_top1_match_at_{args.knn_k}": summarize_distribution(ours_knn_top1)["mean"],
        }

        for prefix, values in (
            ("original_margin", original_adv_margins),
            ("hira_ranpac_mixed_margin", ours_mixed_margins),
            ("hira_baseline_branch_margin", ours_baseline_margins),
            ("ranpac_branch_margin", ours_ranpac_margins),
            ("original_toplogit_norm_margin", original_adv_norm_margins),
            ("hira_ranpac_mixed_toplogit_norm_margin", ours_mixed_norm_margins),
            ("hira_baseline_branch_toplogit_norm_margin", ours_baseline_norm_margins),
            ("ranpac_branch_toplogit_norm_margin", ours_ranpac_norm_margins),
            ("original_prob_margin", original_adv_prob_margins),
            ("hira_ranpac_mixed_prob_margin", ours_mixed_prob_margins),
            ("hira_baseline_branch_prob_margin", ours_baseline_prob_margins),
            ("ranpac_branch_prob_margin", ours_ranpac_prob_margins),
            ("original_feature_drift", original_drift),
            ("hira_penultimate_drift", ours_penultimate_drift),
            ("ranpac_projected_drift", ours_projected_drift),
            ("original_absolute_feature_drift", original_abs_drift),
            ("hira_penultimate_absolute_drift", ours_penultimate_abs_drift),
            ("ranpac_projected_absolute_drift", ours_projected_abs_drift),
            ("original_lipschitz_sensitivity", original_sensitivity),
            ("hira_penultimate_lipschitz_sensitivity", ours_penultimate_sensitivity),
            ("ranpac_projected_lipschitz_sensitivity", ours_projected_sensitivity),
            ("original_centroid_ratio", original_centroid_ratio),
            ("ranpac_projected_centroid_ratio", ours_projected_centroid_ratio),
            (f"original_knn_true_label_frac_at_{args.knn_k}", original_knn_frac),
            (f"ranpac_projected_knn_true_label_frac_at_{args.knn_k}", ours_knn_frac),
        ):
            stats = summarize_distribution(values)
            metric_row[f"{prefix}_p25"] = stats["p25"]
            metric_row[f"{prefix}_p75"] = stats["p75"]

        rows.append(metric_row)

        rescue_mask = (original_adv.predictions != labels) & (ours_adv.predictions == labels)
        for index in range(labels.shape[0]):
            row = {
                "eps": eps,
                "eps_pixel": eps * 255.0,
                "sample_index": int(original_clean.sample_indices[index]),
                "label": int(labels[index]),
                "original_prediction": int(original_adv.predictions[index]),
                "hira_ranpac_prediction": int(ours_adv.predictions[index]),
                "original_clean_margin": float(original_clean_margins[index]),
                "hira_ranpac_clean_mixed_margin": float(ours_clean_mixed_margins[index]),
                "original_clean_toplogit_norm_margin": float(original_clean_norm_margins[index]),
                "hira_ranpac_clean_mixed_toplogit_norm_margin": float(ours_clean_mixed_norm_margins[index]),
                "original_clean_prob_margin": float(original_clean_prob_margins[index]),
                "hira_ranpac_clean_mixed_prob_margin": float(ours_clean_mixed_prob_margins[index]),
                "original_adv_margin": float(original_adv_margins[index]),
                "hira_ranpac_adv_mixed_margin": float(ours_mixed_margins[index]),
                "hira_baseline_branch_adv_margin": float(ours_baseline_margins[index]),
                "ranpac_branch_adv_margin": float(ours_ranpac_margins[index]),
                "margin_delta_hira_ranpac_minus_original": float(ours_mixed_margins[index] - original_adv_margins[index]),
                "original_adv_toplogit_norm_margin": float(original_adv_norm_margins[index]),
                "hira_ranpac_adv_mixed_toplogit_norm_margin": float(ours_mixed_norm_margins[index]),
                "hira_baseline_branch_adv_toplogit_norm_margin": float(ours_baseline_norm_margins[index]),
                "ranpac_branch_adv_toplogit_norm_margin": float(ours_ranpac_norm_margins[index]),
                "toplogit_norm_margin_delta_hira_ranpac_minus_original": float(ours_mixed_norm_margins[index] - original_adv_norm_margins[index]),
                "original_adv_prob_margin": float(original_adv_prob_margins[index]),
                "hira_ranpac_adv_mixed_prob_margin": float(ours_mixed_prob_margins[index]),
                "hira_baseline_branch_adv_prob_margin": float(ours_baseline_prob_margins[index]),
                "ranpac_branch_adv_prob_margin": float(ours_ranpac_prob_margins[index]),
                "prob_margin_delta_hira_ranpac_minus_original": float(ours_mixed_prob_margins[index] - original_adv_prob_margins[index]),
                "original_feature_drift": float(original_drift[index]),
                "hira_penultimate_drift": float(ours_penultimate_drift[index]),
                "ranpac_projected_drift": float(ours_projected_drift[index]),
                "original_absolute_feature_drift": float(original_abs_drift[index]),
                "hira_penultimate_absolute_drift": float(ours_penultimate_abs_drift[index]),
                "ranpac_projected_absolute_drift": float(ours_projected_abs_drift[index]),
                "original_input_l2_perturbation": float(original_adv.input_l2_perturbation[index]),
                "hira_ranpac_input_l2_perturbation": float(ours_adv.input_l2_perturbation[index]),
                "original_lipschitz_sensitivity": float(original_sensitivity[index]),
                "hira_penultimate_lipschitz_sensitivity": float(ours_penultimate_sensitivity[index]),
                "ranpac_projected_lipschitz_sensitivity": float(ours_projected_sensitivity[index]),
                "original_centroid_ratio": float(original_centroid_ratio[index]),
                "ranpac_projected_centroid_ratio": float(ours_projected_centroid_ratio[index]),
                f"original_knn_true_label_frac_at_{args.knn_k}": float(original_knn_frac[index]),
                f"ranpac_projected_knn_true_label_frac_at_{args.knn_k}": float(ours_knn_frac[index]),
                "is_rescue_case": int(rescue_mask[index]),
            }
            per_sample_rows.append(row)
            if rescue_mask[index]:
                rescue_rows.append(row)

    return rows, per_sample_rows, rescue_rows


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def get_series(rows, key):
    return np.asarray([row[key] for row in rows], dtype=np.float64)


def plot_line_with_iqr(ax, rows, mean_key, p25_key=None, p75_key=None, label=None, marker="o"):
    x = get_series(rows, "eps_pixel")
    y = get_series(rows, mean_key)
    ax.plot(x, y, marker=marker, label=label or mean_key)
    if p25_key is not None and p75_key is not None:
        lower = get_series(rows, p25_key)
        upper = get_series(rows, p75_key)
        ax.fill_between(x, lower, upper, alpha=0.15)


def save_plots(run_dir, rows, rescue_rows, args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir.mkdir(parents=True, exist_ok=True)
    x = get_series(rows, "eps_pixel")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, get_series(rows, "original_robust_acc"), marker="o", label="original")
    ax.plot(x, get_series(rows, "hira_ranpac_robust_acc"), marker="o", label="HiRA+RanPAC")
    ax.set_xlabel("PGD epsilon (pixel / 255)")
    ax.set_ylabel("Robust accuracy")
    ax.set_title("Robust accuracy vs PGD epsilon")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "robust_accuracy_vs_eps.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_line_with_iqr(ax, rows, "original_margin_mean", "original_margin_p25", "original_margin_p75", label="original")
    plot_line_with_iqr(ax, rows, "hira_ranpac_mixed_margin_mean", "hira_ranpac_mixed_margin_p25", "hira_ranpac_mixed_margin_p75", label="HiRA+RanPAC mixed")
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_xlabel("PGD epsilon (pixel / 255)")
    ax.set_ylabel("True-class margin")
    ax.set_title("Margin degradation vs PGD epsilon")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "margin_vs_eps.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_line_with_iqr(
        ax,
        rows,
        "original_toplogit_norm_margin_mean",
        "original_toplogit_norm_margin_p25",
        "original_toplogit_norm_margin_p75",
        label="original",
    )
    plot_line_with_iqr(
        ax,
        rows,
        "hira_ranpac_mixed_toplogit_norm_margin_mean",
        "hira_ranpac_mixed_toplogit_norm_margin_p25",
        "hira_ranpac_mixed_toplogit_norm_margin_p75",
        label="HiRA+RanPAC mixed",
    )
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_xlabel("PGD epsilon (pixel / 255)")
    ax.set_ylabel("Top-logit normalized margin")
    ax.set_title("Scale-normalized margin vs PGD epsilon")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "toplogit_norm_margin_vs_eps.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_line_with_iqr(ax, rows, "original_prob_margin_mean", "original_prob_margin_p25", "original_prob_margin_p75", label="original")
    plot_line_with_iqr(ax, rows, "hira_ranpac_mixed_prob_margin_mean", "hira_ranpac_mixed_prob_margin_p25", "hira_ranpac_mixed_prob_margin_p75", label="HiRA+RanPAC mixed")
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_xlabel("PGD epsilon (pixel / 255)")
    ax.set_ylabel("Softmax probability margin")
    ax.set_title("Probability margin vs PGD epsilon")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "prob_margin_vs_eps.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_line_with_iqr(ax, rows, "hira_baseline_branch_margin_mean", "hira_baseline_branch_margin_p25", "hira_baseline_branch_margin_p75", label="HiRA baseline branch")
    plot_line_with_iqr(ax, rows, "ranpac_branch_margin_mean", "ranpac_branch_margin_p25", "ranpac_branch_margin_p75", label="RanPAC branch")
    plot_line_with_iqr(ax, rows, "hira_ranpac_mixed_margin_mean", "hira_ranpac_mixed_margin_p25", "hira_ranpac_mixed_margin_p75", label="mixed logits")
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_xlabel("PGD epsilon (pixel / 255)")
    ax.set_ylabel("True-class margin")
    ax.set_title("HiRA+RanPAC margin decomposition")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "ranpac_margin_decomposition_vs_eps.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_line_with_iqr(
        ax,
        rows,
        "hira_baseline_branch_toplogit_norm_margin_mean",
        "hira_baseline_branch_toplogit_norm_margin_p25",
        "hira_baseline_branch_toplogit_norm_margin_p75",
        label="HiRA baseline branch",
    )
    plot_line_with_iqr(
        ax,
        rows,
        "ranpac_branch_toplogit_norm_margin_mean",
        "ranpac_branch_toplogit_norm_margin_p25",
        "ranpac_branch_toplogit_norm_margin_p75",
        label="RanPAC branch",
    )
    plot_line_with_iqr(
        ax,
        rows,
        "hira_ranpac_mixed_toplogit_norm_margin_mean",
        "hira_ranpac_mixed_toplogit_norm_margin_p25",
        "hira_ranpac_mixed_toplogit_norm_margin_p75",
        label="mixed logits",
    )
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_xlabel("PGD epsilon (pixel / 255)")
    ax.set_ylabel("Top-logit normalized margin")
    ax.set_title("Scale-normalized HiRA+RanPAC margin decomposition")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "toplogit_norm_ranpac_margin_decomposition_vs_eps.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_line_with_iqr(ax, rows, "hira_baseline_branch_prob_margin_mean", "hira_baseline_branch_prob_margin_p25", "hira_baseline_branch_prob_margin_p75", label="HiRA baseline branch")
    plot_line_with_iqr(ax, rows, "ranpac_branch_prob_margin_mean", "ranpac_branch_prob_margin_p25", "ranpac_branch_prob_margin_p75", label="RanPAC branch")
    plot_line_with_iqr(ax, rows, "hira_ranpac_mixed_prob_margin_mean", "hira_ranpac_mixed_prob_margin_p25", "hira_ranpac_mixed_prob_margin_p75", label="mixed logits")
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_xlabel("PGD epsilon (pixel / 255)")
    ax.set_ylabel("Softmax probability margin")
    ax.set_title("HiRA+RanPAC probability margin decomposition")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "prob_ranpac_margin_decomposition_vs_eps.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_line_with_iqr(ax, rows, "original_feature_drift_mean", "original_feature_drift_p25", "original_feature_drift_p75", label="original penultimate")
    plot_line_with_iqr(ax, rows, "hira_penultimate_drift_mean", "hira_penultimate_drift_p25", "hira_penultimate_drift_p75", label="HiRA penultimate")
    plot_line_with_iqr(ax, rows, "ranpac_projected_drift_mean", "ranpac_projected_drift_p25", "ranpac_projected_drift_p75", label="RanPAC projected")
    ax.set_xlabel("PGD epsilon (pixel / 255)")
    ax.set_ylabel("Relative feature drift")
    ax.set_title("Feature drift vs PGD epsilon")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "feature_drift_vs_eps.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_line_with_iqr(
        ax,
        rows,
        "original_absolute_feature_drift_mean",
        "original_absolute_feature_drift_p25",
        "original_absolute_feature_drift_p75",
        label="original penultimate",
    )
    plot_line_with_iqr(
        ax,
        rows,
        "hira_penultimate_absolute_drift_mean",
        "hira_penultimate_absolute_drift_p25",
        "hira_penultimate_absolute_drift_p75",
        label="HiRA penultimate",
    )
    plot_line_with_iqr(
        ax,
        rows,
        "ranpac_projected_absolute_drift_mean",
        "ranpac_projected_absolute_drift_p25",
        "ranpac_projected_absolute_drift_p75",
        label="RanPAC projected",
    )
    ax.set_xlabel("PGD epsilon (pixel / 255)")
    ax.set_ylabel("Absolute L2 feature drift")
    ax.set_title("Absolute feature drift vs PGD epsilon")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "absolute_feature_drift_vs_eps.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_line_with_iqr(
        ax,
        rows,
        "original_lipschitz_sensitivity_mean",
        "original_lipschitz_sensitivity_p25",
        "original_lipschitz_sensitivity_p75",
        label="original penultimate",
    )
    plot_line_with_iqr(
        ax,
        rows,
        "hira_penultimate_lipschitz_sensitivity_mean",
        "hira_penultimate_lipschitz_sensitivity_p25",
        "hira_penultimate_lipschitz_sensitivity_p75",
        label="HiRA penultimate",
    )
    plot_line_with_iqr(
        ax,
        rows,
        "ranpac_projected_lipschitz_sensitivity_mean",
        "ranpac_projected_lipschitz_sensitivity_p25",
        "ranpac_projected_lipschitz_sensitivity_p75",
        label="RanPAC projected",
    )
    ax.set_xlabel("PGD epsilon (pixel / 255)")
    ax.set_ylabel("Feature drift / input L2 perturbation")
    ax.set_title("Lipschitz-style feature sensitivity vs PGD epsilon")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "lipschitz_sensitivity_vs_eps.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, get_series(rows, "original_penultimate_cka_mean"), marker="o", label="original penultimate")
    ax.plot(x, get_series(rows, "hira_penultimate_cka_mean"), marker="o", label="HiRA penultimate")
    ax.plot(x, get_series(rows, "ranpac_projected_cka_mean"), marker="o", label="RanPAC projected")
    ax.set_xlabel("PGD epsilon (pixel / 255)")
    ax.set_ylabel("Linear CKA(clean features, attacked features)")
    ax.set_title("Feature CKA similarity vs PGD epsilon")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "feature_cka_vs_eps.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, get_series(rows, "original_penultimate_mi_mean"), marker="o", label="original penultimate")
    ax.plot(x, get_series(rows, "hira_penultimate_mi_mean"), marker="o", label="HiRA penultimate")
    ax.plot(x, get_series(rows, "ranpac_projected_mi_mean"), marker="o", label="RanPAC projected")
    ax.set_xlabel("PGD epsilon (pixel / 255)")
    ax.set_ylabel("Average clean/adversarial feature MI")
    ax.set_title("Feature mutual information vs PGD epsilon")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "feature_mi_vs_noise.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, get_series(rows, "original_penultimate_nmi_mean"), marker="o", label="original penultimate")
    ax.plot(x, get_series(rows, "hira_penultimate_nmi_mean"), marker="o", label="HiRA penultimate")
    ax.plot(x, get_series(rows, "ranpac_projected_nmi_mean"), marker="o", label="RanPAC projected")
    ax.set_xlabel("PGD epsilon (pixel / 255)")
    ax.set_ylabel("Average clean/adversarial feature normalized MI")
    ax.set_title("Normalized feature mutual information vs PGD epsilon")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "feature_nmi_vs_noise.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_line_with_iqr(ax, rows, "original_centroid_ratio_mean", "original_centroid_ratio_p25", "original_centroid_ratio_p75", label="original penultimate")
    plot_line_with_iqr(ax, rows, "ranpac_projected_centroid_ratio_mean", "ranpac_projected_centroid_ratio_p25", "ranpac_projected_centroid_ratio_p75", label="RanPAC projected")
    ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_xlabel("PGD epsilon (pixel / 255)")
    ax.set_ylabel("dist(true centroid) / dist(nearest wrong centroid)")
    ax.set_title("Class-centroid stability vs PGD epsilon")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "centroid_ratio_vs_eps.png", dpi=300)
    plt.close(fig)

    original_knn_key = f"original_knn_true_label_frac_at_{args.knn_k}"
    ranpac_knn_key = f"ranpac_projected_knn_true_label_frac_at_{args.knn_k}"
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_line_with_iqr(ax, rows, original_knn_key, f"{original_knn_key}_p25", f"{original_knn_key}_p75", label="original penultimate")
    plot_line_with_iqr(ax, rows, ranpac_knn_key, f"{ranpac_knn_key}_p25", f"{ranpac_knn_key}_p75", label="RanPAC projected")
    ax.set_xlabel("PGD epsilon (pixel / 255)")
    ax.set_ylabel(f"True-label fraction among {args.knn_k} clean nearest neighbors")
    ax.set_title("kNN label-neighborhood preservation")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "knn_preservation_vs_eps.png", dpi=300)
    plt.close(fig)

    max_eps = max(row["eps"] for row in rows)
    max_eps_rescue = [row for row in rescue_rows if row["eps"] == max_eps]
    fig, ax = plt.subplots(figsize=(7, 5))
    if max_eps_rescue:
        values = [row["margin_delta_hira_ranpac_minus_original"] for row in max_eps_rescue]
        ax.hist(values, bins=30, alpha=0.85)
        ax.axvline(0.0, color="black", linewidth=1.0, linestyle="--")
        ax.set_title(f"Rescue-case margin gain at eps={format_eps(max_eps)}")
        ax.set_xlabel("HiRA+RanPAC mixed margin - original margin")
        ax.set_ylabel("Rescue-case count")
    else:
        ax.text(0.5, 0.5, f"No rescue cases at eps={format_eps(max_eps)}", ha="center", va="center")
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(run_dir / "rescue_margin_delta_hist_maxeps.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    if max_eps_rescue:
        values = [row["toplogit_norm_margin_delta_hira_ranpac_minus_original"] for row in max_eps_rescue]
        ax.hist(values, bins=30, alpha=0.85)
        ax.axvline(0.0, color="black", linewidth=1.0, linestyle="--")
        ax.set_title(f"Rescue-case normalized margin gain at eps={format_eps(max_eps)}")
        ax.set_xlabel("HiRA+RanPAC normalized margin - original normalized margin")
        ax.set_ylabel("Rescue-case count")
    else:
        ax.text(0.5, 0.5, f"No rescue cases at eps={format_eps(max_eps)}", ha="center", va="center")
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(run_dir / "rescue_toplogit_norm_margin_delta_hist_maxeps.png", dpi=300)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot explanatory RobustBench HiRA+RanPAC diagnostics.")
    parser.add_argument("--model-name", "--model_name", required=True, help="RobustBench ImageNet model name.")
    parser.add_argument("--threat-model", "--threat_model", default="Linf", choices=["Linf", "L2"], help="Threat model used to load the RobustBench model.")
    parser.add_argument("--data-dir", "--data_dir", default="./dataset/imagenet", help="ImageNet root containing train/ and val/.")
    parser.add_argument("--model-dir", "--model_dir", default="./robustbench_models", help="RobustBench checkpoint cache directory.")
    parser.add_argument("--device", default="cuda:0", help="Device, e.g. cuda:0 or cpu.")
    parser.add_argument("--seed", type=int, default=0, help="Global random seed and random class-selection seed.")
    parser.add_argument("--batch-size", "--batch_size", type=int, default=32, help="Batch size for feature collection and PGD.")
    parser.add_argument("--num-workers", "--num_workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--output-dir", "--output_dir", default="visualization/explanation_outputs", help="Directory where figures and metrics are saved.")
    parser.add_argument("--run-name", "--run_name", default="", help="Optional output subdirectory name.")

    parser.add_argument("--eval-examples", "--eval_examples", type=int, default=5000, help="Number of RobustBench ImageNet evaluation examples to load.")
    parser.add_argument("--num-classes", "--num_classes", type=int, default=20, help="Deprecated; visualization now uses the RobustBench evaluation subset.")
    parser.add_argument("--samples-per-class", "--samples_per_class", type=int, default=50, help="Deprecated; visualization now uses --eval-examples.")
    parser.add_argument("--class-ids", "--class_ids", default="", help="Deprecated; visualization now uses the loaded RobustBench classes.")

    parser.add_argument("--eps-list", "--eps_list", default="0,1/255,2/255,4/255,8/255,16/255", help="Comma-separated Linf PGD eps values; accepts fractions like 4/255.")
    parser.add_argument("--pgd-steps", "--pgd_steps", type=int, default=40, help="PGD steps for nonzero eps.")
    parser.add_argument("--pgd-step-size", "--pgd_step_size", type=parse_float_or_fraction, default=None, help="Optional PGD step size. Defaults to 2 * eps / steps for each eps.")
    parser.add_argument("--pgd-random-start", "--pgd_random_start", type=str2bool, default=True, help="Use random-start Linf PGD.")
    parser.add_argument("--mask-pgd-logits", "--mask_pgd_logits", type=str2bool, default=False, help="During PGD, set logits outside loaded RobustBench subset classes to -inf. Defaults to false for full-class attacks.")
    parser.add_argument("--normalize-analysis-features", "--normalize_analysis_features", type=str2bool, default=True, help="L2-normalize features for centroid and kNN analyses.")
    parser.add_argument("--knn-k", "--knn_k", type=int, default=10, help="k used by clean-feature kNN preservation analysis.")
    parser.add_argument("--mi-bins", "--mi_bins", type=int, default=20, help="Number of bins per dimension for clean/adversarial feature mutual information.")
    parser.add_argument("--mi-max-dims", "--mi_max_dims", type=int, default=256, help="Maximum feature dimensions sampled for MI; <=0 uses all dimensions.")
    parser.add_argument("--cka-max-dims", "--cka_max_dims", type=int, default=4096, help="Maximum feature dimensions sampled for linear CKA; <=0 uses all dimensions.")

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
    parser.add_argument("--hira-subspace-rank", "--hira_subspace_rank", type=int, default=0, help="Rank of the HiRA clean hidden subspace kept unshrunk; 0 disables subspace shrinking.")
    parser.add_argument("--hira-subspace-shrink", "--hira_subspace_shrink", type=float, default=1.0, help="Shrink factor for HiRA hidden components orthogonal to the clean subspace.")
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
    if args.threat_model != "Linf":
        raise NotImplementedError("This explanatory script currently implements Linf PGD only.")
    eps_values = parse_eps_list(args.eps_list)
    set_seed(args.seed)
    device = resolve_device(args.device)
    model_preprocessing = resolve_model_preprocessing(args.model_name, args.threat_model)
    dataset = build_imagenet_dataset(args.data_dir, model_preprocessing, n_examples=args.eval_examples)
    selected_indices, selected_class_ids = select_all_indices(dataset)
    args.attack_class_ids = selected_class_ids if args.mask_pgd_logits else None
    loader = build_eval_loader(dataset, selected_indices, args.batch_size, args.num_workers)

    run_name = args.run_name
    if not run_name:
        eps_tag = build_eps_tag(eps_values)
        run_name = f"{sanitize_name(args.model_name)}_examples{len(selected_indices)}_eps{eps_tag}_seed{args.seed}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving explanatory outputs to: {run_dir}")
    print(f"Loaded RobustBench examples: {len(selected_indices)}")
    print(f"Loaded classes: {selected_class_ids}")
    if args.attack_class_ids is not None:
        print("PGD logit mask enabled: attacking only among loaded RobustBench subset classes.")
    print("Loading original RobustBench model...")
    original_model = freeze_model(load_robustbench_model(args.model_name, args.threat_model, args.model_dir, device))
    original_probe = OriginalProbe(original_model)
    print(f"Original feature layer: {original_probe.module_name}")
    original_clean = collect_outputs(original_model, original_probe, loader, selected_indices, device, args, VARIANT_ORIGINAL)
    original_by_eps = {}
    for eps in eps_values:
        original_by_eps[eps] = collect_outputs(
            original_model,
            original_probe,
            loader,
            selected_indices,
            device,
            args,
            VARIANT_ORIGINAL,
            eps=eps,
        )
    del original_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("Loading and wrapping RobustBench model with HiRA+RanPAC...")
    ours_model = freeze_model(build_hira_ranpac_model(args, model_preprocessing, device))
    ours_probe = HiraRanPACProbe(ours_model)
    print(f"HiRA+RanPAC feature layer: {ours_probe.module_name}")
    ours_clean = collect_outputs(ours_model, ours_probe, loader, selected_indices, device, args, VARIANT_HIRA_RANPAC)
    ours_by_eps = {}
    for eps in eps_values:
        ours_by_eps[eps] = collect_outputs(
            ours_model,
            ours_probe,
            loader,
            selected_indices,
            device,
            args,
            VARIANT_HIRA_RANPAC,
            eps=eps,
        )
    del ours_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    rows, per_sample_rows, rescue_rows = build_metric_rows(
        eps_values,
        original_clean,
        ours_clean,
        original_by_eps,
        ours_by_eps,
        selected_class_ids,
        args,
    )
    write_csv(run_dir / "aggregate_metrics.csv", rows)
    write_csv(run_dir / "per_sample_metrics.csv", per_sample_rows)
    write_csv(run_dir / "rescue_cases.csv", rescue_rows)
    save_plots(run_dir, rows, rescue_rows, args)

    summary = {
        "model_name": args.model_name,
        "dataset": DATASET,
        "threat_model": args.threat_model,
        "eval_examples": args.eval_examples,
        "selected_class_ids": selected_class_ids,
        "num_samples": len(selected_indices),
        "num_classes": len(selected_class_ids),
        "eps_values": eps_values,
        "eps_pixels": eps_to_pixel(eps_values).tolist(),
        "pgd_steps": args.pgd_steps,
        "pgd_step_size": args.pgd_step_size,
        "pgd_random_start": args.pgd_random_start,
        "mask_pgd_logits": args.mask_pgd_logits,
        "knn_k": args.knn_k,
        "mi_bins": args.mi_bins,
        "mi_max_dims": args.mi_max_dims,
        "cka_max_dims": args.cka_max_dims,
        "normalize_analysis_features": args.normalize_analysis_features,
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
        "hira_subspace_rank": args.hira_subspace_rank,
        "hira_subspace_shrink": args.hira_subspace_shrink,
        "outputs": [
            "robust_accuracy_vs_eps.png",
            "margin_vs_eps.png",
            "toplogit_norm_margin_vs_eps.png",
            "prob_margin_vs_eps.png",
            "ranpac_margin_decomposition_vs_eps.png",
            "toplogit_norm_ranpac_margin_decomposition_vs_eps.png",
            "prob_ranpac_margin_decomposition_vs_eps.png",
            "feature_drift_vs_eps.png",
            "absolute_feature_drift_vs_eps.png",
            "lipschitz_sensitivity_vs_eps.png",
            "feature_cka_vs_eps.png",
            "feature_mi_vs_noise.png",
            "feature_nmi_vs_noise.png",
            "centroid_ratio_vs_eps.png",
            "knn_preservation_vs_eps.png",
            "rescue_margin_delta_hist_maxeps.png",
            "rescue_toplogit_norm_margin_delta_hist_maxeps.png",
            "aggregate_metrics.csv",
            "per_sample_metrics.csv",
            "rescue_cases.csv",
        ],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
