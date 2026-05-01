#!/usr/bin/env python3
"""
Counterfactual HiRA subspace intervention for RobustBench ImageNet models.

For each PGD adversarial image, this script runs the HiRA+RanPAC model while
replacing all HiRA hidden projected activations with counterfactual mixtures of
clean and adversarial principal/orthogonal components:
  - normal_adv: no intervention
  - fix_parallel: clean principal component + adversarial orthogonal component
  - fix_orthogonal: adversarial principal component + clean orthogonal component
  - fix_both: clean hidden projected activation

The principal subspace is the clean HiRA subspace stored in each HiRA adapter.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from classifiers.hira import HiRAAdapter
from classifiers.mean_sparse import apply_mean_centered_soft_threshold, DEFAULT_MEANSPARSE_STAT_EPS
from classifiers.stability_ridge import DEFAULT_STABILITY_RIDGE_STAT_EPS
from visualization.tsne_robustbench_ranpac import (
    DATASET,
    build_eval_loader,
    build_hira_ranpac_model,
    build_imagenet_dataset,
    freeze_model,
    mask_logits_to_classes,
    parse_class_ids,
    parse_float_or_fraction,
    resolve_device,
    resolve_model_preprocessing,
    sanitize_name,
    select_balanced_indices,
    set_seed,
    str2bool,
)


MODE_NORMAL_ADV = "normal_adv"
MODE_FIX_PARALLEL = "fix_parallel"
MODE_FIX_ORTHOGONAL = "fix_orthogonal"
MODE_FIX_BOTH = "fix_both"
INTERVENTION_MODES = (
    MODE_NORMAL_ADV,
    MODE_FIX_PARALLEL,
    MODE_FIX_ORTHOGONAL,
    MODE_FIX_BOTH,
)


def parse_eps_list(value):
    return [parse_float_or_fraction(item.strip()) for item in str(value).split(",") if item.strip()]


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
        delta = (delta + step_size * grad.sign()).detach().clamp(-eps, eps)
        delta = torch.clamp(x_orig + delta, 0.0, 1.0) - x_orig

    return torch.clamp(x_orig + delta, 0.0, 1.0).detach()


def top_logit_normalized_margin(logits, labels, eps=1e-12):
    true_logits = logits.gather(1, labels.view(-1, 1)).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, labels.view(-1, 1), float("-inf"))
    max_wrong = masked.max(dim=1).values
    top_logits = logits.max(dim=1).values.abs().clamp_min(float(eps))
    return (true_logits - max_wrong) / top_logits


def softmax_probability_margin(logits, labels):
    probabilities = F.softmax(logits.float(), dim=1)
    true_probs = probabilities.gather(1, labels.view(-1, 1)).squeeze(1)
    masked = probabilities.clone()
    masked.scatter_(1, labels.view(-1, 1), float("-inf"))
    return true_probs - masked.max(dim=1).values


def summarize_distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "p25": float("nan"), "p75": float("nan")}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
    }


def find_hira_adapters(model):
    adapters = [(name, module) for name, module in model.named_modules() if isinstance(module, HiRAAdapter)]
    if not adapters:
        raise RuntimeError("No HiRAAdapter modules found. Did HiRA wrapping run successfully?")
    return adapters


def adapter_project_without_clean_subspace(adapter, x):
    token_features = x.reshape(-1, x.shape[-1])
    if getattr(adapter, "force_fp32", False):
        token_features = token_features.float()
        if token_features.device.type == "cuda":
            adapter._ensure_fp32_cache(token_features.device)
            projected = token_features @ adapter._b_rand_fp32
        else:
            projected = token_features @ adapter.b_rand.float()
    elif token_features.device.type == "cuda":
        token_features = token_features.to(dtype=adapter.b_rand.dtype)
        projected = token_features @ adapter.b_rand
    else:
        token_features = token_features.float()
        projected = token_features @ adapter.b_rand.float()
    projected = F.gelu(projected)
    if not adapter.training:
        projected = apply_mean_centered_soft_threshold(
            projected,
            adapter.soft_threshold_mean,
            adapter.soft_threshold_std,
            alpha=adapter.soft_threshold_alpha,
            beta=adapter.soft_threshold_beta,
            stat_eps=adapter.soft_threshold_stat_eps,
            mode=adapter.soft_threshold_mode,
        )
    return projected


def split_parallel_orthogonal(adapter, projected):
    projected_float = projected.float()
    valid_rank = int(adapter.clean_subspace_valid_rank.item())
    if adapter.subspace_rank <= 0 or valid_rank <= 0:
        mean = torch.zeros_like(projected_float)
        parallel = torch.zeros_like(projected_float)
        orthogonal = projected_float
        return mean, parallel, orthogonal

    mean_vector = adapter.clean_subspace_mean.to(device=projected.device, dtype=torch.float32).view(1, -1)
    basis = adapter.clean_subspace_basis[:, :valid_rank].to(device=projected.device, dtype=torch.float32)
    delta = projected_float - mean_vector
    coeff = delta @ basis
    parallel = coeff @ basis.t()
    orthogonal = delta - parallel
    mean = mean_vector.expand_as(projected_float)
    return mean, parallel, orthogonal


def reconstruct_projected(adapter, clean_projected, adv_projected, mode):
    if mode == MODE_NORMAL_ADV:
        return adv_projected


    mean, clean_parallel, clean_orthogonal = split_parallel_orthogonal(adapter, clean_projected)
    _, adv_parallel, adv_orthogonal = split_parallel_orthogonal(adapter, adv_projected)
    if mode == MODE_FIX_PARALLEL:
        projected = mean + clean_parallel + adapter.subspace_shrink * adv_orthogonal
    elif mode == MODE_FIX_ORTHOGONAL:
        projected = mean + adv_parallel + adapter.subspace_shrink * clean_orthogonal
    elif mode == MODE_FIX_BOTH:
        projected = mean + clean_parallel + adapter.subspace_shrink * clean_orthogonal
    else:
        raise ValueError(f"Unknown intervention mode: {mode}")
    return projected.to(dtype=adv_projected.dtype)


def adapter_output_from_projected(adapter, projected, output_shape):
    if projected.device.type == "cuda" and not getattr(adapter, "force_fp32", False):
        output = projected @ adapter.a_weight.t()
    else:
        if projected.device.type == "cuda":
            adapter._ensure_fp32_cache(projected.device)
            output = projected.float() @ adapter._a_weight_fp32.t()
        else:
            output = projected.float() @ adapter.a_weight.float().t()
    return output.view(*output_shape).to(dtype=projected.dtype)


@contextmanager
def collect_clean_projected_context(adapters, store):
    handles = []

    def make_hook(name):
        def hook(module, inputs, output):
            del output
            clean_projected = adapter_project_without_clean_subspace(module, inputs[0]).detach()
            store[name] = clean_projected
        return hook

    for name, adapter in adapters:
        handles.append(adapter.register_forward_hook(make_hook(name)))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def intervention_context(adapters, clean_store, mode):
    handles = []

    def make_hook(name):
        def hook(module, inputs):
            adv_projected = adapter_project_without_clean_subspace(module, inputs[0])
            clean_projected = clean_store[name].to(device=adv_projected.device, dtype=adv_projected.dtype)
            projected = reconstruct_projected(module, clean_projected, adv_projected, mode)
            output = adapter_output_from_projected(module, projected, inputs[0].shape)
            return (output,)
        return hook

    for name, adapter in adapters:
        handles.append(adapter.register_forward_pre_hook(make_hook(name)))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def evaluate_mode(model, adapters, clean_store, inputs, labels, mode):
    if mode == MODE_NORMAL_ADV:
        with torch.no_grad():
            logits = model(inputs).detach().float()
    else:
        with torch.no_grad(), intervention_context(adapters, clean_store, mode):
            logits = model(inputs).detach().float()
    predictions = logits.argmax(dim=1)
    return {
        "predictions": predictions,
        "correct": predictions.eq(labels),
        "norm_margin": top_logit_normalized_margin(logits, labels),
        "prob_margin": softmax_probability_margin(logits, labels),
    }


def update_store(store, mode, result):
    store[mode]["correct"].append(result["correct"].detach().cpu().float())
    store[mode]["norm_margin"].append(result["norm_margin"].detach().cpu().float())
    store[mode]["prob_margin"].append(result["prob_margin"].detach().cpu().float())


def collect_interventions(model, loader, selected_indices, eps_values, device, args):
    adapters = find_hira_adapters(model)
    print(f"Found {len(adapters)} HiRA adapters for all-layer intervention.")
    for name, adapter in adapters:
        print(
            f"  {name}: subspace_rank={adapter.subspace_rank}, "
            f"valid_rank={int(adapter.clean_subspace_valid_rank.item())}, "
            f"subspace_shrink={adapter.subspace_shrink}"
        )

    aggregate_rows = []
    per_sample_rows = []
    pgd_step_sizes = {
        eps: args.pgd_step_size if args.pgd_step_size is not None else 2.0 * eps / max(args.pgd_steps, 1)
        for eps in eps_values
    }

    for eps in eps_values:
        store = defaultdict(lambda: {"correct": [], "norm_margin": [], "prob_margin": []})
        cursor = 0
        progress = tqdm(loader, desc=f"subspace intervention eps={eps * 255.0:g}/255", dynamic_ncols=True)
        for inputs, labels in progress:
            batch_size = labels.size(0)
            batch_indices = selected_indices[cursor:cursor + batch_size]
            cursor += batch_size
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            clean_store = {}
            with torch.no_grad(), collect_clean_projected_context(adapters, clean_store):
                clean_logits = model(inputs).detach().float()
            clean_predictions = clean_logits.argmax(dim=1)
            clean_correct = clean_predictions.eq(labels)

            adv_inputs = pgd_linf_attack(
                model,
                inputs,
                labels,
                eps=eps,
                steps=args.pgd_steps,
                step_size=pgd_step_sizes[eps],
                random_start=args.pgd_random_start,
                attack_class_ids=getattr(args, "attack_class_ids", None),
            )

            batch_results = {}
            for mode in INTERVENTION_MODES:
                result = evaluate_mode(model, adapters, clean_store, adv_inputs, labels, mode)
                batch_results[mode] = result
                update_store(store, mode, result)

            normal_correct = batch_results[MODE_NORMAL_ADV]["correct"]
            for index in range(batch_size):
                row = {
                    "eps": eps,
                    "eps_pixel": eps * 255.0,
                    "sample_index": int(batch_indices[index]),
                    "label": int(labels[index].detach().cpu()),
                    "clean_prediction": int(clean_predictions[index].detach().cpu()),
                    "clean_correct": int(clean_correct[index].detach().cpu()),
                }
                for mode in INTERVENTION_MODES:
                    result = batch_results[mode]
                    row[f"{mode}_prediction"] = int(result["predictions"][index].detach().cpu())
                    row[f"{mode}_correct"] = int(result["correct"][index].detach().cpu())
                    row[f"{mode}_norm_margin"] = float(result["norm_margin"][index].detach().cpu())
                    row[f"{mode}_prob_margin"] = float(result["prob_margin"][index].detach().cpu())
                row["fix_parallel_rescues_normal_adv"] = int((~normal_correct[index] & batch_results[MODE_FIX_PARALLEL]["correct"][index]).detach().cpu())
                row["fix_orthogonal_rescues_normal_adv"] = int((~normal_correct[index] & batch_results[MODE_FIX_ORTHOGONAL]["correct"][index]).detach().cpu())
                row["fix_both_rescues_normal_adv"] = int((~normal_correct[index] & batch_results[MODE_FIX_BOTH]["correct"][index]).detach().cpu())
                per_sample_rows.append(row)

            normal_acc = torch.cat(store[MODE_NORMAL_ADV]["correct"]).float().mean().item()
            fix_orth_acc = torch.cat(store[MODE_FIX_ORTHOGONAL]["correct"]).float().mean().item()
            progress.set_postfix(normal=f"{normal_acc:.3f}", fix_orth=f"{fix_orth_acc:.3f}")

        normal_all = torch.cat(store[MODE_NORMAL_ADV]["correct"]).bool()
        normal_failed = ~normal_all
        metric_row = {"eps": eps, "eps_pixel": eps * 255.0}
        for mode in INTERVENTION_MODES:
            correct = torch.cat(store[mode]["correct"]).bool()
            norm_margin = torch.cat(store[mode]["norm_margin"]).numpy()
            prob_margin = torch.cat(store[mode]["prob_margin"]).numpy()
            metric_row[f"{mode}_accuracy"] = float(correct.float().mean().item())
            metric_row[f"{mode}_rescue_rate_among_normal_failures"] = float((normal_failed & correct).float().sum().item() / max(normal_failed.float().sum().item(), 1.0))
            metric_row[f"{mode}_norm_margin_mean"] = summarize_distribution(norm_margin)["mean"]
            metric_row[f"{mode}_prob_margin_mean"] = summarize_distribution(prob_margin)["mean"]
            for key, value in summarize_distribution(norm_margin).items():
                metric_row[f"{mode}_norm_margin_{key}"] = value
            for key, value in summarize_distribution(prob_margin).items():
                metric_row[f"{mode}_prob_margin_{key}"] = value
        metric_row["normal_failure_count"] = int(normal_failed.sum().item())
        metric_row["num_samples"] = int(normal_all.numel())
        aggregate_rows.append(metric_row)

    return aggregate_rows, per_sample_rows, adapters


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def series(rows, key):
    return np.asarray([row[key] for row in rows], dtype=np.float64)


def save_plots(run_dir, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = series(rows, "eps_pixel")
    labels = {
        MODE_NORMAL_ADV: "normal PGD",
        MODE_FIX_PARALLEL: "fix principal",
        MODE_FIX_ORTHOGONAL: "fix orthogonal",
        MODE_FIX_BOTH: "fix both",
    }

    fig, ax = plt.subplots(figsize=(7, 5))
    for mode in INTERVENTION_MODES:
        ax.plot(x, series(rows, f"{mode}_accuracy"), marker="o", label=labels[mode])
    ax.set_xlabel("PGD epsilon (pixel / 255)")
    ax.set_ylabel("Accuracy")
    ax.set_title("HiRA subspace intervention accuracy")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "subspace_intervention_accuracy_vs_eps.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for mode in (MODE_FIX_PARALLEL, MODE_FIX_ORTHOGONAL, MODE_FIX_BOTH):
        ax.plot(x, series(rows, f"{mode}_rescue_rate_among_normal_failures"), marker="o", label=labels[mode])
    ax.set_xlabel("PGD epsilon (pixel / 255)")
    ax.set_ylabel("Rescue rate among normal-PGD failures")
    ax.set_title("Counterfactual rescue rate")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "subspace_intervention_rescue_rate_vs_eps.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for mode in INTERVENTION_MODES:
        ax.plot(x, series(rows, f"{mode}_norm_margin_mean"), marker="o", label=labels[mode])
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_xlabel("PGD epsilon (pixel / 255)")
    ax.set_ylabel("Top-logit normalized margin")
    ax.set_title("HiRA subspace intervention margin")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "subspace_intervention_norm_margin_vs_eps.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for mode in INTERVENTION_MODES:
        ax.plot(x, series(rows, f"{mode}_prob_margin_mean"), marker="o", label=labels[mode])
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_xlabel("PGD epsilon (pixel / 255)")
    ax.set_ylabel("Softmax probability margin")
    ax.set_title("HiRA subspace intervention probability margin")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "subspace_intervention_prob_margin_vs_eps.png", dpi=300)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Counterfactual intervention on HiRA principal/orthogonal hidden components.")
    parser.add_argument("--model-name", "--model_name", required=True)
    parser.add_argument("--threat-model", "--threat_model", default="Linf", choices=["Linf", "L2"])
    parser.add_argument("--data-dir", "--data_dir", default="./dataset/imagenet")
    parser.add_argument("--model-dir", "--model_dir", default="./robustbench_models")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", "--batch_size", type=int, default=16)
    parser.add_argument("--num-workers", "--num_workers", type=int, default=4)
    parser.add_argument("--output-dir", "--output_dir", default="visualization/hira_subspace_intervention_outputs")
    parser.add_argument("--run-name", "--run_name", default="")
    parser.add_argument("--num-classes", "--num_classes", type=int, default=20)
    parser.add_argument("--samples-per-class", "--samples_per_class", type=int, default=50)
    parser.add_argument("--class-ids", "--class_ids", default="")
    parser.add_argument("--eps-list", "--eps_list", default="0,1/255,2/255,4/255,8/255,16/255")
    parser.add_argument("--pgd-steps", "--pgd_steps", type=int, default=40)
    parser.add_argument("--pgd-step-size", "--pgd_step_size", type=parse_float_or_fraction, default=None)
    parser.add_argument("--pgd-random-start", "--pgd_random_start", type=str2bool, default=True)
    parser.add_argument("--mask-pgd-logits", "--mask_pgd_logits", type=str2bool, default=False)

    parser.add_argument("--hira-expansion-dim", "--hira_expansion_dim", type=int, default=16384)
    parser.add_argument("--hira-num-blocks", "--hira_num_blocks", type=int, default=4)
    parser.add_argument("--hira-batch-size", "--hira_batch_size", type=int, default=128)
    parser.add_argument("--hira-num-workers", "--hira_num_workers", type=int, default=4)
    parser.add_argument("--hira-epochs", "--hira_epochs", type=int, default=1)
    parser.add_argument("--hira-lr", "--hira_lr", type=float, default=1e-4)
    parser.add_argument("--hira-weight-decay", "--hira_weight_decay", type=float, default=1e-4)
    parser.add_argument("--hira-seed", "--hira_seed", type=int, default=0)
    parser.add_argument("--hira-cache-dir", "--hira_cache_dir", default="pretrained/hira_robustbench")
    parser.add_argument("--hira-dataset-root", "--hira_dataset_root", default="")
    parser.add_argument("--hira-max-train-samples", "--hira_max_train_samples", type=int, default=-1)
    parser.add_argument("--hira-force-retrain", "--hira_force_retrain", type=str2bool, default=False)
    parser.add_argument("--adapt-noise-eps", "--adapt_noise_eps", type=parse_float_or_fraction, default=0.0)
    parser.add_argument("--adapt-noise-num", "--adapt_noise_num", type=int, default=1)
    parser.add_argument("--adapt-alpha", "--adapt_alpha", type=float, default=1.0)
    parser.add_argument("--soft-threshold-alpha", "--soft_threshold_alpha", type=float, default=0.9)
    parser.add_argument("--soft-threshold-beta", "--soft_threshold_beta", type=float, default=4.0)
    parser.add_argument("--soft-threshold-stat-eps", "--soft_threshold_stat_eps", type=float, default=DEFAULT_MEANSPARSE_STAT_EPS)
    parser.add_argument("--soft-threshold-mode", "--soft_threshold_mode", choices=["near_mean", "away_from_mean"], default="away_from_mean")
    parser.add_argument("--hira-subspace-rank", "--hira_subspace_rank", type=int, default=1024)
    parser.add_argument("--hira-subspace-shrink", "--hira_subspace_shrink", type=float, default=0.5)
    parser.add_argument("--stability-ridge-gamma", "--stability_ridge_gamma", type=float, default=0.0)
    parser.add_argument("--stability-ridge-stat-eps", "--stability_ridge_stat_eps", type=float, default=DEFAULT_STABILITY_RIDGE_STAT_EPS)

    parser.add_argument("--ranpac-rp-dim", "--ranpac_rp_dim", type=int, default=10000)
    parser.add_argument("--ranpac-fit-batch-size", "--ranpac_fit_batch_size", type=int, default=64)
    parser.add_argument("--ranpac-num-workers", "--ranpac_num_workers", type=int, default=4)
    parser.add_argument("--ranpac-seed", "--ranpac_seed", type=int, default=0)
    parser.add_argument("--ranpac-lambda", "--ranpac_lambda", type=float, default=0.5)
    parser.add_argument("--ranpac-temp", "--ranpac_temp", type=float, default=1.0)
    parser.add_argument("--ranpac-hardneg-topk", "--ranpac_hardneg_topk", type=int, default=0)
    parser.add_argument("--ranpac-hardneg-gamma", "--ranpac_hardneg_gamma", type=float, default=0.0)
    parser.add_argument("--ranpac-cache-dir", "--ranpac_cache_dir", default="pretrained/ranpac_robustbench")
    parser.add_argument("--ranpac-dataset-root", "--ranpac_dataset_root", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.threat_model != "Linf":
        raise NotImplementedError("This script currently implements Linf PGD only.")
    eps_values = parse_eps_list(args.eps_list)
    if not eps_values:
        raise ValueError("--eps-list must contain at least one epsilon value.")

    set_seed(args.seed)
    device = resolve_device(args.device)
    model_preprocessing = resolve_model_preprocessing(args.model_name, args.threat_model)
    dataset = build_imagenet_dataset(args.data_dir, model_preprocessing)
    selected_indices, selected_class_ids = select_balanced_indices(
        dataset,
        num_classes=args.num_classes,
        samples_per_class=args.samples_per_class,
        seed=args.seed,
        class_ids=parse_class_ids(args.class_ids),
    )
    args.attack_class_ids = selected_class_ids if args.mask_pgd_logits else None
    loader = build_eval_loader(dataset, selected_indices, args.batch_size, args.num_workers)

    eps_name = "-".join(sanitize_name(eps) for eps in eps_values)
    run_name = args.run_name or f"{sanitize_name(args.model_name)}_classes{len(selected_class_ids)}_n{args.samples_per_class}_eps{eps_name}_steps{args.pgd_steps}_seed{args.seed}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving HiRA subspace intervention outputs to: {run_dir}")
    print(f"Selected classes: {selected_class_ids}")

    print("Loading HiRA+RanPAC model...")
    model = freeze_model(build_hira_ranpac_model(args, model_preprocessing, device))
    aggregate_rows, per_sample_rows, adapters = collect_interventions(model, loader, selected_indices, eps_values, device, args)

    write_csv(run_dir / "subspace_intervention_metrics.csv", aggregate_rows)
    write_csv(run_dir / "subspace_intervention_per_sample.csv", per_sample_rows)
    save_plots(run_dir, aggregate_rows)

    adapter_summary = [
        {
            "name": name,
            "subspace_rank": int(adapter.subspace_rank),
            "valid_rank": int(adapter.clean_subspace_valid_rank.item()),
            "subspace_shrink": float(adapter.subspace_shrink),
        }
        for name, adapter in adapters
    ]
    summary = {
        "model_name": args.model_name,
        "dataset": DATASET,
        "threat_model": args.threat_model,
        "selected_class_ids": selected_class_ids,
        "samples_per_class": args.samples_per_class,
        "eps_values": eps_values,
        "eps_pixels": [eps * 255.0 for eps in eps_values],
        "pgd_steps": args.pgd_steps,
        "pgd_step_size": args.pgd_step_size,
        "pgd_random_start": args.pgd_random_start,
        "mask_pgd_logits": args.mask_pgd_logits,
        "soft_threshold_alpha": args.soft_threshold_alpha,
        "soft_threshold_beta": args.soft_threshold_beta,
        "soft_threshold_mode": args.soft_threshold_mode,
        "hira_subspace_rank": args.hira_subspace_rank,
        "hira_subspace_shrink": args.hira_subspace_shrink,
        "hira_adapters": adapter_summary,
        "outputs": [
            "subspace_intervention_accuracy_vs_eps.png",
            "subspace_intervention_rescue_rate_vs_eps.png",
            "subspace_intervention_norm_margin_vs_eps.png",
            "subspace_intervention_prob_margin_vs_eps.png",
            "subspace_intervention_metrics.csv",
            "subspace_intervention_per_sample.csv",
        ],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
