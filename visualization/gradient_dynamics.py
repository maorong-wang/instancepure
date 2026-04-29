#!/usr/bin/env python3
"""
Gradient-domain diagnostics for RobustBench ImageNet models with HiRA+RanPAC.

The script traces Linf PGD and measures gradient norms, consecutive-gradient
cosine similarity, sign consistency, gradient-to-perturbation alignment, and
inter-model gradient alignment at shared images.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from classifiers.mean_sparse import DEFAULT_MEANSPARSE_STAT_EPS
from classifiers.stability_ridge import DEFAULT_STABILITY_RIDGE_STAT_EPS
from visualization.tsne_robustbench_ranpac import (
    DATASET,
    build_eval_loader,
    build_hira_ranpac_model,
    build_imagenet_dataset,
    freeze_model,
    load_robustbench_model,
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


VARIANT_ORIGINAL = "original"
VARIANT_HIRA_RANPAC = "hira_ranpac_regression"


def flattened(tensor):
    return tensor.detach().view(tensor.size(0), -1).float()


def batch_cosine(left, right, eps=1e-12):
    left = flattened(left)
    right = flattened(right)
    numerator = (left * right).sum(dim=1)
    denominator = left.norm(p=2, dim=1).clamp_min(eps) * right.norm(p=2, dim=1).clamp_min(eps)
    return numerator / denominator


def batch_sign_consistency(left, right):
    left_sign = flattened(left).sign()
    right_sign = flattened(right).sign()
    return left_sign.eq(right_sign).float().mean(dim=1)


def batch_l1_norm(tensor):
    return flattened(tensor).norm(p=1, dim=1)


def batch_l2_norm(tensor):
    return flattened(tensor).norm(p=2, dim=1)


def batch_linf_norm(tensor):
    return flattened(tensor).abs().max(dim=1).values


def top_logit_normalized_margin_torch(logits, labels, eps=1e-12):
    true_logits = logits.gather(1, labels.view(-1, 1)).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, labels.view(-1, 1), float("-inf"))
    max_wrong = masked.max(dim=1).values
    top_logits = logits.max(dim=1).values.abs().clamp_min(float(eps))
    return (true_logits - max_wrong) / top_logits


def input_gradient(model, inputs, labels, attack_class_ids=None):
    model.zero_grad(set_to_none=True)
    eval_inputs = inputs.detach().requires_grad_(True)
    logits = mask_logits_to_classes(model(eval_inputs), attack_class_ids)
    loss = F.cross_entropy(logits, labels, reduction="sum")
    grad = torch.autograd.grad(loss, eval_inputs, only_inputs=True)[0].detach()
    margin = top_logit_normalized_margin_torch(logits.detach().float(), labels).detach()
    predictions = logits.detach().argmax(dim=1)
    return grad, margin, predictions


def update_stat_store(store, iteration, metrics):
    entry = store.setdefault(iteration, {key: [] for key in metrics})
    for key, value in metrics.items():
        entry[key].append(value.detach().cpu().float())


def summarize_store(store, variant):
    rows = []
    for iteration in sorted(store):
        row = {"variant": variant, "iteration": iteration}
        for key, values in store[iteration].items():
            combined = torch.cat(values, dim=0).numpy()
            row[f"{key}_mean"] = float(np.mean(combined))
            row[f"{key}_median"] = float(np.median(combined))
            row[f"{key}_p25"] = float(np.percentile(combined, 25))
            row[f"{key}_p75"] = float(np.percentile(combined, 75))
        rows.append(row)
    return rows


def trace_variant_gradients(model, loader, device, args, variant_name):
    store = {}
    eps = args.eps
    step_size = args.pgd_step_size if args.pgd_step_size is not None else 2.0 * eps / max(args.max_steps, 1)

    for inputs, labels in tqdm(loader, desc=f"{variant_name}: gradient dynamics", dynamic_ncols=True):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if args.pgd_random_start and eps > 0:
            delta = torch.empty_like(inputs).uniform_(-eps, eps)
            delta = torch.clamp(inputs + delta, 0.0, 1.0) - inputs
        else:
            delta = torch.zeros_like(inputs)
        previous_grad = None
        for iteration in range(1, args.max_steps + 1):
            adv_inputs = torch.clamp(inputs + delta, 0.0, 1.0)
            grad, norm_margin, predictions = input_gradient(
                model,
                adv_inputs,
                labels,
                attack_class_ids=getattr(args, "attack_class_ids", None),
            )
            perturbation = adv_inputs.detach() - inputs
            metrics = {
                "grad_l1": batch_l1_norm(grad),
                "grad_l2": batch_l2_norm(grad),
                "grad_linf": batch_linf_norm(grad),
                "norm_margin": norm_margin,
                "accuracy": predictions.eq(labels).float(),
                "grad_perturb_cosine": batch_cosine(grad, perturbation),
                "grad_perturb_sign_consistency": batch_sign_consistency(grad, perturbation),
            }
            if previous_grad is not None:
                metrics["consecutive_grad_cosine"] = batch_cosine(grad, previous_grad)
                metrics["consecutive_grad_sign_consistency"] = batch_sign_consistency(grad, previous_grad)
            update_stat_store(store, iteration, metrics)
            previous_grad = grad.detach()
            delta = (delta + step_size * grad.sign()).detach().clamp(-eps, eps)
            delta = torch.clamp(inputs + delta, 0.0, 1.0) - inputs

    return summarize_store(store, variant_name)


def trace_inter_model_alignment(original_model, ours_model, loader, device, args):
    store = {}
    eps = args.eps
    step_size = args.pgd_step_size if args.pgd_step_size is not None else 2.0 * eps / max(args.max_steps, 1)

    for inputs, labels in tqdm(loader, desc="inter-model gradient alignment", dynamic_ncols=True):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if args.pgd_random_start and eps > 0:
            delta = torch.empty_like(inputs).uniform_(-eps, eps)
            delta = torch.clamp(inputs + delta, 0.0, 1.0) - inputs
        else:
            delta = torch.zeros_like(inputs)

        for iteration in range(1, args.max_steps + 1):
            shared_inputs = torch.clamp(inputs + delta, 0.0, 1.0)
            original_grad, original_margin, original_predictions = input_gradient(
                original_model,
                shared_inputs,
                labels,
                attack_class_ids=getattr(args, "attack_class_ids", None),
            )
            ours_grad, ours_margin, ours_predictions = input_gradient(
                ours_model,
                shared_inputs,
                labels,
                attack_class_ids=getattr(args, "attack_class_ids", None),
            )
            metrics = {
                "inter_model_grad_cosine": batch_cosine(original_grad, ours_grad),
                "inter_model_grad_sign_consistency": batch_sign_consistency(original_grad, ours_grad),
                "original_norm_margin_on_shared_path": original_margin,
                "hira_ranpac_norm_margin_on_shared_path": ours_margin,
                "original_accuracy_on_shared_path": original_predictions.eq(labels).float(),
                "hira_ranpac_accuracy_on_shared_path": ours_predictions.eq(labels).float(),
            }
            update_stat_store(store, iteration, metrics)
            if args.shared_path_source == "original":
                step_grad = original_grad
            elif args.shared_path_source == "hira_ranpac":
                step_grad = ours_grad
            else:
                step_grad = 0.5 * (original_grad + ours_grad)
            delta = (delta + step_size * step_grad.sign()).detach().clamp(-eps, eps)
            delta = torch.clamp(inputs + delta, 0.0, 1.0) - inputs

    return summarize_store(store, f"shared_path_{args.shared_path_source}")


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rows_for_variant(rows, variant):
    return [row for row in rows if row["variant"] == variant]


def series(rows, key):
    return np.asarray([row[key] for row in rows], dtype=np.float64)


def drop_first_iteration(rows):
    return [row for row in rows if int(row["iteration"]) > 1]


def drop_leading_zero_rows(rows, key):
    mean_key = f"{key}_mean"
    start = 0
    while start < len(rows) and abs(float(rows[start][mean_key])) <= 1e-12:
        start += 1
    return rows[start:]


def plot_metric(ax, rows, key, label, drop_first=False, drop_leading_zero=False):
    if drop_first:
        rows = drop_first_iteration(rows)
    if drop_leading_zero:
        rows = drop_leading_zero_rows(rows, key)
    if not rows:
        return
    x = series(rows, "iteration")
    y = series(rows, f"{key}_mean")
    lower = series(rows, f"{key}_p25")
    upper = series(rows, f"{key}_p75")
    ax.plot(x, y, label=label)
    ax.fill_between(x, lower, upper, alpha=0.15)


def save_plots(run_dir, variant_rows, alignment_rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    original_rows = rows_for_variant(variant_rows, VARIANT_ORIGINAL)
    ours_rows = rows_for_variant(variant_rows, VARIANT_HIRA_RANPAC)

    for metric, ylabel, filename, title in (
        ("grad_l2", "Gradient L2 norm", "gradient_l2_norm_vs_iteration.png", "Gradient norm vs PGD iteration"),
        ("consecutive_grad_cosine", "cos(g_t, g_{t-1})", "consecutive_gradient_cosine_vs_iteration.png", "Consecutive-gradient cosine similarity"),
        ("consecutive_grad_sign_consistency", "sign agreement", "consecutive_gradient_sign_consistency_vs_iteration.png", "Consecutive-gradient sign consistency"),
        ("grad_perturb_cosine", "cos(g_t, x_t - x_clean)", "gradient_perturbation_alignment_vs_iteration.png", "Gradient-to-perturbation alignment"),
        ("norm_margin", "Top-logit normalized margin", "gradient_path_norm_margin_vs_iteration.png", "Margin along PGD path"),
    ):
        fig, ax = plt.subplots(figsize=(7, 5))
        for rows, label in ((original_rows, "original"), (ours_rows, "HiRA+RanPAC")):
            metric_rows = [row for row in rows if f"{metric}_mean" in row]
            if metric_rows:
                plot_metric(ax, metric_rows, metric, label, drop_first=metric.startswith("grad_perturb"))
        ax.set_xlabel("PGD iteration")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(run_dir / filename, dpi=300)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for rows, label in ((original_rows, "original"), (ours_rows, "HiRA+RanPAC")):
        metric = "grad_perturb_cosine"
        metric_rows = [row for row in rows if f"{metric}_mean" in row]
        if metric_rows:
            plot_metric(ax, metric_rows, metric, label, drop_first=True)
    ax.set_xlabel("PGD iteration")
    ax.set_ylabel("cos(g_t, x_t - x_clean)")
    ax.set_title("Gradient-to-perturbation cosine alignment")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "gradient_perturbation_cosine_vs_iteration.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for rows, label in ((original_rows, "original"), (ours_rows, "HiRA+RanPAC")):
        metric = "grad_perturb_sign_consistency"
        metric_rows = [row for row in rows if f"{metric}_mean" in row]
        if metric_rows:
            plot_metric(ax, metric_rows, metric, label, drop_first=True)
    ax.set_xlabel("PGD iteration")
    ax.set_ylabel("sign(g_t), sign(x_t - x_clean) agreement")
    ax.set_title("Gradient-to-perturbation sign agreement")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "gradient_perturbation_sign_agreement_vs_iteration.png", dpi=300)
    plt.close(fig)

    if alignment_rows:
        fig, ax = plt.subplots(figsize=(7, 5))
        plot_metric(ax, alignment_rows, "inter_model_grad_cosine", "gradient cosine")
        ax.set_xlabel("PGD iteration")
        ax.set_ylabel("cos(original grad, HiRA+RanPAC grad)")
        ax.set_title("Inter-model gradient alignment on shared PGD path")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(run_dir / "inter_model_gradient_cosine_vs_iteration.png", dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 5))
        plot_metric(ax, alignment_rows, "inter_model_grad_sign_consistency", "gradient sign agreement")
        ax.set_xlabel("PGD iteration")
        ax.set_ylabel("sign agreement")
        ax.set_title("Inter-model gradient sign consistency on shared PGD path")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(run_dir / "inter_model_gradient_sign_consistency_vs_iteration.png", dpi=300)
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot gradient-domain diagnostics for original vs HiRA+RanPAC RobustBench models.")
    parser.add_argument("--model-name", "--model_name", required=True, help="RobustBench ImageNet model name.")
    parser.add_argument("--threat-model", "--threat_model", default="Linf", choices=["Linf", "L2"], help="Threat model used to load the RobustBench model. Only Linf PGD is implemented here.")
    parser.add_argument("--data-dir", "--data_dir", default="./dataset/imagenet")
    parser.add_argument("--model-dir", "--model_dir", default="./robustbench_models")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", "--batch_size", type=int, default=16)
    parser.add_argument("--num-workers", "--num_workers", type=int, default=4)
    parser.add_argument("--output-dir", "--output_dir", default="visualization/gradient_dynamics_outputs")
    parser.add_argument("--run-name", "--run_name", default="")
    parser.add_argument("--num-classes", "--num_classes", type=int, default=20)
    parser.add_argument("--samples-per-class", "--samples_per_class", type=int, default=50)
    parser.add_argument("--class-ids", "--class_ids", default="")
    parser.add_argument("--eps", type=parse_float_or_fraction, default=4.0 / 255.0)
    parser.add_argument("--max-steps", "--max_steps", type=int, default=40)
    parser.add_argument("--pgd-step-size", "--pgd_step_size", type=parse_float_or_fraction, default=None)
    parser.add_argument("--pgd-random-start", "--pgd_random_start", type=str2bool, default=True)
    parser.add_argument("--mask-pgd-logits", "--mask_pgd_logits", type=str2bool, default=False)
    parser.add_argument("--skip-inter-model", "--skip_inter_model", type=str2bool, default=False)
    parser.add_argument("--shared-path-source", "--shared_path_source", choices=["original", "hira_ranpac", "average"], default="original")

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

    run_name = args.run_name or f"{sanitize_name(args.model_name)}_classes{len(selected_class_ids)}_n{args.samples_per_class}_eps{sanitize_name(args.eps)}_steps{args.max_steps}_seed{args.seed}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving gradient dynamics outputs to: {run_dir}")
    print(f"Selected classes: {selected_class_ids}")

    print("Loading original RobustBench model...")
    original_model = freeze_model(load_robustbench_model(args.model_name, args.threat_model, args.model_dir, device))
    print("Loading and wrapping RobustBench model with HiRA+RanPAC...")
    ours_model = freeze_model(build_hira_ranpac_model(args, model_preprocessing, device))

    variant_rows = []
    variant_rows.extend(trace_variant_gradients(original_model, loader, device, args, VARIANT_ORIGINAL))
    variant_rows.extend(trace_variant_gradients(ours_model, loader, device, args, VARIANT_HIRA_RANPAC))
    alignment_rows = []
    if not args.skip_inter_model:
        alignment_rows = trace_inter_model_alignment(original_model, ours_model, loader, device, args)

    write_csv(run_dir / "gradient_dynamics_metrics.csv", variant_rows)
    write_csv(run_dir / "inter_model_gradient_alignment.csv", alignment_rows)
    save_plots(run_dir, variant_rows, alignment_rows)
    summary = {
        "model_name": args.model_name,
        "dataset": DATASET,
        "threat_model": args.threat_model,
        "selected_class_ids": selected_class_ids,
        "samples_per_class": args.samples_per_class,
        "eps": args.eps,
        "eps_pixel": args.eps * 255.0,
        "max_steps": args.max_steps,
        "pgd_step_size": args.pgd_step_size if args.pgd_step_size is not None else 2.0 * args.eps / max(args.max_steps, 1),
        "pgd_random_start": args.pgd_random_start,
        "mask_pgd_logits": args.mask_pgd_logits,
        "shared_path_source": args.shared_path_source,
        "skip_inter_model": args.skip_inter_model,
        "outputs": [
            "gradient_l2_norm_vs_iteration.png",
            "consecutive_gradient_cosine_vs_iteration.png",
            "consecutive_gradient_sign_consistency_vs_iteration.png",
            "gradient_perturbation_alignment_vs_iteration.png",
            "gradient_perturbation_cosine_vs_iteration.png",
            "gradient_perturbation_sign_agreement_vs_iteration.png",
            "gradient_path_norm_margin_vs_iteration.png",
            "inter_model_gradient_cosine_vs_iteration.png",
            "inter_model_gradient_sign_consistency_vs_iteration.png",
            "gradient_dynamics_metrics.csv",
            "inter_model_gradient_alignment.csv",
        ],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
