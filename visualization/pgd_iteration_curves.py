#!/usr/bin/env python3
"""
PGD iteration curves for RobustBench ImageNet models with and without HiRA+RanPAC.

The main plot is robust accuracy as a function of PGD iterations. The script
records both current-iteration accuracy and best-so-far accuracy, because PGD
can oscillate and a successful adversarial example at any earlier iteration is
still a successful attack.
"""

import argparse
import csv
import json
import random
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


def top_logit_normalized_margin_torch(logits, labels, eps=1e-12):
    true_logits = logits.gather(1, labels.view(-1, 1)).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, labels.view(-1, 1), float("-inf"))
    max_wrong = masked.max(dim=1).values
    top_logits = logits.max(dim=1).values.abs().clamp_min(float(eps))
    return (true_logits - max_wrong) / top_logits


def update_metric_store(store, iteration, logits, labels, ever_wrong):
    predictions = logits.argmax(dim=1)
    current_correct = predictions.eq(labels)
    ever_wrong = ever_wrong | ~current_correct
    best_so_far_correct = ~ever_wrong
    norm_margins = top_logit_normalized_margin_torch(logits.float(), labels)
    entry = store.setdefault(
        iteration,
        {
            "total": 0,
            "current_correct": 0,
            "best_so_far_correct": 0,
            "norm_margins": [],
        },
    )
    entry["total"] += labels.numel()
    entry["current_correct"] += current_correct.sum().item()
    entry["best_so_far_correct"] += best_so_far_correct.sum().item()
    entry["norm_margins"].append(norm_margins.detach().cpu())
    return ever_wrong, predictions


def collect_iteration_curve(model, loader, selected_indices, device, args, variant_name):
    store = {}
    first_success_rows = []
    eps = args.eps
    step_size = args.pgd_step_size if args.pgd_step_size is not None else 2.0 * eps / max(args.max_steps, 1)
    cursor = 0

    for inputs, labels in tqdm(loader, desc=f"{variant_name}: PGD iteration curve", dynamic_ncols=True):
        batch_size = labels.size(0)
        batch_indices = selected_indices[cursor:cursor + batch_size]
        cursor += batch_size

        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.no_grad():
            clean_logits = model(inputs).detach()
        clean_predictions = clean_logits.argmax(dim=1)
        ever_wrong = ~clean_predictions.eq(labels)
        first_success_iter = torch.full_like(labels, fill_value=-1)
        first_success_iter[ever_wrong] = 0
        ever_wrong, _ = update_metric_store(store, 0, clean_logits, labels, ever_wrong)

        if args.pgd_random_start and eps > 0:
            delta = torch.empty_like(inputs).uniform_(-eps, eps)
            delta = torch.clamp(inputs + delta, 0.0, 1.0) - inputs
        else:
            delta = torch.zeros_like(inputs)

        for iteration in range(1, args.max_steps + 1):
            adv_inputs = torch.clamp(inputs + delta, 0.0, 1.0).detach().requires_grad_(True)
            attack_logits = mask_logits_to_classes(model(adv_inputs), getattr(args, "attack_class_ids", None))
            loss = F.cross_entropy(attack_logits, labels, reduction="sum")
            grad = torch.autograd.grad(loss, adv_inputs, only_inputs=True)[0]
            delta = (delta + step_size * grad.sign()).detach().clamp(-eps, eps)
            delta = torch.clamp(inputs + delta, 0.0, 1.0) - inputs

            with torch.no_grad():
                logits = model(torch.clamp(inputs + delta, 0.0, 1.0)).detach()
            previous_ever_wrong = ever_wrong.clone()
            ever_wrong, predictions = update_metric_store(store, iteration, logits, labels, ever_wrong)
            newly_successful = ever_wrong & ~previous_ever_wrong
            first_success_iter[newly_successful] = iteration

        for index, label, clean_pred, first_iter in zip(batch_indices, labels.cpu(), clean_predictions.cpu(), first_success_iter.cpu()):
            first_success_rows.append(
                {
                    "variant": variant_name,
                    "sample_index": int(index),
                    "label": int(label),
                    "clean_prediction": int(clean_pred),
                    "first_success_iter": int(first_iter),
                    "never_successful": int(first_iter < 0),
                }
            )

    rows = []
    for iteration in sorted(store):
        entry = store[iteration]
        margins = torch.cat(entry["norm_margins"], dim=0).numpy()
        rows.append(
            {
                "variant": variant_name,
                "iteration": iteration,
                "current_accuracy": entry["current_correct"] / max(entry["total"], 1),
                "best_so_far_accuracy": entry["best_so_far_correct"] / max(entry["total"], 1),
                "toplogit_norm_margin_mean": float(np.mean(margins)),
                "toplogit_norm_margin_median": float(np.median(margins)),
                "toplogit_norm_margin_p25": float(np.percentile(margins, 25)),
                "toplogit_norm_margin_p75": float(np.percentile(margins, 75)),
                "total": entry["total"],
            }
        )
    return rows, first_success_rows


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def rows_for_variant(rows, variant):
    return [row for row in rows if row["variant"] == variant]


def series(rows, key):
    return np.asarray([row[key] for row in rows], dtype=np.float64)


def save_plots(run_dir, aggregate_rows, first_success_rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    original_rows = rows_for_variant(aggregate_rows, VARIANT_ORIGINAL)
    ours_rows = rows_for_variant(aggregate_rows, VARIANT_HIRA_RANPAC)

    fig, ax = plt.subplots(figsize=(7, 5))
    for rows, label in ((original_rows, "original"), (ours_rows, "HiRA+RanPAC")):
        ax.plot(series(rows, "iteration"), series(rows, "current_accuracy"), linestyle="--", alpha=0.45, label=f"{label} current")
        ax.plot(series(rows, "iteration"), series(rows, "best_so_far_accuracy"), marker="o", markevery=max(len(rows) // 10, 1), label=f"{label} best-so-far")
    ax.set_xlabel("PGD iteration")
    ax.set_ylabel("Robust accuracy")
    ax.set_title("Robust accuracy vs PGD iterations")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "robust_accuracy_vs_pgd_iterations.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for rows, label in ((original_rows, "original"), (ours_rows, "HiRA+RanPAC")):
        x = series(rows, "iteration")
        y = series(rows, "toplogit_norm_margin_mean")
        lower = series(rows, "toplogit_norm_margin_p25")
        upper = series(rows, "toplogit_norm_margin_p75")
        ax.plot(x, y, label=label)
        ax.fill_between(x, lower, upper, alpha=0.15)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_xlabel("PGD iteration")
    ax.set_ylabel("Top-logit normalized margin")
    ax.set_title("Normalized margin vs PGD iterations")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "toplogit_norm_margin_vs_pgd_iterations.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for variant, label in ((VARIANT_ORIGINAL, "original"), (VARIANT_HIRA_RANPAC, "HiRA+RanPAC")):
        values = [row["first_success_iter"] for row in first_success_rows if row["variant"] == variant and row["first_success_iter"] > 0]
        if values:
            ax.hist(values, bins=30, alpha=0.55, label=label)
    ax.set_xlabel("First successful PGD iteration")
    ax.set_ylabel("Sample count")
    ax.set_title("First successful attack iteration")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "first_success_iteration_hist.png", dpi=300)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot robust accuracy vs PGD iterations for original vs HiRA+RanPAC RobustBench models.")
    parser.add_argument("--model-name", "--model_name", required=True, help="RobustBench ImageNet model name.")
    parser.add_argument("--threat-model", "--threat_model", default="Linf", choices=["Linf", "L2"], help="Threat model used to load the RobustBench model. Only Linf PGD is implemented here.")
    parser.add_argument("--data-dir", "--data_dir", default="./dataset/imagenet", help="ImageNet root containing train/ and val/.")
    parser.add_argument("--model-dir", "--model_dir", default="./robustbench_models", help="RobustBench checkpoint cache directory.")
    parser.add_argument("--device", default="cuda:0", help="Device, e.g. cuda:0 or cpu.")
    parser.add_argument("--seed", type=int, default=0, help="Global random seed and class-selection seed.")
    parser.add_argument("--batch-size", "--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--num-workers", "--num_workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--output-dir", "--output_dir", default="visualization/pgd_iteration_outputs", help="Directory where outputs are saved.")
    parser.add_argument("--run-name", "--run_name", default="", help="Optional output subdirectory name.")
    parser.add_argument("--num-classes", "--num_classes", type=int, default=20, help="Number of ImageNet classes to sample when --class-ids is empty.")
    parser.add_argument("--samples-per-class", "--samples_per_class", type=int, default=50, help="Number of validation images per selected class.")
    parser.add_argument("--class-ids", "--class_ids", default="", help="Optional comma-separated ImageNet class IDs.")
    parser.add_argument("--eps", type=parse_float_or_fraction, default=4.0 / 255.0, help="Linf PGD epsilon; accepts fractions like 4/255.")
    parser.add_argument("--max-steps", "--max_steps", type=int, default=100, help="Maximum PGD iterations to trace.")
    parser.add_argument("--pgd-step-size", "--pgd_step_size", type=parse_float_or_fraction, default=None, help="Optional PGD step size. Defaults to 2 * eps / max_steps.")
    parser.add_argument("--pgd-random-start", "--pgd_random_start", type=str2bool, default=True, help="Use random-start PGD.")
    parser.add_argument("--mask-pgd-logits", "--mask_pgd_logits", type=str2bool, default=False, help="During PGD, set logits outside selected classes to -inf.")

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
    print(f"Saving PGD iteration outputs to: {run_dir}")
    print(f"Selected classes: {selected_class_ids}")

    aggregate_rows = []
    first_success_rows = []

    print("Loading original RobustBench model...")
    original_model = freeze_model(load_robustbench_model(args.model_name, args.threat_model, args.model_dir, device))
    rows, first_rows = collect_iteration_curve(original_model, loader, selected_indices, device, args, VARIANT_ORIGINAL)
    aggregate_rows.extend(rows)
    first_success_rows.extend(first_rows)
    del original_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("Loading and wrapping RobustBench model with HiRA+RanPAC...")
    ours_model = freeze_model(build_hira_ranpac_model(args, model_preprocessing, device))
    rows, first_rows = collect_iteration_curve(ours_model, loader, selected_indices, device, args, VARIANT_HIRA_RANPAC)
    aggregate_rows.extend(rows)
    first_success_rows.extend(first_rows)
    del ours_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    write_csv(run_dir / "iteration_metrics.csv", aggregate_rows)
    write_csv(run_dir / "first_success_iterations.csv", first_success_rows)
    save_plots(run_dir, aggregate_rows, first_success_rows)
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
        "outputs": [
            "robust_accuracy_vs_pgd_iterations.png",
            "toplogit_norm_margin_vs_pgd_iterations.png",
            "first_success_iteration_hist.png",
            "iteration_metrics.csv",
            "first_success_iterations.csv",
        ],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
