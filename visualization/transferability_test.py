#!/usr/bin/env python3
"""
PGD transferability test for RobustBench ImageNet models.

For each epsilon, this script generates adversarial images on the original
RobustBench model and evaluates them on both original and HiRA+RanPAC. It then
does the reverse: generates adversarial images on HiRA+RanPAC and evaluates
them on both models.
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
VARIANT_LABELS = {
    VARIANT_ORIGINAL: "original",
    VARIANT_HIRA_RANPAC: "HiRA+RanPAC",
}


def parse_eps_list(value):
    return [parse_float_or_fraction(item.strip()) for item in str(value).split(",") if item.strip()]


def top_logit_normalized_margin_torch(logits, labels, eps=1e-12):
    true_logits = logits.gather(1, labels.view(-1, 1)).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, labels.view(-1, 1), float("-inf"))
    max_wrong = masked.max(dim=1).values
    top_logits = logits.max(dim=1).values.abs().clamp_min(float(eps))
    return (true_logits - max_wrong) / top_logits


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


def make_stat_entry():
    return {
        "total": 0,
        "adv_correct": 0,
        "clean_correct": 0,
        "clean_correct_adv_correct": 0,
        "source_success": 0,
        "source_success_eval_wrong": 0,
        "source_success_eval_clean_correct": 0,
        "source_success_eval_clean_correct_wrong": 0,
        "margins": [],
    }


def safe_divide(numerator, denominator):
    if denominator <= 0:
        return float("nan")
    return numerator / denominator


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


def update_stats(entry, labels, clean_correct, adv_correct, source_success, margins):
    entry["total"] += labels.numel()
    entry["adv_correct"] += adv_correct.sum().item()
    entry["clean_correct"] += clean_correct.sum().item()
    entry["clean_correct_adv_correct"] += (clean_correct & adv_correct).sum().item()
    entry["source_success"] += source_success.sum().item()
    entry["source_success_eval_wrong"] += (source_success & ~adv_correct).sum().item()
    source_success_eval_clean_correct = source_success & clean_correct
    entry["source_success_eval_clean_correct"] += source_success_eval_clean_correct.sum().item()
    entry["source_success_eval_clean_correct_wrong"] += (source_success_eval_clean_correct & ~adv_correct).sum().item()
    entry["margins"].append(margins.detach().cpu().float())


def evaluate_logits(model, inputs):
    with torch.no_grad():
        return model(inputs).detach().float()


def collect_transferability(original_model, ours_model, loader, selected_indices, eps_values, device, args):
    models = {
        VARIANT_ORIGINAL: original_model,
        VARIANT_HIRA_RANPAC: ours_model,
    }
    stats = {}
    clean_stats = {
        variant: {"total": 0, "correct": 0, "margins": []}
        for variant in models
    }
    per_sample_rows = []
    cursor = 0

    for inputs, labels in tqdm(loader, desc="PGD transferability", dynamic_ncols=True):
        batch_size = labels.size(0)
        batch_indices = selected_indices[cursor:cursor + batch_size]
        cursor += batch_size

        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        clean_logits = {variant: evaluate_logits(model, inputs) for variant, model in models.items()}
        clean_predictions = {variant: logits.argmax(dim=1) for variant, logits in clean_logits.items()}
        clean_correct = {variant: prediction.eq(labels) for variant, prediction in clean_predictions.items()}

        for variant, logits in clean_logits.items():
            clean_stats[variant]["total"] += labels.numel()
            clean_stats[variant]["correct"] += clean_correct[variant].sum().item()
            clean_stats[variant]["margins"].append(top_logit_normalized_margin_torch(logits, labels).detach().cpu().float())

        for eps in eps_values:
            step_size = args.pgd_step_size if args.pgd_step_size is not None else 2.0 * eps / max(args.pgd_steps, 1)
            for attack_source, source_model in models.items():
                adv_inputs = pgd_linf_attack(
                    source_model,
                    inputs,
                    labels,
                    eps=eps,
                    steps=args.pgd_steps,
                    step_size=step_size,
                    random_start=args.pgd_random_start,
                    attack_class_ids=getattr(args, "attack_class_ids", None),
                )
                adv_logits = {variant: evaluate_logits(model, adv_inputs) for variant, model in models.items()}
                adv_predictions = {variant: logits.argmax(dim=1) for variant, logits in adv_logits.items()}
                adv_correct = {variant: prediction.eq(labels) for variant, prediction in adv_predictions.items()}
                source_success = clean_correct[attack_source] & ~adv_correct[attack_source]

                for eval_variant, logits in adv_logits.items():
                    entry = stats.setdefault((eps, attack_source, eval_variant), make_stat_entry())
                    margins = top_logit_normalized_margin_torch(logits, labels)
                    update_stats(
                        entry,
                        labels,
                        clean_correct[eval_variant],
                        adv_correct[eval_variant],
                        source_success,
                        margins,
                    )

                for index_in_batch, sample_index in enumerate(batch_indices):
                    row = {
                        "eps": eps,
                        "eps_pixel": eps * 255.0,
                        "sample_index": int(sample_index),
                        "label": int(labels[index_in_batch].detach().cpu()),
                        "attack_source": attack_source,
                        "clean_original_prediction": int(clean_predictions[VARIANT_ORIGINAL][index_in_batch].detach().cpu()),
                        "clean_hira_ranpac_prediction": int(clean_predictions[VARIANT_HIRA_RANPAC][index_in_batch].detach().cpu()),
                        "original_prediction_on_adv": int(adv_predictions[VARIANT_ORIGINAL][index_in_batch].detach().cpu()),
                        "hira_ranpac_prediction_on_adv": int(adv_predictions[VARIANT_HIRA_RANPAC][index_in_batch].detach().cpu()),
                        "original_correct_on_adv": int(adv_correct[VARIANT_ORIGINAL][index_in_batch].detach().cpu()),
                        "hira_ranpac_correct_on_adv": int(adv_correct[VARIANT_HIRA_RANPAC][index_in_batch].detach().cpu()),
                        "source_attack_succeeded": int(source_success[index_in_batch].detach().cpu()),
                    }
                    target_variant = VARIANT_HIRA_RANPAC if attack_source == VARIANT_ORIGINAL else VARIANT_ORIGINAL
                    row["cross_transfer_succeeded"] = int((~adv_correct[target_variant][index_in_batch]).detach().cpu())
                    for eval_variant, logits in adv_logits.items():
                        margin = top_logit_normalized_margin_torch(
                            logits[index_in_batch:index_in_batch + 1],
                            labels[index_in_batch:index_in_batch + 1],
                        )[0]
                        row[f"{eval_variant}_toplogit_norm_margin_on_adv"] = float(margin.detach().cpu())
                    per_sample_rows.append(row)

    aggregate_rows = []
    for (eps, attack_source, eval_variant), entry in sorted(stats.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])):
        margins = torch.cat(entry["margins"], dim=0).numpy()
        margin_summary = summarize_distribution(margins)
        clean_accuracy = safe_divide(entry["clean_correct"], entry["total"])
        adv_accuracy = safe_divide(entry["adv_correct"], entry["total"])
        clean_correct_adv_accuracy = safe_divide(entry["clean_correct_adv_correct"], entry["clean_correct"])
        aggregate_rows.append(
            {
                "eps": eps,
                "eps_pixel": eps * 255.0,
                "attack_source": attack_source,
                "eval_model": eval_variant,
                "is_white_box": int(attack_source == eval_variant),
                "total": entry["total"],
                "clean_accuracy_eval_model": clean_accuracy,
                "adv_accuracy_all": adv_accuracy,
                "attack_success_all": 1.0 - adv_accuracy,
                "adv_accuracy_on_clean_correct_eval_model": clean_correct_adv_accuracy,
                "attack_success_on_clean_correct_eval_model": 1.0 - clean_correct_adv_accuracy,
                "source_success_count": entry["source_success"],
                "source_success_transfer_rate_all": safe_divide(entry["source_success_eval_wrong"], entry["source_success"]),
                "source_success_transfer_rate_eval_clean_correct": safe_divide(
                    entry["source_success_eval_clean_correct_wrong"],
                    entry["source_success_eval_clean_correct"],
                ),
                "toplogit_norm_margin_mean": margin_summary["mean"],
                "toplogit_norm_margin_median": margin_summary["median"],
                "toplogit_norm_margin_p25": margin_summary["p25"],
                "toplogit_norm_margin_p75": margin_summary["p75"],
            }
        )

    clean_rows = []
    for variant, entry in clean_stats.items():
        margins = torch.cat(entry["margins"], dim=0).numpy()
        margin_summary = summarize_distribution(margins)
        clean_rows.append(
            {
                "eval_model": variant,
                "total": entry["total"],
                "clean_accuracy": safe_divide(entry["correct"], entry["total"]),
                "clean_toplogit_norm_margin_mean": margin_summary["mean"],
                "clean_toplogit_norm_margin_median": margin_summary["median"],
                "clean_toplogit_norm_margin_p25": margin_summary["p25"],
                "clean_toplogit_norm_margin_p75": margin_summary["p75"],
            }
        )

    summary_rows = build_summary_rows(aggregate_rows, clean_rows, eps_values)
    return aggregate_rows, summary_rows, clean_rows, per_sample_rows


def row_lookup(aggregate_rows):
    return {
        (float(row["eps"]), row["attack_source"], row["eval_model"]): row
        for row in aggregate_rows
    }


def get_metric(lookup, eps, source, eval_model, metric):
    return lookup[(float(eps), source, eval_model)][metric]


def build_summary_rows(aggregate_rows, clean_rows, eps_values):
    lookup = row_lookup(aggregate_rows)
    clean_lookup = {row["eval_model"]: row for row in clean_rows}
    rows = []
    for eps in eps_values:
        original_on_original = get_metric(lookup, eps, VARIANT_ORIGINAL, VARIANT_ORIGINAL, "adv_accuracy_all")
        original_on_ours = get_metric(lookup, eps, VARIANT_ORIGINAL, VARIANT_HIRA_RANPAC, "adv_accuracy_all")
        ours_on_original = get_metric(lookup, eps, VARIANT_HIRA_RANPAC, VARIANT_ORIGINAL, "adv_accuracy_all")
        ours_on_ours = get_metric(lookup, eps, VARIANT_HIRA_RANPAC, VARIANT_HIRA_RANPAC, "adv_accuracy_all")
        rows.append(
            {
                "eps": eps,
                "eps_pixel": eps * 255.0,
                "clean_original_accuracy": clean_lookup[VARIANT_ORIGINAL]["clean_accuracy"],
                "clean_hira_ranpac_accuracy": clean_lookup[VARIANT_HIRA_RANPAC]["clean_accuracy"],
                "original_pgd_on_original_accuracy": original_on_original,
                "original_pgd_on_hira_ranpac_accuracy": original_on_ours,
                "hira_ranpac_pgd_on_original_accuracy": ours_on_original,
                "hira_ranpac_pgd_on_hira_ranpac_accuracy": ours_on_ours,
                "original_worst_source_accuracy": min(original_on_original, ours_on_original),
                "hira_ranpac_worst_source_accuracy": min(original_on_ours, ours_on_ours),
                "hira_ranpac_advantage_on_original_pgd": original_on_ours - original_on_original,
                "hira_ranpac_advantage_on_hira_ranpac_pgd": ours_on_ours - ours_on_original,
                "hira_ranpac_worst_source_advantage": min(original_on_ours, ours_on_ours) - min(original_on_original, ours_on_original),
                "original_to_hira_ranpac_transfer_rate_eval_clean_correct": get_metric(
                    lookup,
                    eps,
                    VARIANT_ORIGINAL,
                    VARIANT_HIRA_RANPAC,
                    "source_success_transfer_rate_eval_clean_correct",
                ),
                "hira_ranpac_to_original_transfer_rate_eval_clean_correct": get_metric(
                    lookup,
                    eps,
                    VARIANT_HIRA_RANPAC,
                    VARIANT_ORIGINAL,
                    "source_success_transfer_rate_eval_clean_correct",
                ),
                "original_pgd_on_original_toplogit_norm_margin_mean": get_metric(
                    lookup,
                    eps,
                    VARIANT_ORIGINAL,
                    VARIANT_ORIGINAL,
                    "toplogit_norm_margin_mean",
                ),
                "original_pgd_on_hira_ranpac_toplogit_norm_margin_mean": get_metric(
                    lookup,
                    eps,
                    VARIANT_ORIGINAL,
                    VARIANT_HIRA_RANPAC,
                    "toplogit_norm_margin_mean",
                ),
                "hira_ranpac_pgd_on_original_toplogit_norm_margin_mean": get_metric(
                    lookup,
                    eps,
                    VARIANT_HIRA_RANPAC,
                    VARIANT_ORIGINAL,
                    "toplogit_norm_margin_mean",
                ),
                "hira_ranpac_pgd_on_hira_ranpac_toplogit_norm_margin_mean": get_metric(
                    lookup,
                    eps,
                    VARIANT_HIRA_RANPAC,
                    VARIANT_HIRA_RANPAC,
                    "toplogit_norm_margin_mean",
                ),
            }
        )
    return rows


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


def series(rows, key):
    return np.asarray([row[key] for row in rows], dtype=np.float64)


def save_plots(run_dir, aggregate_rows, summary_rows, eps_values):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = series(summary_rows, "eps_pixel")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    axes[0].plot(x, series(summary_rows, "original_pgd_on_original_accuracy"), marker="o", label="eval original")
    axes[0].plot(x, series(summary_rows, "original_pgd_on_hira_ranpac_accuracy"), marker="o", label="eval HiRA+RanPAC")
    axes[0].set_title("PGD crafted on original")
    axes[0].set_xlabel("epsilon / 255")
    axes[0].set_ylabel("Accuracy on adversarial images")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(x, series(summary_rows, "hira_ranpac_pgd_on_original_accuracy"), marker="o", label="eval original")
    axes[1].plot(x, series(summary_rows, "hira_ranpac_pgd_on_hira_ranpac_accuracy"), marker="o", label="eval HiRA+RanPAC")
    axes[1].set_title("PGD crafted on HiRA+RanPAC")
    axes[1].set_xlabel("epsilon / 255")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.suptitle("PGD transferability accuracy")
    fig.tight_layout()
    fig.savefig(run_dir / "transfer_accuracy_vs_eps.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, series(summary_rows, "original_pgd_on_original_accuracy"), marker="o", linestyle="--", alpha=0.65, label="original white-box")
    ax.plot(x, series(summary_rows, "hira_ranpac_pgd_on_hira_ranpac_accuracy"), marker="o", linestyle="--", alpha=0.65, label="HiRA+RanPAC white-box")
    ax.plot(x, series(summary_rows, "original_worst_source_accuracy"), marker="s", label="original worst over both PGD sources")
    ax.plot(x, series(summary_rows, "hira_ranpac_worst_source_accuracy"), marker="s", label="HiRA+RanPAC worst over both PGD sources")
    ax.set_xlabel("epsilon / 255")
    ax.set_ylabel("Accuracy")
    ax.set_title("Worst-source transfer accuracy")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "worst_source_accuracy_vs_eps.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        x,
        series(summary_rows, "original_to_hira_ranpac_transfer_rate_eval_clean_correct"),
        marker="o",
        label="original PGD successes -> HiRA+RanPAC wrong",
    )
    ax.plot(
        x,
        series(summary_rows, "hira_ranpac_to_original_transfer_rate_eval_clean_correct"),
        marker="o",
        label="HiRA+RanPAC PGD successes -> original wrong",
    )
    ax.set_xlabel("epsilon / 255")
    ax.set_ylabel("Transfer success rate")
    ax.set_title("Transfer among source-success cases")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "transfer_success_rate_vs_eps.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        x,
        series(summary_rows, "original_pgd_on_hira_ranpac_toplogit_norm_margin_mean"),
        marker="o",
        label="HiRA+RanPAC on original PGD",
    )
    ax.plot(
        x,
        series(summary_rows, "hira_ranpac_pgd_on_hira_ranpac_toplogit_norm_margin_mean"),
        marker="o",
        label="HiRA+RanPAC on own PGD",
    )
    ax.plot(
        x,
        series(summary_rows, "original_pgd_on_original_toplogit_norm_margin_mean"),
        marker="o",
        linestyle="--",
        alpha=0.65,
        label="original on own PGD",
    )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_xlabel("epsilon / 255")
    ax.set_ylabel("Top-logit normalized margin")
    ax.set_title("Transferred-attack margin")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "transfer_toplogit_norm_margin_vs_eps.png", dpi=300)
    plt.close(fig)

    save_heatmap_grid(run_dir, aggregate_rows, eps_values)


def save_heatmap_grid(run_dir, aggregate_rows, eps_values):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lookup = row_lookup(aggregate_rows)
    num_eps = len(eps_values)
    cols = min(3, num_eps)
    rows = int(np.ceil(num_eps / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.1 * cols, 3.8 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")

    for index, eps in enumerate(eps_values):
        ax = axes.flat[index]
        matrix = np.asarray(
            [
                [
                    lookup[(float(eps), VARIANT_ORIGINAL, VARIANT_ORIGINAL)]["adv_accuracy_all"],
                    lookup[(float(eps), VARIANT_ORIGINAL, VARIANT_HIRA_RANPAC)]["adv_accuracy_all"],
                ],
                [
                    lookup[(float(eps), VARIANT_HIRA_RANPAC, VARIANT_ORIGINAL)]["adv_accuracy_all"],
                    lookup[(float(eps), VARIANT_HIRA_RANPAC, VARIANT_HIRA_RANPAC)]["adv_accuracy_all"],
                ],
            ],
            dtype=np.float64,
        )
        ax.axis("on")
        image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_title(f"eps={eps * 255.0:g}/255")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["eval\noriginal", "eval\nHiRA+RanPAC"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["attack\noriginal", "attack\nHiRA+RanPAC"])
        for row_index in range(2):
            for col_index in range(2):
                value = matrix[row_index, col_index]
                color = "white" if value < 0.5 else "black"
                ax.text(col_index, row_index, f"{value:.3f}", ha="center", va="center", color=color, fontsize=10)
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.82, label="accuracy")
    fig.suptitle("PGD transferability matrix")
    fig.savefig(run_dir / "transfer_accuracy_heatmaps.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="PGD transferability visualization for original vs HiRA+RanPAC RobustBench models.")
    parser.add_argument("--model-name", "--model_name", required=True, help="RobustBench ImageNet model name.")
    parser.add_argument("--threat-model", "--threat_model", default="Linf", choices=["Linf", "L2"], help="Threat model used to load the RobustBench model. Only Linf PGD is implemented here.")
    parser.add_argument("--data-dir", "--data_dir", default="./dataset/imagenet", help="ImageNet root containing train/ and val/.")
    parser.add_argument("--model-dir", "--model_dir", default="./robustbench_models", help="RobustBench checkpoint cache directory.")
    parser.add_argument("--device", default="cuda:0", help="Device, e.g. cuda:0 or cpu.")
    parser.add_argument("--seed", type=int, default=0, help="Global random seed and class-selection seed.")
    parser.add_argument("--batch-size", "--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--num-workers", "--num_workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--output-dir", "--output_dir", default="visualization/transferability_outputs", help="Directory where outputs are saved.")
    parser.add_argument("--run-name", "--run_name", default="", help="Optional output subdirectory name.")
    parser.add_argument("--num-classes", "--num_classes", type=int, default=20, help="Number of ImageNet classes to sample when --class-ids is empty.")
    parser.add_argument("--samples-per-class", "--samples_per_class", type=int, default=50, help="Number of validation images per selected class.")
    parser.add_argument("--class-ids", "--class_ids", default="", help="Optional comma-separated ImageNet class IDs. Overrides --num-classes.")
    parser.add_argument("--eps-list", "--eps_list", default="0,1/255,2/255,4/255,8/255,16/255", help="Comma-separated Linf eps values; fractions like 4/255 are accepted.")
    parser.add_argument("--pgd-steps", "--pgd_steps", type=int, default=40, help="PGD steps used to craft each source attack.")
    parser.add_argument("--pgd-step-size", "--pgd_step_size", type=parse_float_or_fraction, default=None, help="Optional fixed PGD step size. Defaults to 2 * eps / steps for each eps.")
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
    parser.add_argument("--hira-subspace-rank", "--hira_subspace_rank", type=int, default=0)
    parser.add_argument("--hira-subspace-shrink", "--hira_subspace_shrink", type=float, default=1.0)
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

    eps_name = sanitize_name("-".join(str(eps) for eps in eps_values))
    run_name = args.run_name or f"{sanitize_name(args.model_name)}_classes{len(selected_class_ids)}_n{args.samples_per_class}_eps{eps_name}_steps{args.pgd_steps}_seed{args.seed}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving transferability outputs to: {run_dir}")
    print(f"Selected classes: {selected_class_ids}")

    print("Loading original RobustBench model...")
    original_model = freeze_model(load_robustbench_model(args.model_name, args.threat_model, args.model_dir, device))
    print("Loading and wrapping RobustBench model with HiRA+RanPAC...")
    ours_model = freeze_model(build_hira_ranpac_model(args, model_preprocessing, device))

    aggregate_rows, summary_rows, clean_rows, per_sample_rows = collect_transferability(
        original_model,
        ours_model,
        loader,
        selected_indices,
        eps_values,
        device,
        args,
    )

    write_csv(run_dir / "aggregate_transfer_metrics.csv", aggregate_rows)
    write_csv(run_dir / "transfer_summary.csv", summary_rows)
    write_csv(run_dir / "clean_metrics.csv", clean_rows)
    write_csv(run_dir / "per_sample_transfer_metrics.csv", per_sample_rows)
    save_plots(run_dir, aggregate_rows, summary_rows, eps_values)

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
        "outputs": [
            "transfer_accuracy_vs_eps.png",
            "worst_source_accuracy_vs_eps.png",
            "transfer_success_rate_vs_eps.png",
            "transfer_toplogit_norm_margin_vs_eps.png",
            "transfer_accuracy_heatmaps.png",
            "aggregate_transfer_metrics.csv",
            "transfer_summary.csv",
            "clean_metrics.csv",
            "per_sample_transfer_metrics.csv",
        ],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
