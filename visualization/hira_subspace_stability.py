#!/usr/bin/env python3
"""
HiRA principal-vs-orthogonal subspace stability under PGD.

This script measures whether HiRA clean principal components are more stable
than orthogonal components by comparing clean vs PGD hidden projected features
inside all HiRA adapters. It reports relative/absolute drift and linear CKA for
principal and orthogonal components.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from classifiers.mean_sparse import DEFAULT_MEANSPARSE_STAT_EPS
from classifiers.stability_ridge import DEFAULT_STABILITY_RIDGE_STAT_EPS
from visualization.hira_subspace_intervention import (
    adapter_project_without_clean_subspace,
    find_hira_adapters,
    pgd_linf_attack,
    split_parallel_orthogonal,
)
from visualization.tsne_robustbench_ranpac import (
    DATASET,
    build_eval_loader,
    build_hira_ranpac_model,
    build_imagenet_dataset,
    freeze_model,
    parse_float_or_fraction,
    resolve_device,
    resolve_model_preprocessing,
    sanitize_name,
    select_all_indices,
    set_seed,
    str2bool,
)


def parse_eps_list(value):
    return [parse_float_or_fraction(item.strip()) for item in str(value).split(",") if item.strip()]


def build_eps_tag(eps_values):
    eps_pixels = [eps * 255.0 for eps in eps_values]
    return f"n{len(eps_values)}_min{sanitize_name(min(eps_pixels))}_max{sanitize_name(max(eps_pixels))}"


def flatten_batch(tensor, batch_size):
    return tensor.detach().view(batch_size, -1).float()


def linear_cka(clean_features, adv_features, max_dims=4096, seed=0, eps=1e-12):
    clean_features = np.asarray(clean_features, dtype=np.float32)
    adv_features = np.asarray(adv_features, dtype=np.float32)
    if clean_features.shape != adv_features.shape:
        raise ValueError("CKA requires feature matrices with the same shape.")
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


def collect_components(model, adapters, inputs):
    store = defaultdict(dict)
    handles = []

    def make_hook(name):
        def hook(module, hook_inputs, hook_output):
            del hook_output
            batch_size = hook_inputs[0].shape[0]
            projected = adapter_project_without_clean_subspace(module, hook_inputs[0])
            _, parallel, orthogonal = split_parallel_orthogonal(module, projected)
            store[name]["parallel"] = flatten_batch(parallel, batch_size).cpu()
            store[name]["orthogonal"] = flatten_batch(orthogonal, batch_size).cpu()
        return hook

    for name, adapter in adapters:
        handles.append(adapter.register_forward_hook(make_hook(name)))
    try:
        with torch.no_grad():
            logits = model(inputs).detach().float()
    finally:
        for handle in handles:
            handle.remove()
    return logits, store


def update_subspace_sums(sums, clean_store, adv_store, adapter_names):
    batch_size = next(iter(clean_store.values()))["parallel"].shape[0]
    device = clean_store[adapter_names[0]]["parallel"].device
    batch_sums = {
        "parallel_clean_norm_sq": torch.zeros(batch_size, dtype=torch.float64, device=device),
        "parallel_diff_norm_sq": torch.zeros(batch_size, dtype=torch.float64, device=device),
        "orthogonal_clean_norm_sq": torch.zeros(batch_size, dtype=torch.float64, device=device),
        "orthogonal_diff_norm_sq": torch.zeros(batch_size, dtype=torch.float64, device=device),
    }
    for adapter_name in adapter_names:
        clean_parallel = clean_store[adapter_name]["parallel"].double()
        adv_parallel = adv_store[adapter_name]["parallel"].double()
        clean_orthogonal = clean_store[adapter_name]["orthogonal"].double()
        adv_orthogonal = adv_store[adapter_name]["orthogonal"].double()
        batch_sums["parallel_clean_norm_sq"] += torch.sum(clean_parallel * clean_parallel, dim=1)
        batch_sums["parallel_diff_norm_sq"] += torch.sum((adv_parallel - clean_parallel) ** 2, dim=1)
        batch_sums["orthogonal_clean_norm_sq"] += torch.sum(clean_orthogonal * clean_orthogonal, dim=1)
        batch_sums["orthogonal_diff_norm_sq"] += torch.sum((adv_orthogonal - clean_orthogonal) ** 2, dim=1)
    for key, value in batch_sums.items():
        sums[key].append(value.cpu().numpy())


def update_cka_samples(samples, clean_store, adv_store, adapter_names, sample_cap, seed):
    if sample_cap <= 0:
        return
    for component in ("parallel", "orthogonal"):
        clean_parts = []
        adv_parts = []
        for adapter_name in adapter_names:
            clean_parts.append(clean_store[adapter_name][component])
            adv_parts.append(adv_store[adapter_name][component])
        clean = torch.cat(clean_parts, dim=1).numpy()
        adv = torch.cat(adv_parts, dim=1).numpy()
        num_dims = clean.shape[1]
        max_dims = min(int(sample_cap), num_dims)
        if max_dims < num_dims:
            rng = np.random.default_rng(seed + (0 if component == "parallel" else 1000003))
            indices = np.sort(rng.choice(num_dims, size=max_dims, replace=False))
            clean = clean[:, indices]
            adv = adv[:, indices]
        samples[f"clean_{component}"].append(clean.astype(np.float32, copy=False))
        samples[f"adv_{component}"].append(adv.astype(np.float32, copy=False))


def collect_stability(model, loader, selected_indices, eps_values, device, args):
    del selected_indices
    adapters = find_hira_adapters(model)
    adapter_names = [name for name, _ in adapters]
    print(f"Found {len(adapters)} HiRA adapters for subspace stability.")
    for name, adapter in adapters:
        print(
            f"  {name}: subspace_rank={adapter.subspace_rank}, "
            f"valid_rank={int(adapter.clean_subspace_valid_rank.item())}, "
            f"subspace_shrink={adapter.subspace_shrink}"
        )

    rows = []
    per_sample_rows = []
    step_sizes = {
        eps: args.pgd_step_size if args.pgd_step_size is not None else 2.0 * eps / max(args.pgd_steps, 1)
        for eps in eps_values
    }

    for eps in eps_values:
        drift_sums = defaultdict(list)
        cka_samples = defaultdict(list)
        label_batches = []
        clean_prediction_batches = []
        adv_prediction_batches = []
        progress = tqdm(loader, desc=f"HiRA subspace stability eps={eps * 255.0:g}/255", dynamic_ncols=True)
        for inputs, labels in progress:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            clean_logits, clean_store = collect_components(model, adapters, inputs)
            adv_inputs = pgd_linf_attack(
                model,
                inputs,
                labels,
                eps=eps,
                steps=args.pgd_steps,
                step_size=step_sizes[eps],
                random_start=args.pgd_random_start,
                attack_class_ids=getattr(args, "attack_class_ids", None),
            )
            adv_logits, adv_store = collect_components(model, adapters, adv_inputs)
            update_subspace_sums(drift_sums, clean_store, adv_store, adapter_names)
            update_cka_samples(cka_samples, clean_store, adv_store, adapter_names, args.cka_max_dims, args.seed)
            label_batches.append(labels.detach().cpu())
            clean_prediction_batches.append(clean_logits.argmax(dim=1).detach().cpu())
            adv_prediction_batches.append(adv_logits.argmax(dim=1).detach().cpu())

        labels = torch.cat(label_batches, dim=0).numpy()
        clean_predictions = torch.cat(clean_prediction_batches, dim=0).numpy()
        adv_predictions = torch.cat(adv_prediction_batches, dim=0).numpy()
        parallel_clean_norm_sq = np.concatenate(drift_sums["parallel_clean_norm_sq"], axis=0)
        parallel_diff_norm_sq = np.concatenate(drift_sums["parallel_diff_norm_sq"], axis=0)
        orthogonal_clean_norm_sq = np.concatenate(drift_sums["orthogonal_clean_norm_sq"], axis=0)
        orthogonal_diff_norm_sq = np.concatenate(drift_sums["orthogonal_diff_norm_sq"], axis=0)
        parallel_absolute = np.sqrt(parallel_diff_norm_sq)
        orthogonal_absolute = np.sqrt(orthogonal_diff_norm_sq)
        parallel_relative = parallel_absolute / np.maximum(np.sqrt(parallel_clean_norm_sq), 1e-12)
        orthogonal_relative = orthogonal_absolute / np.maximum(np.sqrt(orthogonal_clean_norm_sq), 1e-12)
        ratio = orthogonal_relative / np.maximum(parallel_relative, 1e-12)
        clean_parallel = np.concatenate(cka_samples["clean_parallel"], axis=0)
        adv_parallel = np.concatenate(cka_samples["adv_parallel"], axis=0)
        clean_orthogonal = np.concatenate(cka_samples["clean_orthogonal"], axis=0)
        adv_orthogonal = np.concatenate(cka_samples["adv_orthogonal"], axis=0)
        parallel_cka = linear_cka(clean_parallel, adv_parallel, max_dims=args.cka_max_dims, seed=args.seed)
        orthogonal_cka = linear_cka(clean_orthogonal, adv_orthogonal, max_dims=args.cka_max_dims, seed=args.seed + 1)

        row = {
            "eps": eps,
            "eps_pixel": eps * 255.0,
            "clean_accuracy": float(np.mean(clean_predictions == labels)),
            "adv_accuracy": float(np.mean(adv_predictions == labels)),
            "parallel_cka": parallel_cka,
            "orthogonal_cka": orthogonal_cka,
            "cka_gap_parallel_minus_orthogonal": parallel_cka - orthogonal_cka,
        }
        for prefix, values in (
            ("parallel_relative_drift", parallel_relative),
            ("orthogonal_relative_drift", orthogonal_relative),
            ("parallel_absolute_drift", parallel_absolute),
            ("orthogonal_absolute_drift", orthogonal_absolute),
            ("orthogonal_over_parallel_relative_drift", ratio),
        ):
            for key, value in summarize_distribution(values).items():
                row[f"{prefix}_{key}"] = value
        rows.append(row)

        for index in range(labels.shape[0]):
            per_sample_rows.append(
                {
                    "eps": eps,
                    "eps_pixel": eps * 255.0,
                    "sample_row": index,
                    "label": int(labels[index]),
                    "clean_prediction": int(clean_predictions[index]),
                    "adv_prediction": int(adv_predictions[index]),
                    "clean_correct": int(clean_predictions[index] == labels[index]),
                    "adv_correct": int(adv_predictions[index] == labels[index]),
                    "parallel_relative_drift": float(parallel_relative[index]),
                    "orthogonal_relative_drift": float(orthogonal_relative[index]),
                    "parallel_absolute_drift": float(parallel_absolute[index]),
                    "orthogonal_absolute_drift": float(orthogonal_absolute[index]),
                    "orthogonal_over_parallel_relative_drift": float(ratio[index]),
                }
            )

    return rows, per_sample_rows, adapters


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def series(rows, key):
    return np.asarray([row[key] for row in rows], dtype=np.float64)


def plot_with_iqr(ax, rows, prefix, label):
    x = series(rows, "eps_pixel")
    ax.plot(x, series(rows, f"{prefix}_mean"), marker="o", label=label)
    ax.fill_between(x, series(rows, f"{prefix}_p25"), series(rows, f"{prefix}_p75"), alpha=0.15)


def save_plots(run_dir, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = series(rows, "eps_pixel")

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_with_iqr(ax, rows, "parallel_relative_drift", "principal")
    plot_with_iqr(ax, rows, "orthogonal_relative_drift", "orthogonal")
    ax.set_xlabel("PGD epsilon (pixel / 255)")
    ax.set_ylabel("Relative L2 drift")
    ax.set_title("HiRA subspace relative drift")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "hira_subspace_relative_drift_vs_eps.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_with_iqr(ax, rows, "parallel_absolute_drift", "principal")
    plot_with_iqr(ax, rows, "orthogonal_absolute_drift", "orthogonal")
    ax.set_xlabel("PGD epsilon (pixel / 255)")
    ax.set_ylabel("Absolute L2 drift")
    ax.set_title("HiRA subspace absolute drift")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "hira_subspace_absolute_drift_vs_eps.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, series(rows, "parallel_cka"), marker="o", label="principal")
    ax.plot(x, series(rows, "orthogonal_cka"), marker="o", label="orthogonal")
    ax.set_xlabel("PGD epsilon (pixel / 255)")
    ax.set_ylabel("Linear CKA(clean, PGD)")
    ax.set_title("HiRA subspace CKA stability")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "hira_subspace_cka_vs_eps.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_with_iqr(ax, rows, "orthogonal_over_parallel_relative_drift", "orthogonal / principal")
    ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_xlabel("PGD epsilon (pixel / 255)")
    ax.set_ylabel("Relative drift ratio")
    ax.set_title("Orthogonal/principal relative drift ratio")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "hira_subspace_drift_ratio_vs_eps.png", dpi=300)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Measure clean-vs-PGD HiRA principal/orthogonal subspace drift and CKA.")
    parser.add_argument("--model-name", "--model_name", required=True)
    parser.add_argument("--threat-model", "--threat_model", default="Linf", choices=["Linf", "L2"])
    parser.add_argument("--data-dir", "--data_dir", default="./dataset/imagenet")
    parser.add_argument("--model-dir", "--model_dir", default="./robustbench_models")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", "--batch_size", type=int, default=16)
    parser.add_argument("--num-workers", "--num_workers", type=int, default=4)
    parser.add_argument("--output-dir", "--output_dir", default="visualization/hira_subspace_stability_outputs")
    parser.add_argument("--run-name", "--run_name", default="")
    parser.add_argument("--eval-examples", "--eval_examples", type=int, default=5000)
    parser.add_argument("--num-classes", "--num_classes", type=int, default=20, help="Deprecated; visualization now uses the RobustBench evaluation subset.")
    parser.add_argument("--samples-per-class", "--samples_per_class", type=int, default=50, help="Deprecated; visualization now uses --eval-examples.")
    parser.add_argument("--class-ids", "--class_ids", default="", help="Deprecated; visualization now uses the loaded RobustBench classes.")
    parser.add_argument("--eps-list", "--eps_list", default="0,1/255,2/255,4/255,8/255,16/255")
    parser.add_argument("--pgd-steps", "--pgd_steps", type=int, default=40)
    parser.add_argument("--pgd-step-size", "--pgd_step_size", type=parse_float_or_fraction, default=None)
    parser.add_argument("--pgd-random-start", "--pgd_random_start", type=str2bool, default=True)
    parser.add_argument("--mask-pgd-logits", "--mask_pgd_logits", type=str2bool, default=False)
    parser.add_argument("--cka-max-dims", "--cka_max_dims", type=int, default=4096)

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
    dataset = build_imagenet_dataset(args.data_dir, model_preprocessing, n_examples=args.eval_examples)
    selected_indices, selected_class_ids = select_all_indices(dataset)
    args.attack_class_ids = selected_class_ids if args.mask_pgd_logits else None
    loader = build_eval_loader(dataset, selected_indices, args.batch_size, args.num_workers)

    eps_name = build_eps_tag(eps_values)
    run_name = args.run_name or f"{sanitize_name(args.model_name)}_examples{len(selected_indices)}_eps{eps_name}_steps{args.pgd_steps}_seed{args.seed}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving HiRA subspace stability outputs to: {run_dir}")
    print(f"Loaded RobustBench examples: {len(selected_indices)}")
    print(f"Loaded classes: {selected_class_ids}")

    print("Loading HiRA+RanPAC model...")
    model = freeze_model(build_hira_ranpac_model(args, model_preprocessing, device))
    rows, per_sample_rows, adapters = collect_stability(model, loader, selected_indices, eps_values, device, args)
    write_csv(run_dir / "hira_subspace_stability_metrics.csv", rows)
    write_csv(run_dir / "hira_subspace_stability_per_sample.csv", per_sample_rows)
    save_plots(run_dir, rows)

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
        "eval_examples": args.eval_examples,
        "selected_class_ids": selected_class_ids,
        "num_samples": len(selected_indices),
        "num_classes": len(selected_class_ids),
        "eps_values": eps_values,
        "eps_pixels": [eps * 255.0 for eps in eps_values],
        "pgd_steps": args.pgd_steps,
        "pgd_step_size": args.pgd_step_size,
        "pgd_random_start": args.pgd_random_start,
        "mask_pgd_logits": args.mask_pgd_logits,
        "cka_max_dims": args.cka_max_dims,
        "soft_threshold_alpha": args.soft_threshold_alpha,
        "soft_threshold_beta": args.soft_threshold_beta,
        "soft_threshold_mode": args.soft_threshold_mode,
        "hira_subspace_rank": args.hira_subspace_rank,
        "hira_subspace_shrink": args.hira_subspace_shrink,
        "hira_adapters": adapter_summary,
        "outputs": [
            "hira_subspace_relative_drift_vs_eps.png",
            "hira_subspace_absolute_drift_vs_eps.png",
            "hira_subspace_cka_vs_eps.png",
            "hira_subspace_drift_ratio_vs_eps.png",
            "hira_subspace_stability_metrics.csv",
            "hira_subspace_stability_per_sample.csv",
        ],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
