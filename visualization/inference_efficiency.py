#!/usr/bin/env python3
"""
Forward-time benchmark for ImageNet victims and purification pipelines.

This script measures inference-time overhead only. It times forward passes on the
same RobustBench 5000-example ImageNet subset used by the RobustBench evaluation
scripts, and compares each backbone/purifier with and without HiRA+RanPAC.
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from classifiers.hira import _attach_hira_modules, _freeze_model, _prepare_hira_model_for_eval, _seed_everything
from classifiers.ranpac import (
    RANPAC_CACHE_VERSION,
    RanPACLinear,
    ResidualRanPACLinear,
    _find_last_linear,
    _iter_ranpac_cache_candidate_paths,
)
from purifiers import PurifiedClassifier, build_purifier
from victims import build_imagenet_victim, build_wrapper_config_from_namespace, supports_hira
from visualization.tsne_robustbench_ranpac import (
    DATASET,
    build_imagenet_dataset,
    parse_float_or_fraction,
    resolve_device,
    sanitize_name,
    select_all_indices,
    set_seed,
    str2bool,
)


def parse_csv(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


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


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def cuda_peak_memory_mb(device):
    if device.type != "cuda":
        return float("nan")
    return float(torch.cuda.max_memory_allocated(device) / (1024.0 ** 2))


def reset_cuda_peak_memory(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def build_robustbench_like_dataset(data_dir, eval_examples):
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    return build_imagenet_dataset(data_dir, transform, n_examples=eval_examples)


def build_loader(dataset, batch_size, num_workers):
    indices, class_ids = select_all_indices(dataset)
    return DataLoader(
        dataset,
        batch_size=max(int(batch_size), 1),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    ), indices, class_ids


def make_variant_args(args, classifier, purifier_name, use_ours):
    namespace = SimpleNamespace(**vars(args))
    namespace.classifier = classifier
    namespace.purifier_name = purifier_name
    namespace.use_hira_adapter = bool(use_ours)
    namespace.use_ranpac_head = bool(use_ours)
    if str(purifier_name).lower() == "instantpure":
        strength = float(namespace.strength)
        num_steps = int(namespace.num_inference_step)
        if strength <= 0:
            raise ValueError("InstantPure strength must be positive.")
        if int(num_steps * strength) < 1:
            adjusted_steps = max(num_steps, int(math.ceil(1.0 / strength)))
            print(
                "InstantPure img2img would use zero denoising steps with "
                f"num_inference_step={num_steps}, strength={strength}; "
                f"using num_inference_step={adjusted_steps} for timing."
            )
            namespace.num_inference_step = adjusted_steps
    return namespace


def _load_cached_hira_if_available(model, classifier_name, wrapper_config):
    if wrapper_config.hira_force_retrain:
        return False
    from classifiers.hira import HIRA_CACHE_VERSION, _build_cache_name

    cache_path = os.path.join(
        wrapper_config.hira_cache_dir,
        _build_cache_name(
            classifier_name=classifier_name,
            expansion_dim=wrapper_config.hira_expansion_dim,
            epochs=wrapper_config.hira_epochs,
            lr=wrapper_config.hira_lr,
            weight_decay=wrapper_config.hira_weight_decay,
            max_train_samples=wrapper_config.hira_max_train_samples,
            seed=wrapper_config.hira_seed,
            num_adapter_blocks=wrapper_config.hira_num_blocks,
            adapt_noise_eps=wrapper_config.adapt_noise_eps,
            adapt_noise_num=wrapper_config.adapt_noise_num,
            adapt_alpha=wrapper_config.adapt_alpha,
            soft_threshold_alpha=wrapper_config.soft_threshold_alpha,
            soft_threshold_beta=wrapper_config.soft_threshold_beta,
            soft_threshold_stat_eps=wrapper_config.soft_threshold_stat_eps,
            soft_threshold_mode=wrapper_config.soft_threshold_mode,
            subspace_rank=wrapper_config.hira_subspace_rank,
            subspace_shrink=wrapper_config.hira_subspace_shrink,
            stability_ridge_gamma=wrapper_config.stability_ridge_gamma,
            stability_ridge_stat_eps=wrapper_config.stability_ridge_stat_eps,
        ),
    )
    if not os.path.exists(cache_path):
        return False
    state = torch.load(cache_path, map_location="cpu")
    if state.get("version") != HIRA_CACHE_VERSION or "hira_state" not in state:
        return False
    model.load_state_dict(state["hira_state"], strict=False)
    print(f"Loaded cached HiRA weights for timing: {cache_path}")
    return True


def _apply_hira_cache_or_random(model, classifier_name, wrapper_config, supports_hira_arch):
    if not supports_hira_arch:
        raise ValueError("HiRA timing is only supported for ViT-family backbones.")
    _seed_everything(wrapper_config.hira_seed)
    _attach_hira_modules(
        model,
        wrapper_config.hira_expansion_dim,
        wrapper_config.hira_num_blocks,
        soft_threshold_alpha=wrapper_config.soft_threshold_alpha,
        soft_threshold_beta=wrapper_config.soft_threshold_beta,
        soft_threshold_stat_eps=wrapper_config.soft_threshold_stat_eps,
        soft_threshold_mode=wrapper_config.soft_threshold_mode,
        subspace_rank=wrapper_config.hira_subspace_rank,
        subspace_shrink=wrapper_config.hira_subspace_shrink,
    )
    _freeze_model(model)
    loaded = _load_cached_hira_if_available(model, classifier_name, wrapper_config)
    if not loaded:
        print("HiRA cache missing for timing; using random HiRA weights and default calibration statistics.")
    return _prepare_hira_model_for_eval(model), loaded


def _load_cached_ranpac_state_if_available(classifier_name, wrapper_config):
    candidate_paths = list(
        _iter_ranpac_cache_candidate_paths(
            cache_dir=wrapper_config.ranpac_cache_dir,
            classifier_name=classifier_name,
            rp_dim=wrapper_config.ranpac_rp_dim,
            seed=wrapper_config.ranpac_seed,
            adapt_noise_eps=wrapper_config.adapt_noise_eps,
            adapt_noise_num=wrapper_config.adapt_noise_num,
            adapt_alpha=wrapper_config.adapt_alpha,
            hardneg_topk=wrapper_config.ranpac_hardneg_topk,
            hardneg_gamma=wrapper_config.ranpac_hardneg_gamma,
            stability_ridge_gamma=wrapper_config.stability_ridge_gamma,
            stability_ridge_stat_eps=wrapper_config.stability_ridge_stat_eps,
        )
    )
    for cache_path in candidate_paths:
        if not os.path.exists(cache_path):
            continue
        state = torch.load(cache_path, map_location="cpu")
        if state.get("version") == RANPAC_CACHE_VERSION and "weight" in state and "w_rand" in state:
            print(f"Loaded cached RanPAC weights for timing: {cache_path}")
            return state, True
    return None, False


def _random_ranpac_state(linear_layer, wrapper_config):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(wrapper_config.ranpac_seed))
    w_rand = torch.randn(
        linear_layer.in_features,
        wrapper_config.ranpac_rp_dim,
        generator=generator,
        dtype=torch.float32,
    ) / max(float(linear_layer.in_features) ** 0.5, 1.0)
    weight = torch.empty(
        linear_layer.out_features,
        wrapper_config.ranpac_rp_dim,
        dtype=torch.float32,
    )
    torch.nn.init.kaiming_uniform_(weight, a=5 ** 0.5)
    return {
        "version": RANPAC_CACHE_VERSION,
        "layer_name": None,
        "in_features": linear_layer.in_features,
        "out_features": linear_layer.out_features,
        "rp_dim": wrapper_config.ranpac_rp_dim,
        "weight": weight,
        "w_rand": w_rand,
    }


def _apply_ranpac_cache_or_random(model, classifier_name, wrapper_config):
    layer_name, linear_layer = _find_last_linear(model)
    state, loaded = _load_cached_ranpac_state_if_available(classifier_name, wrapper_config)
    if state is None:
        print("RanPAC cache missing for timing; using random RanPAC head weights.")
        state = _random_ranpac_state(linear_layer, wrapper_config)
    elif state.get("layer_name") != layer_name:
        print(
            f"Cached RanPAC layer name {state.get('layer_name')} differs from current {layer_name}; "
            "using cache anyway because dimensions match."
        )
    if state["in_features"] != linear_layer.in_features or state["out_features"] != linear_layer.out_features:
        print("Cached RanPAC dimensions do not match current classifier; using random RanPAC head weights.")
        state = _random_ranpac_state(linear_layer, wrapper_config)
        loaded = False
    ranpac_branch = RanPACLinear(
        in_features=state["in_features"],
        out_features=state["out_features"],
        rp_dim=state["rp_dim"],
        weight=state["weight"],
        w_rand=state["w_rand"],
    )
    ranpac_head = ResidualRanPACLinear(
        original_linear=linear_layer,
        ranpac_linear=ranpac_branch,
        ranpac_lambda=wrapper_config.ranpac_lambda,
        ranpac_temp=wrapper_config.ranpac_temp,
    )
    parent = model
    parts = layer_name.split(".")
    for attr in parts[:-1]:
        parent = getattr(parent, attr)
    setattr(parent, parts[-1], ranpac_head)
    return model, loaded


def apply_wrappers_cache_or_random(classifier, classifier_name, victim_spec, wrapper_config):
    wrapped_name = classifier_name
    hira_loaded = None
    ranpac_loaded = None
    if wrapper_config.use_hira:
        classifier, hira_loaded = _apply_hira_cache_or_random(
            classifier,
            classifier_name=wrapped_name,
            wrapper_config=wrapper_config,
            supports_hira_arch=supports_hira(victim_spec),
        )
        wrapped_name = f"{wrapped_name}-hira-timing"
    if wrapper_config.use_ranpac:
        classifier, ranpac_loaded = _apply_ranpac_cache_or_random(
            classifier,
            classifier_name=wrapped_name,
            wrapper_config=wrapper_config,
        )
        wrapped_name = f"{wrapped_name}-ranpac-timing"
    return classifier, wrapped_name, hira_loaded, ranpac_loaded


def build_classifier(args, classifier_name, use_ours, device):
    classifier, base_classifier_name, victim_spec = build_imagenet_victim(classifier_name, pretrained=True)
    wrapper_source = "none"
    hira_cache_loaded = None
    ranpac_cache_loaded = None
    if use_ours:
        wrapper_args = deepcopy(args)
        wrapper_args.use_hira_adapter = True
        wrapper_args.use_ranpac_head = True
        wrapper_config = build_wrapper_config_from_namespace(wrapper_args, dataset="imagenet")
        classifier, wrapped_name, hira_cache_loaded, ranpac_cache_loaded = apply_wrappers_cache_or_random(
            classifier,
            base_classifier_name,
            victim_spec,
            wrapper_config,
        )
        wrapper_source = "cache_or_random"
    else:
        wrapped_name = base_classifier_name
    return classifier.to(device).eval(), wrapped_name, victim_spec, wrapper_source, hira_cache_loaded, ranpac_cache_loaded


def build_timed_model(args, classifier_name, purifier_name, use_ours, device):
    variant_args = make_variant_args(args, classifier_name, purifier_name, use_ours)
    classifier, wrapped_name, victim_spec, wrapper_source, hira_cache_loaded, ranpac_cache_loaded = build_classifier(variant_args, classifier_name, use_ours, device)
    if purifier_name == "none":
        model = classifier
        purifier_label = "none"
    else:
        purifier = build_purifier(variant_args, device).to(device).eval()
        model = PurifiedClassifier(purifier, classifier).to(device).eval()
        purifier_label = purifier_name
    return model, wrapped_name, victim_spec, purifier_label, wrapper_source, hira_cache_loaded, ranpac_cache_loaded


def time_model(model, loader, device, args, variant_name):
    model.eval()
    warmup_batches = max(int(args.warmup_batches), 0)
    max_batches = None if args.max_batches <= 0 else int(args.max_batches)
    timed_batches = []
    batch_rows = []
    total_samples = 0

    reset_cuda_peak_memory(device)
    with torch.no_grad():
        if warmup_batches > 0:
            warmup_progress = tqdm(
                enumerate(loader),
                total=min(warmup_batches, len(loader)),
                desc=f"warmup {variant_name}",
                dynamic_ncols=True,
            )
            for warmup_index, (inputs, _) in warmup_progress:
                if warmup_index >= warmup_batches:
                    break
                inputs = inputs.to(device, non_blocking=True)
                synchronize(device)
                _ = model(inputs)
                synchronize(device)

        progress = tqdm(loader, desc=f"timing {variant_name}", dynamic_ncols=True)
        for batch_index, (inputs, _) in enumerate(progress):
            if max_batches is not None and batch_index >= max_batches:
                break
            inputs = inputs.to(device, non_blocking=True)
            synchronize(device)
            start = time.perf_counter()
            _ = model(inputs)
            synchronize(device)
            elapsed = time.perf_counter() - start
            batch_size = int(inputs.shape[0])
            timed_batches.append(elapsed)
            total_samples += batch_size
            batch_rows.append(
                {
                    "variant": variant_name,
                    "batch_index": batch_index,
                    "timed_batch_index": len(timed_batches) - 1,
                    "batch_size": batch_size,
                    "forward_time_sec": elapsed,
                    "sec_per_sample": elapsed / max(batch_size, 1),
                }
            )
            progress.set_postfix(samples=total_samples, sec_per_img=f"{elapsed / max(batch_size, 1):.4f}")

    total_time = float(np.sum(timed_batches)) if timed_batches else 0.0
    sec_per_sample = total_time / max(total_samples, 1)
    batch_times = np.asarray(timed_batches, dtype=np.float64)
    return {
        "timed_samples": int(total_samples),
        "timed_batches": int(len(timed_batches)),
        "total_forward_time_sec": total_time,
        "sec_per_sample": sec_per_sample,
        "samples_per_sec": float(total_samples / max(total_time, 1e-12)),
        "batch_time_mean_sec": float(np.mean(batch_times)) if batch_times.size else float("nan"),
        "batch_time_median_sec": float(np.median(batch_times)) if batch_times.size else float("nan"),
        "batch_time_p25_sec": float(np.percentile(batch_times, 25)) if batch_times.size else float("nan"),
        "batch_time_p75_sec": float(np.percentile(batch_times, 75)) if batch_times.size else float("nan"),
        "peak_cuda_memory_mb": cuda_peak_memory_mb(device),
    }, batch_rows


def add_common_args(parser):
    parser.add_argument("--data-dir", "--data_dir", default="./dataset/imagenet")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-examples", "--eval_examples", type=int, default=5000)
    parser.add_argument("--batch-size", "--batch_size", type=int, default=32)
    parser.add_argument("--purifier-batch-size", "--purifier_batch_size", type=int, default=1)
    parser.add_argument("--num-workers", "--num_workers", type=int, default=4)
    parser.add_argument("--warmup-batches", "--warmup_batches", type=int, default=2)
    parser.add_argument("--max-batches", "--max_batches", type=int, default=-1, help="Optional cap on timed batches after warmup; <=0 times all loaded samples.")
    parser.add_argument("--output-dir", "--output_dir", default="visualization/inference_efficiency_outputs")
    parser.add_argument("--run-name", "--run_name", default="")
    parser.add_argument("--classifiers", default="swin_b,vit_base", help="Comma-separated timm/ImageNet victim aliases for non-purified timing.")
    parser.add_argument("--purifier-classifier", "--purifier_classifier", default="vit_base", help="Victim backbone used behind purifier pipelines.")
    parser.add_argument("--purifiers", default="mimicdiffusion,instantpure", help="Comma-separated purifier names to time with --purifier-classifier.")
    parser.add_argument("--variants", default="baseline,ours", choices=None, help="Comma-separated variants: baseline,ours.")
    parser.add_argument("--random-init-missing-wrappers", "--random_init_missing_wrappers", type=str2bool, default=True, help="Legacy no-op. Timing never fits missing HiRA/RanPAC caches and always falls back to random wrapper weights.")


def add_wrapper_args(parser):
    parser.add_argument("--hira-expansion-dim", "--hira_expansion_dim", type=int, default=16384)
    parser.add_argument("--hira-num-blocks", "--hira_num_blocks", type=int, default=4)
    parser.add_argument("--hira-batch-size", "--hira_batch_size", type=int, default=128)
    parser.add_argument("--hira-num-workers", "--hira_num_workers", type=int, default=4)
    parser.add_argument("--hira-epochs", "--hira_epochs", type=int, default=1)
    parser.add_argument("--hira-lr", "--hira_lr", type=float, default=1e-4)
    parser.add_argument("--hira-weight-decay", "--hira_weight_decay", type=float, default=1e-4)
    parser.add_argument("--hira-seed", "--hira_seed", type=int, default=0)
    parser.add_argument("--hira-cache-dir", "--hira_cache_dir", default="pretrained/hira")
    parser.add_argument("--hira-dataset-root", "--hira_dataset_root", default=None)
    parser.add_argument("--hira-max-train-samples", "--hira_max_train_samples", type=int, default=-1)
    parser.add_argument("--hira-force-retrain", "--hira_force_retrain", type=str2bool, default=False)
    parser.add_argument("--adapt-noise-eps", "--adapt_noise_eps", type=parse_float_or_fraction, default=0.0)
    parser.add_argument("--adapt-noise-num", "--adapt_noise_num", type=int, default=1)
    parser.add_argument("--adapt-alpha", "--adapt_alpha", type=float, default=1.0)
    parser.add_argument("--soft-threshold-alpha", "--soft_threshold_alpha", type=float, default=0.9)
    parser.add_argument("--soft-threshold-beta", "--soft_threshold_beta", type=float, default=4.0)
    parser.add_argument("--soft-threshold-stat-eps", "--soft_threshold_stat_eps", type=float, default=1e-6)
    parser.add_argument("--soft-threshold-mode", "--soft_threshold_mode", choices=["near_mean", "away_from_mean"], default="away_from_mean")
    parser.add_argument("--hira-subspace-rank", "--hira_subspace_rank", type=int, default=1024)
    parser.add_argument("--hira-subspace-shrink", "--hira_subspace_shrink", type=float, default=0.5)
    parser.add_argument("--stability-ridge-gamma", "--stability_ridge_gamma", type=float, default=0.0)
    parser.add_argument("--stability-ridge-stat-eps", "--stability_ridge_stat_eps", type=float, default=1e-6)

    parser.add_argument("--ranpac-rp-dim", "--ranpac_rp_dim", type=int, default=10000)
    parser.add_argument("--ranpac-fit-batch-size", "--ranpac_fit_batch_size", type=int, default=64)
    parser.add_argument("--ranpac-batch-size", "--ranpac_batch_size", type=int, default=64)
    parser.add_argument("--ranpac-num-workers", "--ranpac_num_workers", type=int, default=4)
    parser.add_argument("--ranpac-seed", "--ranpac_seed", type=int, default=0)
    parser.add_argument("--ranpac-selection-method", "--ranpac_selection_method", default="regression")
    parser.add_argument("--ranpac-lambda", "--ranpac_lambda", type=float, default=0.5)
    parser.add_argument("--ranpac-temp", "--ranpac_temp", type=float, default=1.0)
    parser.add_argument("--ranpac-hardneg-topk", "--ranpac_hardneg_topk", type=int, default=0)
    parser.add_argument("--ranpac-hardneg-gamma", "--ranpac_hardneg_gamma", type=float, default=0.0)
    parser.add_argument("--ranpac-cache-dir", "--ranpac_cache_dir", default="pretrained/ranpac")
    parser.add_argument("--ranpac-dataset-root", "--ranpac_dataset_root", default=None)


def add_purifier_args(parser):
    parser.add_argument("--model", default="LCM", help="InstantPure model, e.g. LCM or TCD.")
    parser.add_argument("--load-origin-lora", "--load_origin_lora", type=str2bool, default=False)
    parser.add_argument("--lora-input-dir", "--lora_input_dir", default=None)
    parser.add_argument("--num-inference-step", "--num_inference_step", type=int, default=1)
    parser.add_argument("--strength", type=float, default=0.1)
    parser.add_argument("--guidance-scale", "--guidance_scale", type=float, default=1.0)
    parser.add_argument("--control-scale", "--control_scale", type=float, default=0.8)
    parser.add_argument("--diffusion-respace", "--diffusion_respace", default="ddim50")
    parser.add_argument("--diffusion-timestep", "--diffusion_timestep", type=int, default=150)
    parser.add_argument("--guided-diffusion-pretrained-root", "--guided_diffusion_pretrained_root", default="pretrained")
    parser.add_argument("--guided-diffusion-checkpoint-path", "--guided_diffusion_checkpoint_path", default=None)
    parser.add_argument("--guided-diffusion-use-fp16", "--guided_diffusion_use_fp16", type=str2bool, default=False)
    parser.add_argument("--mimicdiffusion-max-timesteps", "--mimicdiffusion_max_timesteps", default="1000")
    parser.add_argument("--mimicdiffusion-num-denoising-steps", "--mimicdiffusion_num_denoising_steps", default="100")
    parser.add_argument("--mimicdiffusion-sampling-method", "--mimicdiffusion_sampling_method", choices=["ddpm", "ddim"], default="ddpm")
    parser.add_argument("--mimicdiffusion-rho-scale", "--mimicdiffusion_rho_scale", type=float, default=3000.0)
    parser.add_argument("--mimicdiffusion-guidance-start-step", "--mimicdiffusion_guidance_start_step", type=int, default=20)
    parser.add_argument("--mimicdiffusion-guidance-end-step", "--mimicdiffusion_guidance_end_step", type=int, default=90)
    parser.add_argument("--mimicdiffusion-projection-scale", "--mimicdiffusion_projection_scale", type=int, default=4)


def parse_args():
    parser = argparse.ArgumentParser(description="Measure forward-time inference efficiency for victims and purifier pipelines.")
    add_common_args(parser)
    add_wrapper_args(parser)
    add_purifier_args(parser)
    return parser.parse_args()


def variant_enabled(variants, name):
    return name in variants


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    dataset = build_robustbench_like_dataset(args.data_dir, args.eval_examples)
    _, selected_class_ids = select_all_indices(dataset)

    run_name = args.run_name or f"efficiency_examples{len(dataset)}_bs{args.batch_size}_seed{args.seed}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    classifiers = parse_csv(args.classifiers)
    purifiers = parse_csv(args.purifiers)
    variants = set(parse_csv(args.variants))
    invalid_variants = variants.difference({"baseline", "ours"})
    if invalid_variants:
        raise ValueError(f"Unknown variants: {sorted(invalid_variants)}")

    summary_rows = []
    all_batch_rows = []
    print(f"Saving inference-efficiency outputs to: {run_dir}")
    print(f"Loaded RobustBench-like ImageNet examples: {len(dataset)}")
    print(f"Loaded classes: {selected_class_ids}")

    for classifier_name in classifiers:
        loader, _, _ = build_loader(dataset, args.batch_size, args.num_workers)
        for variant_name, use_ours in (("baseline", False), ("ours", True)):
            if not variant_enabled(variants, variant_name):
                continue
            label = f"victim_{classifier_name}_{variant_name}"
            print(f"\nBuilding {label}...")
            (
                model,
                wrapped_name,
                victim_spec,
                purifier_label,
                wrapper_source,
                hira_cache_loaded,
                ranpac_cache_loaded,
            ) = build_timed_model(
                args,
                classifier_name=classifier_name,
                purifier_name="none",
                use_ours=use_ours,
                device=device,
            )
            metrics, batch_rows = time_model(model, loader, device, args, label)
            summary_rows.append(
                {
                    "setting": "victim_only",
                    "variant": variant_name,
                    "classifier": classifier_name,
                    "wrapped_classifier": wrapped_name,
                    "victim_timm_model": victim_spec.timm_model_name,
                    "purifier": purifier_label,
                    "wrapper_source": wrapper_source,
                    "hira_cache_loaded": hira_cache_loaded,
                    "ranpac_cache_loaded": ranpac_cache_loaded,
                    "batch_size": args.batch_size,
                    **metrics,
                }
            )
            all_batch_rows.extend(batch_rows)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    for purifier_name in purifiers:
        loader, _, _ = build_loader(dataset, args.purifier_batch_size, args.num_workers)
        for variant_name, use_ours in (("baseline", False), ("ours", True)):
            if not variant_enabled(variants, variant_name):
                continue
            label = f"{purifier_name}_{args.purifier_classifier}_{variant_name}"
            print(f"\nBuilding {label}...")
            (
                model,
                wrapped_name,
                victim_spec,
                purifier_label,
                wrapper_source,
                hira_cache_loaded,
                ranpac_cache_loaded,
            ) = build_timed_model(
                args,
                classifier_name=args.purifier_classifier,
                purifier_name=purifier_name,
                use_ours=use_ours,
                device=device,
            )
            metrics, batch_rows = time_model(model, loader, device, args, label)
            summary_rows.append(
                {
                    "setting": "purified_classifier",
                    "variant": variant_name,
                    "classifier": args.purifier_classifier,
                    "wrapped_classifier": wrapped_name,
                    "victim_timm_model": victim_spec.timm_model_name,
                    "purifier": purifier_label,
                    "wrapper_source": wrapper_source,
                    "hira_cache_loaded": hira_cache_loaded,
                    "ranpac_cache_loaded": ranpac_cache_loaded,
                    "batch_size": args.purifier_batch_size,
                    **metrics,
                }
            )
            all_batch_rows.extend(batch_rows)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    write_csv(run_dir / "inference_efficiency_summary.csv", summary_rows)
    write_csv(run_dir / "inference_efficiency_batches.csv", all_batch_rows)

    summary = {
        "dataset": DATASET,
        "eval_examples": args.eval_examples,
        "loaded_examples": len(dataset),
        "num_classes": len(selected_class_ids),
        "classifiers": classifiers,
        "purifier_classifier": args.purifier_classifier,
        "purifiers": purifiers,
        "variants": sorted(variants),
        "batch_size": args.batch_size,
        "purifier_batch_size": args.purifier_batch_size,
        "warmup_batches": args.warmup_batches,
        "max_batches": args.max_batches,
        "random_init_missing_wrappers": args.random_init_missing_wrappers,
        "outputs": [
            "inference_efficiency_summary.csv",
            "inference_efficiency_batches.csv",
            "summary.json",
        ],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
