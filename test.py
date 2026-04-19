# modify on top of https://github.com/xavihart/Diff-PGD

import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from attacks import build_attack, gen_pgd_confs
from classifiers.stability_ridge import DEFAULT_STABILITY_RIDGE_STAT_EPS, build_stability_ridge_tag
from dataset import get_dataset
from purifiers import PurifiedClassifier, build_purifier
from utils import finish_wandb, init_wandb, log_wandb_metrics, str2bool
from victims import IMAGENET_MODEL, apply_victim_wrappers, build_imagenet_victim, build_wrapper_config_from_namespace, supports_hira


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate purified or unpurified ImageNet victims with optional HiRA/RanPAC protection.")
    parser.add_argument("--model", default="LCM", type=str, help="InstantPure purifier model: lcm or tcd. Ignored when --purifier_name none.")
    parser.add_argument("--purifier_name", type=str, choices=["none", "instantpure", "instancepure", "puriflow"], default="instantpure", help="Purifier backend applied before the wrapped victim.")
    parser.add_argument("--load_origin_lora", default=False, action="store_true", help="Use the original LoRA mixed with the adversarial LoRA for InstantPure.")
    parser.add_argument("--lora_input_dir", type=str, help="Input LoRA directory for the purifier backend.")
    parser.add_argument("--output_dir", default="vis_and_stat/", type=str, help="Output directory for metrics and optional artifacts.")
    parser.add_argument("--num_validation_set", default=1000, type=int, help="Size of the validation subset.")
    parser.add_argument("--num_inference_step", default=1, type=int, help="Purifier inference steps.")
    parser.add_argument("--strength", default=0.1, type=float, help="Purifier noise strength.")
    parser.add_argument("--seed", default=3407, type=int, help="Seed for all RNGs.")
    parser.add_argument("--guidance_scale", default=1.0, type=float, help="Purifier guidance scale.")
    parser.add_argument("--control_scale", default=0.8, type=float, help="Purifier control scale.")
    parser.add_argument("--input_image", default="./image_net", type=str, help="Unused legacy compatibility flag.")
    parser.add_argument("--classifier", default="resnet50", type=str, help=f"Victim classification model. Available aliases: {', '.join(IMAGENET_MODEL)}")
    parser.add_argument("--use_hira_adapter", "--use_hira", type=str2bool, default=False, help="Attach HiRA before the MLP of the last N transformer blocks for ViT-family backbones.")
    parser.add_argument("--hira_expansion_dim", type=int, default=4096, help="Hidden expansion dimension for HiRA adapters.")
    parser.add_argument("--hira_num_blocks", type=int, default=2, help="Number of final transformer MLP blocks that receive HiRA adapters.")
    parser.add_argument("--hira_batch_size", type=int, default=32, help="Batch size used to fit closed-form HiRA weights.")
    parser.add_argument("--hira_num_workers", type=int, default=8, help="Number of workers used to fit closed-form HiRA weights.")
    parser.add_argument("--hira_epochs", type=int, default=1, help="Legacy compatibility flag; closed-form HiRA ignores this value.")
    parser.add_argument("--hira_lr", type=float, default=1e-4, help="Legacy compatibility flag; closed-form HiRA ignores this value.")
    parser.add_argument("--hira_weight_decay", type=float, default=1e-4, help="Legacy compatibility flag; closed-form HiRA ignores this value.")
    parser.add_argument("--hira_seed", type=int, default=0, help="Seed used for HiRA projection sampling and dataset split.")
    parser.add_argument("--hira_cache_dir", type=str, default="pretrained/hira", help="Cache directory for fitted HiRA weights.")
    parser.add_argument("--hira_dataset_root", type=str, default=None, help="ImageNet root used when fitting closed-form HiRA weights.")
    parser.add_argument("--hira_max_train_samples", type=int, default=-1, help="Optional cap on ImageNet train samples used to fit closed-form HiRA.")
    parser.add_argument("--hira_force_retrain", type=str2bool, default=False, help="Ignore cached HiRA weights and fit again.")
    parser.add_argument("--adapt_noise_eps", type=float, default=0.0, help="Linf noise radius used while fitting HiRA and RanPAC adaptation statistics.")
    parser.add_argument("--adapt_noise_num", type=int, default=0, help="Number of noisy samples per training image used while fitting HiRA and RanPAC.")
    parser.add_argument("--adapt_alpha", type=float, default=1.0, help="Total weight assigned to noisy adaptation statistics relative to clean statistics.")
    parser.add_argument("--soft_threshold_alpha", type=float, default=0.0, help="HiRA-only width of the smooth mean-centered threshold in units of hidden-feature std; 0 disables it.")
    parser.add_argument("--soft_threshold_beta", type=float, default=8.0, help="HiRA-only sharpness of the smooth mean-centered threshold.")
    parser.add_argument("--soft_threshold_stat_eps", type=float, default=1e-6, help="HiRA-only minimum hidden-feature std used by the smooth mean-centered threshold.")
    parser.add_argument("--soft_threshold_mode", type=str, choices=["near_mean", "away_from_mean"], default="away_from_mean", help="HiRA-only inference sparsification target: pull ambiguous hidden features toward the mean or toward the nearest mean +/- alpha*std boundary.")
    parser.add_argument("--stability_ridge_gamma", type=float, default=0.0, help="Strength of the stability-aware diagonal ridge prior; 0 disables it.")
    parser.add_argument("--stability_ridge_stat_eps", type=float, default=DEFAULT_STABILITY_RIDGE_STAT_EPS, help="Minimum projected-feature std used by the stability-aware ridge prior.")
    parser.add_argument("--attack_method", default="Linf_pgd", type=str, help="Attack backend. Use diff_pgd for the current SDEdit-based diffusion PGD attack. Standard attacks hit the selected attack target.")
    parser.add_argument("--attack_target", choices=["victim", "purified"], default="victim", help="Which composed model standard attacks should target.")
    parser.add_argument("--device", default="cuda:0", help="Device, e.g. cuda:0")
    parser.add_argument("--use_ranpac_head", "--use_ranpac", type=str2bool, default=False, help="Replace the final linear layer with a RanPAC ridge head.")
    parser.add_argument("--ranpac_rp_dim", type=int, default=5000, help="Random projection dimension for RanPAC.")
    parser.add_argument("--ranpac_batch_size", "--ranpac_fit_batch_size", type=int, default=256, help="Batch size used to fit the RanPAC head.")
    parser.add_argument("--ranpac_num_workers", type=int, default=8, help="Number of workers used to fit the RanPAC head.")
    parser.add_argument("--ranpac_seed", type=int, default=0, help="Seed used to build the RanPAC random projection.")
    parser.add_argument("--ranpac_lambda", type=float, default=1.0, help="Convex mixing weight between the original classifier head and the temperature-scaled RanPAC head.")
    parser.add_argument("--ranpac_temp", type=float, default=1.0, help="Temperature applied to the RanPAC logits before ensembling.")
    parser.add_argument("--ranpac_hardneg_topk", type=int, default=9, help="Number of top confusing non-ground-truth classes to suppress in the RanPAC regression targets.")
    parser.add_argument("--ranpac_hardneg_gamma", type=float, default=1.0, help="Total suppression weight assigned across the RanPAC hard-negative classes.")
    parser.add_argument("--ranpac_selection_method", type=str, choices=["regression"], default="regression", help="Which cached RanPAC head to apply; only regression-loss ridge selection is supported.")
    parser.add_argument("--ranpac_cache_dir", type=str, default="pretrained/ranpac", help="Cache directory for fitted RanPAC heads.")
    parser.add_argument("--ranpac_dataset_root", type=str, default=None, help="ImageNet root used when fitting a RanPAC head.")
    parser.add_argument("--stadv_num_iterations", type=int, default=100, help="Number of optimization steps for the DiffPure stadv attack.")
    parser.add_argument("--stadv_eot_iter", type=int, default=20, help="EOT iterations for the DiffPure stadv attack.")
    parser.add_argument("--use_wandb", type=str2bool, default=False, help="Log final metrics to Weights & Biases.")
    parser.add_argument("--wandb_project", type=str, default="instantpure", help="Weights & Biases project name.")
    parser.add_argument("--wandb_entity", type=str, default="", help="Weights & Biases entity.")
    parser.add_argument("--wandb_name", type=str, default="", help="Weights & Biases run name.")
    parser.add_argument("--wandb_group", type=str, default="", help="Weights & Biases run group.")
    parser.add_argument("--wandb_mode", type=str, default="online", help="Weights & Biases mode: online, offline, or disabled.")
    parser.add_argument("--attack_version", type=str, choices=["v1", "v2"], default="v1", help="Attack pipeline version: v1 attacks the selected attack target directly; v2 uses the current SDEdit-based diffusion PGD backend.")
    parser.add_argument("--atk_iter", type=int, default=40, help="Attack steps.")
    parser.add_argument("--eps", type=int, default=4, help="Attack epsilon in pixel-space units out of 255.")
    parser.add_argument("--diffusion_respace", type=str, default="ddim50", help="Guided diffusion timestep respacing used by the InstantPure backend.")
    parser.add_argument("--diffusion_timestep", type=int, default=150, help="Guided diffusion timestep used by the current SDEdit-based diffusion attack backend.")
    return parser.parse_args()


def resolve_device(device):
    if isinstance(device, torch.device):
        return device
    if isinstance(device, int):
        return torch.device(f"cuda:{device}")
    if isinstance(device, str) and device.isdigit():
        return torch.device(f"cuda:{device}")
    return torch.device(device)


def _format_variant_noise_value(value):
    text = str(value)
    for old, new in (("/", "_"), (" ", ""), (".", "p"), ("-", "m")):
        text = text.replace(old, new)
    return text


def build_adapt_noise_tag(args):
    if args.adapt_noise_eps <= 0 or args.adapt_noise_num <= 0 or args.adapt_alpha <= 0:
        return ""
    return (
        f"_neps{_format_variant_noise_value(args.adapt_noise_eps)}"
        f"_nnum{args.adapt_noise_num}"
        f"_na{_format_variant_noise_value(args.adapt_alpha)}"
    )


def build_meansparse_tag(args):
    if args.soft_threshold_alpha <= 0:
        return ""
    tag = (
        f"_msa{_format_variant_noise_value(args.soft_threshold_alpha)}"
        f"_msb{_format_variant_noise_value(args.soft_threshold_beta)}"
        f"_msm{'near' if args.soft_threshold_mode == 'near_mean' else 'away'}"
    )
    if args.soft_threshold_stat_eps != 1e-6:
        tag = f"{tag}_mseps{_format_variant_noise_value(args.soft_threshold_stat_eps)}"
    return tag


def build_stability_ridge_variant_tag(args):
    return build_stability_ridge_tag(
        gamma=args.stability_ridge_gamma,
        stat_eps=args.stability_ridge_stat_eps,
        separator="_",
    )


def build_classifier_variant_name(args):
    classifier_variant = args.classifier
    adapt_noise_tag = build_adapt_noise_tag(args)
    meansparse_tag = build_meansparse_tag(args)
    stability_ridge_tag = build_stability_ridge_variant_tag(args)
    if args.use_hira_adapter:
        classifier_variant = f"{classifier_variant}_hira"
        if args.hira_num_blocks != 2:
            classifier_variant = f"{classifier_variant}_blk{args.hira_num_blocks}"
    if args.use_ranpac_head:
        classifier_variant = f"{classifier_variant}_ranpac_{args.ranpac_selection_method}_bbias"
        if args.ranpac_lambda != 1.0:
            classifier_variant = f"{classifier_variant}_lam{_format_variant_noise_value(args.ranpac_lambda)}"
        if args.ranpac_temp != 1.0:
            classifier_variant = f"{classifier_variant}_temp{_format_variant_noise_value(args.ranpac_temp)}"
        if args.ranpac_hardneg_topk != 9 or args.ranpac_hardneg_gamma != 1.0:
            classifier_variant = (
                f"{classifier_variant}_htk{args.ranpac_hardneg_topk}"
                f"_hg{_format_variant_noise_value(args.ranpac_hardneg_gamma)}"
            )
    if adapt_noise_tag and (args.use_hira_adapter or args.use_ranpac_head):
        classifier_variant = f"{classifier_variant}{adapt_noise_tag}"
    if meansparse_tag and args.use_hira_adapter:
        classifier_variant = f"{classifier_variant}{meansparse_tag}"
    if stability_ridge_tag and (args.use_hira_adapter or args.use_ranpac_head):
        classifier_variant = f"{classifier_variant}{stability_ridge_tag}"
    return classifier_variant


def build_purifier_variant_name(args):
    if args.purifier_name == "none":
        return "no_purifier"
    if args.purifier_name == "instantpure":
        lora_dir = (args.lora_input_dir or "no_lora").replace("/", "_")
        origin_tag = "origin_lora_1" if args.load_origin_lora else "origin_lora_0"
        return (
            f"instantpure_{args.model.lower()}_{origin_tag}_{lora_dir}"
            f"_nstep{args.num_inference_step}"
            f"_strength{int(args.strength * 1000)}"
            f"_g{_format_variant_noise_value(args.guidance_scale)}"
            f"_c{_format_variant_noise_value(args.control_scale)}"
        )
    return args.purifier_name


def sample_eval_subset(dataset, num_samples, seed):
    if num_samples >= len(dataset):
        return dataset
    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:num_samples].tolist()
    return Subset(dataset, indices)


def seed_everything(seed=3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_victim_pipeline(args, device):
    classifier, classifier_name, victim_spec = build_imagenet_victim(args.classifier)
    wrapper_config = build_wrapper_config_from_namespace(args, dataset="imagenet")
    if wrapper_config.use_hira or wrapper_config.use_ranpac:
        print("Preparing HiRA/RanPAC on the clean ImageNet training split before purifier construction.")
    classifier, wrapped_name = apply_victim_wrappers(
        classifier,
        classifier_name=classifier_name,
        supports_hira_arch=supports_hira(victim_spec),
        config=wrapper_config,
        device=device,
    )
    classifier = classifier.to(device).eval()
    return classifier, wrapped_name, victim_spec


def evaluate_pipeline(args):
    device = resolve_device(args.device)
    seed_everything(args.seed)

    classifier_variant = build_classifier_variant_name(args)
    purifier_variant = build_purifier_variant_name(args)
    save_path = os.path.join(
        args.output_dir,
        args.attack_method,
        classifier_variant,
        purifier_variant,
        f"{args.num_validation_set}_samples",
    )
    os.makedirs(save_path, exist_ok=True)
    wandb_run = init_wandb(args, save_path)

    classifier, wrapped_classifier_name, victim_spec = build_victim_pipeline(args, device)
    purifier = build_purifier(args, device)
    purifier = purifier.to(device).eval()
    purified_classifier = PurifiedClassifier(purifier, classifier).to(device).eval()

    dataset = get_dataset("imagenet", split="test", adv=False)
    dataset = sample_eval_subset(dataset, args.num_validation_set, args.seed)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)

    pgd_conf = gen_pgd_confs(eps=args.eps, alpha=1, iter=args.atk_iter, input_range=(0, 1))
    attack = build_attack(args, classifier, purified_classifier, purifier, pgd_conf, device)

    raw_clean_correct = 0
    raw_robust_correct = 0
    purified_clean_correct = 0
    purified_robust_correct = 0

    for x, y in tqdm(test_loader):
        x = x.to(device)
        y = y.to(device)

        raw_clean_correct += (classifier(x).argmax(1) == y).sum().item()
        purified_clean_correct += (purified_classifier(x).argmax(1) == y).sum().item()

        x_adv = attack(x, y)
        raw_robust_correct += (classifier(x_adv).argmax(1) == y).sum().item()
        purified_robust_correct += (purified_classifier(x_adv).argmax(1) == y).sum().item()

    num_eval_samples = len(dataset)
    metrics = {
        "victim_name": wrapped_classifier_name,
        "victim_timm_model": victim_spec.timm_model_name,
        "purifier_name": args.purifier_name,
        "attack_method": args.attack_method,
        "attack_target": args.attack_target,
        "attack_version": args.attack_version,
        "classifier_accuracy": raw_clean_correct / num_eval_samples,
        "original_classifier_robust_accuracy": raw_robust_correct / num_eval_samples,
        "attack_fail_rate": raw_robust_correct / num_eval_samples,
        "clean_accuracy": purified_clean_correct / num_eval_samples,
        "robust_accuracy": purified_robust_correct / num_eval_samples,
        "evaluated_examples": num_eval_samples,
        "adapt_noise_eps": args.adapt_noise_eps,
        "adapt_noise_num": args.adapt_noise_num,
        "adapt_alpha": args.adapt_alpha,
        "soft_threshold_alpha": args.soft_threshold_alpha,
        "soft_threshold_beta": args.soft_threshold_beta,
        "soft_threshold_stat_eps": args.soft_threshold_stat_eps,
        "soft_threshold_mode": args.soft_threshold_mode,
        "ranpac_lambda": args.ranpac_lambda,
        "ranpac_temp": args.ranpac_temp,
        "ranpac_baseline_bias_centered": args.use_ranpac_head,
        "ranpac_hardneg_topk": args.ranpac_hardneg_topk,
        "ranpac_hardneg_gamma": args.ranpac_hardneg_gamma,
    }

    stat = pd.DataFrame(metrics, index=[0])
    stat.to_csv(os.path.join(save_path, "stat.csv"), index=False)

    if wandb_run is not None:
        wandb_run.summary["save_path"] = save_path
        log_wandb_metrics(wandb_run, metrics)
        finish_wandb(wandb_run)

    print(stat)
    return metrics


def Global(classifier, device, respace, t, args, eps=16, iter=10, name="attack_global", alpha=2, version="v1"):
    del classifier, device, respace, t, eps, iter, name, alpha
    args.attack_version = version
    return evaluate_pipeline(args)


if __name__ == "__main__":
    args = parse_args()
    evaluate_pipeline(args)
