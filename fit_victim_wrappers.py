import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from classifiers.stability_ridge import DEFAULT_STABILITY_RIDGE_STAT_EPS
from dataset import get_dataset
from utils import str2bool
from victims import IMAGENET_MODEL, apply_victim_wrappers, build_imagenet_victim, build_wrapper_config_from_namespace, supports_hira


def parse_args():
    parser = argparse.ArgumentParser(description="Fit HiRA and/or RanPAC on clean ImageNet without any purifier.")
    parser.add_argument("--classifier", default="resnet50", type=str, help=f"Victim classification model. Available aliases: {', '.join(IMAGENET_MODEL)}")
    parser.add_argument("--device", default="cuda:0", help="Device, e.g. cuda:0")
    parser.add_argument("--seed", default=3407, type=int, help="Seed for all RNGs.")
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
    parser.add_argument("--soft_threshold_mode", type=str, choices=["near_mean", "away_from_mean"], default="away_from_mean", help="HiRA-only inference sparsification target.")
    parser.add_argument("--stability_ridge_gamma", type=float, default=0.0, help="Strength of the stability-aware diagonal ridge prior; 0 disables it.")
    parser.add_argument("--stability_ridge_stat_eps", type=float, default=DEFAULT_STABILITY_RIDGE_STAT_EPS, help="Minimum projected-feature std used by the stability-aware ridge prior.")
    parser.add_argument("--use_ranpac_head", "--use_ranpac", type=str2bool, default=False, help="Replace the final linear layer with a RanPAC ridge head.")
    parser.add_argument("--ranpac_rp_dim", type=int, default=5000, help="Random projection dimension for RanPAC.")
    parser.add_argument("--ranpac_batch_size", "--ranpac_fit_batch_size", type=int, default=256, help="Batch size used to fit the RanPAC head.")
    parser.add_argument("--ranpac_num_workers", type=int, default=8, help="Number of workers used to fit the RanPAC head.")
    parser.add_argument("--ranpac_seed", type=int, default=0, help="Seed used to build the RanPAC random projection.")
    parser.add_argument("--ranpac_lambda", type=float, default=1.0, help="Mixing weight between the original classifier head and the temperature-scaled RanPAC head.")
    parser.add_argument("--ranpac_temp", type=float, default=1.0, help="Temperature applied to the RanPAC logits before ensembling.")
    parser.add_argument("--ranpac_hardneg_topk", type=int, default=9, help="Number of top confusing non-ground-truth classes to suppress in the RanPAC regression targets.")
    parser.add_argument("--ranpac_hardneg_gamma", type=float, default=1.0, help="Total suppression weight assigned across the RanPAC hard-negative classes.")
    parser.add_argument("--ranpac_selection_method", type=str, choices=["regression"], default="regression", help="Which cached RanPAC head to apply; only regression-loss ridge selection is supported.")
    parser.add_argument("--ranpac_cache_dir", type=str, default="pretrained/ranpac", help="Cache directory for fitted RanPAC heads.")
    parser.add_argument("--ranpac_dataset_root", type=str, default=None, help="ImageNet root used when fitting a RanPAC head.")
    parser.add_argument("--num_eval_samples", type=int, default=0, help="If positive, also run a clean validation check on this many ImageNet val examples after fitting.")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Batch size for the optional clean validation check.")
    parser.add_argument("--eval_num_workers", type=int, default=8, help="Number of workers for the optional clean validation check.")
    return parser.parse_args()


def resolve_device(device):
    if isinstance(device, torch.device):
        return device
    if isinstance(device, int):
        return torch.device(f"cuda:{device}")
    if isinstance(device, str) and device.isdigit():
        return torch.device(f"cuda:{device}")
    return torch.device(device)


def seed_everything(seed=3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sample_eval_subset(dataset, num_samples, seed):
    if num_samples <= 0 or num_samples >= len(dataset):
        return dataset
    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:num_samples].tolist()
    return Subset(dataset, indices)


def evaluate_clean_accuracy(model, device, num_samples, batch_size, num_workers, seed):
    dataset = get_dataset("imagenet", split="test", adv=False)
    dataset = sample_eval_subset(dataset, num_samples, seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Clean validation"):
            x = x.to(device)
            y = y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)

    accuracy = correct / total if total > 0 else 0.0
    print(f"Clean accuracy over {total} samples: {accuracy:.4%}")
    return accuracy


def main():
    args = parse_args()
    device = resolve_device(args.device)
    seed_everything(args.seed)

    classifier, classifier_name, victim_spec = build_imagenet_victim(args.classifier)
    wrapper_config = build_wrapper_config_from_namespace(args, dataset="imagenet")
    classifier, wrapped_name = apply_victim_wrappers(
        classifier,
        classifier_name=classifier_name,
        supports_hira_arch=supports_hira(victim_spec),
        config=wrapper_config,
        device=device,
    )
    classifier = classifier.to(device).eval()

    print(f"Built victim: {victim_spec.timm_model_name}")
    print(f"Wrapped victim: {wrapped_name}")

    if args.num_eval_samples > 0:
        evaluate_clean_accuracy(
            classifier,
            device=device,
            num_samples=args.num_eval_samples,
            batch_size=args.eval_batch_size,
            num_workers=args.eval_num_workers,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
