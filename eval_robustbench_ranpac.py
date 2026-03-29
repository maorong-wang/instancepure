"""
Brief evaluation for RobustBench models with and without a RanPAC head.

Examples:
  python eval_robustbench_ranpac.py \
    --model-names Rebuffi2021Fixing_70_16_cutmix_extra \
    --dataset cifar10 \
    --threat-model Linf \
    --data-dir ./dataset \
    --eval-examples 1000

  python eval_robustbench_ranpac.py \
    --model-names Salman2020Do_R18 \
    --dataset imagenet \
    --threat-model Linf \
    --data-dir /path/to/imagenet \
    --eval-examples 256 \
    --use-ranpac true
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from autoattack import AutoAttack
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from classifiers.hira import apply_hira_adaptation
from classifiers.ranpac import apply_ranpac_head
from dataset import get_dataset as instantpure_get_dataset

try:
    from robustbench import load_model
    from robustbench.data import get_preprocessing
    from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel
except ImportError:
    fallback_root = Path(__file__).resolve().parent.parent / "adversarial-attacks-pytorch"
    if str(fallback_root) not in sys.path:
        sys.path.insert(0, str(fallback_root))
    from robustbench import load_model
    from robustbench.data import get_preprocessing
    from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel


DEFAULT_EPS = {
    ("cifar10", "Linf"): 8.0 / 255.0,
    ("cifar100", "Linf"): 8.0 / 255.0,
    ("imagenet", "Linf"): 4.0 / 255.0,
    ("cifar10", "L2"): 0.5,
    ("cifar100", "L2"): 0.5,
    ("imagenet", "L2"): 3.0,
}


def str2bool(v):
    if isinstance(v, bool):
        return v
    value = str(v).lower()
    if value in {"yes", "true", "t", "y", "1"}:
        return True
    if value in {"no", "false", "f", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def _format_cache_value(value):
    text = str(value)
    for old, new in (("/", "_"), (" ", ""), (".", "p"), ("-", "m")):
        text = text.replace(old, new)
    return text


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RobustBench models with and without RanPAC.")
    parser.add_argument("--model-names", "--model_names", nargs="+", required=True, help="RobustBench model names to evaluate.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=[BenchmarkDataset.cifar_10.value, BenchmarkDataset.cifar_100.value, BenchmarkDataset.imagenet.value],
    )
    parser.add_argument(
        "--threat-model",
        "--threat_model",
        required=True,
        choices=[ThreatModel.Linf.value, ThreatModel.L2.value],
        help="Threat model used to load the RobustBench model.",
    )
    parser.add_argument(
        "--data-dir",
        "--data_dir",
        required=True,
        help="Dataset root. For ImageNet it should contain train/ and val/; for CIFAR it is the download root.",
    )
    parser.add_argument("--model-dir", "--model_dir", default="./robustbench_models", help="RobustBench checkpoint cache directory.")
    parser.add_argument("--device", default="cuda:0", help="Device, e.g. cuda:0 or cpu.")
    parser.add_argument("--seed", type=int, default=0, help="Global random seed.")
    parser.add_argument("--eval-examples", "--eval_examples", type=int, default=1000, help="Number of evaluation images to sample.")
    parser.add_argument("--eval-batch-size", "--eval_batch_size", type=int, default=64, help="Evaluation batch size.")
    parser.add_argument("--num-workers", "--num_workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument(
        "--eval-split",
        "--eval_split",
        default="test",
        choices=["test", "val", "validation"],
        help="Evaluation split. Defaults to test.",
    )
    parser.add_argument(
        "--attacks",
        default="pgd,autoattack",
        help="Comma-separated list from: clean,pgd,autoattack. Clean is always reported.",
    )
    parser.add_argument(
        "--attack-method",
        "--attack_method",
        default="",
        choices=["", "pgd", "autoattack", "both"],
        help="Convenience alias for sweeps. Overrides --attacks when set.",
    )
    parser.add_argument("--eps", type=float, default=None, help="Attack epsilon. Defaults depend on dataset and norm.")
    parser.add_argument("--pgd-steps", "--pgd_steps", type=int, default=20, help="PGD steps.")
    parser.add_argument("--pgd-step-size", "--pgd_step_size", type=float, default=None, help="PGD step size. Defaults to 2 * eps / steps.")
    parser.add_argument("--pgd-random-start", "--pgd_random_start", type=str2bool, default=False, help="Enable random-start PGD.")
    parser.add_argument("--autoattack-version", "--autoattack_version", default="standard", help="AutoAttack version.")
    parser.add_argument(
        "--use-hira",
        "--use_hira",
        type=str2bool,
        default=None,
        help="If set, evaluate only the original model (false) or only the HiRA model (true).",
    )
    parser.add_argument("--hira-expansion-dim", "--hira_expansion_dim", type=int, default=4096, help="Hidden expansion dimension for HiRA adapters.")
    parser.add_argument("--hira-num-blocks", "--hira_num_blocks", type=int, default=2, help="Number of final transformer MLP blocks that receive HiRA adapters.")
    parser.add_argument("--hira-batch-size", "--hira_batch_size", type=int, default=32, help="HiRA fitting batch size.")
    parser.add_argument("--hira-num-workers", "--hira_num_workers", type=int, default=4, help="HiRA fitting DataLoader workers.")
    parser.add_argument("--hira-epochs", "--hira_epochs", type=int, default=1, help="Number of epochs used to fine-tune HiRA.")
    parser.add_argument("--hira-lr", "--hira_lr", type=float, default=1e-4, help="Learning rate used to fine-tune HiRA.")
    parser.add_argument("--hira-weight-decay", "--hira_weight_decay", type=float, default=1e-4, help="Weight decay used to fine-tune HiRA.")
    parser.add_argument("--hira-seed", "--hira_seed", type=int, default=0, help="Seed used to fine-tune and cache HiRA.")
    parser.add_argument("--hira-cache-dir", "--hira_cache_dir", default="pretrained/hira_robustbench", help="HiRA cache directory.")
    parser.add_argument("--hira-dataset-root", "--hira_dataset_root", default="", help="Optional ImageNet root used to fine-tune HiRA. Defaults to --data-dir.")
    parser.add_argument("--hira-max-train-samples", "--hira_max_train_samples", type=int, default=-1, help="Optional cap on ImageNet train samples used to fine-tune HiRA.")
    parser.add_argument("--hira-force-retrain", "--hira_force_retrain", type=str2bool, default=False, help="Ignore cached HiRA weights and fine-tune again.")
    parser.add_argument(
        "--use-ranpac",
        "--use_ranpac",
        type=str2bool,
        default=None,
        help="If set, evaluate only the original model (false) or only the RanPAC model (true).",
    )
    parser.add_argument("--ranpac-rp-dim", "--ranpac_rp_dim", type=int, default=5000, help="RanPAC random projection dimension.")
    parser.add_argument(
        "--ranpac-fit-batch-size",
        "--ranpac_fit_batch_size",
        type=int,
        default=256,
        help="RanPAC fitting batch size.",
    )
    parser.add_argument(
        "--ranpac-selection-method",
        "--ranpac_selection_method",
        choices=["regression", "val_acc", "both"],
        default="both",
        help="Which cached RanPAC head(s) to evaluate after fitting both ridge-selection variants.",
    )
    parser.add_argument("--ranpac-cache-dir", "--ranpac_cache_dir", default="pretrained/ranpac_robustbench", help="RanPAC cache directory.")
    parser.add_argument("--output-path", "--output_path", default="", help="Optional JSON output path.")
    parser.add_argument("--use-wandb", "--use_wandb", type=str2bool, default=False, help="Log evaluation metrics to Weights & Biases.")
    parser.add_argument("--wandb-project", "--wandb_project", default="robustbench-ranpac", help="Weights & Biases project name.")
    parser.add_argument("--wandb-entity", "--wandb_entity", default="", help="Weights & Biases entity.")
    parser.add_argument("--wandb-name", "--wandb_name", default="", help="Weights & Biases run name.")
    parser.add_argument("--wandb-group", "--wandb_group", default="", help="Weights & Biases run group.")
    parser.add_argument("--wandb-mode", "--wandb_mode", default="online", help="Weights & Biases mode: online, offline, or disabled.")
    parser.add_argument("--wandb-dir", "--wandb_dir", default="", help="Optional Weights & Biases directory.")
    return parser.parse_args()


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


def init_wandb(args):
    if not args.use_wandb:
        return None

    try:
        import wandb
    except ImportError as exc:
        raise ImportError("wandb is not installed. Install it or run without --use-wandb.") from exc

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_name or None,
        group=args.wandb_group or None,
        mode=args.wandb_mode,
        dir=args.wandb_dir or None,
        config=vars(args),
        reinit=True,
    )


def log_wandb_result(run, result, step):
    if run is None:
        return

    run.log(result, step=step)
    prefix = f"{result['model_name']}/{result['variant']}"
    for key, value in result.items():
        if isinstance(value, (int, float)):
            run.summary[f"{prefix}/{key}"] = value


def log_wandb_results_table(run, results):
    if run is None or not results:
        return

    import wandb

    columns = sorted({key for result in results for key in result.keys()})
    table = wandb.Table(columns=columns)
    for result in results:
        table.add_data(*[result.get(column) for column in columns])
    run.log({"results_table": table})


def finish_wandb(run):
    if run is not None:
        run.finish()


def default_eps(dataset, threat_model):
    return DEFAULT_EPS[(dataset, threat_model)]


def default_pgd_step_size(eps, steps):
    return 2.0 * eps / max(steps, 1)


def build_dataset(dataset, split, data_dir, transform):
    if dataset == BenchmarkDataset.cifar_10.value:
        return torchvision.datasets.CIFAR10(
            root=data_dir,
            train=(split == "train"),
            transform=transform,
            download=True,
        )
    if dataset == BenchmarkDataset.cifar_100.value:
        return torchvision.datasets.CIFAR100(
            root=data_dir,
            train=(split == "train"),
            transform=transform,
            download=True,
        )
    if dataset == BenchmarkDataset.imagenet.value:
        image_net_root = str(Path(data_dir))
        import os

        os.environ["IMAGENET_LOC_ENV"] = image_net_root
        instantpure_split = "train" if split == "train" else "test"
        dataset_obj = instantpure_get_dataset("imagenet", split=instantpure_split, adv=False)
        if hasattr(dataset_obj, "transform") and transform is not None:
            dataset_obj.transform = transform
        return dataset_obj
    raise NotImplementedError(f"Unsupported dataset: {dataset}")


def random_subset(dataset, n_examples, seed):
    if n_examples is None or n_examples < 0 or n_examples >= len(dataset):
        return dataset
    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:n_examples].tolist()
    return Subset(dataset, indices)


def build_eval_loader(dataset, threat_model, model_name, data_dir, n_examples, seed, batch_size, num_workers, eval_split):
    transform = get_preprocessing(BenchmarkDataset(dataset), ThreatModel(threat_model), model_name, None)
    eval_dataset = build_dataset(dataset, eval_split, data_dir, transform)
    eval_dataset = random_subset(eval_dataset, n_examples, seed)
    return DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def batch_l2_norm(x):
    return x.view(x.size(0), -1).norm(p=2, dim=1).view(-1, 1, 1, 1)


def random_l2_delta(x, eps):
    delta = torch.randn_like(x)
    delta_norm = batch_l2_norm(delta).clamp_min(1e-12)
    radius = torch.rand(x.size(0), 1, 1, 1, device=x.device)
    delta = delta / delta_norm * radius * eps
    return delta


def project_l2(delta, eps):
    delta_norm = batch_l2_norm(delta).clamp_min(1e-12)
    factor = torch.minimum(torch.ones_like(delta_norm), eps / delta_norm)
    return delta * factor


def pgd_attack(model, inputs, targets, norm, eps, steps, step_size, random_start):
    model.eval()
    x_orig = inputs.detach()
    if norm == ThreatModel.Linf.value:
        if random_start:
            delta = torch.empty_like(x_orig).uniform_(-eps, eps)
        else:
            delta = torch.zeros_like(x_orig)
        delta = torch.clamp(x_orig + delta, 0.0, 1.0) - x_orig
    elif norm == ThreatModel.L2.value:
        delta = random_l2_delta(x_orig, eps) if random_start else torch.zeros_like(x_orig)
        delta = torch.clamp(x_orig + delta, 0.0, 1.0) - x_orig
    else:
        raise NotImplementedError(f"Unsupported norm: {norm}")

    for _ in range(steps):
        adv = (x_orig + delta).clamp(0.0, 1.0).detach().requires_grad_(True)
        logits = model(adv)
        loss = F.cross_entropy(logits, targets, reduction="sum")
        grad = torch.autograd.grad(loss, adv)[0]

        if norm == ThreatModel.Linf.value:
            delta = delta + step_size * grad.sign()
            delta = delta.clamp(-eps, eps)
        else:
            grad_norm = batch_l2_norm(grad).clamp_min(1e-12)
            delta = delta + step_size * grad / grad_norm
            delta = project_l2(delta, eps)

        delta = torch.clamp(x_orig + delta, 0.0, 1.0) - x_orig

    return (x_orig + delta).clamp(0.0, 1.0).detach()


def evaluate_clean(model, loader, device, desc):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc=desc, leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)
            predictions = model(inputs).argmax(1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)
    return correct / max(total, 1)


def evaluate_pgd(model, loader, device, norm, eps, steps, step_size, random_start, desc):
    model.eval()
    correct = 0
    total = 0
    for inputs, targets in tqdm(loader, desc=desc, leave=False):
        inputs = inputs.to(device)
        targets = targets.to(device)
        adv_inputs = pgd_attack(model, inputs, targets, norm, eps, steps, step_size, random_start)
        with torch.no_grad():
            predictions = model(adv_inputs).argmax(1)
        correct += (predictions == targets).sum().item()
        total += targets.size(0)
    return correct / max(total, 1)


def evaluate_autoattack(model, loader, device, norm, eps, version, desc):
    model.eval()
    adversary = AutoAttack(model, norm=norm, eps=eps, version=version, device=device, verbose=False)
    correct = 0
    total = 0
    for inputs, targets in tqdm(loader, desc=desc, leave=False):
        inputs = inputs.to(device)
        targets = targets.to(device)
        adv_inputs = adversary.run_standard_evaluation(inputs, targets, bs=inputs.size(0))
        with torch.no_grad():
            predictions = model(adv_inputs).argmax(1)
        correct += (predictions == targets).sum().item()
        total += targets.size(0)
    return correct / max(total, 1)


def load_robustbench_model(model_name, dataset, threat_model, model_dir, device):
    model = load_model(
        model_name=model_name,
        model_dir=model_dir,
        dataset=dataset,
        threat_model=threat_model,
    )
    return model.to(device).eval()


def evaluate_variant(model, variant_name, loader, attacks, args, eps, pgd_step_size, device):
    metrics = {
        "variant": variant_name,
        "clean_acc": evaluate_clean(model, loader, device, desc=f"{variant_name} clean"),
    }
    if "pgd" in attacks:
        metrics["pgd_robust_acc"] = evaluate_pgd(
            model,
            loader,
            device,
            norm=args.threat_model,
            eps=eps,
            steps=args.pgd_steps,
            step_size=pgd_step_size,
            random_start=args.pgd_random_start,
            desc=f"{variant_name} pgd",
        )
    if "autoattack" in attacks:
        metrics["autoattack_robust_acc"] = evaluate_autoattack(
            model,
            loader,
            device,
            norm=args.threat_model,
            eps=eps,
            version=args.autoattack_version,
            desc=f"{variant_name} autoattack",
        )
    robust_values = [metrics[key] for key in ("pgd_robust_acc", "autoattack_robust_acc") if key in metrics]
    if robust_values:
        metrics["robust_acc"] = min(robust_values)
    return metrics


def resolve_attacks(args):
    if args.attack_method == "pgd":
        return {"pgd"}
    if args.attack_method == "autoattack":
        return {"autoattack"}
    if args.attack_method == "both":
        return {"pgd", "autoattack"}
    return {attack.strip().lower() for attack in args.attacks.split(",") if attack.strip()}


def build_hira_variant_name(base_name, args):
    sample_tag = "full" if args.hira_max_train_samples is None or args.hira_max_train_samples < 0 else str(args.hira_max_train_samples)
    block_tag = "" if args.hira_num_blocks == 2 else f"-blk{args.hira_num_blocks}"
    return (
        f"{base_name}-hira-v10-mlp-residual-randact{block_tag}"
        f"-exp{args.hira_expansion_dim}"
        f"-ep{args.hira_epochs}"
        f"-lr{_format_cache_value(args.hira_lr)}"
        f"-wd{_format_cache_value(args.hira_weight_decay)}"
        f"-ns{sample_tag}"
        f"-seed{args.hira_seed}"
    )


def resolve_variants(args):
    hira_values = [False, True] if args.use_hira is None else [args.use_hira]
    ranpac_values = [False, True] if args.use_ranpac is None else [args.use_ranpac]
    ranpac_methods = ["regression", "val_acc"] if args.ranpac_selection_method == "both" else [args.ranpac_selection_method]
    variant_order = [
        "original",
        "hira",
        "ranpac_regression",
        "ranpac_val_acc",
        "hira_ranpac_regression",
        "hira_ranpac_val_acc",
    ]
    variants = []
    for use_hira in hira_values:
        for use_ranpac in ranpac_values:
            selection_methods = ranpac_methods if use_ranpac else [None]
            for ranpac_selection_method in selection_methods:
                if not use_ranpac:
                    variant = "hira" if use_hira else "original"
                else:
                    base_name = "hira_ranpac" if use_hira else "ranpac"
                    variant = f"{base_name}_{ranpac_selection_method}"
                variants.append(
                    {
                        "variant": variant,
                        "use_hira": use_hira,
                        "use_ranpac": use_ranpac,
                        "ranpac_selection_method": ranpac_selection_method,
                    }
                )
    variants.sort(key=lambda item: variant_order.index(item["variant"]))
    return variants


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    eps = args.eps if args.eps is not None else default_eps(args.dataset, args.threat_model)
    pgd_step_size = args.pgd_step_size if args.pgd_step_size is not None else default_pgd_step_size(eps, args.pgd_steps)
    attacks = resolve_attacks(args)
    variants = resolve_variants(args)
    wandb_run = init_wandb(args)

    if any(variant["use_hira"] for variant in variants) and args.dataset != BenchmarkDataset.imagenet.value:
        raise NotImplementedError("HiRA evaluation is currently implemented only for ImageNet RobustBench models.")

    try:
        results = []
        step = 0
        for model_name in args.model_names:
            print(f"Evaluating {model_name} on {args.dataset} ({args.threat_model}), eps={eps}")
            eval_loader = build_eval_loader(
                dataset=args.dataset,
                threat_model=args.threat_model,
                model_name=model_name,
                data_dir=args.data_dir,
                n_examples=args.eval_examples,
                seed=args.seed,
                batch_size=args.eval_batch_size,
                num_workers=args.num_workers,
                eval_split=args.eval_split,
            )
            eval_examples = len(eval_loader.dataset)

            for variant_cfg in variants:
                model = load_robustbench_model(
                    model_name=model_name,
                    dataset=args.dataset,
                    threat_model=args.threat_model,
                    model_dir=args.model_dir,
                    device=device,
                )
                classifier_name = f"{args.dataset}_{args.threat_model}_{model_name}"
                if variant_cfg["use_hira"]:
                    model = apply_hira_adaptation(
                        model,
                        classifier_name=classifier_name,
                        dataset_root=args.hira_dataset_root or args.data_dir,
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
                    ).to(device).eval()
                    classifier_name = build_hira_variant_name(classifier_name, args)
                if variant_cfg["use_ranpac"]:
                    model = apply_ranpac_head(
                        model,
                        classifier_name=classifier_name,
                        dataset_root=args.data_dir,
                        rp_dim=args.ranpac_rp_dim,
                        batch_size=args.ranpac_fit_batch_size,
                        num_workers=args.num_workers,
                        seed=args.seed,
                        selection_method=variant_cfg["ranpac_selection_method"],
                        device=device,
                        cache_dir=args.ranpac_cache_dir,
                    ).to(device).eval()

                metrics = evaluate_variant(
                    model,
                    variant_name=variant_cfg["variant"],
                    loader=eval_loader,
                    attacks=attacks,
                    args=args,
                    eps=eps,
                    pgd_step_size=pgd_step_size,
                    device=device,
                )
                metrics.update(
                    {
                        "model_name": model_name,
                        "dataset": args.dataset,
                        "threat_model": args.threat_model,
                        "eps": eps,
                        "eval_examples": eval_examples,
                        "attack_method": args.attack_method or args.attacks,
                        "use_hira": variant_cfg["use_hira"],
                        "hira_num_blocks": args.hira_num_blocks if variant_cfg["use_hira"] else None,
                        "use_ranpac": variant_cfg["use_ranpac"],
                        "ranpac_selection_method": variant_cfg["ranpac_selection_method"],
                    }
                )
                results.append(metrics)
                log_wandb_result(wandb_run, metrics, step=step)
                step += 1
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        log_wandb_results_table(wandb_run, results)
        print(json.dumps(results, indent=2))
        if args.output_path:
            output_path = Path(args.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(results, indent=2) + "\n")
    finally:
        finish_wandb(wandb_run)


if __name__ == "__main__":
    main()
