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
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from classifiers.hira import apply_hira_adaptation, build_hira_variant_name
from classifiers.ranpac import apply_ranpac_head
from dataset import get_dataset as instantpure_get_dataset

try:
    from autoattack import AutoAttack
except ImportError:
    AutoAttack = None

try:
    from robustbench import benchmark, load_model
    from robustbench.data import get_preprocessing
    from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel
except ImportError:
    fallback_root = Path(__file__).resolve().parent.parent / "adversarial-attacks-pytorch"
    if str(fallback_root) not in sys.path:
        sys.path.insert(0, str(fallback_root))
    from robustbench import benchmark, load_model
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

OFFICIAL_AUTOATTACK_VERSIONS = {"standard"}
CUSTOM_AUTOATTACK_VERSIONS = {"rand", "full", "apgdt"}


def str2bool(v):
    if isinstance(v, bool):
        return v
    value = str(v).lower()
    if value in {"yes", "true", "t", "y", "1"}:
        return True
    if value in {"no", "false", "f", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


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
    parser.add_argument(
        "--autoattack-version",
        "--autoattack_version",
        default="standard",
        help="AutoAttack mode. Use 'standard' for official RobustBench benchmark(), 'full' for local APGD-CE/APGD-DLR/FAB/Square, 'rand' for local APGD-CE/APGD-DLR, or 'apgdt' for local APGD-T with RobustBench-standard targeted settings.",
    )
    parser.add_argument(
        "--autoattack-eot-iter",
        "--autoattack_eot_iter",
        type=int,
        default=1,
        help="EOT iterations for local 'rand' AutoAttack APGD components. Official RobustBench benchmark() ignores this.",
    )
    parser.add_argument(
        "--use-hira",
        "--use_hira",
        type=str2bool,
        default=None,
        help="If set, evaluate only the original model (false) or only the HiRA model (true).",
    )
    parser.add_argument("--hira-expansion-dim", "--hira_expansion_dim", type=int, default=4096, help="Hidden expansion dimension for HiRA adapters.")
    parser.add_argument("--hira-num-blocks", "--hira_num_blocks", type=int, default=2, help="Number of final transformer MLP blocks that receive HiRA adapters.")
    parser.add_argument("--hira-batch-size", "--hira_batch_size", type=int, default=32, help="HiRA closed-form fitting batch size.")
    parser.add_argument("--hira-num-workers", "--hira_num_workers", type=int, default=4, help="HiRA closed-form fitting DataLoader workers.")
    parser.add_argument("--hira-epochs", "--hira_epochs", type=int, default=1, help="Legacy compatibility flag. Closed-form HiRA ignores this value.")
    parser.add_argument("--hira-lr", "--hira_lr", type=float, default=1e-4, help="Legacy compatibility flag. Closed-form HiRA ignores this value.")
    parser.add_argument("--hira-weight-decay", "--hira_weight_decay", type=float, default=1e-4, help="Legacy compatibility flag. Closed-form HiRA ignores this value.")
    parser.add_argument("--hira-seed", "--hira_seed", type=int, default=0, help="Seed used to sample the closed-form HiRA projection and dataset split.")
    parser.add_argument("--hira-cache-dir", "--hira_cache_dir", default="pretrained/hira_robustbench", help="HiRA cache directory.")
    parser.add_argument("--hira-dataset-root", "--hira_dataset_root", default="", help="Optional ImageNet root used to fit closed-form HiRA. Defaults to --data-dir.")
    parser.add_argument("--hira-max-train-samples", "--hira_max_train_samples", type=int, default=-1, help="Optional cap on ImageNet train samples used to fit closed-form HiRA.")
    parser.add_argument("--hira-force-retrain", "--hira_force_retrain", type=str2bool, default=False, help="Ignore cached HiRA weights and fit again.")
    parser.add_argument("--adapt-noise-eps", "--adapt_noise_eps", type=float, default=0.0, help="Linf noise radius used while fitting HiRA and RanPAC adaptation statistics.")
    parser.add_argument("--adapt-noise-num", "--adapt_noise_num", type=int, default=0, help="Number of noisy samples per training image used while fitting HiRA and RanPAC.")
    parser.add_argument("--adapt-alpha", "--adapt_alpha", type=float, default=1.0, help="Total weight assigned to noisy adaptation statistics relative to clean statistics.")
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
        choices=["regression"],
        default="regression",
        help="Which cached RanPAC head to evaluate. Only regression-loss ridge selection is supported.",
    )
    parser.add_argument(
        "--ranpac-lambda",
        "--ranpac_lambda",
        type=float,
        default=1.0,
        help="Residual weight for the RanPAC branch added on top of the original classifier head.",
    )
    parser.add_argument("--ranpac-cache-dir", "--ranpac_cache_dir", default="pretrained/ranpac_robustbench", help="RanPAC cache directory.")
    parser.add_argument(
        "--diagnostic-topk",
        "--diagnostic_topk",
        type=int,
        default=9,
        help="Top-k confusing classes used for RanPAC targeted-margin and rank-preservation diagnostics.",
    )
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


def freeze_backbone(model):
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model.eval()


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


def resolve_model_preprocessing(dataset, threat_model, model_name):
    return get_preprocessing(BenchmarkDataset(dataset), ThreatModel(threat_model), model_name, None)


def build_eval_loader(dataset, threat_model, model_name, data_dir, n_examples, seed, batch_size, num_workers, eval_split, transform=None):
    if transform is None:
        transform = resolve_model_preprocessing(dataset, threat_model, model_name)
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


def evaluate_autoattack_benchmark(model, benchmark_model_name, args, eps, device, preprocessing):
    model.eval()
    if args.autoattack_eot_iter != 1:
        print(
            "RobustBench benchmark() uses standard AutoAttack and does not expose the "
            f"configured EOT setting; ignoring eot_iter={args.autoattack_eot_iter}."
        )
    clean_acc, robust_acc = benchmark(
        model,
        model_name=benchmark_model_name,
        n_examples=args.eval_examples,
        dataset=args.dataset,
        threat_model=args.threat_model,
        eps=eps,
        device=device,
        batch_size=args.eval_batch_size,
        data_dir=args.data_dir,
        to_disk=True,
        preprocessing=preprocessing,
    )
    return clean_acc, robust_acc


def _loader_to_tensors(loader, desc):
    input_batches = []
    target_batches = []
    for inputs, targets in tqdm(loader, desc=desc, leave=False):
        input_batches.append(inputs.cpu())
        target_batches.append(targets.cpu())
    if not input_batches:
        raise ValueError("AutoAttack evaluation loader is empty.")
    return torch.cat(input_batches, dim=0), torch.cat(target_batches, dim=0)


def _tensor_accuracy(model, inputs, targets, batch_size, device):
    model.eval()
    correct = 0
    total = inputs.size(0)
    with torch.no_grad():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_inputs = inputs[start:end].to(device)
            batch_targets = targets[start:end].to(device)
            predictions = model(batch_inputs).argmax(1)
            correct += (predictions == batch_targets).sum().item()
    return correct / max(total, 1)


def _collect_logits(model, loader, device, desc):
    model.eval()
    logits_batches = []
    target_batches = []
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc=desc, leave=False):
            inputs = inputs.to(device)
            logits_batches.append(model(inputs).float().cpu())
            target_batches.append(targets.cpu())
    if not logits_batches:
        raise ValueError("Diagnostic loader is empty.")
    return torch.cat(logits_batches, dim=0), torch.cat(target_batches, dim=0)


def _confusing_class_order(logits, targets):
    masked_logits = logits.clone()
    masked_logits[torch.arange(masked_logits.size(0)), targets] = float("-inf")
    return masked_logits.argsort(dim=1, descending=True)


def _compute_ranpac_diagnostics(reference_logits, variant_logits, targets, topk):
    if reference_logits.shape != variant_logits.shape:
        raise ValueError("Reference and variant logits must have identical shapes for diagnostics.")
    if targets.ndim != 1 or targets.size(0) != reference_logits.size(0):
        raise ValueError("Targets must be a 1D tensor aligned with the diagnostic logits.")

    reference_predictions = reference_logits.argmax(dim=1)
    variant_predictions = variant_logits.argmax(dim=1)
    joint_correct_mask = (reference_predictions == targets) & (variant_predictions == targets)

    metrics = {
        "diagnostic_num_samples": int(targets.size(0)),
        "diagnostic_joint_clean_count": int(joint_correct_mask.sum().item()),
    }
    if not joint_correct_mask.any():
        return metrics

    reference_logits = reference_logits[joint_correct_mask]
    variant_logits = variant_logits[joint_correct_mask]
    targets = targets[joint_correct_mask]
    topk = min(topk, reference_logits.size(1) - 1)
    if topk <= 0:
        return metrics

    reference_order = _confusing_class_order(reference_logits, targets)
    variant_order = _confusing_class_order(variant_logits, targets)
    reference_topk = reference_order[:, :topk]
    variant_topk = variant_order[:, :topk]

    target_positions = torch.arange(topk, dtype=torch.float32).view(1, -1)
    variant_positions = torch.empty_like(variant_order)
    variant_positions.scatter_(
        1,
        variant_order,
        torch.arange(variant_order.size(1), dtype=torch.long).view(1, -1).expand_as(variant_order),
    )

    true_logits_reference = reference_logits.gather(1, targets.view(-1, 1))
    true_logits_variant = variant_logits.gather(1, targets.view(-1, 1))
    reference_target_logits_reference = reference_logits.gather(1, reference_topk)
    reference_target_logits_variant = variant_logits.gather(1, reference_topk)
    variant_target_logits_variant = variant_logits.gather(1, variant_topk)

    overlap = (reference_topk.unsqueeze(2) == variant_topk.unsqueeze(1)).any(dim=2).float()
    rank_shift = (
        variant_positions.gather(1, reference_topk).float() - target_positions
    ).abs()

    metrics.update(
        {
            f"apgdt_ref_targets_margin_before_top{topk}": (
                true_logits_reference - reference_target_logits_reference
            ).mean().item(),
            f"apgdt_ref_targets_margin_after_top{topk}": (
                true_logits_variant - reference_target_logits_variant
            ).mean().item(),
            f"apgdt_self_targets_margin_after_top{topk}": (
                true_logits_variant - variant_target_logits_variant
            ).mean().item(),
            "apgdt_ref_target1_margin_before": (
                true_logits_reference - reference_target_logits_reference[:, :1]
            ).mean().item(),
            "apgdt_ref_target1_margin_after": (
                true_logits_variant - reference_target_logits_variant[:, :1]
            ).mean().item(),
            f"confusing_top{topk}_overlap": overlap.mean().item(),
            "confusing_top1_preservation": (reference_topk[:, 0] == variant_topk[:, 0]).float().mean().item(),
            f"confusing_top{topk}_ordered_match": (reference_topk == variant_topk).all(dim=1).float().mean().item(),
            f"confusing_top{topk}_rank_shift_mean": rank_shift.mean().item(),
        }
    )
    metrics[f"apgdt_ref_targets_margin_delta_top{topk}"] = (
        metrics[f"apgdt_ref_targets_margin_after_top{topk}"] - metrics[f"apgdt_ref_targets_margin_before_top{topk}"]
    )
    metrics["apgdt_ref_target1_margin_delta"] = (
        metrics["apgdt_ref_target1_margin_after"] - metrics["apgdt_ref_target1_margin_before"]
    )
    return metrics


def evaluate_autoattack_custom(model, loader, device, norm, eps, version, eot_iter, batch_size, desc):
    if AutoAttack is None:
        raise ImportError("autoattack is not installed. Install it or use --autoattack_version standard.")

    model.eval()
    clean_inputs, clean_targets = _loader_to_tensors(loader, desc=f"{desc}_materialize")
    adversary = AutoAttack(model, norm=norm, eps=eps, version="custom", device=device, verbose=False)

    if version == "rand":
        adversary.attacks_to_run = ["apgd-ce", "apgd-dlr"]
        adversary.apgd.n_restarts = 1
        adversary.apgd.eot_iter = eot_iter
    elif version == "full":
        adversary.attacks_to_run = ["apgd-ce", "apgd-dlr", "fab", "square"]
    elif version == "apgdt":
        adversary.attacks_to_run = ["apgd-t"]
        if norm in {"Linf", "L2"}:
            adversary.apgd_targeted.n_restarts = 1
            adversary.apgd_targeted.n_target_classes = 9
        else:
            adversary.apgd_targeted.use_largereps = True
            adversary.apgd_targeted.n_target_classes = 5
        adversary.apgd_targeted.eot_iter = 1
    else:
        raise ValueError(f"Unsupported local autoattack version '{version}'.")

    clean_acc = _tensor_accuracy(model, clean_inputs, clean_targets, batch_size=batch_size, device=device)
    adv_inputs = adversary.run_standard_evaluation(clean_inputs, clean_targets, bs=batch_size)
    robust_acc = _tensor_accuracy(model, adv_inputs.cpu(), clean_targets, batch_size=batch_size, device=device)
    return clean_acc, robust_acc


def load_robustbench_model(model_name, dataset, threat_model, model_dir, device):
    model = load_model(
        model_name=model_name,
        model_dir=model_dir,
        dataset=dataset,
        threat_model=threat_model,
    )
    return model.to(device).eval()


def evaluate_variant(model, variant_name, benchmark_model_name, loader, attacks, args, eps, pgd_step_size, device, preprocessing):
    metrics = {"variant": variant_name}
    if "autoattack" in attacks:
        autoattack_version = args.autoattack_version.lower()
        if autoattack_version in OFFICIAL_AUTOATTACK_VERSIONS:
            clean_acc, autoattack_robust_acc = evaluate_autoattack_benchmark(
                model,
                benchmark_model_name=benchmark_model_name,
                args=args,
                eps=eps,
                device=device,
                preprocessing=preprocessing,
            )
        elif autoattack_version in CUSTOM_AUTOATTACK_VERSIONS:
            clean_acc, autoattack_robust_acc = evaluate_autoattack_custom(
                model,
                loader,
                device,
                norm=args.threat_model,
                eps=eps,
                version=autoattack_version,
                eot_iter=args.autoattack_eot_iter,
                batch_size=args.eval_batch_size,
                desc=f"{variant_name} autoattack_{autoattack_version}",
            )
        else:
            raise ValueError(
                f"Unsupported autoattack version '{args.autoattack_version}'. "
                "Use one of: standard, full, rand, apgdt."
            )
        metrics["clean_acc"] = clean_acc
        metrics["autoattack_robust_acc"] = autoattack_robust_acc
    else:
        metrics["clean_acc"] = evaluate_clean(model, loader, device, desc=f"{variant_name} clean")
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


def resolve_variants(args):
    hira_values = [False, True] if args.use_hira is None else [args.use_hira]
    ranpac_values = [False, True] if args.use_ranpac is None else [args.use_ranpac]
    variant_order = [
        "original",
        "hira",
        "ranpac_regression",
        "hira_ranpac_regression",
    ]
    variants = []
    for use_hira in hira_values:
        for use_ranpac in ranpac_values:
            ranpac_selection_method = args.ranpac_selection_method if use_ranpac else None
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
            model_preprocessing = resolve_model_preprocessing(args.dataset, args.threat_model, model_name)
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
                transform=model_preprocessing,
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
                reference_logits = None
                reference_targets = None
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
                        train_transform=model_preprocessing,
                        adapt_noise_eps=args.adapt_noise_eps,
                        adapt_noise_num=args.adapt_noise_num,
                        adapt_alpha=args.adapt_alpha,
                    ).to(device).eval()
                    classifier_name = build_hira_variant_name(
                        classifier_name,
                        expansion_dim=args.hira_expansion_dim,
                        epochs=args.hira_epochs,
                        lr=args.hira_lr,
                        weight_decay=args.hira_weight_decay,
                        max_train_samples=args.hira_max_train_samples,
                        seed=args.hira_seed,
                        num_adapter_blocks=args.hira_num_blocks,
                        adapt_noise_eps=args.adapt_noise_eps,
                        adapt_noise_num=args.adapt_noise_num,
                        adapt_alpha=args.adapt_alpha,
                    )
                if variant_cfg["use_ranpac"]:
                    reference_logits, reference_targets = _collect_logits(
                        model,
                        eval_loader,
                        device,
                        desc=f"{variant_cfg['variant']} pre_ranpac_logits",
                    )
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
                        train_transform=model_preprocessing,
                        adapt_noise_eps=args.adapt_noise_eps,
                        adapt_noise_num=args.adapt_noise_num,
                        adapt_alpha=args.adapt_alpha,
                        ranpac_lambda=args.ranpac_lambda,
                    ).to(device).eval()
                model = freeze_backbone(model).to(device).eval()
                benchmark_model_name = f"{model_name}-{variant_cfg['variant']}"
                if variant_cfg["use_ranpac"] and args.ranpac_lambda != 1.0:
                    benchmark_model_name = f"{benchmark_model_name}-lam{str(args.ranpac_lambda).replace('/', '_')}"

                metrics = evaluate_variant(
                    model,
                    variant_name=variant_cfg["variant"],
                    benchmark_model_name=benchmark_model_name,
                    loader=eval_loader,
                    attacks=attacks,
                    args=args,
                    eps=eps,
                    pgd_step_size=pgd_step_size,
                    device=device,
                    preprocessing=model_preprocessing,
                )
                if variant_cfg["use_ranpac"]:
                    variant_logits, variant_targets = _collect_logits(
                        model,
                        eval_loader,
                        device,
                        desc=f"{variant_cfg['variant']} post_ranpac_logits",
                    )
                    if not torch.equal(reference_targets, variant_targets):
                        raise ValueError("Diagnostic targets changed between pre- and post-RanPAC logits.")
                    metrics.update(
                        _compute_ranpac_diagnostics(
                            reference_logits,
                            variant_logits,
                            reference_targets,
                            topk=args.diagnostic_topk,
                        )
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
                        "ranpac_lambda": args.ranpac_lambda,
                        "adapt_noise_eps": args.adapt_noise_eps,
                        "adapt_noise_num": args.adapt_noise_num,
                        "adapt_alpha": args.adapt_alpha,
                    }
                )
                results.append(metrics)
                log_wandb_result(wandb_run, metrics, step=step)
                step += 1
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        log_wandb_results_table(wandb_run, results)
        output_payload = results
        print(json.dumps(output_payload, indent=2))
        if args.output_path:
            output_path = Path(args.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(output_payload, indent=2) + "\n")
    finally:
        finish_wandb(wandb_run)


if __name__ == "__main__":
    main()
