# Repository Guidelines

## Project Structure & Module Organization
Top-level scripts drive most workflows: `train_lora.py` handles LoRA distillation training, `test.py` runs ImageNet evaluation, and `get_args.py` centralizes training CLI arguments. Shared helpers live in `dataset.py`, `utils.py`, `load_dm.py`, `archs.py`, and `ddim_solver.py`. The `guided_diffusion/` package contains modified diffusion internals. Keep paper figures in `asset/` and treat `pretrained/` as local checkpoint storage, not a place for new large artifacts.

## Build, Test, and Development Commands
Install dependencies with `pip install -r requirements.txt`. Configure `accelerate` once before training with `accelerate config`. Start training with `bash train_lora.sh`; it launches `accelerate launch train_lora.py` with the repository’s default LoRA settings. Run evaluation with `bash test.sh`, or inspect options with `python test.py --help`. For a quick syntax smoke test after edits, run `python -m compileall .`.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, `snake_case` for functions and variables, `PascalCase` for classes, and `UPPER_CASE` for constants. Keep new CLI flags in `get_args.py` when they affect training. Prefer small helper functions over long inline blocks, and add brief comments only where tensor flow or checkpoint logic is non-obvious. No formatter or linter is checked in, so match the surrounding style closely and keep imports grouped consistently.

## Testing Guidelines
This repository uses script-level validation rather than unit tests. For evaluation changes, run `bash test.sh` or a reduced check such as `python test.py --model LCM --num_validation_set 32 ...` before opening a PR. For training changes, run a short `accelerate launch train_lora.py ... --max_train_steps 10 --max_train_samples 32` smoke test when possible. Document the exact command, GPU, and key metrics you used.

## Commit & Pull Request Guidelines
Recent history uses short imperative subjects like `Update train_lora.py` and `Add checkpoint link to README`. Keep commit titles concise, present tense, and scoped to the changed area. PRs should explain the behavioral impact, list the commands you ran, note any dataset or checkpoint assumptions, and include sample metrics or output images when evaluation behavior changes.

## Configuration Tips
Dataset paths default to local folders in `dataset.py` such as `./image_net`; avoid committing machine-specific path changes. Keep secrets, private dataset locations, and large checkpoints out of version control.

## Current Research Notes
Shared working idea: improve robustness by modifying the victim classifier rather than retraining the backbone or changing the purifier. The two main victim-side methods are:
- `RanPAC`: replace the final linear head with a fixed Gaussian random projection + `GELU` + closed-form ridge regression head.
- `HiRA`: attach a pre-MLP adapter to the last `hira_num_blocks` ViT/Swin blocks, fit it in closed form to reconstruct the frozen original MLP input, and optionally apply MeanSparse-style inference-only sparsification in the hidden projected space.

Keep these design choices unless explicitly revisiting an ablation:
- Use fixed Gaussian initialization for both HiRA `A` / `b_rand` and RanPAC `W_rand`.
- Do not switch to block-orthogonal projection by default.
- Keep MeanSparse behavior inference-only for HiRA; do not apply it during HiRA fitting.
- Do not apply MeanSparse / soft-thresholding inside RanPAC.

Current active code state:
- `RanPAC` cache version is `17` in `classifiers/ranpac.py`.
- `HiRA` cache version is `32` in `classifiers/hira.py`.
- RanPAC supports only the `regression` ridge-selection variant; the older validation-accuracy ridge search was removed.
- RanPAC target construction supports hard negatives: GT one-hot by default, with optional suppression of top confusing non-GT classes via `ranpac_hardneg_topk` and `ranpac_hardneg_gamma`.
- Adaptation noise is supported for both RanPAC and HiRA with `adapt_noise_eps`, `adapt_noise_num`, and `adapt_alpha`. When enabled, the current code uses noisy-only adaptation statistics rather than mixing clean and noisy stats.
- Stability-aware diagonal ridge is supported for both RanPAC and HiRA with `stability_ridge_gamma` and `stability_ridge_stat_eps`. The diagonal prior is computed from projected train-feature `mean_abs / std` statistics and inserted into the closed-form ridge solve.
- HiRA attaches before the target MLPs. Fitting uses `B(GELU(A(x)))` against the frozen original MLP input. Inference uses `B(sparse(GELU(A(x))))`, where `sparse` is MeanSparse-style soft thresholding in hidden space.
- HiRA inference behavior is selectable with `soft_threshold_mode`:
  - `near_mean`: pull ambiguous in-band hidden features toward the mean.
  - `away_from_mean`: push ambiguous in-band hidden features toward the nearest `mean +/- alpha * std` boundary.
- HiRA cache naming intentionally ignores pure soft-threshold changes. Changing `soft_threshold_alpha`, `soft_threshold_beta`, `soft_threshold_stat_eps`, or `soft_threshold_mode` should not force HiRA retraining; the cached weights are reused and the new threshold behavior is applied only at inference.
- Current RanPAC inference formula is:
  `logit = (1 - ranpac_lambda) * logit_baseline + ranpac_lambda * (logit_ranpac / ranpac_temp)`.
- `ranpac_lambda = 1` means pure RanPAC head output. The code allows `ranpac_lambda >= 0`; it is not restricted to `[0, 1]` even if some CLI help still describes it as a convex mixing weight.
- RanPAC no longer subtracts any baseline-logit mean at inference. Old `v17` caches may still contain an unused `baseline_logit_mean` field, but they remain valid because the fitted RanPAC weights are unchanged.
- Current result naming no longer uses the old `bbias` suffix.
- HiRA fitting uses half precision on CUDA via `HiRAHalfPrecisionWrapper` / autocast, but evaluation is forced back to fp32 by `_prepare_hira_model_for_eval`.

## New Thread Handoff
This repo now has two evaluation tracks:

1. Purification-based ImageNet evaluation around `InstantPure`.
2. Adversarial-training-based evaluation on RobustBench models.

### 1. Purification-based evaluation flow

Purification evaluation is centered on [test.py](/home_fmg/maorong/python/InstantPure/test.py). The current architecture is explicitly split into:
- victim model loader: [victims/imagenet.py](/home_fmg/maorong/python/InstantPure/victims/imagenet.py)
- optional victim wrappers (HiRA / RanPAC): [victims/wrappers.py](/home_fmg/maorong/python/InstantPure/victims/wrappers.py)
- purifier layer: [purifiers/](/home_fmg/maorong/python/InstantPure/purifiers)
- attack layer: [attacks/](/home_fmg/maorong/python/InstantPure/attacks)

The evaluation pipeline in `test.py` is:
1. Build a clean timm ImageNet victim from `victims/imagenet.py`.
2. Optionally fit/apply HiRA and/or RanPAC through `victims/wrappers.py`.
3. Build a purifier through `purifiers/factory.py`.
4. Compose `PurifiedClassifier(purifier, classifier)`.
5. Build an attack through `attacks/factory.py`.
6. Report raw-victim and purified clean/robust metrics.

Important behavior:
- `test.py` will fit HiRA/RanPAC on the clean ImageNet training split automatically before purifier construction if wrapper caches are missing. The user does not need to run `fit_victim_wrappers.py` first.
- `fit_victim_wrappers.py` exists as a dedicated clean-only wrapper fitting / cache warmup script. It can also run a small clean validation check.
- `archs.py` is now only a compatibility wrapper used by `train_lora.py`; it delegates to `victims/` and `victims/wrappers.py`.

Current victim-side modules:
- [victims/imagenet.py](/home_fmg/maorong/python/InstantPure/victims/imagenet.py): timm-backed ImageNet victim loader plus canonical aliases. Accepts both known aliases and arbitrary valid timm model names.
- [victims/wrappers.py](/home_fmg/maorong/python/InstantPure/victims/wrappers.py): shared wrapper config and application logic for HiRA / RanPAC.
- Supported HiRA backbones are currently ViT-family models only in the code sense of names starting with `vit` or `swin`.

Current purifier layer:
- [purifiers/base.py](/home_fmg/maorong/python/InstantPure/purifiers/base.py): base purifier abstractions and `PurifiedClassifier`.
- [purifiers/instantpure.py](/home_fmg/maorong/python/InstantPure/purifiers/instantpure.py): current InstantPure backend with batched Canny control-image generation, diffusion `sdedit(...)`, and optional LoRA loading.
- [purifiers/factory.py](/home_fmg/maorong/python/InstantPure/purifiers/factory.py): registry.
- `purifier_name=none` gives identity purification.
- `purifier_name=instantpure` is implemented.
- `purifier_name=instancepure` and `purifier_name=puriflow` are placeholders that currently raise `NotImplementedError`; the external repos/backends are not present locally.

Current attack layer:
- [attacks/factory.py](/home_fmg/maorong/python/InstantPure/attacks/factory.py): registry / dispatch.
- [attacks/standard.py](/home_fmg/maorong/python/InstantPure/attacks/standard.py): standard attacks (`Linf_pgd`, `L2_pgd`, `stadv`, `AutoAttack`, `target_Linf_pgd`).
- [attacks/sdedit.py](/home_fmg/maorong/python/InstantPure/attacks/sdedit.py): current diffusion PGD attack (`diff_pgd`) that uses `purifier.sdedit(...)` inside the attack loop.

Attack semantics in `test.py`:
- `attack_method=diff_pgd` always uses the SDEdit-based diffusion PGD branch, even if `attack_version=v1`. This was a source of user confusion; the effective attack is chosen by `attack_method`.
- `attack_version=v2` also forces the same diffusion PGD branch for legacy compatibility.
- Standard attacks use `attack_target` to attack either the raw wrapped victim or the purified composed model.
- `diff_pgd` attacks the raw wrapped victim through the purifier’s `sdedit(...)`; it does not use `attack_target`.

Metric semantics in `test.py`:
- `classifier_accuracy`: clean accuracy of the wrapped victim without purification.
- `original_classifier_robust_accuracy`: robust accuracy of the wrapped victim under the chosen attack.
- `clean_accuracy`: clean accuracy after purification.
- `robust_accuracy`: robust accuracy after purification.
- `attack_fail_rate`: currently identical to `original_classifier_robust_accuracy`; it is not a separate fail-rate computation.
- `ranpac_baseline_bias_centered` is currently always `False`.

### 2. RobustBench evaluation flow

Adversarial-training evaluation is in [eval_robustbench_ranpac.py](/home_fmg/maorong/python/InstantPure/eval_robustbench_ranpac.py). It:
1. Loads a RobustBench model.
2. Optionally applies HiRA and/or RanPAC.
3. Evaluates clean accuracy and PGD / AutoAttack robust accuracy.
4. Optionally computes RanPAC diagnostics from pre/post clean logits.

RobustBench variants currently evaluated are:
- `original`
- `hira`
- `ranpac_regression`
- `hira_ranpac_regression`

AutoAttack modes:
- `standard`: official RobustBench `benchmark(...)` with `to_disk=True`
- `full`: local AutoAttack with `apgd-ce`, `apgd-dlr`, `fab`, `square`
- `rand`: local AutoAttack with `apgd-ce`, `apgd-dlr`, one restart, configurable `autoattack_eot_iter`
- `apgdt`: local AutoAttack with `apgd-t` and RobustBench-style targeted settings

EOT semantics:
- Local PGD uses no EOT.
- Local AutoAttack `full` uses no configurable EOT.
- Local AutoAttack `rand` uses `autoattack_eot_iter`.
- Local AutoAttack `apgdt` explicitly uses `eot_iter = 1`.
- Official `standard` uses upstream RobustBench / AutoAttack behavior; this wrapper does not expose EOT control there.

RanPAC diagnostics in RobustBench:
- Before applying RanPAC, the script may collect `pre_ranpac_logits`.
- After applying RanPAC, it may collect `post_ranpac_logits`.
- These extra passes feed `_compute_ranpac_diagnostics(...)` and produce metrics such as confusing-class overlap, rank shift, and reference-target margin changes.
- The tqdm labeled `pre_ranpac_logits` is diagnostic overhead only. It does not change the clean/robust evaluation result itself.

### Current active experiment defaults

RobustBench sweep in [sweeps/robustbench_ranpac_imagenet.yaml](/home_fmg/maorong/python/InstantPure/sweeps/robustbench_ranpac_imagenet.yaml):
- ImageNet / `Linf`
- `autoattack_version=standard`
- `use_hira=true`
- `use_ranpac=true`
- `hira_expansion_dim=16384`
- `hira_num_blocks=4`
- `adapt_noise_num=1`
- `soft_threshold_alpha=0.9`
- `soft_threshold_beta=4.0`
- `soft_threshold_mode=away_from_mean`
- `ranpac_rp_dim=10000`
- `ranpac_lambda=0.5`
- `ranpac_hardneg_topk=0`
- `ranpac_hardneg_gamma=0.0`
- `ranpac_fit_batch_size=64`

Purification sweep in [sweeps/imagenet_vit_swin_ranpac.yaml](/home_fmg/maorong/python/InstantPure/sweeps/imagenet_vit_swin_ranpac.yaml):
- `purifier_name=instantpure`
- `model=LCM`
- `attack_method=diff_pgd`
- currently targets `vit_small`, `vit_tiny`, `swin_b`, `swin_s`
- `use_hira_adapter=true`
- `use_ranpac_head=true`
- same main HiRA / RanPAC hyperparameters as the RobustBench sweep

### Important caveats / non-obvious behavior

- `test.py --help` and some old comments may still mention legacy `attack_version` semantics, but `attack_method=diff_pgd` is the clearer current entry point for diffusion attacks.
- Some CLI help still calls `ranpac_lambda` a convex weight, but the implementation allows `ranpac_lambda >= 0`.
- The purification refactor introduced new modules under `victims/`, `purifiers/`, `attacks/`, and `fit_victim_wrappers.py`. If moving to another machine or another git-based workspace, make sure those files are actually included; earlier stale-job issues were caused by a worker running an older tree.
- `instancepure`, `puriflow`, `diffhammer`, and `diffattack` are currently only placeholders. The abstraction is ready, but the external backends are not wired.
- The current purification evaluation loop in `test.py` uses `batch_size=1` for the ImageNet eval loader and can be slow.

## Suggested First Prompt
Open `AGENTS.md`, `classifiers/hira.py`, `classifiers/ranpac.py`, `victims/wrappers.py`, `test.py`, and `eval_robustbench_ranpac.py`, then summarize the current victim-wrapper architecture, the active HiRA / RanPAC behavior, the purification and RobustBench evaluation flows, and the current sweep defaults before making changes.
