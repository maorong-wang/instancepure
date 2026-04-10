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
Shared working idea: robustness gains may come from injecting large random projections while preserving the base backbone. Treat RanPAC as the reference head. For HiRA experiments, prefer ViT-family post-MLP attachment on the last `hira_num_blocks` transformer blocks (default `2`): keep the original MLP, then attach `A(GELU(B(soft_threshold(x))))` after the MLP output with fixed Gaussian `B`. Fit `A` by closed-form ridge regression against the frozen original MLP outputs. Do not switch HiRA `B` or RanPAC `W_rand` to block-orthogonal initialization unless explicitly revisiting that ablation.

Current active code state:
- `RanPAC` cache version is `17` in `classifiers/ranpac.py`.
- `HiRA` cache version is `31` in `classifiers/hira.py`.
- RanPAC now fits only the `regression` ridge-selection variant; the older validation-accuracy ridge search was removed.
- RanPAC supports a projected feature-mode switch. The default is the original `GELU(B(x))` feature. The new non-default mode computes `z = GELU(B(x))`, estimates clean projected mean/std, builds nonnegative standardized projected-distance features, and centers them with `phi = r - mean_train(r)` before ridge regression.
- RanPAC target construction supports hard negatives: the ridge targets are GT one-hot by default, but when `ranpac_hardneg_topk > 0` and `ranpac_hardneg_gamma > 0`, the top confusing non-GT classes from the teacher logits receive total suppression weight `-ranpac_hardneg_gamma`.
- Adaptation noise is supported for both RanPAC and HiRA with `adapt_noise_eps`, `adapt_noise_num`, and `adapt_alpha`. The current implementation uses noisy-only adaptation statistics when noise is enabled, rather than mixing clean and noisy stats.
- Stability-aware diagonal ridge is supported for both RanPAC and HiRA with `stability_ridge_gamma` and `stability_ridge_stat_eps`. The current one-pass implementation builds a diagonal prior from projected train-feature `mean_abs / std` statistics and uses it inside the closed-form ridge solve.
- Current RanPAC inference formula is:
  `logit = (1 - ranpac_lambda) * (logit_baseline - mean_train_logits) + ranpac_lambda * (logit_ranpac / ranpac_temp)`.
- `mean_train_logits` is currently a single global scalar mean over all baseline logits on the clean RanPAC train split, not a per-class vector.
- The baseline-logit scalar bias is cached inside the RanPAC state as `baseline_logit_mean`; changing this behavior requires a cache-version bump.
- Important mismatch to remember: the `RanPAC` code currently allows `ranpac_lambda >= 0`, but the CLI help text in `test.py` and `eval_robustbench_ranpac.py` still describes it as a convex mixing weight. Check `classifiers/ranpac.py` before assuming `ranpac_lambda <= 1`.
- Current InstantPure and RobustBench result naming adds a `bbias` suffix for RanPAC variants because the baseline-logit scalar bias subtraction changed the effective head behavior.
- HiRA currently uses half precision during fitting on CUDA via `HiRAHalfPrecisionWrapper` / autocast, but evaluation is forced back to fp32 by `_prepare_hira_model_for_eval`.

## New Thread Handoff
This project is an ImageNet adversarial-defense codebase centered on `InstantPure`, a purification-based method that denoises inputs with a diffusion pipeline before classification. The repo now also evaluates adversarially trained models from RobustBench under the same RanPAC and HiRA ideas.

Our main method is to modify the victim classifier rather than the purifier. `RanPAC` replaces the final linear layer with a random-projection ridge head from [classifiers/ranpac.py](/home_fmg/maorong/python/InstantPure/classifiers/ranpac.py): backbone features are optionally soft-thresholded at inference, then projected by fixed Gaussian `W_rand`. The default projected feature is `GELU(B(x))`, while the new non-default projected-distance mode first computes `z = GELU(B(x))`, then uses train-estimated projected mean/std to build centered distance features before ridge regression. The cached state now includes the RP weights, the chosen ridge, the hard-negative target settings, the noise-adaptation settings, the clean backbone-feature mean/std used by soft thresholding, the projected-feature mode/statistics, the stability-aware diagonal ridge settings, and a scalar clean-train baseline-logit mean used at inference time. `HiRA` lives in [classifiers/hira.py](/home_fmg/maorong/python/InstantPure/classifiers/hira.py): for ViT-family backbones only, keep the original MLP and attach `A(GELU(B(soft_threshold(x))))` after the MLP output in the last `hira_num_blocks` transformer blocks. `B` is fixed Gaussian, and each `A` is solved by closed-form ridge regression to match the original frozen MLP outputs, with optional stability-aware diagonal ridge regularization from the same train-pass projected statistics. Current preference is Gaussian initialization; do not use block-orthogonal init.

Purification-based evaluation is in [test.py](/home_fmg/maorong/python/InstantPure/test.py). It loads a victim classifier through [archs.py](/home_fmg/maorong/python/InstantPure/archs.py), optionally applies HiRA and/or RanPAC, samples a random ImageNet `test` subset, attacks the classifier directly, and reports both baseline and purified metrics. Key outputs are raw classifier clean accuracy (`classifier_accuracy`), raw classifier robust accuracy under the attack (`original_classifier_robust_accuracy`), purified clean accuracy, and purified robust accuracy after the diffusion denoiser. Attacks include Foolbox PGD, AutoAttack, and `stadv` imported from `../DiffPure`. `advertorch` is not used. The InstantPure CLI currently exposes the RanPAC projected-feature mode and mixing / hard-negative knobs, the inference-only soft-threshold knobs, and the stability-aware ridge knobs, and RanPAC variants are tagged with `_bbias` in output paths.

Adversarial-training evaluation is in [eval_robustbench_ranpac.py](/home_fmg/maorong/python/InstantPure/eval_robustbench_ranpac.py). It loads RobustBench models, evaluates `original`, `hira`, `ranpac_regression`, and combined variants, on a random sampled eval set with clean accuracy plus PGD and/or AutoAttack robust accuracy. RobustBench preprocessing is taken from `robustbench.data.get_preprocessing(...)` and passed into the official `benchmark(...)` call for `autoattack_version=standard`. Local AutoAttack variants are also supported:
- `standard`: official RobustBench `benchmark(...)` with `to_disk=True`
- `full`: local AutoAttack with `apgd-ce`, `apgd-dlr`, `fab`, `square`
- `rand`: local AutoAttack with `apgd-ce`, `apgd-dlr`, one restart, configurable `autoattack_eot_iter`
- `apgdt`: local AutoAttack with `apgd-t`, `n_restarts=1`, `n_target_classes=9` for `Linf`/`L2`

The RobustBench evaluation code also computes RanPAC diagnostics from clean logits:
- reference-target margin changes for the baseline top confusing classes
- top-1 confusing-class preservation
- top-k overlap / ordered match / rank-shift statistics

Current sweep and cache assumptions:
- Existing RanPAC caches before version `17` and HiRA caches before version `31` are stale for the current GELU / projected-distance feature modes with pre-projection soft-threshold statistics plus stability-aware diagonal ridge support, and should be retrained.
- Existing RobustBench benchmark result files can collide when the `benchmark_model_name` is reused, so the current code appends `-bbias` to RanPAC variant names to avoid mixing old and new on-disk AutoAttack outputs.
- The default RobustBench sweep currently targets ImageNet, `autoattack_version=standard`, and includes `ranpac_temp`.
- The current `sweeps/robustbench_ranpac_imagenet.yaml` project name still reflects an older experiment label (`robustbench-ranpac-hira-rc-rp_normalization`) and is not guaranteed to match the current head behavior; update it explicitly if that label matters for a fresh sweep.

## Suggested First Prompt
Open `AGENTS.md`, `classifiers/hira.py`, `classifiers/ranpac.py`, `test.py`, and `eval_robustbench_ranpac.py`, then summarize the current InstantPure and RobustBench evaluation flows, the active RanPAC/HiRA design choices, and any cache/version assumptions before making changes.
