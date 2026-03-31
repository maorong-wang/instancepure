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
Shared working idea: robustness gains may come from injecting large random projections while preserving the base backbone. Treat RanPAC as the reference head. For HiRA experiments, prefer ViT-family post-MLP attachment on the last `hira_num_blocks` transformer blocks (default `2`): keep the original MLP, then attach `A(GELU(Bx))` after the MLP output with fixed Gaussian `B`. Fit `A` by closed-form ridge regression against the frozen original MLP outputs. Do not switch HiRA `B` or RanPAC `W_rand` to block-orthogonal initialization unless explicitly revisiting that ablation.

## New Thread Handoff
This project is an ImageNet adversarial-defense codebase centered on `InstantPure`, a purification-based method that denoises inputs with a diffusion pipeline before classification. The repo now also evaluates adversarially trained models from RobustBench under the same RanPAC and HiRA ideas.

Our main method is to modify the victim classifier rather than the purifier. `RanPAC` replaces the final linear layer with a random-projection ridge head from [classifiers/ranpac.py](/home_fmg/maorong/python/InstantPure/classifiers/ranpac.py): features are projected by fixed Gaussian `W_rand`, passed through `GELU`, then fit by ridge regression; both `regression` and `val_acc` ridge-selection variants are cached. `HiRA` lives in [classifiers/hira.py](/home_fmg/maorong/python/InstantPure/classifiers/hira.py): for ViT-family backbones only, keep the original MLP and attach `A(GELU(Bx))` after the MLP output in the last `hira_num_blocks` transformer blocks. `B` is fixed Gaussian, and each `A` is solved by closed-form ridge regression to match the original frozen MLP outputs. Current preference is Gaussian initialization; do not use block-orthogonal init.

Purification-based evaluation is in [test.py](/home_fmg/maorong/python/InstantPure/test.py). It loads a victim classifier through [archs.py](/home_fmg/maorong/python/InstantPure/archs.py), optionally applies HiRA and/or RanPAC, samples a random ImageNet `test` subset, attacks the classifier directly, and reports both baseline and purified metrics. Key outputs are raw classifier clean accuracy (`classifier_accuracy`), raw classifier robust accuracy under the attack (`original_classifier_robust_accuracy`), purified clean accuracy, and purified robust accuracy after the diffusion denoiser. Attacks include Foolbox PGD, AutoAttack, and `stadv` imported from `../DiffPure`. `advertorch` is not used.

Adversarial-training evaluation is in [eval_robustbench_ranpac.py](/home_fmg/maorong/python/InstantPure/eval_robustbench_ranpac.py). It loads RobustBench models, evaluates `original`, `hira`, `ranpac_regression`, `ranpac_val_acc`, and combined variants, on a random sampled eval set with clean accuracy plus PGD and/or AutoAttack robust accuracy. W&B logging and sweeps are already wired in `sweeps/robustbench_ranpac_imagenet.yaml`.

## Suggested First Prompt
Open `AGENTS.md`, `classifiers/hira.py`, `classifiers/ranpac.py`, `test.py`, and `eval_robustbench_ranpac.py`, then summarize the current InstantPure and RobustBench evaluation flows, the active RanPAC/HiRA design choices, and any cache/version assumptions before making changes.
