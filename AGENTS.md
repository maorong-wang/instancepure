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
