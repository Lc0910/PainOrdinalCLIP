# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a fork/adaptation of **OrdinalCLIP** — a PyTorch implementation of "OrdinalCLIP: Learning Rank Prompts for Language-Guided Ordinal Regression". The project applies ordinal regression using CLIP's vision-language model with learnable rank prompts, adapted here for pain assessment tasks.

## Environment Setup

```bash
conda env create -f environment.yaml
conda activate ordinalclip
pip install -r requirements.txt
pip install -e CLIP/        # OpenAI CLIP as git submodule
pip install -e .            # install ordinalclip package
```

Dev tools:
```bash
pip install bandit==1.7.0 black==22.3.0 flake8==3.9.1 isort==5.8.0 mypy==0.902 pre-commit==2.13.0 pytest ipython
pre-commit install
```

## Commands

**Training (single run):**
```bash
python scripts/run.py --config configs/default.yaml --config configs/base_cfgs/runner_cfg/model/ordinalclip.yaml --config configs/base_cfgs/data_cfg/datasets/morph/morph.yaml
```

**Test only:**
```bash
python scripts/run.py --config configs/default.yaml ... --test_only
```

**Sweep experiments (multi-run):**
```bash
python scripts/experiments/meta_config_generator.py -c $meta_config_file
python scripts/experiments/config_sweeper.py --max_num_gpus_used_in_parallel 8 --num_jobs_per_gpu 1 -d $sweep_config_dir
```

**Visualize ordinality:**
```bash
CUDA_VISIBLE_DEVICES=-1 find results/ -name 'config.yaml' -exec python scripts/vis_ordinality.py -c {} \;
```

**Parse results:**
```bash
python scripts/experiments/parse_results.py -d $result_dir -p 'test_stats.json'
```

**Run tests:**
```bash
pytest tests/
pytest tests/test_runner.py   # single test file
```

**Linting / formatting:**
```bash
black --line-length 120 .
isort .
flake8 .
```

## Architecture

### Core Data Flow

```
Image → image_encoder → image_features [B, D]
                                              → cosine similarity → logits [B, num_ranks]
PromptLearner() → text_encoder → text_features [num_ranks, D]
```

The model computes softmax over `num_ranks` class logits, then predicts the ordinal rank either via expectation (`exp`) or argmax (`max`).

### Key Components

**`ordinalclip/models/ordinalclip.py` — `OrdinalCLIP`**
- Wraps CLIP's image encoder and text encoder
- Holds a `PromptLearner` (either `PlainPromptLearner` or `RankPromptLearner`)
- `forward(images)` returns `(logits [B, num_ranks], image_features, text_features)`
- All CLIP weights are cast to float32 at init

**`ordinalclip/models/prompt_leaners/plain_prompt_learner.py` — `PlainPromptLearner`**
- Learns `context_embeds` (shared or rank-specific) and `rank_embeds`
- Builds pseudo-sentences: `<sot> [context] [rank_i] <full_stop> <eot>` with positional variants (`tail`/`front`/`middle`)
- `forward()` returns `sentence_embeds [num_ranks, 77, D]`

**`ordinalclip/models/prompt_leaners/rank_prompt_learner.py` — `RankPromptLearner`**
- Extends `PlainPromptLearner` with interpolation between base rank embeddings (`num_base_ranks`)
- Interpolation types: `linear`, `cosine`, etc.

**`ordinalclip/runner/runner.py` — `Runner` (pl.LightningModule)**
- Combines CE loss + KL divergence loss (weighted)
- Metrics: MAE and accuracy computed via both `exp` (expectation) and `max` (argmax) strategies
- Checkpoints saved by best `val_mae_max_metric`; results written to `{output_dir}/val_stats.json` and `test_stats.json`

**`ordinalclip/runner/data.py` — `RegressionDataModule`**
- Data files are plain text: `<image_path> <label1> [label2 ...]` (multiple rater labels supported)
- At test/val: uses median label; at train: samples label randomly
- Supports few-shot sampling and label distribution shifting

### Config System

Configs use **OmegaConf** with hierarchical YAML merging:
```bash
python scripts/run.py --config configs/default.yaml --config configs/base_cfgs/...
```
`default.yaml` defines all keys with default values. Override via additional `--config` files or `--cfg_options key=value`.

Main config sections: `runner_cfg`, `data_cfg`, `trainer_cfg`, `test_only`.

Models and prompt learners use a **registry** pattern (`MODELS.register_module()`, `PROMPT_LEARNERS.register_module()`) from MMCV. Instantiated via `MODELS.build(cfg)` where `cfg.type` is the class name.

### Required Files

- **CLIP weights** → `.cache/clip/RN50.pt` or `ViT-B-16.pt`
- **Dataset** → `data/MORPH/` with annotation files `data_list/*.txt`
- **CLIP submodule** → `CLIP/` (installed as editable package)

### Experiment Outputs

Each run creates a versioned directory:
```
results/<experiment_name>/version_N/
  config.yaml, run.log
  ckpts/          # best + last checkpoint
  csv_logger/     # pytorch_lightning CSV logs
  val_stats.json, test_stats.json, ordinality.json
```
