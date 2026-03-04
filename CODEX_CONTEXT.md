# CODEX_CONTEXT

Last updated: 2026-03-04
Project: PainOrdinalCLIP
Scope: BioVid Part A pain intensity experiments (Baseline / CoOp / OrdinalCLIP / 2-class diagnostics)

## 1) Current Goal

- Improve performance on BioVid Part A pain intensity estimation.
- Validate whether CLIP visual features contain usable pain signal before adding heavier heads (e.g., Siamese ranking).

## 2) Environment & Repo State

### Evidence
- Main branch latest commit: `0e62496` (pushed to `origin/main`).
- Recent key commits:
  - `2abb4f4`: Baseline `last_project` added to optimizer.
  - `8e6a532`: CLIP normalization and diagnosis tools.
  - `0752d3e`: Baseline AMP fp16/fp32 fix.
  - `47dc485`: baseline script LF line endings fix.
  - `0e62496`: sync Claude updates and diagnostics.
- Working tree currently only shows submodule dirtiness: `m CLIP`.

### Inference
- Main repo code is synchronized and pushed; submodule local state should be checked separately before release tagging.

## 3) Data & Split Decisions

### Evidence
- Dataset used: BioVid Part A.
- Config keeps `val = test` (`configs/base_cfgs/data_cfg/datasets/biovid/biovid.yaml`).
- 5-class labels: BLN/PA1/PA2/PA3/PA4 mapped to 0..4.
- Added 2-class pipeline:
  - `scripts/data/build_2class.py`
  - `configs/base_cfgs/data_cfg/datasets/biovid/biovid_2class.yaml`
  - `classnames_2class.txt` (`no pain`, `intense pain`)
- 2-class preprocessing rule: keep BLN/PA4, remap to {0,1}, skip early frames by `frame_idx < 50` default.

### Inference
- 2-class setup is for feature separability diagnosis, not final benchmark.

## 4) Implemented Training/Config Changes

- CLIP normalization config added and used:
  - `configs/base_cfgs/data_cfg/transforms/clip-normalize.yaml`
- Freeze/fine-tune image encoder configs:
  - `freeze-image.yaml`, `tune-image-1e-4.yaml`, `tune-image-1e-5.yaml`
- Prompt tuning config:
  - `tune-ctx-rank-2e-3.yaml`
- Ordinal soft label option added in runner:
  - `ordinal_soft_label`, `ordinal_soft_label_sigma`
  - file: `ordinalclip/runner/runner.py`
- Baseline linear head optimization bug fixed:
  - `build_param_dict` now includes `last_project` when prompt learner is absent.
- Baseline AMP stability:
  - CLIP model cast to fp32 in `ordinalclip/models/baseline.py`

## 5) Experiment Evidence (from `diagnosis_report.txt`)

### Evidence
- 5-class results remain near random baseline (20%):
  - Best video `acc_max`: `0.2335` (baseline-vitb16-v2)
  - CoOp-frozen v2 `acc_max`: `0.2331`
  - OrdinalCLIP-frozen vitb16 `acc_max`: `0.2304`
- `pred_exp` MAE improved more clearly than accuracy:
  - CoOp-frozen-softlabel vitb16 `mae_exp = 1.1954` (better than many baselines).
- Old unfixed runs include sub-random behavior (~0.19), indicating earlier config/training issues were real.

### Inference
- Current bottleneck is likely feature separability rather than prompt wording only.
- 2-class BLN vs PA4 should be run first to verify whether CLIP can separate extremes.

## 6) Scripts to Run

- Full 5-class matrix:
  - `bash scripts/experiments/run_biovid_matrix.sh --backbone vitb16 --max_epochs 50`
- Baseline only:
  - `bash scripts/experiments/run_biovid_baseline_only.sh --max_epochs 30`
- Build and run 2-class diagnostics:
  - `python3 scripts/data/build_2class.py`
  - `bash scripts/experiments/run_biovid_2class.sh --backbone vitb16 --max_epochs 50`
- Parse and diagnose:
  - `python3 scripts/experiments/parse_video_results.py -d results/ -r test`
  - `python3 scripts/diagnosis/diagnose_biovid.py -d results/biovid-2cls-baseline-vitb16`

## 7) Decision Log

- Date: 2026-03-04
- Decision: Keep CLIP normalization + freeze-image baseline as default controls.
- Why: Prevents known degradation from ImageNet stats mismatch and unstable image encoder updates.
- Alternatives considered: direct aggressive fine-tune first, immediate Siamese head.
- Impact: Stable baseline for ablation; clearer diagnosis path.
- Revisit trigger: if 2-class BLN/PA4 still near 50% after tuning.

- Date: 2026-03-04
- Decision: Prioritize 2-class diagnostic before Siamese ranking head.
- Why: Ranking head is useful only after basic separability exists.
- Alternatives considered: add Siamese immediately.
- Impact: Avoids expensive optimization on potentially non-separable features.
- Revisit trigger: if 2-class > 60-65%, then move to ranking/Siamese on 5-class.

## 8) Handoff

- What changed:
  - Multiple BioVid configs/scripts, diagnosis pipeline, 2-class pipeline, normalization/optimizer fixes.
- What was validated:
  - Code-level fixes are in repo and pushed to `main`.
  - `diagnosis_report.txt` confirms current 5-class performance regime.
- What failed / uncertain:
  - 5-class accuracy still low (~22-23%).
  - 2-class experiment results not yet included in diagnosis report.
- Next 1-3 actions:
  1. Run full 2-class suite (A/B/C/D) on vitb16.
  2. Compare rn50 vs vitb16 under same 2-class protocol.
  3. If 2-class weak, pivot to AU/temporal features before Siamese expansion.
- Risks/blockers:
  - BioVid subtle expression differences + subject shift may limit frame-only CLIP discrimination.
