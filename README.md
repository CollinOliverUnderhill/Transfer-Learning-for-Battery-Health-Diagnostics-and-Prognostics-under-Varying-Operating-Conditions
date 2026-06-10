# Transfer Learning for Battery Health Diagnostics and Prognostics under Varying Operating Conditions

This repository contains the thesis code used for battery health diagnostics and lifetime prognostics under heterogeneous operating conditions. The main workflow is a week-based remaining useful life (RUL) transfer-learning pipeline under an end-of-life threshold defined at 70% state of health (EOL70). It is supported by upstream feature extraction, state of health (SOH) baselines, alternative EOL-threshold processing scripts, and thesis figure-generation scripts.

## Repository Layout

- `week_based/`
  - Main thesis workflow.
  - `Final/EOL70/codes/`: week-based feature construction, split generation, correlation analysis, and the MLP transfer-learning runner.
  - `Final/EOL70/3step/`: staged Optuna workflow for source search, fine-tuning search, and final held-out target evaluation.
  - `Feature_extraction/`, `Feature_engineering/`, `SOHest/`: shared feature and SOH baseline utilities required by the final workflow.
- `dataprocess/Lifetime_RUL_prediction/`
  - Workspaces and scripts for alternative EOL thresholds: EOL50, EOL55, EOL60, EOL65, EOL70, EOL75, and EOL80. In this notation, EOLxx means the lifetime label is defined by the cell reaching xx% SOH.
- `figures/` and `figurecaptions/`
  - Scripts used to regenerate thesis figures and appendix material.
- `metadata/`
  - Lightweight metadata copied from the IVAS workspace, including `Groupcondi.csv` and `Valid_cells.csv`.
- `legacy_scripts/`
  - Root-level IVAS preprocessing and benchmark refresh scripts.
- `references/`
  - Reference feature-extraction scripts adapted from prior early-prediction work.

Large runtime artifacts are intentionally excluded: model checkpoints, Optuna databases, generated figures, raw JSON/RPT data, and full result folders.

## Thesis Terminology

The terminology follows the thesis definitions:

- `SOH` means state of health. In this work it is a capacity-based diagnostic label derived from reference performance test (RPT) measurements and describes the current degradation state of a cell relative to its fresh condition.
- `EOL` means end of life. An EOL threshold is the SOH level at which the cell is treated as having reached the end of its usable life for the selected modelling definition.
- `EOL70` means the EOL threshold is defined at 70% SOH. The main RUL experiments use this threshold because it provides the lifetime labels for the final early-life prognostics workflow.
- `RUL` means remaining useful life. For an observation at week `t`, the RUL label is the remaining time until the cell reaches the selected EOL threshold. In this repository, `lifetime_weeks_EOL70` and related labels use weeks as the time unit.
- `Week-based` or `early-life` features are features extracted at a selected observation week, such as `w3`, `w5`, or `w10`. The main thesis configuration uses early-life features before the cell approaches EOL; week 5 is the main reported feature week.
- `Domain` refers to the battery data distribution associated with cells or cell groups under particular operating conditions. In the cross-condition setting, source and target domains are separated by cell group and operating-condition structure, especially charge C-rate, discharge C-rate, and depth of discharge.
- `Source domain` is the cell-group data used for source model training or pretraining. `Target domain` is the held-out related cell-group data used for limited fine-tuning and final target-test evaluation.
- `Fine-tuning transfer learning` means first training an MLP on source-domain data and then adapting the learned parameters using a limited number of labelled target-domain cells.
- `HPO` means hyperparameter optimization. The staged workflow uses Optuna with the TPE sampler to select feature subsets, MLP architecture and training settings, and fine-tuning settings.

## Main RUL Workflow Under the 70% SOH End-of-Life Threshold

Run commands from the repository root unless a script requires another working directory.

1. Build or refresh the multi-week feature table used for RUL labels under the 70% SOH EOL threshold:

```powershell
python .\week_based\Final\EOL70\codes\extract_ivas_lifetime_multiweek_and_augment_eol.py
```

2. Generate source/target domain splits with target fine-tuning cells spread across the target lifetime range:

```powershell
python .\week_based\Final\EOL70\codes\build_weekly_target_spread_splits.py
```

3. Analyze correlations between early-life features and lifetime labels:

```powershell
python .\week_based\Final\EOL70\codes\analyze_feature_lifetime_correlations_multiweek.py
```

4. Run the fixed MLP transfer-learning baseline:

```powershell
python .\week_based\Final\EOL70\codes\run_lifetime_transfer_mlp.py
```

5. Run the staged HPO and final held-out target-test evaluation:

```powershell
python .\week_based\Final\EOL70\3step\build_feature_candidates.py
python .\week_based\Final\EOL70\3step\stage1_source_search_optuna.py
python .\week_based\Final\EOL70\3step\stage2_finetune_search_optuna.py
python .\week_based\Final\EOL70\3step\stage3_final_evaluate.py
```

See `week_based/Final/EOL70/README.md` and `week_based/Final/EOL70/3step/README.md` for workflow-specific notes.

## Data Assumptions

The experimental data comes from the Iowa State University / Iowa Lakes Community College dataset:

- Dataset: [ISU-ILCC Battery Aging Dataset](https://iastate.figshare.com/articles/dataset/_b_ISU-ILCC_Battery_Aging_Dataset_b_/22582234)
- DOI: [10.25380/iastate.22582234](https://doi.org/10.25380/iastate.22582234)
- License: CC BY 4.0

The original scripts were developed in the local IVAS workspace. The main RUL workflow under the 70% SOH end-of-life threshold now uses repository-relative paths for copied feature tables, split files, and `metadata/Groupcondi.csv`. Raw cycling/RPT data is not included, and some legacy scripts still point to local absolute paths, especially:

- `E:\Datasets\IVAS\Data`

If running on a different machine, update script arguments or defaults to point at the local raw-data location and Python environment.

## Environment

Core dependencies are listed in `requirements.txt`. The deep-learning workflows require PyTorch; install the CPU or CUDA build appropriate for the machine.

```powershell
python -m pip install -r requirements.txt
```

## Version-Control Policy

The repository is intended to track source code, experiment definitions, split CSVs, feature summaries, and documentation. Generated outputs should stay out of Git unless they are deliberately promoted as small, curated artifacts.
