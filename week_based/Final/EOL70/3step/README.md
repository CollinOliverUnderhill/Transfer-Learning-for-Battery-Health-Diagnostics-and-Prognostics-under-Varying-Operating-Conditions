# 3-Step Optuna Pipeline for EOL70 / week5

This folder is the week-based 3-step tuning workflow for the final `EOL70 / w5` setup.

It is adapted from the old `Dropped/EFC180/EOL70/3Step` flow, but all defaults now point to the week-based final workspace:

- data: `features/feature_table_all_cells_multiweek_EOL70.csv`
- split: `domain_split/cell_split_targetspread_w5_EOL70.csv`
- target: `lifetime_weeks_EOL70`
- runner: `codes/run_lifetime_transfer_mlp.py`

The reporting metrics now include:

- `MAE`
- `RMSE`
- `MAPE`
- `SMAPE`
- `WMAPE`
- `R2`

## Files

- `three_step_common.py`
  - shared defaults and helpers
- `three_step_transfer_runner.py`
  - standalone runner used by Stage 1/2/3
- `stage1_source_search_optuna.py`
  - source-only search over feature sets and source model hyperparameters
- `stage2_finetune_search_optuna.py`
  - fine-tune search over transfer hyperparameters
- `stage3_final_evaluate.py`
  - final untouched-target-test evaluation using the best Stage 2 config
- `build_feature_candidates.py`
  - rebuilds `features/informed_feature_candidates_w5_EOL70.csv`

## Default search behavior

The defaults are aligned with the validated `fixed_8x8x8_freeze2_w5` direction:

- activation: `relu`
- dropout: `0.0`
- batch size: `16`
- source epochs: `800`
- early-stop patience: `25`

Stage 1 searches:

- `features`
- `hidden_dims`
- `lr`
- `weight_decay`

Stage 2 searches:

- `ft_lr`
- `ft_weight_decay`
- `ft_epochs`
- `ft_freeze_hidden_layers`
- `target_support_ratio`
- `transfer_replay_weight`

## Typical run order

### 1. Build candidate feature sets

```powershell
python E:\Datasets\IVAS\Codes\chunqiu_codes\week_based\Final\EOL70\3step\build_feature_candidates.py
```

### 2. Stage 1

```powershell
D:\Anaconda\envs\torchenv\python.exe `
  E:\Datasets\IVAS\Codes\chunqiu_codes\week_based\Final\EOL70\3step\stage1_source_search_optuna.py `
  --out_dir E:\Datasets\IVAS\Codes\chunqiu_codes\week_based\Final\EOL70\3step\outputs\stage1
```

Main outputs:

- `outputs/stage1/stage1_trials.csv`
- `outputs/stage1/stage1_top_source_checkpoints.csv`

### 3. Stage 2

```powershell
D:\Anaconda\envs\torchenv\python.exe `
  E:\Datasets\IVAS\Codes\chunqiu_codes\week_based\Final\EOL70\3step\stage2_finetune_search_optuna.py `
  --stage1_top_csv E:\Datasets\IVAS\Codes\chunqiu_codes\week_based\Final\EOL70\3step\outputs\stage1\stage1_top_source_checkpoints.csv `
  --out_dir E:\Datasets\IVAS\Codes\chunqiu_codes\week_based\Final\EOL70\3step\outputs\stage2
```

Main outputs:

- `outputs/stage2/stage2_trials.csv`
- `outputs/stage2/stage2_best_configs.csv`

### 4. Stage 3

```powershell
D:\Anaconda\envs\torchenv\python.exe `
  E:\Datasets\IVAS\Codes\chunqiu_codes\week_based\Final\EOL70\3step\stage3_final_evaluate.py `
  --stage2_best_csv E:\Datasets\IVAS\Codes\chunqiu_codes\week_based\Final\EOL70\3step\outputs\stage2\stage2_best_configs.csv `
  --out_dir E:\Datasets\IVAS\Codes\chunqiu_codes\week_based\Final\EOL70\3step\outputs\stage3_final
```

Main outputs:

- `outputs/stage3_final/stage3_final_report.json`
- `outputs/stage3_final/target_test_overall_metrics.csv`

## Notes

- Stage 2 does not evaluate the untouched target test set.
- Stage 3 is the only step that should be used for the final held-out test decision.
- `SMAPE` and `WMAPE` are exported in both the trial-level CSVs and the final JSON report.
