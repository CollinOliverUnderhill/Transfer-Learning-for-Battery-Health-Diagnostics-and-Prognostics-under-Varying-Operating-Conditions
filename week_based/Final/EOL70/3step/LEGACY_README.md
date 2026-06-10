Legacy 3-step entry points

These wrappers preserve the earlier week5 EOL70 workflow behavior without overwriting the current expanded scripts.

Files
- stage1_source_search_optuna_legacy.py
- stage2_finetune_search_optuna_legacy.py
- stage3_final_evaluate_legacy.py

Legacy defaults
- Stage 1: narrow search, fixed dropout=0.0, fixed activation=relu, 80 trials, top_k=5, max_feature_candidates=20, width_candidates=8,16,32,64, max_depth=4
- Stage 2: early-stop driven selection, 80 trials, top_k=5, support_ratios=0.67,1.0, ft_epoch_choices=200,400,800, replay_weight_choices=0.0,0.1,0.3,1.0
- Stage 3: earlier final-evaluation defaults with dropout=0.0, activation=relu, and no forced minimum epoch / no forced minimum early-stop gate

Recommended legacy commands
1. Stage 1
python stage1_source_search_optuna_legacy.py --out_dir outputs\\stage1_legacy

2. Stage 2
python stage2_finetune_search_optuna_legacy.py --stage1_top_csv outputs\\stage1_legacy\\stage1_top_source_checkpoints.csv --out_dir outputs\\stage2_legacy

3. Stage 3
python stage3_final_evaluate_legacy.py --stage2_best_csv outputs\\stage2_legacy\\stage2_best_configs.csv --out_dir outputs\\stage3_final_legacy
