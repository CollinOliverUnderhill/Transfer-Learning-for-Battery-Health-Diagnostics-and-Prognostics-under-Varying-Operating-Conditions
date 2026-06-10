# SOH Estimation Results

This folder contains SOH estimation code and restored SOH results from:

`E:\Datasets\IVAS\Originaltrails\chunqiu_codes\Dropped\SOHestimation_results`

The SOH experiments are RPT-sample-level predictions. Each row is an RPT sample
with `time_week`, `rpt_idx`, and target `soh`. These results are not fixed
early-week checkpoint experiments like the week-based RUL results.

## Structure

- `Ridge_codes/`
  - Ridge SOH training/evaluation scripts.
- `MLP_codes/`
  - MLP SOH training/evaluation scripts.
- `results/ridge_single_cell/`
  - Ridge single-cell SOH estimation results.
  - `seed_sweep_by_cell/`: per-cell seed sweep results.
  - `rolling_by_cell/`: rolling single-cell SOH prediction example.
- `results/ridge_cross_cell/`
  - Ridge cross-cell/cross-group SOH estimation results.
  - `domain_shift/`: explicit train-group/test-group domain split results.
  - `single_cell_train_multi_cell_test/`: one-cell train, multi-cell test result.
  - `random_cell_holdout_baseline/`: release-cell holdout baseline from the original Ridge root.
- `results/mlp_cross_cell/`
  - MLP cross-cell/cross-group SOH estimation results, including source-only and
    target-domain fine-tuned experiments.
- `results/soh_results_index.csv`
  - Summary index of available SOH results and key metrics.

## Recommended Results For Paper Use

- Ridge single-cell baseline:
  - `results/ridge_single_cell/seed_sweep_by_cell/_seed_summary_mape/singlecell_singlefeature_summary_stats.csv`
  - Cell-level summary: MAE mean `0.02059`, RMSE mean `0.02313`, R2 mean `0.89665`, MAPE mean `2.636%`.
- Ridge cross-cell baseline:
  - `results/ridge_cross_cell/domain_shift/scheme2_37train_12test/`
  - Test: 45 cells / 12 groups, MAE `0.04857`, RMSE `0.05675`, R2 `0.88598`, MAPE `5.811%`.
- MLP cross-cell source-only result:
  - `results/mlp_cross_cell/MLP_8L_3F178/scheme2_37train_12test/`
  - Test: 40 cells / 11 groups, MAE `0.02063`, RMSE `0.02595`, R2 `0.934`, MAPE `2.45%`.
- MLP cross-cell transfer result:
  - `results/mlp_cross_cell/MLP_8L_3F178/scheme2_37train_13ft_12test_ft800_128x8_newlogic/`
  - Target-test finetuned: 40 cells / 11 groups, MAE `0.02054`, RMSE `0.02569`, R2 `0.935`, MAPE `2.43%`.

## Notes

- Target column is `soh`.
- Time axis is `time_week`, but the model is trained on all available RPT samples
  in the selected train cells/groups rather than on one fixed week checkpoint.
- Cross-cell experiments split at cell/group level, not by random row-level split.
