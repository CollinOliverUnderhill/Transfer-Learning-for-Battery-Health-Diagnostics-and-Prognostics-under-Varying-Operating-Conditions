# Thesis Appendix

This file collects the proposed appendix material for the thesis. The generated tables and figures are stored under `Figurecaption/Appendixfile` in section-specific folders. Appendix figures are provided as PNG and PDF files.

Important boundary: this appendix reports data availability, engineered features, model configuration, sensitivity checks, and reproducibility notes. It does not define a new target-cell selection metric; random target-cell selection is reported only as an empirical sensitivity result.

## Appendix A: Dataset and Cell Availability

Purpose: document cell-level availability and operating conditions used by the RUL experiments under the end-of-life (EOL) threshold defined at 70% state of health (SOH). This section supports the statement that week-based sensitivity is affected by changing sample availability.

# Appendix A Generated Tables

## Dataset Summary

| item                                                                                                           |   value | source                                                     |
|:---------------------------------------------------------------------------------------------------------------|--------:|:-----------------------------------------------------------|
| Valid cells listed before 70% SOH EOL threshold feature-table construction                                     |     251 | Valid_cells.csv                                            |
| Operating-condition groups                                                                                     |      63 | Groupcondi.csv                                             |
| Cells with available lifetime label under the end-of-life (EOL) threshold defined at 70% state of health (SOH) |     247 | ivas_lifetime_eol_availability.csv                         |
| Cells missing lifetime label under the end-of-life (EOL) threshold defined at 70% state of health (SOH)        |       4 | ivas_lifetime_eol_availability.csv                         |
| Cells retained in final multi-week feature table for 70% SOH EOL threshold                                     |     242 | final feature table for 70% SOH EOL threshold              |
| Groups retained in final multi-week feature table for 70% SOH EOL threshold                                    |      63 | final feature table for 70% SOH EOL threshold              |
| Valid-list cells not retained in final feature table for 70% SOH EOL threshold                                 |       9 | comparison between Valid_cells.csv and final feature table |

## Label Availability at 70% SOH EOL threshold

For the end-of-life (EOL) threshold defined at 70% state of health (SOH), 247 of 251 cells have an available lifetime label; 4 cells are missing this label.

## Week-Based Feature Availability

| week_label   |   usable_non_nan_cells |   status_ok_cells |   feature_or_nan_unusable_cells |   total_cells |
|:-------------|-----------------------:|------------------:|--------------------------------:|--------------:|
| w3           |                    237 |               240 |                               5 |           242 |
| w5           |                    231 |               236 |                              11 |           242 |
| w6           |                    228 |               232 |                              14 |           242 |
| w7           |                    223 |               228 |                              19 |           242 |
| w8           |                    217 |               224 |                              25 |           242 |
| w9           |                    207 |               221 |                              35 |           242 |
| w10          |                    192 |               212 |                              50 |           242 |
| w15          |                    117 |               148 |                             125 |           242 |

## Cell-Retention Summary

| availability_status                                 |   cell_count |   fraction_of_valid_cells | example_cells                                  |
|:----------------------------------------------------|-------------:|--------------------------:|:-----------------------------------------------|
| retained_final_70pct_soh_eol_feature_table          |          242 |                0.964143   | G1C1, G1C4, G2C1, G2C2, G2C3, G2C4, G3C1, G3C2 |
| not_retained_final_feature_table_reason_not_encoded |            5 |                0.0199203  | G1C2, G1C3, G6C3, G18C1, G26C3                 |
| missing_70pct_soh_eol_label_and_week_features       |            1 |                0.00398406 | G14C4                                          |
| missing_70pct_soh_eol_label                         |            3 |                0.0119522  | G57C1, G57C2, G57C4                            |

Suggested figures:

- `Figurecaption/Appendixfile/AppendixA/appendix_A_week_availability.png` / `.pdf`
- `Figurecaption/Appendixfile/AppendixA/appendix_A_eol_label_availability.png` / `.pdf`

## Appendix B: Engineered Feature Definitions

Purpose: list the engineered early-cycle features used for RUL prediction and provide the feature-correlation context for the main w5 experiments.

Notation:

- Let `w0` denote the baseline diagnostic week and `wk` denote a later early-life diagnostic week. In the main experiment reported here, `k = 5`.
- For any descriptor `x`, `Delta x_{wk-w0} = x_{wk} - x_{w0}`.
- `dQ/dV` denotes the incremental-capacity curve.
- `CV time` denotes the constant-voltage charging time.
- `C_chg` denotes charge C-rate.
- `DoD` denotes depth of discharge.

Basic feature categories:

- IC-curve features quantify early electrochemical response changes from `dQ/dV` curves in selected voltage windows.
- CV-time features describe changes in constant-voltage charging behaviour.
- Usage-condition features include `DoD`, `C_chg`, and their interaction term.
- Capacity-response features describe early capacity-related changes through `Delta Q^1` and `Delta Q^3` descriptors.

Files:

- `Figurecaption/Appendixfile/AppendixB/appendix_B_feature_notation_categories.csv`
- `Figurecaption/Appendixfile/AppendixB/appendix_B_feature_definitions.csv`
- `Figurecaption/Appendixfile/AppendixB/appendix_B_w5_feature_correlations.csv`
- `Figurecaption/Appendixfile/AppendixB/appendix_B_w5_feature_value_summary.csv`
- `Figurecaption/Appendixfile/AppendixB/appendix_B_w5_correlation_heatmap.png` / `.pdf`

Feature notation, category, and interpretation:

1. `f1_w5` (IC-curve feature)

   $\log(|\mathrm{mean}(\Delta(dQ/dV)_{w_5-w_0}^{3.6-3.9V})|)$

   Average early-life change in the incremental-capacity curve within the 3.6-3.9 V window.

2. `f2_w5` (CV-time feature)

   $\log(|\Delta CV\ time_{w_5-w_0}|)$

   Absolute early-life change in constant-voltage charging time.

3. `f3_w5` (Usage-condition feature)

   $DoD$

   Depth of discharge, representing the fraction of available capacity used during cycling.

4. `f4_w5` (Capacity-response feature)

   $\Delta Q^1_{w_5-w_0}$

   Early-life change in the first capacity-related descriptor.

5. `f5_w5` (Usage-condition feature)

   $C_{chg}^{0.5}DoD^{0.5}$

   Interaction between charge C-rate and depth of discharge.

6. `f6_w5` (Usage-condition feature)

   $C_{chg}$

   Charge C-rate, describing charging current relative to nominal cell capacity.

7. `f7_w5` (IC-curve feature)

   $\log(\mathrm{var}(\Delta(dQ/dV)_{w_5-w_0}^{3.0-3.6V}))$

   Variance of early-life incremental-capacity curve change within the 3.0-3.6 V window.

8. `f8_w5` (Capacity-response feature)

   $\Delta Q^3_{w_5-w_0}$

   Early-life change in the third capacity-related descriptor, complementary to the first descriptor.

9. `f9_w5` (IC-curve feature)

   $\log(|\mathrm{mean}(\Delta(dQ/dV)_{w_5-w_0}^{3.0-3.6V})|)$

   Average early-life change in the incremental-capacity curve within the lower-voltage 3.0-3.6 V window.

10. `f10_w5` (CV-time feature)

   $\log(|CV\ time_{w_0}|)$

   Initial constant-voltage charging time before the diagnostic ageing window.

Current numerical values for the week-5 feature table:

| feature   |   n_non_nan |    mean |    std |      min |   median |     max |   pearson_r_with_lifetime |   spearman_r_with_lifetime |
|:----------|------------:|--------:|-------:|---------:|---------:|--------:|--------------------------:|---------------------------:|
| f1_w5     |         236 | -2.8026 | 0.3153 |  -3.6162 |  -2.7476 | -1.4751 |                   -0.8278 |                    -0.8876 |
| f2_w5     |         236 |  5.572  | 0.6503 |   2.3979 |   5.7071 |  8.1131 |                    0.2857 |                     0.449  |
| f3_w5     |         242 |  0.6298 | 0.2622 |   0.0346 |   0.6164 |  0.9965 |                   -0.7012 |                    -0.7152 |
| f4_w5     |         231 | -0.0092 | 0.0177 |  -0.0775 |  -0.0043 |  0.0279 |                    0.2159 |                     0.218  |
| f5_w5     |         242 |  0.9545 | 0.2909 |   0.2202 |   0.9636 |  1.556  |                   -0.7088 |                    -0.7218 |
| f6_w5     |         242 |  1.6217 | 0.6829 |   0.5    |   1.6875 |  3      |                   -0.2118 |                    -0.2206 |
| f7_w5     |         236 | -6.7651 | 0.7613 |  -8.4527 |  -6.8125 | -4.1603 |                   -0.4256 |                    -0.4487 |
| f8_w5     |         231 |  0.018  | 0.0121 |   0.0021 |   0.0155 |  0.0832 |                   -0.408  |                    -0.4835 |
| f9_w5     |         236 | -4.6251 | 1.0739 | -11.3856 |  -4.4304 | -2.4982 |                   -0.1652 |                    -0.258  |
| f10_w5    |         242 |  6.9559 | 0.1307 |   6.6958 |   6.9527 |  8.3687 |                    0.2072 |                     0.2946 |

Feature-lifetime correlations for w5:

| feature   | feature_description                           |   pearson_r |   spearman_r |   n |
|:----------|:----------------------------------------------|------------:|-------------:|----:|
| f1_w5     | Step 1: log\|mean dQ/dV delta 3.6-3.9V\| [w5] |      -0.828 |       -0.888 | 241 |
| f2_w5     | Step 2: log\|delta CV time\| [w5]             |       0.286 |        0.449 | 241 |
| f3_w5     | Step 3: DoD [w5]                              |      -0.701 |       -0.715 | 247 |
| f4_w5     | Step 4: delta Q1 DVA [w5]                     |       0.216 |        0.218 | 236 |
| f5_w5     | Step 5: sqrt(Cchg)*sqrt(DoD) [w5]             |      -0.709 |       -0.722 | 247 |
| f6_w5     | Step 6: Cchg [w5]                             |      -0.212 |       -0.221 | 247 |
| f7_w5     | Step 7: log\|var dQ/dV delta 3.0-3.6V\| [w5]  |      -0.426 |       -0.449 | 241 |
| f8_w5     | Step 8: delta Q3 DVA [w5]                     |      -0.408 |       -0.483 | 236 |
| f9_w5     | Step 9: log\|mean dQ/dV delta 3.0-3.6V\| [w5] |      -0.165 |       -0.258 | 241 |
| f10_w5    | Step 10: log\|CV time w0\| [w5]               |       0.207 |        0.295 | 247 |

Suggested caption: Supplementary feature correlation matrix for the week-5 RUL feature set under the end-of-life (EOL) threshold defined at 70% state of health (SOH). The heatmap is used only to document the degree of feature redundancy and does not introduce an additional selection metric.

## Appendix C: Hyperparameter Search and Final Model Configuration

Purpose: document the Stage 1 source search, Stage 2 fine-tuning search, and the final selected model configuration used in the main w5 RUL experiment.

Files:

- `Figurecaption/Appendixfile/AppendixC/appendix_C_hpo_search_space.csv`
- `Figurecaption/Appendixfile/AppendixC/appendix_C_selected_stage2_config.csv`
- `Figurecaption/Appendixfile/AppendixC/appendix_C_final_model_config.csv`

Final selected configuration:

| section            | parameter               | value                                                                |
|:-------------------|:------------------------|:---------------------------------------------------------------------|
| source pretraining | features                | f1_w5,f6_w5                                                          |
| source pretraining | hidden_dims             | 64,64,64,64                                                          |
| source pretraining | dropout                 | 0.0                                                                  |
| source pretraining | activation              | relu                                                                 |
| source pretraining | epochs                  | 800                                                                  |
| source pretraining | lr                      | 0.0009741957660925079                                                |
| source pretraining | weight_decay            | 0.00013369738668013456                                               |
| source pretraining | source_stage1_val_mae   | 1.794578163840554                                                    |
| target fine-tuning | ft_lr                   | 0.0002703005473042                                                   |
| target fine-tuning | ft_weight_decay         | 3.380298296224284e-06                                                |
| target fine-tuning | ft_epochs               | 400                                                                  |
| target fine-tuning | ft_freeze_hidden_layers | 3                                                                    |
| target fine-tuning | target_support_ratio    | 0.67                                                                 |
| target fine-tuning | transfer_replay_weight  | 1.0                                                                  |
| target fine-tuning | target_ft_val_mae       | 1.6633552307128905                                                   |
| final evaluation   | feature_week            | w5                                                                   |
| final evaluation   | EOL threshold           | the end-of-life (EOL) threshold defined at 70% state of health (SOH) |

## Appendix D: Sensitivity Check Results

Purpose: collect the two sensitivity checks retained in the appendix: observation-week sensitivity and random target-cell selection sensitivity. These results are empirical checks only and do not introduce a new selection metric.

Files:

- `Figurecaption/Appendixfile/AppendixD/appendix_D_week_sensitivity_overall_metrics.csv`
- `Figurecaption/Appendixfile/AppendixD/appendix_D_week_sensitivity_metrics.png` / `.pdf`
- `Figurecaption/Appendixfile/AppendixD/appendix_D_random_target_selection_seed_results.csv`
- `Figurecaption/Appendixfile/AppendixD/appendix_D_random_target_selection_numeric_aggregate.csv`
- `Figurecaption/Appendixfile/AppendixD/appendix_D_random_target_selection_mae_improvement.png` / `.pdf`

Week-level overall metrics:

| model               |   week |   mae |   rmse |     r2 |   mape_percent |   smape_percent |   wmape_percent |   n_cells |   n_groups |
|:--------------------|-------:|------:|-------:|-------:|---------------:|----------------:|----------------:|----------:|-----------:|
| benchmark           |      5 | 4.181 |  5.421 |  0.257 |         14.621 |          15.909 |          15.522 |        29 |         16 |
| source_only         |      5 | 5.104 |  6.491 | -0.065 |         17.408 |          19.467 |          18.947 |        29 |         16 |
| fine_tuned_transfer |      5 | 3.773 |  4.993 |  0.37  |         13.666 |          14.109 |          14.004 |        29 |         16 |
| benchmark           |      6 | 3.769 |  5.106 |  0.332 |         13.804 |          13.966 |          13.904 |        30 |         17 |
| source_only         |      6 | 3.896 |  5.15  |  0.321 |         13.671 |          14.588 |          14.372 |        30 |         17 |
| fine_tuned_transfer |      6 | 3.739 |  4.91  |  0.383 |         13.351 |          13.913 |          13.793 |        30 |         17 |
| benchmark           |      7 | 4.526 |  5.956 |  0.315 |         14.811 |          15.709 |          16.287 |        31 |         17 |
| source_only         |      7 | 5.548 |  7.179 |  0.005 |         18.146 |          20.468 |          19.964 |        31 |         17 |
| fine_tuned_transfer |      7 | 4.572 |  5.826 |  0.345 |         15.762 |          16.922 |          16.452 |        31 |         17 |
| benchmark           |      8 | 4.065 |  5.647 |  0.384 |         13.492 |          14.341 |          14.628 |        31 |         17 |
| source_only         |      8 | 4.237 |  5.937 |  0.319 |         13.644 |          14.536 |          15.245 |        31 |         17 |
| fine_tuned_transfer |      8 | 4.54  |  5.887 |  0.331 |         16.934 |          15.682 |          16.336 |        31 |         17 |
| benchmark           |      9 | 5.064 |  8.679 | -0.253 |         16.862 |          15.363 |          17.705 |        33 |         17 |
| source_only         |      9 | 5.079 |  7.176 |  0.143 |         15.788 |          16.77  |          17.757 |        33 |         17 |
| fine_tuned_transfer |      9 | 4.696 |  7.793 | -0.01  |         15.347 |          14.481 |          16.417 |        33 |         17 |
| benchmark           |     10 | 4.591 |  7.826 | -0.08  |         15.127 |          13.924 |          16.039 |        35 |         17 |
| source_only         |     10 | 3.957 |  5.711 |  0.425 |         12.1   |          13.108 |          13.822 |        35 |         17 |
| fine_tuned_transfer |     10 | 3.768 |  5.347 |  0.496 |         11.769 |          12.461 |          13.163 |        35 |         17 |

Random target-cell selection seed-level results:

| seed    | stage3_dir           |   bench_test_n_cells |   bench_test_mae |   bench_test_rmse |   bench_test_mape |   bench_test_r2 |   transfer_test_n_cells |   transfer_test_mae |   transfer_test_rmse |   transfer_test_mape |   transfer_test_r2 |   test_transfer_vs_bench_mae_improve_percent |   test_transfer_vs_bench_mape_improve_percent |   test_transfer_vs_bench_r2_delta |
|:--------|:---------------------|---------------------:|-----------------:|------------------:|------------------:|----------------:|------------------------:|--------------------:|---------------------:|---------------------:|-------------------:|---------------------------------------------:|----------------------------------------------:|----------------------------------:|
| seed000 | seed000/stage3_final |                   29 |            5.162 |             6.579 |            18.147 |          -0.124 |                      29 |               4.376 |                5.72  |               15.814 |              0.15  |                                       15.235 |                                        12.855 |                             0.274 |
| seed001 | seed001/stage3_final |                   29 |            4.623 |             6.129 |            17.175 |           0.443 |                      29 |               6.142 |               10.382 |               23.516 |             -0.597 |                                      -32.861 |                                       -36.924 |                            -1.04  |
| seed002 | seed002/stage3_final |                   29 |            5.46  |             7.121 |            17.521 |           0.265 |                      29 |               5.691 |                7.587 |               17.803 |              0.166 |                                       -4.219 |                                        -1.612 |                            -0.099 |
| seed003 | seed003/stage3_final |                   29 |            4.275 |             5.689 |            14.62  |           0.121 |                      29 |               3.998 |                5.405 |               13.632 |              0.207 |                                        6.465 |                                         6.76  |                             0.086 |
| seed004 | seed004/stage3_final |                   29 |            5.236 |             6.484 |            20.436 |           0.196 |                      29 |               6.389 |                7.54  |               26.606 |             -0.086 |                                      -22.019 |                                       -30.194 |                            -0.283 |
| seed005 | seed005/stage3_final |                   29 |            4.272 |             5.699 |            15.976 |           0.479 |                      29 |               4.086 |                5.301 |               16.719 |              0.549 |                                        4.359 |                                        -4.653 |                             0.07  |
| seed006 | seed006/stage3_final |                   29 |            4.547 |             5.772 |            17.622 |           0.528 |                      29 |               4.553 |                5.48  |               18.768 |              0.575 |                                       -0.131 |                                        -6.508 |                             0.046 |
| seed007 | seed007/stage3_final |                   29 |            4.154 |             5.747 |            15.69  |           0.476 |                      29 |               4.608 |                6.225 |               16.314 |              0.386 |                                      -10.932 |                                        -3.983 |                            -0.091 |
| seed008 | seed008/stage3_final |                   29 |            4.242 |             5.413 |            17.293 |           0.327 |                      29 |               3.849 |                4.726 |               15.193 |              0.487 |                                        9.272 |                                        12.142 |                             0.16  |
| seed009 | seed009/stage3_final |                   29 |            5.088 |             6.327 |            19.531 |           0.078 |                      29 |               4.266 |                5.478 |               16.761 |              0.309 |                                       16.167 |                                        14.181 |                             0.231 |
| seed010 | seed010/stage3_final |                   29 |            5.087 |             6.412 |            19.286 |           0.453 |                      29 |               5.117 |                6.357 |               20.074 |              0.463 |                                       -0.585 |                                        -4.088 |                             0.009 |
| seed011 | seed011/stage3_final |                   29 |            4.67  |             5.763 |            18.567 |           0.32  |                      29 |               4.457 |                6.265 |               15.921 |              0.197 |                                        4.563 |                                        14.254 |                            -0.124 |
| seed012 | seed012/stage3_final |                   29 |            4.871 |             6.327 |            16.916 |           0.223 |                      29 |               4.816 |                6.263 |               16.502 |              0.239 |                                        1.115 |                                         2.45  |                             0.016 |
| seed013 | seed013/stage3_final |                   29 |            4.161 |             5.433 |            15.345 |           0.201 |                      29 |               3.931 |                4.861 |               14.516 |              0.361 |                                        5.52  |                                         5.406 |                             0.159 |
| seed014 | seed014/stage3_final |                   29 |            4.637 |             6.087 |            16.642 |           0.548 |                      29 |               5.034 |                6.514 |               17.348 |              0.483 |                                       -8.548 |                                        -4.241 |                            -0.066 |
| seed015 | seed015/stage3_final |                   29 |            5.108 |             6.721 |            17.883 |           0.474 |                      29 |               5.176 |                6.582 |               19.207 |              0.496 |                                       -1.33  |                                        -7.406 |                             0.022 |
| seed016 | seed016/stage3_final |                   29 |            5.363 |             6.819 |            21.444 |           0.153 |                      29 |               4.781 |                6.331 |               18.97  |              0.27  |                                       10.844 |                                        11.534 |                             0.117 |
| seed017 | seed017/stage3_final |                   29 |            3.638 |             5.016 |            13.721 |           0.348 |                      29 |               3.209 |                4.001 |               12.605 |              0.586 |                                       11.775 |                                         8.138 |                             0.237 |
| seed018 | seed018/stage3_final |                   29 |            5.994 |             8.941 |            21.616 |          -0.424 |                      29 |               6.32  |               10.615 |               22.86  |             -1.008 |                                       -5.45  |                                        -5.755 |                            -0.584 |
| seed019 | seed019/stage3_final |                   29 |            4.787 |             6.36  |            15.696 |           0.418 |                      29 |               4.709 |                6.038 |               15.916 |              0.475 |                                        1.639 |                                        -1.399 |                             0.057 |

Suggested caption: Sensitivity checks for RUL prediction under the end-of-life (EOL) threshold defined at 70% state of health (SOH). The week-based plot reports performance changes from w5 to w10, while the random-selection plot reports seed-level changes caused by changing target fine-tuning cells.

## Appendix E: Code and Reproducibility Notes

Purpose: provide a concise code map and reproducibility checklist for locating the inputs, generated appendix files, and experiment outputs.

Files:

- `Figurecaption/Appendixfile/AppendixE/appendix_E_code_map.csv`
- `Figurecaption/Appendixfile/AppendixE/appendix_E_reproducibility_checklist.csv`

Code map:

| purpose                                                   | path                                                                                                                    |
|:----------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------|
| 70% SOH EOL threshold feature table and week availability | week_based/Final/[70% SOH EOL-threshold results]/features                                                               |
| Feature-lifetime correlation analysis                     | week_based/Final/[70% SOH EOL-threshold results]/feature_engineering                                                    |
| Hyperparameter search and final RUL model configuration   | week_based/Final/[70% SOH EOL-threshold results]/3step                                                                  |
| Main w5 final RUL run                                     | week_based/Final/[70% SOH EOL-threshold results]/3step/outputs_400/BasicModel/stage3_final_rerun_400                    |
| Week-sensitivity runs                                     | week_based/Final/[70% SOH EOL-threshold results]/3step/outputs_400/protocol_w6_10_from_stage3_final_rerun_400_legacy400 |
| Random target-cell selection runs                         | week_based/Final/[70% SOH EOL-threshold results]/3step/outputs_400/random_w5_[70% SOH EOL threshold]_10seeds_legacy400  |
| Appendix generation scripts and assets                    | Figurecaption/Appendixfile                                                                                              |

Reproducibility checklist:

| item                           | value                                                                                |
|:-------------------------------|:-------------------------------------------------------------------------------------|
| Python environment             | local conda base environment                                                         |
| Figure format                  | PNG and PDF; no SVG generated for appendix figures                                   |
| Main endpoint                  | RUL label under the end-of-life (EOL) threshold defined at 70% state of health (SOH) |
| Main feature week              | w5                                                                                   |
| Supplementary week sensitivity | w5-w10                                                                               |
| Random selection analysis      | reported as seed-level sensitivity only; no new metric defined                       |
| Appendix script                | Figurecaption/Appendixfile/generate_appendix_B_to_I.py                               |
