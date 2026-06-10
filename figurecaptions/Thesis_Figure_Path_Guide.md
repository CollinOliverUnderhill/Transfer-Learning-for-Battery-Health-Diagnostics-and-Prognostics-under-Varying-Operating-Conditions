# 论文图片路径参考

根目录：`E:\Datasets\IVAS`

本文件用于把 `Figurecaption/Thesis_Figure_Formal_Captions.md` 中的主要论文图片，对应到当前 Windows 工作区里已经存在的图片文件，或最相关的数据、配置和结果文件。除非特别说明，下面所有路径都相对于上述根目录。macOS 产生的 `._*` 辅助文件已忽略。

校验说明：本文件中的路径已在当前工作区 `E:\Datasets\IVAS` 下重新检查。主要 RUL 实验结果集中在 `week_based/Final/EOL70/`。

## 状态说明

- `已有图片`：已找到可以直接查看的图片文件。
- `候选图片`：已有图片与目标论文图接近，但可能还需要统一格式或组合成多面板图。
- `仅有源数据`：已有数据或配置文件，但尚未找到可直接用于论文的成品图。
- `需要示意图`：该图主要是概念图或方法流程图，建议重新绘制成清晰的论文示意图。

## 主要图片

### Fig. 1 | 迁移学习范式概览

状态：需要示意图。

- 本地搜索中未找到可直接使用的图片。
- 设计参考：`Figurecaption/Thesis_Figure_Plan_Captions.md`
- caption 参考：`Figurecaption/Thesis_Figure_Formal_Captions.md`

### Fig. 2 | 可用电池组的工况分布

状态：已有图片 / 源数据。

- 主图：`Figure/figure2/figure2.png`
- 主图 PDF：`Figure/figure2/figure2.pdf`
- 绘图脚本：`Figure/figurecodes/plot_figure2_operating_condition_distribution.py`
- 旧版候选图：`plot_condition_split_selection_3d.png`
- 源数据：`Groupcondi.csv`
- 源数据：`Valid_cells.csv`

### Fig. 3 | 基于 70% SOH 寿命终点阈值的 RUL 标签定义

状态：需要示意图。

- 本地搜索中未找到可直接使用的图片。
- 建议参考数据：`week_based/Final/EOL70/features/feature_table_all_cells_multiweek_EOL70.csv`
- 设计参考：`Figurecaption/Thesis_Figure_Plan_Captions.md`

### Fig. 4 | SOH 与 RUL 建模的数据处理流程

状态：需要示意图 / 源目录。

- 源目录：`Data/capacity_fade/`
- 源目录：`Data/Q_interpolated/`
- 源目录：`Data/Processing_Data/SOHest/`
- 源目录：`week_based/Final/EOL70/features/`

### Fig. 5 | week-based 早期 RUL 样本构建

状态：需要示意图 / 源数据。

- 源数据：`week_based/Final/EOL70/features/feature_table_all_cells_multiweek_EOL70.csv`
- 源数据：`week_based/Final/EOL70/features/week_availability_summary_EOL70.csv`

### Fig. 6 | 十个工程特征与 SOH 的整体相关性

状态：候选图片 / 源数据。

- 候选图：`Data/Processing_Data/SOHest/Pearson_all_cell.png`
- 源数据：`Data/Processing_Data/SOHest/10F_correlation/pearson_overall_10features.csv`
- 补充散点图目录：`Data/Processing_Data/SOHest/10F_correlation/`

### Fig. 7 | Ridge SOH 分析中使用的特征组合

状态：仅有源数据。

- 源数据：`Data/Processing_Data/SOHest/10F_correlation/pearson_overall_10features.csv`
- 源数据：`Data/Processing_Data/SOHest/domain_gap_3F178/domain_gap_3f178_summary.csv`
- 候选诊断图：`Data/Processing_Data/SOHest/domain_gap_3F178/domain_gap_f1_f7_f8_vs_soh.png`
- 候选诊断图：`Data/Processing_Data/SOHest/domain_gap_3F178/domain_gap_f1_f7_f8_pairwise.png`

### Fig. 8 | 主线三阶段 RUL 流程的候选输入特征相关性矩阵

状态：已有图片。

- 主图：`week_based/Final/EOL70/features/correlation_heatmap_w5_EOL70.png`
- 相关性矩阵：`week_based/Final/EOL70/features/correlation_matrix_w5_EOL70.csv`
- 源表：`week_based/Final/EOL70/features/feature_table_all_cells_multiweek_EOL70.csv`

### Fig. 9 | 主线 RUL 实验中的最终输入特征组合

状态：仅有源数据。

- 配置文件：`week_based/Final/EOL70/3step/outputs_400/BasicModel/stage3_final_rerun_400/final_selection.json`
- 配置文件：`week_based/Final/EOL70/3step/outputs_400/BasicModel/stage3_final_rerun_400/selected_stage2_config.csv`
- 配置文件：`week_based/Final/EOL70/3step/outputs_400/BasicModel/stage3_final_rerun_400/transfer_model/config.json`

### Fig. 10 | 基于观测第 5 周和 70% SOH 寿命终点定义的跨组 domain split

状态：候选图片 / 源数据。

- 候选图：`week_based/Final/EOL70/domain_split/w5_EOL70_random_seed000/plot_condition_split_selection_3d_cell_jitter_w5_EOL70_seed000.png`
- 候选图：`week_based/Final/EOL70/domain_split/w5_EOL70_random_seed000/diagnostics/operating_condition_3d_w5_EOL70_seed000.png`
- target-spread 源数据：`week_based/Final/EOL70/domain_split/group_split_targetspread_w5_EOL70.csv`
- target-spread 源数据：`week_based/Final/EOL70/domain_split/cell_split_targetspread_w5_EOL70.csv`
- 主实验样本：`week_based/Final/EOL70/3step/outputs_400/BasicModel/stage3_final_rerun_400/source_train_samples.csv`
- 主实验样本：`week_based/Final/EOL70/3step/outputs_400/BasicModel/stage3_final_rerun_400/target_finetune_samples.csv`
- 主实验样本：`week_based/Final/EOL70/3step/outputs_400/BasicModel/stage3_final_rerun_400/target_test_samples.csv`

### Fig. 11 | source、target fine-tuning 与 target-test 电池的寿命分布

状态：已有候选图片。

- 候选图：`week_based/Final/EOL70/domain_split/w5_EOL70_random_seed000/diagnostics/split_lifetime_distribution_w5_EOL70_seed000.png`
- 源摘要：`week_based/Final/EOL70/domain_split/w5_EOL70_random_seed000/diagnostics/split_lifetime_summary_w5_EOL70_seed000.csv`
- 主实验样本：`week_based/Final/EOL70/3step/outputs_400/BasicModel/stage3_final_rerun_400/source_train_samples.csv`
- 主实验样本：`week_based/Final/EOL70/3step/outputs_400/BasicModel/stage3_final_rerun_400/target_finetune_samples.csv`
- 主实验样本：`week_based/Final/EOL70/3step/outputs_400/BasicModel/stage3_final_rerun_400/target_test_samples.csv`

### Fig. 12 | target-test RUL 预测的 baseline 协议

状态：需要示意图。

- source-only 指标/数据：`week_based/Final/EOL70/3step/outputs_400/BasicModel/stage3_final_rerun_400/transfer_model/test_overall_metrics_source_only.csv`
- benchmark 目录：`week_based/Final/EOL70/3step/outputs_400/BasicModel/stage3_final_rerun_400/benchmark/`
- 设计参考：`Figurecaption/Thesis_Figure_Plan_Captions.md`

### Fig. 13 | 跨组 RUL 预测的 fine-tuning 迁移学习流程

状态：需要示意图。

- transfer model 目录：`week_based/Final/EOL70/3step/outputs_400/BasicModel/stage3_final_rerun_400/transfer_model/`
- source 训练历史图：`week_based/Final/EOL70/3step/outputs_400/BasicModel/stage3_final_rerun_400/transfer_model/plot_training_history_source.png`
- fine-tuning 训练历史图：`week_based/Final/EOL70/3step/outputs_400/BasicModel/stage3_final_rerun_400/transfer_model/plot_training_history_finetune.png`

### Fig. 14 | 用于模型与 fine-tuning 选择的 staged Optuna/TPE 搜索

状态：需要示意图 / 源数据。

- Stage 1 目录：`week_based/Final/EOL70/3step/outputs_400/BasicModel/stage1_expanded/`
- Stage 2 摘要：`week_based/Final/EOL70/3step/outputs_400/BasicModel/stage2_expanded/stage2_best_configs.csv`
- Stage 3 报告：`week_based/Final/EOL70/3step/outputs_400/BasicModel/stage3_final_rerun_400/report_summary.txt`

### Fig. 15 | within-cell SOH 估计结果汇总

状态：已有图片。

- 主图：`Originaltrails/chunqiu_codes/Dropped/SOHestimation_results/Ridge_Results/Ridge_Results_SingleCell/_seed_summary/plot_all_groups_cell_mae_box.png`
- MAPE 图：`Originaltrails/chunqiu_codes/Dropped/SOHestimation_results/Ridge_Results/Ridge_Results_SingleCell/_seed_summary_mape/plot_all_groups_cell_mape_box.png`
- 源摘要：`Originaltrails/chunqiu_codes/Dropped/SOHestimation_results/Ridge_Results/Ridge_Results_SingleCell/_seed_summary/group_cell_summary.csv`

### Fig. 16 | within-cell Ridge 估计中代表性的预测 SOH 与真实 SOH 对比

状态：已有图片。

- 主图：`Originaltrails/chunqiu_codes/Dropped/SOHestimation_results/Ridge_Results/Ridge_Results_SingleCell/G1C1/plot_pred_vs_true.png`
- 备选图：`Originaltrails/chunqiu_codes/Dropped/SOHestimation_results/Ridge_Results/Ridge_Results_SingleCell/G1C1/plot_allp_vs_true.png`
- seed sweep 图：`Originaltrails/chunqiu_codes/Dropped/SOHestimation_results/Ridge_Results/Ridge_Results_SingleCell/G1C1/seed_sweep_pred_vs_true_allpoints.png`

### Fig. 17 | 超出 within-cell self-prediction 的 SOH 估计结果

状态：源数据 / 候选图片。

- single-cell-to-multi-cell 图：`Originaltrails/chunqiu_codes/Dropped/SOHestimation_results/Ridge_Results/Ridge_Results_SingleCell_1fTrain_MultiCellTest/G1C1/test_summary.png`
- domain-shift 候选图：`Originaltrails/chunqiu_codes/Dropped/SOHestimation_results/Ridge_Results/Ridge_Results_DomainShift/scheme2_37train_12test/plot_test_summary_metrics.png`
- domain-shift 候选图：`Originaltrails/chunqiu_codes/Dropped/SOHestimation_results/Ridge_Results/Ridge_Results_DomainShift/scheme2_37train_8ft_12test/plot_test_summary_metrics.png`

### Fig. 18 | 单电池训练、多电池测试的 SOH 估计

状态：已有图片。

- 主图：`Originaltrails/chunqiu_codes/Dropped/SOHestimation_results/Ridge_Results/Ridge_Results_SingleCell_1fTrain_MultiCellTest/G1C1/test_summary.png`
- 源摘要：`Originaltrails/chunqiu_codes/Dropped/SOHestimation_results/Ridge_Results/Ridge_Results_SingleCell_1fTrain_MultiCellTest/G1C1/test_summary.csv`
- cell 指标：`Originaltrails/chunqiu_codes/Dropped/SOHestimation_results/Ridge_Results/Ridge_Results_SingleCell_1fTrain_MultiCellTest/G1C1/test_cell_metrics.csv`

### Fig. 19 | domain shift 下 subset-to-subset SOH 估计

状态：候选图片 / 源数据。

- 候选图：`Originaltrails/chunqiu_codes/Dropped/SOHestimation_results/Ridge_Results/Ridge_Results_DomainShift/g1234_to_g5678/plot_test_group_mae_mape.png`
- 候选图：`Originaltrails/chunqiu_codes/Dropped/SOHestimation_results/Ridge_Results/Ridge_Results_DomainShift/g1234_to_g5678/plot_test_pred_vs_true.png`
- 候选图：`Originaltrails/chunqiu_codes/Dropped/SOHestimation_results/Ridge_Results/Ridge_Results_DomainShift/scheme1_lowMidDoD_to_highDoD/plot_test_summary_metrics.png`
- 候选图：`Originaltrails/chunqiu_codes/Dropped/SOHestimation_results/Ridge_Results/Ridge_Results_DomainShift/scheme2_37train_12test/plot_test_summary_metrics.png`
- 候选图：`Originaltrails/chunqiu_codes/Dropped/SOHestimation_results/Ridge_Results/Ridge_Results_DomainShift/scheme2_37train_8ft_12test/plot_test_summary_metrics.png`

### Fig. 20 | 主线 70% SOH 寿命终点 RUL 实验的 target-test 指标

状态：已有图片 / 源数据。

- benchmark 图：`week_based/Final/EOL70/3step/outputs_400/BasicModel/stage3_final_rerun_400/benchmark/plot_test_summary_metrics.png`
- transfer 图：`week_based/Final/EOL70/3step/outputs_400/BasicModel/stage3_final_rerun_400/transfer_model/plot_test_summary_metrics.png`
- benchmark 指标：`week_based/Final/EOL70/3step/outputs_400/BasicModel/stage3_final_rerun_400/benchmark/test_overall_metrics.csv`
- source-only 指标：`week_based/Final/EOL70/3step/outputs_400/BasicModel/stage3_final_rerun_400/transfer_model/test_overall_metrics_source_only.csv`
- transfer 指标：`week_based/Final/EOL70/3step/outputs_400/BasicModel/stage3_final_rerun_400/transfer_model/test_overall_metrics.csv`

### Fig. 21 | 主线 70% SOH 寿命终点实验中预测 RUL 与真实 RUL 对比

状态：已有图片。

- benchmark 图：`week_based/Final/EOL70/3step/outputs_400/BasicModel/stage3_final_rerun_400/benchmark/plot_test_pred_vs_true.png`
- transfer 图：`week_based/Final/EOL70/3step/outputs_400/BasicModel/stage3_final_rerun_400/transfer_model/plot_test_pred_vs_true.png`
- benchmark 预测：`week_based/Final/EOL70/3step/outputs_400/BasicModel/stage3_final_rerun_400/benchmark/predictions_test.csv`
- transfer 预测：`week_based/Final/EOL70/3step/outputs_400/BasicModel/stage3_final_rerun_400/transfer_model/predictions_test.csv`

### Fig. 22 | 主线 70% SOH 寿命终点实验的 group-level target-test 误差分布

状态：已有图片。

- benchmark 图：`week_based/Final/EOL70/3step/outputs_400/BasicModel/stage3_final_rerun_400/benchmark/plot_test_group_mae_mape.png`
- transfer 图：`week_based/Final/EOL70/3step/outputs_400/BasicModel/stage3_final_rerun_400/transfer_model/plot_test_group_mae_mape.png`
- benchmark 指标：`week_based/Final/EOL70/3step/outputs_400/BasicModel/stage3_final_rerun_400/benchmark/test_group_metrics.csv`
- source-only 指标：`week_based/Final/EOL70/3step/outputs_400/BasicModel/stage3_final_rerun_400/transfer_model/test_group_metrics_source_only.csv`
- transfer 指标：`week_based/Final/EOL70/3step/outputs_400/BasicModel/stage3_final_rerun_400/transfer_model/test_group_metrics.csv`

### Fig. 23 | 主线 70% SOH 寿命终点实验的 cell-level target-test 误差分布

状态：已有图片。

- benchmark 图：`week_based/Final/EOL70/3step/outputs_400/BasicModel/stage3_final_rerun_400/benchmark/plot_test_cell_mae_mape.png`
- transfer 图：`week_based/Final/EOL70/3step/outputs_400/BasicModel/stage3_final_rerun_400/transfer_model/plot_test_cell_mae_mape.png`
- benchmark 指标：`week_based/Final/EOL70/3step/outputs_400/BasicModel/stage3_final_rerun_400/benchmark/test_cell_metrics.csv`
- source-only 指标：`week_based/Final/EOL70/3step/outputs_400/BasicModel/stage3_final_rerun_400/transfer_model/test_cell_metrics_source_only.csv`
- transfer 指标：`week_based/Final/EOL70/3step/outputs_400/BasicModel/stage3_final_rerun_400/transfer_model/test_cell_metrics.csv`

### Fig. 24 | 不同早期观测 week 下的有效电池与样本数量

状态：仅有源数据。

- 源数据：`week_based/Final/EOL70/features/week_availability_summary_EOL70.csv`
- 源表：`week_based/Final/EOL70/features/feature_table_all_cells_multiweek_EOL70.csv`
- 补充可用性数据：`Data/Processing_Data/Lifetime_prediction/ivas_lifetime_eol_availability.csv`

### Fig. 25 | target-test MAE 对早期 week 数据可用性的敏感性

状态：源数据 / 结果目录。

- 摘要数据：`week_based/Final/week6_10_stage3_detailed_summary.csv`
- 结果目录：`week_based/Final/EOL70/3step/outputs_400/protocol_w6_10_from_stage3_final_rerun_400_legacy400/`
- 对比数据：`week_based/Final/EOL70/3step/outputs_400/BasicModel/stage3_benchmark_improvement_comparison.csv`

### Fig. 26 | fine-tuning 随 week 变化带来的改进或退化

状态：仅有源数据。

- 摘要数据：`week_based/Final/week6_10_stage3_detailed_summary.csv`
- 对比数据：`week_based/Final/EOL70/3step/outputs_400/BasicModel/stage3_benchmark_improvement_comparison.csv`

### Fig. 27 | target fine-tuning coverage index 与 seed-dependent 迁移性能

状态：源数据 / 分析说明。

- 分析说明：`Figurecaption/Seed_Coverage_Index_Analysis.md`
- seed 摘要：`week_based/Final/EOL70/3step/outputs_400/random_w5_EOL70_10seeds_legacy400/random_w5_EOL70_10seeds_legacy400_benchmark_transfer_summary_all_splits.csv`
- 数值汇总：`week_based/Final/EOL70/3step/outputs_400/random_w5_EOL70_10seeds_legacy400/random_w5_EOL70_10seeds_legacy400_benchmark_transfer_summary_numeric_aggregate.csv`
- 单 seed 样本目录：`week_based/Final/EOL70/3step/outputs_400/random_w5_EOL70_10seeds_legacy400/seed*/stage3_final/`

### Fig. 28 | 低容量目标域 stress-test 的 source/fine-tuning/test split

状态：源数据 / 可能需要重新绘制 split map。

- low-capacity 输出目录：`week_based/Final/EOL70/3step/outputs_lowcapacity/`
- Stage 3 样本：`week_based/Final/EOL70/3step/outputs_lowcapacity/lowcapacity_grid/stage3_final/source_train_samples.csv`
- Stage 3 样本：`week_based/Final/EOL70/3step/outputs_lowcapacity/lowcapacity_grid/stage3_final/target_finetune_samples.csv`
- Stage 3 样本：`week_based/Final/EOL70/3step/outputs_lowcapacity/lowcapacity_grid/stage3_final/target_test_samples.csv`
- benchmark train cells：`week_based/Final/EOL70/3step/outputs_lowcapacity/lowcapacity_grid/stage3_final/benchmark/train_cells.csv`
- benchmark test cells：`week_based/Final/EOL70/3step/outputs_lowcapacity/lowcapacity_grid/stage3_final/benchmark/test_cells.csv`
- transfer target fine-tuning cells：`week_based/Final/EOL70/3step/outputs_lowcapacity/lowcapacity_grid/stage3_final/transfer_model/target_finetune_cells.csv`
- transfer test cells：`week_based/Final/EOL70/3step/outputs_lowcapacity/lowcapacity_grid/stage3_final/transfer_model/test_cells.csv`
- support-cell 摘要：`week_based/Final/EOL70/3step/outputs_lowcapacity/lowcapacity_grid/stage3_final/support_cells.csv`

### Fig. 29 | 低容量目标域 stress-test 指标

状态：已有图片 / 源数据。

- benchmark 图：`week_based/Final/EOL70/3step/outputs_lowcapacity/lowcapacity_grid/stage3_final/benchmark/plot_test_summary_metrics.png`
- transfer 图：`week_based/Final/EOL70/3step/outputs_lowcapacity/lowcapacity_grid/stage3_final/transfer_model/plot_test_summary_metrics.png`
- 汇总摘要：`week_based/Final/EOL70/3step/outputs_lowcapacity/outputs_lowcapacity_benchmark_transfer_summary_numeric_aggregate.csv`
- Stage 3 报告：`week_based/Final/EOL70/3step/outputs_lowcapacity/lowcapacity_grid/stage3_final/report_summary.txt`

### Fig. 30 | 低容量目标域 stress-test 中预测 RUL 与真实 RUL 对比

状态：已有图片。

- benchmark 图：`week_based/Final/EOL70/3step/outputs_lowcapacity/lowcapacity_grid/stage3_final/benchmark/plot_test_pred_vs_true.png`
- transfer 图：`week_based/Final/EOL70/3step/outputs_lowcapacity/lowcapacity_grid/stage3_final/transfer_model/plot_test_pred_vs_true.png`
- benchmark 预测：`week_based/Final/EOL70/3step/outputs_lowcapacity/lowcapacity_grid/stage3_final/benchmark/predictions_test.csv`
- transfer 预测：`week_based/Final/EOL70/3step/outputs_lowcapacity/lowcapacity_grid/stage3_final/transfer_model/predictions_test.csv`

### Fig. 31 | 低容量目标域 negative transfer 背后的误差分布

状态：已有图片。

- benchmark group 图：`week_based/Final/EOL70/3step/outputs_lowcapacity/lowcapacity_grid/stage3_final/benchmark/plot_test_group_mae_mape.png`
- transfer group 图：`week_based/Final/EOL70/3step/outputs_lowcapacity/lowcapacity_grid/stage3_final/transfer_model/plot_test_group_mae_mape.png`
- benchmark cell 图：`week_based/Final/EOL70/3step/outputs_lowcapacity/lowcapacity_grid/stage3_final/benchmark/plot_test_cell_mae_mape.png`
- transfer cell 图：`week_based/Final/EOL70/3step/outputs_lowcapacity/lowcapacity_grid/stage3_final/transfer_model/plot_test_cell_mae_mape.png`
- benchmark group 指标：`week_based/Final/EOL70/3step/outputs_lowcapacity/lowcapacity_grid/stage3_final/benchmark/test_group_metrics.csv`
- transfer group 指标：`week_based/Final/EOL70/3step/outputs_lowcapacity/lowcapacity_grid/stage3_final/transfer_model/test_group_metrics.csv`
- benchmark cell 指标：`week_based/Final/EOL70/3step/outputs_lowcapacity/lowcapacity_grid/stage3_final/benchmark/test_cell_metrics.csv`
- transfer cell 指标：`week_based/Final/EOL70/3step/outputs_lowcapacity/lowcapacity_grid/stage3_final/transfer_model/test_cell_metrics.csv`

## 备注

- Fig. 1、3、4、5、12、13、14 即使有支撑数据，也更适合作为干净的方法或流程示意图重新绘制。
- 多个结果图已经存在自动生成的 PNG。最终用于论文时，可能仍需要统一字体大小、坐标标签、配色和版式。
- Fig. 20-23 与 Fig. 29-31 中，benchmark 和 transfer 图目前多为分开的文件。最终论文图可能需要组合成一个多面板图。
