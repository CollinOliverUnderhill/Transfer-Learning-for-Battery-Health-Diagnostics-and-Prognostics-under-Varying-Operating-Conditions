## Main Figures

### Fig. 1. Overview of Representative Transfer-Learning Strategies

放置位置：Section 2.3, Transfer Learning.

建议画法：

- 使用概念性 schematic，对比几类常见 transfer-learning 范式。
- 可并列展示 instance-based transfer、feature-representation transfer、parameter transfer / fine-tuning。
- 明确标出 source domain 与 target domain 的关系，并高亮本论文采用的 fine-tuning route。
- 这张图承担背景解释功能，不报导结果。

Caption:

**Fig. 1 | Overview of representative transfer-learning paradigms and the fine-tuning route adopted in this thesis.** Several common transfer-learning strategies are compared schematically, including instance-based transfer, feature-representation transfer, and parameter-transfer or fine-tuning-based adaptation. The comparison situates the thesis method within the broader transfer-learning literature and clarifies why a fine-tuning-based strategy is selected for cross-domain battery prognostics.

### Fig. 2. Operating-Condition Space of All Cell Groups

放置位置：Section 3.1, Dataset Description.

建议画法：

- 3D scatter map，坐标为 charging C-rate, discharging C-rate, DoD。
- 点按 group 显示；颜色或点大小表示 group-level mean lifetime。
- 不放 source/fine-tune/test split，先单独说明整个数据集的 operating-condition 分布。

可用数据/图：

- `Groupcondi.csv`
- `Valid_cells.csv`

Caption:

**Fig. 2 | Operating-condition distribution of the available battery groups.** The cell groups are shown in the operating-condition space defined by charging C-rate, discharging C-rate, and mean depth of discharge (DoD). Group-level lifetime is used as the degradation outcome, illustrating that cells tested under different operating conditions can have substantially different degradation behaviour.

### Fig. 3. RUL Label Definition Relative to the 70% SOH End-of-Life Threshold

放置位置：Section 3.2, Label Definition.

建议画法：

- 单独 schematic。
- 横轴为 cycle/week/time，纵轴为 capacity or SOH。
- 从 observation week 到 SOH reaches 70% threshold 的时间差标为 RUL。

Caption:

**Fig. 3 | Definition of the remaining useful life label relative to the 70% SOH end-of-life threshold.** For each cell, end of life is defined as the week at which SOH first decreases to 70% of the reference capacity. RUL is then calculated as the time interval from an early observation week to this 70% SOH threshold, converting the degradation trajectory into a supervised prognostic target for early-life lifetime prediction.

### Fig. 4. Dataset and Sample-Construction Pipeline

放置位置：Section 3.3, Sample Construction.

建议画法：

- 单独画流程图，不再和 operating-condition map 合并。
- 从 raw RPT/cycling/Q-interpolated data 开始，连接到 SOH samples、RUL samples、domain split、model evaluation。
- 这张图承担“数据怎么变成建模样本”的解释功能。

可用数据/图：

- `Data/capacity_fade/`
- `Data/Q_interpolated/`
- `Data/Processing_Data/SOHest/`
- `week_based/Final/EOL70/features/`

Caption:

**Fig. 4 | Dataset processing workflow for SOH and RUL modelling.** Raw RPT, cycling, capacity-fade, and Q-interpolated data are processed into two types of modelling samples: diagnostic SOH samples and prognostic RUL samples. The workflow separates preliminary SOH-oriented feature analysis from the final week-based RUL transfer-learning experiments.

### Fig. 5. Week-Based Early-Feature Sample Construction

放置位置：Section 3.3, Sample Construction.

建议画法：

- 单独流程图。
- 展示 w3/w5/w6...w10 feature extraction 与 lifetime/RUL label pairing。
- 说明每个 cell 在某个 observation week 生成一条 RUL sample。

可用数据/图：

- `week_based/Final/EOL70/features/feature_table_all_cells_multiweek_EOL70.csv`
- `week_based/Final/EOL70/features/week_availability_summary_EOL70.csv`

Caption:

**Fig. 5 | Construction of week-based early-life RUL samples.** Features extracted at a selected early observation week are paired with the corresponding lifetime or RUL label defined by the 70% SOH end-of-life threshold. This week-based construction allows the model to be evaluated under different levels of early-life data availability.

### Fig. 6. Overall SOH Correlation Ranking of the Ten Engineered Features

放置位置：Section 3.4, Feature Engineering.

建议画法：

- 单独 ranked bar chart。
- 使用 ten engineered features 与 SOH 的 Pearson correlation。
- 突出 f8、f1、f7 等与 SOH 绝对相关性最高的特征。
- 这张图属于 SOH feature engineering 结果，应放在前面的 feature engineering 部分而不是第 5 章性能结果部分。

可用数据/图：

- `Data/Processing_Data/SOHest/10F_correlation/pearson_overall_10features.csv`

关键数值：

- f8-SOH Pearson r = -0.9573
- f1-SOH Pearson r = -0.9244
- f7-SOH Pearson r = -0.8998

Caption:

**Fig. 6 | Overall SOH correlations of the ten engineered features.** Pearson correlations between each engineered feature and SOH are calculated across valid RPT samples. The IC/DVA-derived features f8, f1, and f7 show the strongest absolute correlations with SOH, indicating that the engineered voltage-derived features encode battery health information.

### Fig. 7. Selected SOH Feature Combination Used in the Ridge-Based SOH Estimation Analysis

放置位置：Section 3.4, Feature Engineering.

建议画法：

- 使用 compact table/tabular figure。
- 汇总 SOH estimation analysis 中最终采用的 feature combination。
- 可以附一列说明各特征的 physical meaning 或 feature family。
- 这张图的任务是把 SOH feature engineering 的“结果”明确落到输入组合上。

可用数据/图：

- `Data/Processing_Data/SOHest/10F_correlation/pearson_overall_10features.csv`
- `Data/Processing_Data/SOHest/domain_gap_3F178/domain_gap_3f178_summary.csv`
- final SOH Ridge-analysis feature list used in the reported SOH experiments

Caption:

**Fig. 7 | Selected SOH feature combination used in the Ridge-based SOH estimation analysis.** The SOH-oriented feature subset adopted in the reported Ridge-based SOH analysis is summarized in a compact tabular form. This figure clarifies the exact inputs used for the SOH feature-engineering stage.

### Fig. 8. Input Feature Correlation Matrix for the Main 3-Step RUL Pipeline

放置位置：Section 3.4, Feature Engineering.

建议画法：

- 单独使用 feature-feature Pearson correlation matrix。
- 行列都放主线 candidate input features，颜色表示相关性强弱。
- 可以用边框、星号或浅色遮罩标出最终进入主线 RUL 实验的 selected features。
- 这张图的任务是说明主线 RUL 输入特征空间结构与冗余性，不再展开各种协议的对比。

可用数据/图：

- pairwise Pearson correlation matrix computed from the main EOL70 week-5 feature table
- `week_based/Final/EOL70/features/feature_table_all_cells_multiweek_EOL70.csv`

Caption:

**Fig. 8 | Correlation structure of the candidate input features for the main three-stage RUL pipeline.** A Pearson-correlation matrix is used to visualize the relationships among the engineered early-life features for the 70% SOH end-of-life RUL task. The matrix highlights redundant and complementary inputs before selecting the final feature subset for the reported main experiment.

### Fig. 9. Final Input Feature Combination Used in the Main RUL Experiment

放置位置：Section 3.4, Feature Engineering.

建议画法：

- 使用 compact table/tabular figure。
- 只报导主线 RUL 实验的最终 selected feature combination，不再在主文前段并列展开其它协议。
- 可加一列简短说明 each feature 的 physical meaning 或所属 feature family，但保持简洁。
- 这张图的任务是明确主线三阶段 RUL transfer-learning pipeline 到底用了哪些输入。

可用数据/图：

- `week_based/Final/EOL70/3step/outputs_400/BasicModel/stage2_expanded/stage2_best_configs.csv`
- `week_based/Final/EOL70/3step/outputs_400/BasicModel/stage3_final_rerun_400/final_selection.json`
- reported main-experiment `config.json` / feature list files

Caption:

**Fig. 9 | Final input feature combination used in the main RUL experiment.** The selected feature subset for the reported observation-week-5 RUL protocol, with lifetime defined at the 70% SOH end-of-life threshold, is summarized in a compact tabular form. This figure clarifies the exact inputs used by the main three-stage transfer-learning pipeline.

### Fig. 10. Source, Target Fine-Tuning, and Target Test Split for the 70% SOH End-of-Life RUL Task

放置位置：Section 3.5, Data Preparation and Domain Definition for Modelling.

建议画法：

- 使用 3D condition map 单独展示正式 EOL70 split。
- source train、target fine-tune、target test 用三种颜色。
- 如果想展示 cell jitter，可以使用 cell-jitter 版本；如果正文更简洁，group-level 版本即可。

可用数据/图：

- `week_based/Final/EOL70/domain_split/group_split_targetspread_w5_EOL70.csv`
- `week_based/Final/EOL70/domain_split/cell_split_targetspread_w5_EOL70.csv`
- `week_based/Final/EOL70/domain_split/plot_condition_split_selection_3d_cell_jitter_w5_EOL70_seed000.png`

Caption:

**Fig. 10 | Cross-group domain split used in the main observation-week-5 RUL transfer-learning experiment.** Source-domain training groups, target-domain fine-tuning groups, and target-test groups are shown in the same operating-condition space for the 70% SOH end-of-life RUL task. The split is defined at the group/cell level so that target-test evaluation measures cross-group generalization rather than random interpolation within the same operating condition.

### Fig. 11. Lifetime Distribution Across the Three Data Partitions

放置位置：Section 3.5, Data Preparation and Domain Definition for Modelling.

建议画法：

- 单独 box/violin + scatter。
- 横轴为 source train、target fine-tune、target test。
- 纵轴为 lifetime 或 EOL70 lifetime weeks。

可用数据/图：

- `week_based/Final/EOL70/domain_split/group_split_targetspread_w5_EOL70.csv`
- `week_based/Final/EOL70/domain_split/cell_split_targetspread_w5_EOL70.csv`

Caption:

**Fig. 11 | Lifetime distribution of source, target fine-tuning, and target-test cells.** The lifetime distributions of the three partitions are compared using the 70% SOH end-of-life definition. The comparison verifies that the transfer-learning task involves both operating-condition shift and lifetime-distribution variation, supporting the use of a cross-group evaluation protocol.

### Fig. 12. Source-Only and Benchmark Baseline Protocols

放置位置：Section 4.3, Cross-Domain RUL Prediction with Fine-Tuning-Based Transfer Learning.

建议画法：

- 单独方法流程图。
- 左侧 source-only: source train -> target test。
- 右侧 benchmark: target-aware benchmark training protocol -> target test。
- 两个 baseline 可以并列，因为都是 baseline protocol。

Caption:

**Fig. 12 | Baseline protocols for target-test RUL prediction.** The source-domain-only baseline is trained on source-domain cells and evaluated directly on the target-test partition without target-domain adaptation. The benchmark protocol uses the corresponding non-transfer training setting for comparison on the same target-test cells. These baselines define whether fine-tuning improves cross-group RUL prediction.

### Fig. 13. Fine-Tuning-Based Transfer-Learning Pipeline

放置位置：Section 4.3, Cross-Domain RUL Prediction with Fine-Tuning-Based Transfer Learning.

建议画法：

- 单独 transfer pipeline。
- source pretraining -> checkpoint selection -> partial freezing -> target fine-tuning -> target test。
- 可以标注 freeze layers、target fine-tuning cells、replay weight。

Caption:

**Fig. 13 | Fine-tuning-based transfer-learning pipeline for cross-group RUL prediction.** A neural network is first pretrained using source-domain cells. The pretrained checkpoint is then adapted using a limited number of target fine-tuning cells before final evaluation on held-out target-test cells. Partial freezing, replay weighting, and fine-tuning duration control how much source knowledge is retained during target adaptation.

### Fig. 14. Staged Optuna/TPE Hyperparameter-Search Workflow

放置位置：Section 4.4, Hyperparameter Optimization.

建议画法：

- 单独 HPO workflow。
- Stage 1 source model search; Stage 2 fine-tuning search; Stage 3 final retraining/evaluation。
- 不和模型 pipeline 合并。

可用数据/图：

- `week_based/Final/EOL70/3step/outputs_400/BasicModel/stage2_expanded/stage2_best_configs.csv`
- `week_based/Final/EOL70/3step/outputs_400/BasicModel/stage3_final_rerun_400/report_summary.txt`

Caption:

**Fig. 14 | Staged Optuna/TPE search used for model and fine-tuning selection.** Hyperparameters are selected through a staged search procedure. Stage 1 identifies source-pretraining configurations, Stage 2 optimizes target fine-tuning settings, and Stage 3 retrains and evaluates the selected configuration on the target-test partition.

### Fig. 15. Within-Cell SOH Estimation Summary Across Cells

放置位置：Section 5.1, Within-Cell SOH Estimation Results.

建议画法：

- 使用 aggregated MAE/MAPE boxplot、heatmap 或 summary panel。
- 汇总所有 cell 的 within-cell Ridge SOH estimation result。
- 可同时展示 mean error 和 seed robustness。
- 这张图应真正对应 SOH prediction result，而不是 feature evidence。

可用数据/图：

- `Originaltrails/chunqiu_codes/Dropped/SOHestimation_results/Ridge_Results/Ridge_Results_SingleCell/_seed_summary/plot_all_groups_cell_mae_box.png`
- `Originaltrails/chunqiu_codes/Dropped/SOHestimation_results/Ridge_Results/Ridge_Results_SingleCell/_seed_summary_mape/plot_all_groups_cell_mape_box.png`
- `Originaltrails/chunqiu_codes/Dropped/SOHestimation_results/Ridge_Results/Ridge_Results_SingleCell/_seed_summary/group_cell_summary.csv`
- `Originaltrails/chunqiu_codes/Dropped/SOHestimation_results/Ridge_Results/Ridge_Results_SingleCell/_seed_summary_mape/group_cell_summary.csv`

Caption:

**Fig. 15 | Within-cell SOH estimation summary across cells.** Ridge-based SOH estimation performance is summarized across individual cells using aggregate error statistics. The figure shows the accuracy and robustness of the easiest SOH prediction setting, where training and testing are both performed within the same cell.

### Fig. 16. Representative Predicted-Versus-True SOH for Within-Cell Ridge Estimation

放置位置：Section 5.1, Within-Cell SOH Estimation Results.

建议画法：

- 使用代表性单 cell 或汇总版本的 predicted-vs-true SOH scatter。
- 如果版面允许，可附一张 all-points plot。
- 这张图承担 within-cell SOH prediction 的直观拟合展示功能。

可用数据/图：

- representative `plot_pred_vs_true.png` or `seed_sweep_pred_vs_true_allpoints.png` under `Ridge_Results_SingleCell/*`
- `Originaltrails/chunqiu_codes/Dropped/SOHestimation_results/Ridge_Results/Ridge_Results_SingleCell/G1C1/plot_pred_vs_true.png`
- `Originaltrails/chunqiu_codes/Dropped/SOHestimation_results/Ridge_Results/Ridge_Results_SingleCell/G1C1/plot_allp_vs_true.png`

Caption:

**Fig. 16 | Representative predicted-versus-true SOH for within-cell Ridge estimation.** Predicted and measured SOH values are compared for a representative within-cell Ridge model. The scatter pattern illustrates the calibration and residual spread of SOH estimation when no cell-to-cell generalization is required.

### Fig. 17. Summary of Cross-Cell and Cross-Group SOH Estimation Beyond Within-Cell Self-Prediction

放置位置：Section 5.2, Cross-Cell and Cross-Group SOH Estimation Results.

建议画法：

- 用 grouped bar chart 或 summary panel 汇总 5.2 的全部 SOH estimation results。
- 同一张图里明确区分 `single-cell-to-multi-cell` 与 `subset-to-subset / group-to-group` 两类 protocol。
- 可以把 `Ridge_Results_SingleCell_1fTrain_MultiCellTest` 作为一类，把 `Ridge_Results_DomainShift` 下各 scheme 作为另一类。
- 这张图承担 5.2 总览功能，明确这一节收纳的是所有非 self-prediction 的 SOH 结果。

可用数据/图：

- `Originaltrails/chunqiu_codes/Dropped/SOHestimation_results/Ridge_Results/Ridge_Results_SingleCell_1fTrain_MultiCellTest/G1C1/test_summary.csv`
- `Originaltrails/chunqiu_codes/Dropped/SOHestimation_results/Ridge_Results/Ridge_Results_SingleCell_1fTrain_MultiCellTest/G1C1/test_summary.png`
- `Originaltrails/chunqiu_codes/Dropped/SOHestimation_results/Ridge_Results/Ridge_Results_DomainShift/scheme2_37train_12test/test_overall_metrics.csv`
- `Originaltrails/chunqiu_codes/Dropped/SOHestimation_results/Ridge_Results/Ridge_Results_DomainShift/scheme2_37train_8ft_12test/test_overall_metrics.csv`
- `Originaltrails/chunqiu_codes/Dropped/SOHestimation_results/Ridge_Results/Ridge_Results_DomainShift/scheme2_37train_8ft_12test/test_overall_metrics_source_only.csv`
- `Originaltrails/chunqiu_codes/Dropped/SOHestimation_results/Ridge_Results/Ridge_Results_DomainShift/g1234_to_g5678/test_overall_metrics.csv`
- `Originaltrails/chunqiu_codes/Dropped/SOHestimation_results/Ridge_Results/Ridge_Results_DomainShift/scheme1_lowMidDoD_to_highDoD/test_overall_metrics.csv`
- representative `plot_test_summary_metrics.png`

Caption:

**Fig. 17 | Summary of cross-cell and cross-group SOH estimation beyond within-cell self-prediction.** Target-test metrics are summarized for SOH settings that require generalization beyond the training cell, including models trained on a single source cell and evaluated on different target cells, as well as protocols that train on one subset of cells or groups and test on another. This figure defines Section 5.2 as the part of the SOH study devoted to cell-to-cell and group-to-group generalization.

### Fig. 18. Cross-Cell SOH Estimation Results

放置位置：Section 5.2, Cross-Cell and Cross-Group SOH Estimation Results.

建议画法：

- 单独展示 `one-cell-train, multi-cell-test` 这一支结果。
- 可使用 `G1C1` 作为训练 cell，对所有其它测试 cell 的 `test_summary.png`、`test_cell_metrics.csv` 或 heatmap/排序图。
- 如果版面允许，可附代表性的 predicted-vs-true 或 test-cell ranking 子图。
- 这张图的任务是明确说明：5.2 不只是 group-to-group，也包括“一个电池训练后去预测别的电池”。

可用数据/图：

- `Originaltrails/chunqiu_codes/Dropped/SOHestimation_results/Ridge_Results/Ridge_Results_SingleCell_1fTrain_MultiCellTest/G1C1/test_summary.png`
- `Originaltrails/chunqiu_codes/Dropped/SOHestimation_results/Ridge_Results/Ridge_Results_SingleCell_1fTrain_MultiCellTest/G1C1/test_summary.csv`
- `Originaltrails/chunqiu_codes/Dropped/SOHestimation_results/Ridge_Results/Ridge_Results_SingleCell_1fTrain_MultiCellTest/G1C1/test_cell_metrics.csv`

Caption:

**Fig. 18 | Cross-cell SOH estimation using a model trained on one source cell and tested on other target cells.** A Ridge model trained on a single battery cell is evaluated on different cells to quantify cell-to-cell SOH generalization beyond within-cell self-prediction. This figure isolates the one-source-cell to multiple-target-cell setting and shows how performance changes when the training signal comes from only one battery.

### Fig. 19. Subset-to-Subset SOH Estimation Results Under Domain Shift

放置位置：Section 5.2, Cross-Cell and Cross-Group SOH Estimation Results.

建议画法：

- 单独展示 train-on-some-cells / test-on-other-cells 这一类结果。
- 可对几类 domain-shift protocol 做并列 summary：Groups 1-4 train to Groups 5-8 test, low-to-moderate DoD train to high-DoD test, 37-cell train to 12-cell held-out test, and 37-cell source training plus 8-cell target fine-tuning before 12-cell held-out testing。
- 若版面允许，可在主图中加入 representative predicted-vs-true 或 group-wise MAE/MAPE 子图。
- 这张图的任务是明确说明 5.2 的另一支是 subset-to-subset / group-to-group SOH generalization。

可用数据/图：

- `Originaltrails/chunqiu_codes/Dropped/SOHestimation_results/Ridge_Results/Ridge_Results_DomainShift/g1234_to_g5678/test_overall_metrics.csv`
- `Originaltrails/chunqiu_codes/Dropped/SOHestimation_results/Ridge_Results/Ridge_Results_DomainShift/g1234_to_g5678/plot_test_group_mae_mape.png`
- `Originaltrails/chunqiu_codes/Dropped/SOHestimation_results/Ridge_Results/Ridge_Results_DomainShift/scheme1_lowMidDoD_to_highDoD/test_overall_metrics.csv`
- `Originaltrails/chunqiu_codes/Dropped/SOHestimation_results/Ridge_Results/Ridge_Results_DomainShift/scheme2_37train_12test/test_overall_metrics.csv`
- `Originaltrails/chunqiu_codes/Dropped/SOHestimation_results/Ridge_Results/Ridge_Results_DomainShift/scheme2_37train_8ft_12test/test_overall_metrics.csv`
- representative `plot_test_pred_vs_true.png` / `plot_test_group_mae_mape.png`

Caption:

**Fig. 19 | SOH estimation results when one subset of cells is used to predict another subset under domain shift.** SOH estimation performance is summarized for protocols that train on one subset of cells or groups and evaluate on a disjoint subset. The evaluated splits include training on Groups 1-4 and testing on Groups 5-8, training on low-to-moderate DoD conditions and testing on high-DoD conditions, and training on 37 cells followed by evaluation on 12 held-out cells. These protocols show the difficulty of SOH generalization under stronger cell-group and operating-condition shifts.

### Fig. 20. Main Target-Test Metrics for the 70% SOH End-of-Life RUL Task

放置位置：Section 5.3, Performance of Cross-Domain RUL Prediction with Fine-Tuning-Based Transfer Learning.

建议画法：

- MAE, RMSE, MAPE, R2 的 grouped bar chart。
- benchmark、source-only、fine-tuned transfer 三种模型并列是合理的，因为比较对象一致。
- 不再把 predicted-vs-true、group error、cell error 放在同一张图里。

可用数据/图：

- `week_based/Final/EOL70/3step/outputs_400/BasicModel/stage3_final_rerun_400/benchmark/test_overall_metrics.csv`
- `week_based/Final/EOL70/3step/outputs_400/BasicModel/stage3_final_rerun_400/transfer_model/test_overall_metrics_source_only.csv`
- `week_based/Final/EOL70/3step/outputs_400/BasicModel/stage3_final_rerun_400/transfer_model/test_overall_metrics.csv`

关键数值：

- Benchmark: MAE = 4.181 weeks, RMSE = 5.421, R2 = 0.257, MAPE = 14.621%.
- Source-only: MAE = 5.104 weeks, RMSE = 6.491, R2 = -0.065, MAPE = 17.408%.
- Fine-tuned transfer: MAE = 3.773 weeks, RMSE = 4.993, R2 = 0.370, MAPE = 13.666%.

Caption:

**Fig. 20 | Target-test metrics for the main RUL prediction experiment using the 70% SOH end-of-life definition.** The benchmark model, the source-domain-only model, and the fine-tuned transfer model are evaluated on the same target-test partition under the reported main configuration. Fine-tuning reduces target-test MAE from 4.181 weeks for the benchmark and 5.104 weeks for the source-domain-only model to 3.773 weeks, corresponding to an MAE reduction of about 9.8% relative to the benchmark and 26.1% relative to the source-domain-only model.

### Fig. 21. Predicted Versus True RUL for the Main 70% SOH End-of-Life Run

放置位置：Section 5.3, Performance of Cross-Domain RUL Prediction with Fine-Tuning-Based Transfer Learning.

建议画法：

- benchmark、source-only、fine-tuned transfer 可以做并列 scatter，统一坐标范围和 y=x reference line。
- 如果版面有限，主文放 benchmark vs fine-tuned transfer，source-only 放补充。

可用数据/图：

- `week_based/Final/EOL70/3step/outputs_400/BasicModel/stage3_final_rerun_400/benchmark/plot_test_pred_vs_true.png`
- `week_based/Final/EOL70/3step/outputs_400/BasicModel/stage3_final_rerun_400/transfer_model/plot_test_pred_vs_true.png`

Caption:

**Fig. 21 | Predicted versus true target-test RUL for the main 70% SOH end-of-life run.** Predicted and true RUL values are compared using the same axis limits and identity reference line. The plot shows the calibration and dispersion of the benchmark and transfer predictions for the reported main configuration beyond the aggregate error metrics.

### Fig. 22. Group-Level Error Breakdown for the Main 70% SOH End-of-Life Run

放置位置：Section 5.3, Performance of Cross-Domain RUL Prediction with Fine-Tuning-Based Transfer Learning.

建议画法：

- 单独 group-wise MAE/MAPE plot。
- benchmark、source-only、fine-tuned transfer 可以并列或用同一图比较。
- 重点解释哪些 target test groups 改善明显，哪些仍有残差。

可用数据/图：

- `plot_test_group_mae_mape.png` under the main stage3 benchmark and transfer directories

Caption:

**Fig. 22 | Group-level target-test error distribution for the main 70% SOH end-of-life run.** Target-test errors are decomposed by cell group for the benchmark, source-domain-only, and fine-tuned transfer models under the reported main configuration. The group-level comparison shows whether the average improvement from fine-tuning is shared across target operating conditions or concentrated in a subset of groups.

### Fig. 23. Cell-Level Error Breakdown for the Main 70% SOH End-of-Life Run

放置位置：Section 5.3, Performance of Cross-Domain RUL Prediction with Fine-Tuning-Based Transfer Learning.

建议画法：

- 单独 cell-wise absolute error / MAE / MAPE distribution。
- violin/box + points 或 bar chart。
- 不和 group-level plot 合并。

可用数据/图：

- `plot_test_cell_mae_mape.png` under the main stage3 benchmark and transfer directories

Caption:

**Fig. 23 | Cell-level target-test error distribution for the main 70% SOH end-of-life run.** Absolute prediction errors are summarized at the cell level for the reported main configuration. The cell-wise breakdown reveals the spread of errors within target groups and identifies individual cells for which the transfer model remains difficult to generalize.

### Fig. 24. Valid Sample Availability Across Feature Weeks

放置位置：Section 5.5, Impact of Data Availability on Cross-Domain RUL Prediction.

建议画法：

- 每个 feature week 的 valid cells/samples 数量。
- 可以和 week sensitivity 文字配合，解释为什么 week choice 会影响结果。

可用数据/图：

- `week_based/Final/EOL70/features/week_availability_summary_EOL70.csv`
- `week_based/Final/EOL70/features/feature_table_all_cells_multiweek_EOL70.csv`

Caption:

**Fig. 24 | Valid cell and sample availability across early observation weeks.** The number of valid cells and samples is summarized for each observation week used in the 70% SOH end-of-life RUL experiments. Sample availability is an important practical constraint when comparing early-week RUL prediction results because later weeks can provide richer degradation information but may also exclude cells that fail earlier.

### Fig. 25. Early-Week Sensitivity of Target-Test Error

放置位置：Section 5.5, Impact of Data Availability on Cross-Domain RUL Prediction.

建议画法：

- target-test MAE vs feature week 的 line plot。
- benchmark、source-only、fine-tuned transfer 三条线可以在同一图，因为纵轴和含义一致。
- 如果版面允许，可加入 RMSE/MAPE 子图；如果拆得更彻底，则 MAE 主文，RMSE/MAPE 放补充。

可用数据/图：

- `week_based/Final/week6_10_stage3_detailed_summary.csv`
- `week_based/Final/EOL70/3step/outputs_400/protocol_w6_10_from_stage3_final_rerun_400_legacy400/week*/stage3_final/`

关键数值：

- Benchmark MAE decreases from 4.887 weeks at w6 to 3.607 weeks at w10.
- Fine-tuned transfer MAE decreases from 6.077 weeks at w6 to 4.075 weeks at w10.

Caption:

**Fig. 25 | Sensitivity of target-test MAE to early-week data availability.** Target-test MAE is compared from observation week 6 to week 10 for the benchmark, source-domain-only, and fine-tuned transfer models under the same 70% SOH end-of-life RUL protocol. Later observation weeks generally provide more degradation information and reduce prediction error, but fine-tuning does not consistently outperform the baselines in this sensitivity experiment.

### Fig. 26. Transfer Improvement Across Feature Weeks

放置位置：Section 5.5, Impact of Data Availability on Cross-Domain RUL Prediction.

建议画法：

- transfer-vs-benchmark MAE improvement percent vs week。
- 单独画，方便讨论 fine-tuning 不稳定和 negative transfer。
- 可用 0% horizontal reference line。

可用数据/图：

- `week_based/Final/week6_10_stage3_detailed_summary.csv`

Caption:

**Fig. 26 | Week-dependent improvement or degradation from fine-tuning.** The percentage change in MAE between the fine-tuned transfer model and the benchmark model is plotted for each feature week. Positive values indicate improvement from transfer, whereas negative values indicate that fine-tuning worsens target-test error. This figure highlights the protocol-dependent nature of transfer learning for early-life RUL prediction.

### Fig. 27. Target Fine-Tuning Coverage Index Across Random Seeds

放置位置：Section 5.6, Coverage Index Analysis of Random-Seed Sensitivity.

建议画法：

- 推荐使用双 panel scatter。
- 左图：x 轴为 target fine-tuning coverage index，y 轴为 MAE improvement relative to benchmark。
- 右图：x 轴为同一个 index，y 轴为 R2 delta。
- 每个点对应一个 random seed，可用颜色区分 positive transfer 与 negative transfer。
- 可标出 best/worst seeds，并在图注中说明该 index 由 centroid gap 与 nearest-neighbour coverage 定义。

可用数据/图：

- `week_based/Final/EOL70/3step/outputs_400/random_w5_EOL70_10seeds_legacy400/random_w5_EOL70_10seeds_legacy400_benchmark_transfer_summary_all_splits.csv`
- `week_based/Final/EOL70/3step/outputs_400/random_w5_EOL70_10seeds_legacy400/seed*/stage3_final/target_finetune_samples.csv`
- `week_based/Final/EOL70/3step/outputs_400/random_w5_EOL70_10seeds_legacy400/seed*/stage3_final/target_test_samples.csv`
- `Figurecaption/Seed_Coverage_Index_Analysis.md`

Caption:

**Fig. 27 | Relationship between the target fine-tuning coverage index and seed-dependent transfer performance.** A target fine-tuning coverage index, defined from the centroid gap and nearest-neighbour coverage between target fine-tuning cells and target-test cells in the standardized condition space, is compared across random seeds. Higher index values indicate better geometric coverage of the target-test cells and are generally associated with more positive transfer relative to the benchmark.

### Fig. 28. Low-Capacity Target-Domain Split

放置位置：Section 6.4, Subjectivity of Domain Definition.

建议画法：

- 单独 source/fine-tune/test domain split for low-capacity target domain。
- 使用 3D operating-condition map 或 group split map。
- 不和结果 metrics 合并。

可用数据/图：

- `week_based/Final/EOL70/3step/outputs_lowcapacity/`
- low-capacity split files under the corresponding protocol directory

Caption:

**Fig. 28 | Low-capacity target-domain split used for the stress-test experiment.** The source, target fine-tuning, and target-test partitions are shown for a stress-test protocol in which the target domain is defined by lower-capacity cells. This split is used to test whether the fine-tuning strategy remains beneficial when the target domain differs more strongly from the main observation-week-5, 70% SOH end-of-life setting.

### Fig. 29. Low-Capacity Target-Test Metrics

放置位置：Section 6.4, Subjectivity of Domain Definition.

建议画法：

- benchmark vs source-only vs fine-tuned target-test metrics。
- grouped bar chart，与 Fig. 16 风格一致。
- 这张图单独承担 stress-test quantitative result。

可用数据/图：

- `week_based/Final/EOL70/3step/outputs_lowcapacity/outputs_lowcapacity_benchmark_transfer_summary_numeric_aggregate.csv`
- `week_based/Final/EOL70/3step/outputs_lowcapacity/lowcapacity_grid/stage3_final/report_summary.txt`

关键数值：

- Benchmark target test: MAE = 3.882 weeks, RMSE = 5.139, R2 = 0.332, MAPE = 13.608%.
- Fine-tuned target test: MAE = 4.450 weeks, RMSE = 6.671, R2 = -0.125, MAPE = 15.785%.
- Transfer-vs-benchmark MAE change = -14.63%, meaning transfer worsened MAE in this stress test.

Caption:

**Fig. 29 | Low-capacity target-domain stress-test metrics.** Benchmark and transfer models are evaluated under a stress-test protocol in which the target domain is defined by lower-capacity cells. Unlike the main observation-week-5, 70% SOH end-of-life result, fine-tuning does not improve target-test performance in this setting: MAE increases from 3.882 weeks for the benchmark to 4.450 weeks for the fine-tuned transfer model. This result indicates a risk of negative transfer.

### Fig. 30. Predicted Versus True RUL in the Low-Capacity Stress Test

放置位置：Section 6.4, Subjectivity of Domain Definition.

建议画法：

- benchmark and fine-tuned transfer predicted-vs-true 并列，统一坐标和 identity line。
- 作为 Fig. 29 的误差来源解释，不和 metrics 合并。

可用数据/图：

- `week_based/Final/EOL70/3step/outputs_lowcapacity/lowcapacity_grid/stage3_final/benchmark/`
- `week_based/Final/EOL70/3step/outputs_lowcapacity/lowcapacity_grid/stage3_final/transfer_model/`

Caption:

**Fig. 30 | Predicted versus true RUL for the lower-capacity target-domain stress test.** Benchmark and fine-tuned transfer predictions are compared with the identity line under the stress-test protocol in which the target domain is defined by lower-capacity cells. The scatter distribution helps explain why fine-tuning worsens aggregate error in this stress-test setting.

### Fig. 31. Group- or Cell-Level Error Breakdown in the Low-Capacity Stress Test

放置位置：Section 6.5, Influence of Fine-Tuning Cell Selection and Quantity.

建议画法：

- 单独 error by target test group/cell。
- 如果 group-level 和 cell-level 都要展示，建议分成 Fig. 31 与额外补充图；如果篇幅有限，只保留 group-level。
- 用于说明哪些 target groups/cells 贡献 negative transfer。

可用数据/图：

- `week_based/Final/EOL70/3step/outputs_lowcapacity/lowcapacity_grid/stage3_final/benchmark/*.csv`
- `week_based/Final/EOL70/3step/outputs_lowcapacity/lowcapacity_grid/stage3_final/transfer_model/*.csv`

Caption:

**Fig. 31 | Error distribution behind negative transfer in the lower-capacity target-domain stress test.** Target-test errors are decomposed by group or cell for the benchmark and fine-tuned transfer models. The breakdown identifies which lower-capacity target conditions contribute most to the degradation in transfer performance.

## Supplementary Figures

### Fig. S1. Representative Capacity-Fade Curves

建议画法：选择几个 group/cell，单独展示 capacity fade 随 cycle/week 的变化。不要和 Q-V 或 dQ/dV 曲线合并。

Caption:

**Fig. S1 | Representative capacity-fade trajectories.** Capacity retention is shown over cycling for selected cells under different operating conditions. The examples illustrate the diversity of degradation trajectories in the dataset.

### Fig. S2. Representative Q-V Curves

建议画法：选择同一批或典型 cells，展示 Q-V curve 随 RPT/cycle/week 变化。

Caption:

**Fig. S2 | Representative Q-V curves during degradation.** Q-V profiles are shown at different degradation stages for selected cells. The evolution of the voltage-capacity relationship provides physical context for the engineered voltage-derived features.

### Fig. S3. Representative dQ/dV or IC/DVA Curves

建议画法：单独展示 dQ/dV 或 DVA-derived signatures，不和 capacity fade 合并。

Caption:

**Fig. S3 | Representative voltage-derivative signatures during degradation.** Incremental-capacity or differential-voltage signatures are shown for selected cells across degradation stages. The curve changes motivate the use of IC/DVA-derived features in SOH and RUL modelling.

### Fig. S4. Full Ten-Feature SOH Scatter Plots

建议画法：把 10 个 SOH feature scatter 放补充。因为是同类 scatter，可以做成多图并列。

Caption:

**Fig. S4 | Full SOH scatter-plot analysis for the ten engineered features.** Scatter plots between SOH and all ten engineered features are shown using a consistent plotting style. The full set supports the feature-correlation ranking reported in the main text.

### Fig. S5. Full SOH Correlation Tables

建议画法：Pearson/Spearman by cell/release/overall 汇总表。

Caption:

**Fig. S5 | Detailed SOH correlation summaries for the engineered features.** Overall, cell-level, and release-level correlation results are summarized for the ten engineered features. These supplementary statistics document the robustness and variability of the SOH feature relationships.

### Fig. S6. Domain-Split Lifetime Diagnostics

建议画法：targetrandom/targetspread split 的 lifetime distribution 放补充。

Caption:

**Fig. S6 | Lifetime diagnostics for alternative domain splits.** Lifetime distributions are shown for alternative source, target fine-tuning, and target-test split definitions. These diagnostics support the final choice of a cross-group partition in which the target fine-tuning and target-test cells are selected to span the target operating-condition space.

### Fig. S7. Domain-Split Feature Diagnostics

建议画法：feature boxplots/correlation heatmaps 作为补充，不和 split 3D map 合并。

Caption:

**Fig. S7 | Feature-distribution diagnostics for domain splits.** Feature distributions and correlation matrices are compared across the source, target fine-tuning, and target-test partitions. These diagnostics verify that target-test evaluation is performed under a meaningful domain-shift setting.

### Fig. S8. Stage-1 Source-Model Search Results

建议画法：top 10 Stage 1 source models 或 source validation curves。

Caption:

**Fig. S8 | Ranked source-pretraining configurations from Stage 1.** The top source-pretraining trials are listed with feature subset, network architecture, optimizer settings, and source-validation performance. These results document the source model candidates used before target-domain fine-tuning.

### Fig. S9. Full Stage-2 Fine-Tuning Search Table

建议画法：top 10 or top 20 Stage 2 trials。由于主文不再单独展开 Stage-2 top-k 结果，这张图承担更完整的 HPO 细节汇总。

Caption:

**Fig. S9 | Detailed Stage-2 fine-tuning hyperparameter search results.** The leading fine-tuning trials are summarized with feature subset, network architecture, freezing depth, support ratio, replay weight, learning rate, weight decay, validation MAE, and checkpoint-selection settings.

### Fig. S10. Training and Fine-Tuning Histories

建议画法：source pretraining loss/MAE curve，fine-tuning train/val curve；如果有 early stopping/SWA window 可以标出。

Caption:

**Fig. S10 | Training histories for source pretraining and target-domain fine-tuning.** Loss and validation metrics are shown across epochs for the source and fine-tuning stages. The curves document convergence behaviour and support the selected checkpoint or last-window/SWA model-selection strategy.

### Fig. S11. Full Per-Group Error Tables for the Main 70% SOH End-of-Life Runs

建议画法：benchmark、source-only、fine-tuned 的 group-level MAE/MAPE 全量表或图。

Caption:

**Fig. S11 | Full per-group error summaries for the main 70% SOH end-of-life evaluation.** Target-test errors are decomposed by group for the benchmark, source-domain-only, and fine-tuned transfer models. The full table supplements the main group-level error figure in Fig. 22.

### Fig. S12. Full Per-Cell Error Tables for the Main 70% SOH End-of-Life Runs

建议画法：benchmark、source-only、fine-tuned 的 cell-level MAE/MAPE 全量表或图。

Caption:

**Fig. S12 | Full per-cell error summaries for the main 70% SOH end-of-life evaluation.** Target-test errors are decomposed by individual cell for the benchmark, source-domain-only, and fine-tuned transfer models. The full table supplements the main cell-level error figure in Fig. 23.

### Fig. S13. RMSE and MAPE Early-Week Sensitivity

建议画法：如果主文只放 MAE，RMSE/MAPE sensitivity 放这里。

Caption:

**Fig. S13 | Early-week sensitivity of RMSE and MAPE.** RMSE and MAPE are compared across early observation weeks for the benchmark, source-domain-only, and fine-tuned transfer models. These supplementary metrics support the MAE-based sensitivity analysis in the main text.

## Figures to Avoid as Main Figures

- 不建议把所有自动生成的 `plot_test_summary_metrics.png` 原样作为主图。它们可以作为数据来源，但正式毕设图应统一字体、颜色、坐标范围和 legend。
- 不建议再使用 4-panel 大拼图作为主图，除非几个 panel 是完全同类的横向比较。
- 不建议把 SOH scatter 全部放主文。主文中的 SOH feature engineering 更适合保留 SOH correlation ranking 与 SOH selected-feature table；完整 scatter 放补充。
- 不建议把 RUL label、week-based sample construction、feature correlation ranking 合成一张图。它们分别服务于 label definition、sample construction、feature screening 三个不同叙事点。
- 不建议把 model protocol 和 HPO workflow 合并。模型流程与参数搜索流程应分开解释。
- 不建议只展示 fine-tuning 成功的 w5 结果。week6-10 和 low-capacity 结果应保留，用于支撑 Discussion 中关于 domain definition、fine-tuning cell selection、negative transfer 的严谨讨论。
- 不建议把所有 EOL thresholds (50/55/60/65/70/75/80) 都做成主文图，除非论文结果章节明确扩展到多 EOL threshold。当前 outline 以 EOL70 和 early-week sensitivity 为主，其他 EOL 更适合作 supplementary。

## Recommended Figure Order in Thesis

按 outline section 摆放时，建议正文中出现顺序如下：

1. Section 2.3, Transfer Learning: Fig. 1, Overview of Representative Transfer-Learning Strategies.
2. Section 3.1, Dataset Description: Fig. 2, Operating-Condition Space of All Cell Groups.
3. Section 3.2, Label Definition: Fig. 3, RUL Label Definition Relative to the 70% SOH End-of-Life Threshold.
4. Section 3.3, Sample Construction: Fig. 4, Dataset and Sample-Construction Pipeline.
5. Section 3.3, Sample Construction: Fig. 5, Week-Based Early-Feature Sample Construction.
6. Section 3.4, Feature Engineering: Fig. 6, Overall SOH Correlation Ranking of the Ten Engineered Features.
7. Section 3.4, Feature Engineering: Fig. 7, Selected SOH Feature Combination Used in the Ridge-Based SOH Estimation Analysis.
8. Section 3.4, Feature Engineering: Fig. 8, Input Feature Correlation Matrix for the Main 3-Step RUL Pipeline.
9. Section 3.4, Feature Engineering: Fig. 9, Final Input Feature Combination Used in the Main RUL Experiment.
10. Section 3.5, Data Preparation and Domain Definition for Modelling: Fig. 10, Source, Target Fine-Tuning, and Target Test Split for the 70% SOH End-of-Life RUL Task.
11. Section 3.5, Data Preparation and Domain Definition for Modelling: Fig. 11, Lifetime Distribution Across the Three Data Partitions.
12. Section 4.3, Cross-Domain RUL Prediction with Fine-Tuning-Based Transfer Learning: Fig. 12, Source-Only and Benchmark Baseline Protocols.
13. Section 4.3, Cross-Domain RUL Prediction with Fine-Tuning-Based Transfer Learning: Fig. 13, Fine-Tuning-Based Transfer-Learning Pipeline.
14. Section 4.4, Hyperparameter Optimization: Fig. 14, Staged Optuna/TPE Hyperparameter-Search Workflow.
15. Section 5.1, Within-Cell SOH Estimation Results: Fig. 15, Within-Cell SOH Estimation Summary Across Cells.
16. Section 5.1, Within-Cell SOH Estimation Results: Fig. 16, Representative Predicted-Versus-True SOH for Within-Cell Ridge Estimation.
17. Section 5.2, Cross-Cell and Cross-Group SOH Estimation Results: Fig. 17, Summary of Cross-Cell and Cross-Group SOH Estimation Beyond Within-Cell Self-Prediction.
18. Section 5.2, Cross-Cell and Cross-Group SOH Estimation Results: Fig. 18, Cross-Cell SOH Estimation Results.
19. Section 5.2, Cross-Cell and Cross-Group SOH Estimation Results: Fig. 19, Subset-to-Subset SOH Estimation Results Under Domain Shift.
20. Section 5.3, Performance of Cross-Domain RUL Prediction with Fine-Tuning-Based Transfer Learning: Fig. 20, Main Target-Test Metrics for the 70% SOH End-of-Life RUL Task.
21. Section 5.3, Performance of Cross-Domain RUL Prediction with Fine-Tuning-Based Transfer Learning: Fig. 21, Predicted Versus True RUL for the Main 70% SOH End-of-Life Run.
22. Section 5.3, Performance of Cross-Domain RUL Prediction with Fine-Tuning-Based Transfer Learning: Fig. 22, Group-Level Error Breakdown for the Main 70% SOH End-of-Life Run.
23. Section 5.3, Performance of Cross-Domain RUL Prediction with Fine-Tuning-Based Transfer Learning: Fig. 23, Cell-Level Error Breakdown for the Main 70% SOH End-of-Life Run.
24. Section 5.5, Impact of Data Availability on Cross-Domain RUL Prediction: Fig. 24, Valid Sample Availability Across Feature Weeks.
25. Section 5.5, Impact of Data Availability on Cross-Domain RUL Prediction: Fig. 25, Early-Week Sensitivity of Target-Test Error.
26. Section 5.5, Impact of Data Availability on Cross-Domain RUL Prediction: Fig. 26, Transfer Improvement Across Feature Weeks.
27. Section 5.6, Coverage Index Analysis of Random-Seed Sensitivity: Fig. 27, Target Fine-Tuning Coverage Index Across Random Seeds.
28. Section 6.4, Subjectivity of Domain Definition: Fig. 28, Low-Capacity Target-Domain Split.
29. Section 6.4, Subjectivity of Domain Definition: Fig. 29, Low-Capacity Target-Test Metrics.
30. Section 6.4, Subjectivity of Domain Definition: Fig. 30, Predicted Versus True RUL in the Low-Capacity Stress Test.
31. Section 6.5, Influence of Fine-Tuning Cell Selection and Quantity: Fig. 31, Group- or Cell-Level Error Breakdown in the Low-Capacity Stress Test.
