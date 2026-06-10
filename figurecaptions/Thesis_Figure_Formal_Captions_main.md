1.6.3

Fig. 1 | Overview of representative transfer-learning paradigms and the fine-tuning route adopted in this thesis. Several common transfer-learning strategies are compared schematically, including instance-based transfer, feature-representation transfer, and parameter-transfer or fine-tuning-based adaptation. The comparison situates the thesis method within the broader transfer-learning literature and clarifies why a fine-tuning-based strategy is selected for cross-domain battery prognostics.

3.1.1

Fig. 2 | Operating-condition distribution of the available battery groups. The cell groups are shown in the operating-condition space defined by charging C-rate, discharging C-rate, and mean depth of discharge (DoD). Group-level lifetime is used as the degradation outcome, illustrating that cells tested under different operating conditions can have substantially different degradation behaviour.

3.1.2

Fig. 3 | Definition of the remaining useful life label relative to the 70% SOH end-of-life threshold. For each cell, end of life is defined as the week at which SOH first decreases to 70% of the reference capacity. RUL is then calculated as the time interval from an early observation week to this 70% SOH threshold, converting the degradation trajectory into a supervised prognostic target for early-life lifetime prediction.

3.1.3

Fig. 4 | Dataset processing workflow for SOH and RUL modelling. Raw RPT, cycling, capacity-fade, and Q-interpolated data are processed into two types of modelling samples: diagnostic SOH samples and prognostic RUL samples. The workflow separates preliminary SOH-oriented feature analysis from the final week-based RUL transfer-learning experiments.

3.1.3

Fig. 5 | Construction of week-based early-life RUL samples. Features extracted at a selected early observation week are paired with the corresponding lifetime or RUL label defined by the 70% SOH end-of-life threshold. This week-based construction allows the model to be evaluated under different levels of early-life data availability.

3.1.4

Fig. 6 | Overall SOH correlations of the ten engineered features. Pearson correlations between each engineered feature and SOH are calculated across valid RPT samples. The IC/DVA-derived features f8, f1, and f7 show the strongest absolute correlations with SOH, indicating that the engineered voltage-derived features encode battery health information.

3.1.4

Fig. 7 | Selected SOH feature combination used in the Ridge-based SOH estimation analysis. The SOH-oriented feature subset adopted in the reported Ridge-based SOH analysis is summarized in a compact tabular form. This figure clarifies the exact inputs used for the SOH feature-engineering stage.

3.1.4

Fig. 8 | Correlation structure of the candidate input features for the main three-stage RUL pipeline. A Pearson-correlation matrix is used to visualize the relationships among the engineered early-life features for the 70% SOH end-of-life RUL task. The matrix highlights redundant and complementary inputs before selecting the final feature subset for the reported main experiment.

3.1.4

Fig. 9 | Final input feature combination used in the main RUL experiment. The selected feature subset for the reported observation-week-5 RUL protocol, with lifetime defined at the 70% SOH end-of-life threshold, is summarized in a compact tabular form. This figure clarifies the exact inputs used by the main three-stage transfer-learning pipeline.

3.1.5

Fig. 10 | Cross-group domain split used in the main observation-week-5 RUL transfer-learning experiment. Source-domain training groups, target-domain fine-tuning groups, and target-test groups are shown in the same operating-condition space for the 70% SOH end-of-life RUL task. The split is defined at the group/cell level so that target-test evaluation measures cross-group generalization rather than random interpolation within the same operating condition.

3.1.5

Fig. 11 | Lifetime distribution of source, target fine-tuning, and target-test cells. The lifetime distributions of the three partitions are compared using the 70% SOH end-of-life definition. The comparison verifies that the transfer-learning task involves both operating-condition shift and lifetime-distribution variation, supporting the use of a cross-group evaluation protocol.

3.2.3

Fig. 12 | Baseline protocols for target-test RUL prediction. The source-domain-only baseline is trained on source-domain cells and evaluated directly on the target-test partition without target-domain adaptation. The benchmark protocol uses the corresponding non-transfer training setting for comparison on the same target-test cells. These baselines define whether fine-tuning improves cross-group RUL prediction.

3.2.3

Fig. 13 | Fine-tuning-based transfer-learning pipeline for cross-group RUL prediction. A neural network is first pretrained using source-domain cells. The pretrained checkpoint is then adapted using a limited number of target fine-tuning cells before final evaluation on held-out target-test cells. Partial freezing, replay weighting, and fine-tuning duration control how much source knowledge is retained during target adaptation.

3.2.4

Fig. 14 | Staged Optuna/TPE search used for model and fine-tuning selection. Hyperparameters are selected through a staged search procedure. Stage 1 identifies source-pretraining configurations, Stage 2 optimizes target fine-tuning settings, and Stage 3 retrains and evaluates the selected configuration on the target-test partition.

4.1

Fig. 15 | Within-cell SOH estimation summary across cells. Ridge-based SOH estimation performance is summarized across individual cells using aggregate error statistics. The figure shows the accuracy and robustness of the easiest SOH prediction setting, where training and testing are both performed within the same cell.

4.1

Fig. 16 | Representative predicted-versus-true SOH for within-cell Ridge estimation. Predicted and measured SOH values are compared for a representative within-cell Ridge model. The scatter pattern illustrates the calibration and residual spread of SOH estimation when no cell-to-cell generalization is required.

4.2

Fig. 17 | Summary of cross-cell and cross-group SOH estimation beyond within-cell self-prediction. Target-test metrics are summarized for SOH settings that require generalization beyond the training cell, including models trained on a single source cell and evaluated on different target cells, as well as protocols that train on one subset of cells or groups and test on another. This figure defines Section 4.2 as the part of the SOH study devoted to cell-to-cell and group-to-group generalization.

4.2

Fig. 18 | Cross-cell SOH estimation using a model trained on one source cell and tested on other target cells. A Ridge model trained on a single battery cell is evaluated on different cells to quantify cell-to-cell SOH generalization beyond within-cell self-prediction. This figure isolates the one-source-cell to multiple-target-cell setting and shows how performance changes when the training signal comes from only one battery.

4.2

Fig. 19 | SOH estimation results when one subset of cells is used to predict another subset under domain shift. SOH estimation performance is summarized for protocols that train on one subset of cells or groups and evaluate on a disjoint subset. The evaluated splits include training on Groups 1-4 and testing on Groups 5-8, training on low-to-moderate DoD conditions and testing on high-DoD conditions, and training on 37 cells followed by evaluation on 12 held-out cells. These protocols show the difficulty of SOH generalization under stronger cell-group and operating-condition shifts.

4.3

Fig. 20 | Target-test metrics for the main RUL prediction experiment using the 70% SOH end-of-life definition. The benchmark model, the source-domain-only model, and the fine-tuned transfer model are evaluated on the same target-test partition under the reported main configuration. Fine-tuning reduces target-test MAE from 4.181 weeks for the benchmark and 5.104 weeks for the source-domain-only model to 3.773 weeks, corresponding to an MAE reduction of about 9.8% relative to the benchmark and 26.1% relative to the source-domain-only model.

4.3

Fig. 21 | Predicted versus true target-test RUL for the main 70% SOH end-of-life run. Predicted and true RUL values are compared using the same axis limits and identity reference line. The plot shows the calibration and dispersion of the benchmark and transfer predictions for the reported main configuration beyond the aggregate error metrics.

4.3

Fig. 22 | Group-level target-test error distribution for the main 70% SOH end-of-life run. Target-test errors are decomposed by cell group for the benchmark, source-domain-only, and fine-tuned transfer models under the reported main configuration. The group-level comparison shows whether the average improvement from fine-tuning is shared across target operating conditions or concentrated in a subset of groups.

4.3

Fig. 23 | Cell-level target-test error distribution for the main 70% SOH end-of-life run. Absolute prediction errors are summarized at the cell level for the reported main configuration. The cell-wise breakdown reveals the spread of errors within target groups and identifies individual cells for which the transfer model remains difficult to generalize.

4.5.1

Fig. 24 | Valid cell and sample availability across early observation weeks. The number of valid cells and samples is summarized for each observation week used in the 70% SOH end-of-life RUL experiments. Sample availability is an important practical constraint when comparing early-week RUL prediction results because later weeks can provide richer degradation information but may also exclude cells that fail earlier.

4.5.1

Fig. 25 | Sensitivity of target-test MAE to early-week data availability. Target-test MAE is compared from observation week 6 to week 10 for the benchmark, source-domain-only, and fine-tuned transfer models under the same 70% SOH end-of-life RUL protocol. Later observation weeks generally provide more degradation information and reduce prediction error, but fine-tuning does not consistently outperform the baselines in this sensitivity experiment.

4.5.1

Fig. 26 | Week-dependent improvement or degradation from fine-tuning. The percentage change in MAE between the fine-tuned transfer model and the benchmark model is plotted for each feature week. Positive values indicate improvement from transfer, whereas negative values indicate that fine-tuning worsens target-test error. This figure highlights the protocol-dependent nature of transfer learning for early-life RUL prediction.

4.6

Fig. 27 | Relationship between the target fine-tuning coverage index and seed-dependent transfer performance. A target fine-tuning coverage index, defined from the centroid gap and nearest-neighbour coverage between target fine-tuning cells and target-test cells in the standardized condition space, is compared across random seeds. Higher index values indicate better geometric coverage of the target-test cells and are generally associated with more positive transfer relative to the benchmark.

5.4

Fig. 28 | Lower-capacity target-domain split used for the stress-test experiment. The source, target fine-tuning, and target-test partitions are shown for a stress-test protocol in which the target domain is defined by lower-capacity cells. This split is used to test whether the fine-tuning strategy remains beneficial when the target domain differs more strongly from the main observation-week-5, 70% SOH end-of-life setting.

5.4

Fig. 29 | Lower-capacity target-domain stress-test metrics. Benchmark and transfer models are evaluated under a stress-test protocol in which the target domain is defined by lower-capacity cells. Unlike the main observation-week-5, 70% SOH end-of-life result, fine-tuning does not improve target-test performance in this setting: MAE increases from 3.882 weeks for the benchmark to 4.450 weeks for the fine-tuned transfer model. This result indicates a risk of negative transfer.

5.4

Fig. 30 | Predicted versus true RUL for the lower-capacity target-domain stress test. Benchmark and fine-tuned transfer predictions are compared with the identity line under the stress-test protocol in which the target domain is defined by lower-capacity cells. The scatter distribution helps explain why fine-tuning worsens aggregate error in this stress-test setting.

5.4

Fig. 31 | Error distribution behind negative transfer in the lower-capacity target-domain stress test. Target-test errors are decomposed by group or cell for the benchmark and fine-tuned transfer models. The breakdown identifies which lower-capacity target conditions contribute most to the degradation in transfer performance.
