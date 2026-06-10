# 论文主图中文说明版

这份文件用于解释每张主图在论文里承担什么作用、读者应该从图里看出什么，以及为什么放在对应 section。它不是正式英文 caption，而是写作和画图时的中文叙事参考。

## Fig. 1. Overview of Representative Transfer-Learning Strategies

放置位置：Section 2.3, Transfer Learning.

这张图放在第二章的背景部分，用来解释 transfer learning 不是单一做法，而是有几种典型路线，例如 instance-based transfer、feature-representation transfer，以及 parameter transfer 或 fine-tuning-based transfer。

它的作用是给读者一个方法谱系图。后面正文真正采用的是 fine-tuning-based route，所以这张图可以先把“我们的方法在 transfer-learning 文献里属于哪一类”讲清楚，再进入后面的具体建模流程。

读图重点：这张图不是结果图，而是概念图。重点是让读者看懂不同 transfer-learning 路线的区别，并明确本论文采用的是 source pretraining 加上 target fine-tuning 这一类 parameter-transfer strategy。

## Fig. 2. Operating-Condition Space of All Cell Groups

放置位置：Section 3.1, Dataset Description.

这张图先让读者看到整个数据集的实验条件空间。横纵深三个轴分别对应充电 C-rate、放电 C-rate 和平均 DoD，每个点代表一个 cell group，颜色或点大小可以表示该 group 的平均寿命。

它的作用是交代“数据不是来自同一种工况”，而是覆盖了多个充放电和 DoD 条件。后面讨论 source domain、target fine-tuning domain 和 target-test domain 时，读者已经知道这些 domain 是从一个有明显工况差异的数据空间里划分出来的。

读图重点：不同 group 在 3D 条件空间中分布不重合，且寿命差异明显。这为后面的跨 group 泛化问题做铺垫。

## Fig. 3. EOL70 RUL Label Definition

放置位置：Section 3.2, Label Definition.

这张图解释最终 RUL 任务的标签是怎么定义的。可以画一条容量或 SOH 随 cycle/week 下降的曲线，标出观测周，例如 week 5，再标出 SOH 到达 70% 阈值的时间点，两者之间的时间差就是 EOL70 RUL。

它的作用是把抽象的 RUL label 变成可视化定义。读者需要先明白模型预测的不是当前 SOH，而是从早期观测点到 70% 退化阈值之间还剩多少时间。

读图重点：RUL 是一个从 early observation week 出发的剩余时间标签，而不是整段 lifetime 本身。

## Fig. 4. Dataset and Sample-Construction Pipeline

放置位置：Section 3.3, Sample Construction.

这张图说明原始数据如何变成建模样本。流程可以从 raw RPT、cycling、capacity fade、Q-interpolated data 开始，经过清洗、SOH 样本构造、RUL 样本构造、domain split，最后进入模型训练和评价。

它的作用是给读者一个完整的数据处理地图。论文里既有前期 SOH 分析，也有最终 RUL transfer-learning 实验，这张图可以解释两条任务线分别从哪里来，又如何使用同一批电池退化数据。

读图重点：SOH 样本和 RUL 样本不是同一种数据组织方式，SOH 用于健康状态诊断分析，RUL 用于最终跨 group 预测任务。

## Fig. 5. Week-Based Early-Feature Sample Construction

放置位置：Section 3.3, Sample Construction.

这张图专门解释 week-based RUL 样本如何构造。每个 cell 在某个早期 observation week，例如 w5、w6、w10，被提取一组早期特征，然后和该 cell 的 EOL70 lifetime 或 RUL label 配对。

它的作用是说明“早期预测”的实验设定。模型不能使用完整寿命曲线，只能使用某个早期周数之前的信息来预测未来剩余寿命。

读图重点：不同 feature week 代表不同数据可用程度。week 越晚，通常信息越多，但可用样本数量和实际早期预测意义也会变化。

## Fig. 6. Overall SOH Correlation Ranking of the Ten Engineered Features

放置位置：Section 3.4, Feature Engineering.

这张图展示十个 engineered features 与 SOH 的整体相关性排序。重点突出 f8、f1、f7 等与 SOH 绝对相关性最高的特征，用一个很直接的 ranked bar chart 先把 SOH feature engineering 的结果摆出来。

它的作用是证明前面做的 IC/DVA-derived feature engineering 不是“纯构造”，而是真的能捕捉 battery health information。因为 SOH 这条线本来就是先做的，所以它应该先出现在第 3 部分，作为 feature engineering 的第一层结果。

读图重点：强 SOH 相关性说明这些特征对健康退化敏感，也为后面为什么会选它们进入 SOH estimation analysis 提供依据。

## Fig. 7. Selected SOH Feature Combination Used in the Ridge-Based SOH Estimation Analysis

放置位置：Section 3.4, Feature Engineering.

这张图更适合用一个紧凑的表格来表达，列出 SOH estimation analysis 最终采用的 feature combination。必要时可以补一列 physical meaning 或 feature family，但整体保持简洁。

它的作用是把 SOH feature engineering 的“结果”落到具体输入组合上。Fig. 6 说明哪些特征和 SOH 关系强，Fig. 7 则明确告诉读者：真正报导的 SOH Ridge analysis 最后用了哪些特征。

读图重点：这张图回答的是“SOH prediction 这一支到底用了什么输入”，它让 feature engineering 和后面的 SOH prediction result 连得更顺。

## Fig. 8. Input Feature Correlation Matrix for the Main 3-Step RUL Pipeline

放置位置：Section 3.4, Feature Engineering.

这张图转到主线 RUL feature engineering，使用 candidate input features 之间的 Pearson correlation matrix。行和列都放主线特征，颜色表示相关性强弱，必要时可以标出最终进入主线 RUL 实验的 selected features。

它的作用是说明 RUL 输入特征空间的结构和冗余性。因为前面已经先交代了 SOH feature engineering，这里就可以自然过渡到最终 3-step RUL pipeline 所关心的 early-life feature space。

读图重点：高度相关的特征可能信息重复，相关性较低的特征可能提供补充信息。这张图回答的是“主线 RUL 输入空间长什么样”。

## Fig. 9. Final Input Feature Combination Used in the Main RUL Experiment

放置位置：Section 3.4, Feature Engineering.

这张图同样更适合做成一个 compact table，专门列出主线 RUL 实验最终采用的输入特征组合。它不再在正文前段并列展开各种 auxiliary protocols，而是只服务于主线。

它的作用是把 RUL feature engineering 的结果明确落到主线 RUL 实验上。这样第 3 部分的逻辑就很完整了: 先 SOH feature engineering，再 RUL feature engineering，最后进入正式建模。

读图重点：这张图回答的是“主报导的三阶段 RUL transfer-learning 模型到底用了哪些输入”，是主线方法叙事里的关键承接点。

## Fig. 10. EOL70 Source, Target Fine-Tuning, and Target Test Split

放置位置：Section 3.5, Data Preparation and Domain Definition for Modelling.

这张图展示最终 EOL70 week-5 transfer-learning 实验的数据划分。仍然使用 3D operating-condition space，但用不同颜色标出 source train、target fine-tuning 和 target test。

它的作用是说明论文的核心评价不是随机划分，而是跨 group、跨工况条件的泛化评价。target-test group 在训练和 fine-tuning 中都不直接用于最终拟合，因此能测试模型是否真的迁移到新 group。

读图重点：source、target fine-tuning 和 target test 在工况空间中的位置不同，这就是 domain shift 的来源。

## Fig. 11. Lifetime Distribution Across the Three Data Partitions

放置位置：Section 3.5, Data Preparation and Domain Definition for Modelling.

这张图比较 source train、target fine-tuning 和 target test 三个 partition 的 lifetime 或 EOL70 lifetime weeks 分布。可以用 boxplot、violin plot 加散点。

它的作用是补充 Fig. 10。Fig. 10 讲工况空间差异，Fig. 11 讲寿命分布差异。两者合起来说明 transfer-learning 任务既有 operating-condition shift，也有 degradation outcome 的分布差异。

读图重点：如果 target-test 的寿命分布和 source 不完全一致，那么模型在 target-test 上的误差更能反映跨域预测难度。

## Fig. 12. Source-Only and Benchmark Baseline Protocols

放置位置：Section 4.3, Cross-Domain RUL Prediction with Fine-Tuning-Based Transfer Learning.

这张图解释两个 baseline protocol。source-only 是只用 source domain 训练，然后直接预测 target test；benchmark 是对应的非 transfer 设置，用于和 fine-tuned transfer model 比较。

它的作用是明确“transfer 有没有用”需要和什么对象比较。没有 baseline，单独报告 fine-tuning 模型误差无法判断迁移学习是否真的带来改进。

读图重点：source-only 衡量直接跨域泛化能力，benchmark 衡量非迁移训练设定下的参考性能，后面的 transfer result 都要和它们比较。

## Fig. 13. Fine-Tuning-Based Transfer-Learning Pipeline

放置位置：Section 4.3, Cross-Domain RUL Prediction with Fine-Tuning-Based Transfer Learning.

这张图展示本文采用的 fine-tuning transfer-learning 流程。流程包括 source pretraining、checkpoint selection、partial freezing、target fine-tuning，以及最后在 target test 上评价。

它的作用是把 transfer-learning 方法讲清楚。读者需要知道模型不是直接把 source model 拿去测试，而是先从 source 学到通用表示，再用有限 target fine-tuning cells 做适配。

读图重点：fine-tuning 的关键控制因素包括冻结层数、target fine-tuning cell 数量、replay weight 和 fine-tuning epoch，这些设置会影响最终迁移效果。

## Fig. 14. Staged Optuna/TPE Hyperparameter-Search Workflow

放置位置：Section 4.4, Hyperparameter Optimization.

这张图说明超参数搜索不是一次性完成，而是 staged workflow。Stage 1 搜 source pretraining 配置，Stage 2 搜 target fine-tuning 配置，Stage 3 用选出的配置重新训练并测试。

它的作用是解释模型参数怎么选出来，避免读者误以为最终结果是手工调出来的。因为详细 HPO 结果已经不再占主图位，这张图主要承担方法解释功能。

读图重点：Stage 2 对 transfer performance 很关键，因为 fine-tuning 的学习率、冻结层、support ratio 等都会显著影响 target-domain adaptation。

## Fig. 15. Within-Cell SOH Estimation Summary Across Cells

放置位置：Section 5.1, Within-Cell SOH Estimation Results.

这张图汇总 within-cell Ridge SOH estimation 在不同 cell 上的整体表现。可以用 aggregated MAE/MAPE boxplot、heatmap 或 summary panel，把准确性和 seed robustness 一起表达出来。

它的作用是让第 5 章真正开始进入 SOH prediction result，而不是继续停留在 feature evidence。也就是说，这里报导的是“在同一电池内做 SOH 估计时，模型表现如何”。

读图重点：看 overall error 水平和不同 cells 之间的波动，理解 within-cell SOH estimation 是一个相对容易、但也能作为后续跨 cell/generalization 对照的 setting。

## Fig. 16. Representative Predicted-Versus-True SOH for Within-Cell Ridge Estimation

放置位置：Section 5.1, Within-Cell SOH Estimation Results.

这张图展示代表性 within-cell Ridge model 的 predicted-versus-true SOH 散点。可以选一个 representative cell，也可以用一个汇总 all-points 版本。

它的作用是补充 Fig. 15 的 aggregate result。Fig. 15 说明整体误差水平，Fig. 16 则让读者直观看到 within-cell SOH prediction 的拟合形态、校准程度和残差分布。

读图重点：点越靠近 identity line，说明 within-cell SOH estimation 越稳定；如果残差模式很规整，也说明特征和 SOH 关系在单 cell 内部比较清晰。

## Fig. 17. Summary of SOH Estimation Results Beyond Within-Cell Self-Prediction

放置位置：Section 5.2, Cross-Cell and Cross-Group SOH Estimation Results.

这张图是 5.2 的总览图，用来汇总所有“不是单电池预测自己”的 SOH estimation 结果。也就是说，这一张图里要同时让读者看到两类结果：一类是一个电池训练后去预测别的电池，另一类是用一部分电池或一部分 group 去预测另一部分电池或另一部分 group。

它的作用是先把 5.2 的边界说清楚。第 5.1 节只管 self-prediction，而第 5.2 节收纳的是其余所有 SOH generalization experiments。这样读者不会误以为 5.2 只是在讲某一个 cross-group protocol。

读图重点：先看两类 setting 相比 Fig. 15 的 within-cell 结果下降了多少，再看哪一类 generalization 更难。这张图回答的是“离开 self-prediction 以后，SOH estimation 整体会变成什么样”。

## Fig. 18. Single-Cell-to-Multi-Cell SOH Estimation Results

放置位置：Section 5.2, Cross-Cell and Cross-Group SOH Estimation Results.

这张图专门对应“一个电池测别的电池”这一支结果。最直接的做法是拿 `G1C1` 这种单个 train cell 的结果，展示它在其它测试电池上的 `test_summary`、`test_cell_metrics`，或者做成按测试 cell 排序的 summary 图。

它的作用是把 5.2 里的第一类 generalization 单独讲清楚。这里的难点不是部分 group 对部分 group，而是训练信息只来自一块电池，却要泛化到很多其它电池，所以这是最“稀疏训练信息”的 SOH setting。

读图重点：看不同 test cells 的误差分布差异。如果有些电池预测得很好、有些明显变差，就说明单电池训练得到的 SOH relationship 对跨电池泛化并不稳定。

## Fig. 19. Subset-to-Subset SOH Estimation Results Under Domain Shift

放置位置：Section 5.2, Cross-Cell and Cross-Group SOH Estimation Results.

这张图专门对应“部分电池测另一部分电池”这一支结果。这里可以汇总几类 domain-shift protocol：Groups 1-4 train to Groups 5-8 test，low-to-moderate DoD train to high-DoD test，37-cell train to 12-cell held-out test，以及 37-cell source training plus 8-cell target fine-tuning before 12-cell held-out testing，用 summary metrics 加上 representative group-wise error 或 predicted-vs-true 子图来表达。

它的作用是把 5.2 里的第二类 generalization 单独讲清楚。相比 Fig. 18，这里训练集不再只是一个 cell，而是一批 cells 或 groups；但测试集同样来自另一批 cells 或 groups，因此更接近真正的 subset-to-subset / group-to-group SOH estimation。

读图重点：看不同 split 或不同 target groups 之间的误差差异。如果某些 domain-shift protocol 明显更难，说明 SOH estimation 的 generalization 不仅取决于是否跨电池，还取决于跨的是哪一类 cell subset 或 operating-condition subset。

## Fig. 20. Main Target-Test Metrics for the 70% SOH End-of-Life RUL Task

放置位置：Section 5.3, Performance of Cross-Domain RUL Prediction with Fine-Tuning-Based Transfer Learning.

这张图是主报导点里最核心的定量结果。用 grouped bar chart 比较 benchmark、source-domain-only 和 fine-tuned transfer 在 target test 上的 MAE、RMSE、MAPE 和 R2，焦点明确放在主线 RUL 实验设定。

它的作用是回答论文核心问题：在以观测第 5 周特征和 70% SOH 寿命终点为标签定义的主线 RUL 设定下，fine-tuning transfer learning 是否提升了 RUL prediction。当前结果中 fine-tuned transfer 的 MAE 为 3.773 weeks，低于 benchmark 的 4.181 weeks 和 source-domain-only 的 5.104 weeks。

读图重点：重点看 target-test MAE 和 R2。fine-tuning 相对 benchmark 约降低 9.8% MAE，相对 source-only 约降低 26.1% MAE，说明主实验中迁移学习有效。

## Fig. 21. Predicted Versus True RUL for the Main 70% SOH End-of-Life Run

放置位置：Section 5.3, Performance of Cross-Domain RUL Prediction with Fine-Tuning-Based Transfer Learning.

这张图展示主线 70% SOH 寿命终点设置下预测 RUL 和真实 RUL 的散点关系。可以把 benchmark、source-domain-only 和 fine-tuned transfer 并列展示，统一坐标范围，并加 y=x identity line。

它的作用是补充 Fig. 20 的 aggregate metrics。指标告诉读者误差有多大，predicted-vs-true 图告诉读者模型是否系统性高估、低估，以及误差是否集中在某些 RUL 区间。

读图重点：点越靠近 y=x 线，预测越准确；如果某个模型的点更集中且偏差更小，说明 calibration 和稳定性更好。

## Fig. 22. Group-Level Error Breakdown for the Main 70% SOH End-of-Life Run

放置位置：Section 5.3, Performance of Cross-Domain RUL Prediction with Fine-Tuning-Based Transfer Learning.

这张图把主线 70% SOH 寿命终点设置下的 target-test 误差按 group 拆开。可以比较 benchmark、source-domain-only 和 fine-tuned transfer 在每个 target group 上的 MAE 或 MAPE。

它的作用是回答“平均改进是不是每个 group 都有”。有时 overall MAE 下降可能主要来自少数 group，大多数 group 改善不明显；这张图可以帮助讨论 transfer learning 的稳定性。

读图重点：看哪些 target groups 改善最大，哪些 group 仍然误差高。这些 group 往往对应更困难的 operating conditions 或更强 domain shift。

## Fig. 23. Cell-Level Error Breakdown for the Main 70% SOH End-of-Life Run

放置位置：Section 5.3, Performance of Cross-Domain RUL Prediction with Fine-Tuning-Based Transfer Learning.

这张图进一步把主线 70% SOH 寿命终点设置下的误差拆到 cell level。可以画 cell-wise absolute error、MAE 或 MAPE 分布，用 violin、boxplot 或 bar chart。

它的作用是展示同一个 group 内部的误差离散程度。group-level 平均值可能掩盖个别 cell 的大误差，cell-level 图能指出模型在哪些具体电池上仍然不稳定。

读图重点：如果 fine-tuned transfer 降低了大部分 cell 的误差，说明改进更稳健；如果只改善少数 cell，则需要在 Discussion 中谨慎解释。

## Fig. 24. Valid Sample Availability Across Feature Weeks

放置位置：Section 5.5, Impact of Data Availability on Cross-Domain RUL Prediction.

这张图展示不同 feature week 下可用 cell 或 sample 数量。可以画每个 week 的 valid cell count 或 sample count。

它的作用是为 early-week sensitivity 分析做铺垫。不同 week 的结果不仅受信息量影响，也受样本可用性影响。越晚的 week 可能有更多退化信息，但也可能排除早期失效或数据不足的 cell。

读图重点：看 week 变化时样本数量是否变化明显。样本数量变化会影响后面 Fig. 25 和 Fig. 26 的性能解释。

## Fig. 25. Early-Week Sensitivity of Target-Test Error

放置位置：Section 5.5, Impact of Data Availability on Cross-Domain RUL Prediction.

这张图展示 target-test MAE 随 feature week 变化的趋势。可以把 benchmark、source-only 和 fine-tuned transfer 三条线画在同一图中。

它的作用是回答“早期数据用到第几周会影响预测性能多少”。当前结果显示 benchmark MAE 从 w6 的 4.887 weeks 降到 w10 的 3.607 weeks，fine-tuned transfer MAE 从 w6 的 6.077 weeks 降到 w10 的 4.075 weeks。

读图重点：通常 week 越晚，模型看到的信息越多，误差会降低。但这里 fine-tuning 不一定始终优于 baseline，说明 transfer learning 的效果依赖具体 protocol。

## Fig. 26. Transfer Improvement Across Feature Weeks

放置位置：Section 5.5, Impact of Data Availability on Cross-Domain RUL Prediction.

这张图直接画 fine-tuned transfer 相对 benchmark 的 MAE 改善百分比，横轴是 feature week，并加 0% 参考线。

它的作用是把 Fig. 25 中的多条误差曲线转化成“迁移到底是改善还是变差”的结论图。正值表示 transfer improves，负值表示 negative transfer。

读图重点：如果某些 week 出现负值，说明 fine-tuning 在这些 early-week 设置下反而伤害 target-test performance，这对 Discussion 中讨论 negative transfer 很重要。

## Fig. 27. Target Fine-Tuning Coverage Index Across Random Seeds

放置位置：Section 5.6, Coverage Index Analysis of Random-Seed Sensitivity.

这张图把前面 random-seed sensitivity 里观察到的 target fine-tuning cell distribution 问题进一步量化。图里可以把每个 seed 的 target fine-tuning coverage index 画出来，再和 MAE improvement 或 R2 delta 做 scatter 对比。

它的作用是把“哪些 seed 的 fine-tuning cell 分布更有利”从定性观察变成定量结果。这里强调的不是简单的平铺或聚集，而是 target fine-tuning cells 对 target-test cells 的有效覆盖程度。

读图重点：如果 index 越高时，MAE improvement 越容易为正，或者 R2 delta 越大，就说明 fine-tuning cells 在 target domain 里的覆盖质量确实和 transfer 成败有关。这张图是后面解释 fine-tuning cell selection 机制的重要桥梁。

## Fig. 28. Low-Capacity Target-Domain Split

放置位置：Section 6.4, Subjectivity of Domain Definition.

这张图展示 low-capacity stress-test 的 source、target fine-tuning 和 target-test 划分。可以沿用 3D operating-condition map 或 group split map。

它的作用是说明 low-capacity 实验使用了另一种 domain definition。这个实验不是主结果的重复，而是用更有挑战性的 target domain 检验 fine-tuning 是否仍然有效。

读图重点：看 low-capacity target domain 和主 EOL70 week-5 设置相比是否更偏、更难。如果 target domain 定义变了，transfer performance 也可能改变。

## Fig. 29. Low-Capacity Target-Test Metrics

放置位置：Section 6.4, Subjectivity of Domain Definition.

这张图展示 low-capacity stress-test 下 benchmark 和 fine-tuned transfer 的 target-test metrics。风格应与 Fig. 20 保持一致，便于比较。

它的作用是呈现 negative transfer 证据。当前结果中 benchmark MAE 为 3.882 weeks，而 fine-tuned transfer MAE 为 4.450 weeks，fine-tuning 使误差增加，MAE change 为 -14.63%。

读图重点：这张图说明 fine-tuning 在主实验有效，但在 low-capacity target-domain protocol 下不一定有效。论文可以据此讨论 domain definition 对结论的影响。

## Fig. 30. Predicted Versus True RUL in the Low-Capacity Stress Test

放置位置：Section 6.4, Subjectivity of Domain Definition.

这张图展示 low-capacity stress-test 中 benchmark 和 fine-tuned transfer 的 predicted-vs-true scatter。应统一坐标和 identity line。

它的作用是解释 Fig. 29 中 negative transfer 的误差来源。aggregate metrics 告诉我们 fine-tuning 变差了，而 scatter 图可以看出变差来自系统性偏差、离散度增加，还是某些 RUL 区间预测失败。

读图重点：如果 fine-tuned transfer 的点更远离 y=x 线，或出现明显高估/低估，就能直观看出为什么 stress-test 中 transfer worsens performance。

## Fig. 31. Group- or Cell-Level Error Breakdown in the Low-Capacity Stress Test

放置位置：Section 6.5, Influence of Fine-Tuning Cell Selection and Quantity.

这张图把 low-capacity stress-test 的误差按 group 或 cell 拆开。篇幅有限时优先保留 group-level；如果需要更细，可以另拆 cell-level。

它的作用是进一步解释 negative transfer 由哪些 target groups 或 cells 贡献。它也连接 Section 6.5 中关于 fine-tuning cell selection 和 quantity 的讨论。

读图重点：如果只有少数 group/cell 误差特别大，说明问题可能来自 target fine-tuning cell selection 或某些极端 target condition；如果大多数 group 都变差，则说明该 domain definition 下 fine-tuning 策略整体不稳。
