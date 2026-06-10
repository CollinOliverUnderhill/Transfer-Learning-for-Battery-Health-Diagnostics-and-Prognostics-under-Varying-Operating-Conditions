# Target Fine-Tuning Support Index for Battery Selection

This document formalizes a unified index for selecting target fine-tuning cells in the transfer-learning experiments. The aim is to describe when a selected target fine-tuning set is expected to provide better support for the target-test cells, and therefore to increase the probability of improving MAE and MAPE relative to the benchmark model.

The earlier file `Seed_Coverage_Index_Analysis.md` should be treated as a working note. The present document is the formal version intended for supervisor review.

## Scientific Motivation

In the target-domain transfer-learning setting, the fine-tuning cells are not only additional training samples. They define the local target-domain support available to the transferred model. If the selected fine-tuning cells are close to the target-test cells in the relevant input space, fine-tuning is more likely to adapt the source model toward the target-test distribution. If some target-test cells are far from all selected fine-tuning cells, fine-tuning may not support those cells and can even degrade aggregate MAE/MAPE.

The central hypothesis is:

```text
A target fine-tuning set is more likely to improve target-test MAE and MAPE
when it provides stronger geometric support for the target-test cells in the
representation space used to describe the target domain.
```

The same support index can be evaluated at three levels of representation precision:

1. Operating-condition level: `charging_crate`, `discharging_crate`, and `dod_pct`.
2. Full 10-feature level: `f1, ..., f10` from the stage-3 sample tables.
3. Active model-input level: the actual candidate input columns used by each trained model, such as `f1_w5`, `f1_w5+f5_w5+f4_w5`, or `f1_w5+f6_w5`.

The expected trend is not that every higher-level index must be numerically larger. The scientific expectation is that indices computed in a representation closer to the model-relevant degradation information should have a stronger relationship with target-test MAE/MAPE improvement.

## Unified Index Definition

Let `F = {f_1, ..., f_m}` be the target fine-tuning cells and `T = {t_1, ..., t_n}` be the target-test cells. For a representation level `l`, define a vector mapping:

$$
\phi_l(c) \in \mathbb{R}^{p_l}
$$

where `c` is a cell and `p_l` is the dimensionality of the representation. Each coordinate is standardized before distance calculation:

$$
z_j(c) = \frac{\phi_{l,j}(c) - \mu_{l,j}}{\sigma_{l,j}}
$$

For fixed representation spaces, such as the operating-condition space and the full 10-feature space, `mu_l,j` and `sigma_l,j` are computed from the unique target-domain points appearing across the seed analysis. For the active model-input level, the selected input dimensions differ between experiments, so standardization is performed within the active input subspace of each experiment.

The standardized fine-tuning and test sets are:

$$
Z_F^{(l)} = \{z_l(f_i)\}_{i=1}^{m},
\qquad
Z_T^{(l)} = \{z_l(t_k)\}_{k=1}^{n}
$$

Three support-gap terms are computed:

$$
\begin{aligned}
d_{\mathrm{centroid}}^{(l)}
&= \left\|
\frac{1}{m}\sum_{i=1}^{m} z_l(f_i)
{}- \frac{1}{n}\sum_{k=1}^{n} z_l(t_k)
\right\|_2
\end{aligned}
$$

$$
\begin{aligned}
d_{\mathrm{NN,mean}}^{(l)}
&= \frac{1}{n}\sum_{k=1}^{n}
\min_{1 \le i \le m}
\left\| z_l(t_k) - z_l(f_i) \right\|_2
\end{aligned}
$$

$$
\begin{aligned}
d_{\mathrm{NN,p90}}^{(l)}
&= P_{90}\left(
\left\{
\min_{1 \le i \le m}
\left\| z_l(t_k) - z_l(f_i) \right\|_2
\right\}_{k=1}^{n}
\right)
\end{aligned}
$$

The unified Target Fine-Tuning Support Index is:

$$
\begin{aligned}
\mathrm{TFSI}^{(l)}
&=
\frac{1}{
1
{}+ d_{\mathrm{centroid}}^{(l)}
{}+ d_{\mathrm{NN,mean}}^{(l)}
{}+ d_{\mathrm{NN,p90}}^{(l)}
}
\end{aligned}
$$

Interpretation:

- `0 < TFSI_l <= 1`.
- Larger `TFSI_l` means stronger support of target-test cells by the selected target fine-tuning cells.
- The `p90` term makes the index tail-aware, so a few poorly supported target-test cells are penalized.
- `TFSI_l` is a selection-support index, not a direct performance metric.

For conservative prior selection, the following risk term is also retained:

$$
\begin{aligned}
d_{\mathrm{NN,max}}^{(l)}
&=
\max_{1 \le k \le n}
\min_{1 \le i \le m}
\left\| z_l(t_k) - z_l(f_i) \right\|_2
\end{aligned}
$$

This term is useful when the representation is coarse. It checks whether any target-test cell is left far from all target fine-tuning cells.

## Practical Selection Rule

The index should be used differently depending on which representation is available at the time of fine-tuning cell selection.

At the operating-condition level, the index is a split-quality constraint rather than a strong performance predictor. The recommended rule is:

```text
Avoid selecting fine-tuning sets with large condition-space d_NN,max.
Prefer splits with broad group coverage and small worst-case condition distance.
```

At the full 10-feature level, the index becomes a stronger diagnostic and can be used more directly:

```text
Prefer fine-tuning sets with larger TFSI_10f and smaller d_NN,p90_10f.
```

At the active model-input level, the index checks whether the final selected feature subset also supports the target-test cells. This level is model-specific and depends on the chosen candidate input columns. It is useful as a consistency check, but the full 10-feature level remains more stable because it evaluates target support before information is discarded by feature-subset selection.

## Validation Data

The validation uses 21 samples:

- 20 random target fine-tuning splits: `seed000` to `seed019`.
- One reference rerun experiment: `stage3_final_rerun_400`, reported as `rerun400`.

The performance labels are based on target-test improvement relative to the benchmark:

$$ I_{\mathrm{MAE}}(\%) = 100 \times \frac{\mathrm{MAE}_{\mathrm{bench}} - \mathrm{MAE}_{\mathrm{transfer}}}{\mathrm{MAE}_{\mathrm{bench}}} $$

$$ I_{\mathrm{MAPE}}(\%) = 100 \times \frac{\mathrm{MAPE}_{\mathrm{bench}} - \mathrm{MAPE}_{\mathrm{transfer}}}{\mathrm{MAPE}_{\mathrm{bench}}} $$

A sample is labeled `positive` only when both MAE improvement and MAPE improvement are positive. Under this definition, the 21 samples contain 10 positive and 11 negative cases.

## Empirical Results

The table below summarizes how the three support levels relate to target-test improvement.

| Metric | Representation | Corr. with MAE improvement | Corr. with MAPE improvement | Best threshold accuracy | Interpretation |
|---|---|---:|---:|---:|---|
| `TFSI_cond` | operating conditions | 0.1670 | 0.1737 | 0.6667 | weak continuous predictor |
| `d_NN,max_cond` | operating conditions | -0.2828 | -0.3997 | 0.7619 | useful coarse exclusion rule |
| `TFSI_10f` | full 10-feature space | 0.6192 | 0.6142 | 0.7143 | strongest positive index form |
| `d_NN,p90_10f` | full 10-feature space | -0.7550 | -0.7695 | 0.7143 | strongest linear diagnostic |
| `TFSI_input` | active model-input space | 0.2966 | 0.3274 | 0.7143 | model-specific consistency check |
| `d_NN,p90_input` | active model-input space | -0.3010 | -0.3262 | 0.7143 | moderate diagnostic |

The final two rows above use only the final model input columns selected for each Stage3 experiment and compare them with final target-test improvement. This is the valid test-level active-input-space check currently supported by the available Stage3 results.

The condition-space index has weak correlations with MAE/MAPE improvement. However, the worst-case condition-space distance is still useful as a threshold-style rule. In the 21-sample set, the best threshold for the conservative condition risk term is approximately:

$$ d_{\mathrm{NN,max}}^{(\mathrm{cond})} \le 0.901 $$

which gives 16 correct classifications out of 21 samples under the strict MAE+MAPE positive definition.

The full 10-feature representation gives the strongest relationship with performance. The feature-space tail distance has strong negative correlations with both MAE and MAPE improvement:

$$ \mathrm{Corr}\left(d_{\mathrm{NN,p90}}^{(10f)}, I_{\mathrm{MAE}}\right) = -0.7550 $$

$$ \mathrm{Corr}\left(d_{\mathrm{NN,p90}}^{(10f)}, I_{\mathrm{MAPE}}\right) = -0.7695 $$

This supports the core hypothesis: fine-tuning selection is better explained by support in degradation-feature space than by operating conditions alone.

## Candidate Input-Space Validation Status

A stricter third-level validation would require evaluating every Stage1-derived candidate feature space on the final target-test set. This is the correct endpoint for the present hypothesis, because the purpose of selecting fine-tuning cells is to improve final target-test MAE and MAPE, not only to reduce target fine-tuning validation error.

The available Stage2 candidate runs contain target fine-tuning training and validation metrics, but they do not contain target-test metrics. Therefore, Stage2 validation MAE/MAPE should not be used as primary evidence for whether a candidate feature space leads to better fine-tuning-cell selection. Validation performance can show that a configuration fits or adapts to the fine-tuning/validation subset, but it does not demonstrate that the selected fine-tuning cells improve the held-out target-test cells.

Consequently, the completed test-level evidence currently consists of:

- operating-condition support versus final target-test MAE/MAPE improvement;
- full 10-feature support versus final target-test MAE/MAPE improvement;
- active-input support for the final selected Stage3 input columns versus final target-test MAE/MAPE improvement.

The full candidate-level third step remains an additional experiment:

1. Select a set of Stage1-derived candidate feature spaces.
2. For each candidate feature space, compute `TFSI_candidate` using the target fine-tuning and target-test cells in that candidate input space.
3. For each candidate feature space, run the corresponding final Stage3 evaluation on the same held-out target-test cells.
4. Correlate `TFSI_candidate` and `d_NN,p90_candidate` with final target-test MAE/MAPE improvement relative to the benchmark.

Until this candidate-level Stage3 test evaluation is run, `TFSI_candidate` should be described only as a proposed extension, not as a validated selection rule.

## All-Sample Results

| Sample | MAE improve % | MAPE improve % | Status | TFSI_cond | dmax_cond | TFSI_10f | p90_10f | TFSI_input | p90_input | input cols |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| seed000 | 15.2350 | 12.8547 | positive | 0.4319 | 0.8727 | 0.1975 | 1.7631 | 0.6313 | 0.3736 | `f1_w5` |
| seed001 | -32.8608 | -36.9238 | negative | 0.4265 | 1.8514 | 0.1203 | 4.5456 | 0.2990 | 0.9923 | `f1_w5+f3_w5+f10_w5` |
| seed002 | -4.2194 | -1.6118 | negative | 0.5871 | 0.0000 | 0.2255 | 1.4495 | 0.3762 | 0.7873 | `f1_w5+f5_w5+f2_w5` |
| seed003 | 6.4646 | 6.7597 | positive | 0.8549 | 0.0000 | 0.1878 | 1.9311 | 0.3579 | 0.9961 | `f1_w5+f5_w5+f2_w5` |
| seed004 | -22.0193 | -30.1941 | negative | 0.6215 | 1.1904 | 0.1227 | 4.6999 | 0.3103 | 1.1137 | `f1_w5+f5_w5+f2_w5` |
| seed005 | 4.3588 | -4.6527 | negative | 0.3883 | 1.5691 | 0.2105 | 1.9344 | 0.3528 | 0.9494 | `f1_w5+f5_w5+f4_w5` |
| seed006 | -0.1311 | -6.5079 | negative | 0.5831 | 1.0119 | 0.2226 | 1.5221 | 0.3013 | 1.0149 | `f1_w5+f5_w5+f2_w5` |
| seed007 | -10.9316 | -3.9826 | negative | 0.2890 | 1.5691 | 0.1737 | 2.3470 | 0.3138 | 1.1757 | `f1_w5+f5_w5+f2_w5` |
| seed008 | 9.2719 | 12.1419 | positive | 0.7337 | 0.0000 | 0.2140 | 1.6794 | 0.4028 | 0.7597 | `f1_w5+f5_w5+f4_w5` |
| seed009 | 16.1672 | 14.1811 | positive | 0.4573 | 0.8888 | 0.2076 | 2.0409 | 0.3439 | 0.9895 | `f1_w5+f5_w5+f4_w5` |
| seed010 | -0.5848 | -4.0879 | negative | 0.4269 | 1.2129 | 0.1644 | 2.9099 | 0.3546 | 0.9642 | `f1_w5+f5_w5+f4_w5` |
| seed011 | 4.5626 | 14.2542 | positive | 0.4744 | 0.7877 | 0.1784 | 2.5171 | 0.3504 | 0.9843 | `f1_w5+f5_w5+f2_w5` |
| seed012 | 1.1152 | 2.4497 | positive | 0.7430 | 0.0000 | 0.2328 | 1.5702 | 0.8123 | 0.1399 | `f1_w5` |
| seed013 | 5.5202 | 5.4059 | positive | 0.9175 | 0.0000 | 0.2446 | 1.6401 | 0.7652 | 0.1426 | `f1_w5` |
| seed014 | -8.5477 | -4.2408 | negative | 0.7822 | 0.0000 | 0.1937 | 2.3951 | 0.4707 | 0.5677 | `f1_w5+f3_w5+f10_w5` |
| seed015 | -1.3302 | -7.4057 | negative | 0.6844 | 0.9135 | 0.1757 | 2.6363 | 0.3605 | 0.8907 | `f1_w5+f5_w5+f4_w5` |
| seed016 | 10.8437 | 11.5339 | positive | 0.9205 | 0.0000 | 0.1898 | 2.3496 | 0.4194 | 0.6475 | `f1_w5+f5_w5+f2_w5` |
| seed017 | 11.7752 | 8.1381 | positive | 0.4234 | 1.8514 | 0.1975 | 2.0669 | 0.3289 | 0.9687 | `f1_w5+f3_w5+f10_w5` |
| seed018 | -5.4500 | -5.7552 | negative | 0.8524 | 0.0000 | 0.1625 | 3.0581 | 0.4158 | 0.5084 | `f1_w5+f3_w5+f10_w5` |
| seed019 | 1.6394 | -1.3990 | negative | 0.8744 | 0.0000 | 0.2367 | 1.6016 | 0.3933 | 0.8592 | `f1_w5+f5_w5+f4_w5` |
| rerun400 | 9.7800 | 6.5340 | positive | 0.8474 | 0.7877 | 0.1748 | 2.1169 | 0.4942 | 0.4981 | `f1_w5+f6_w5` |

## Interpretation

The results support a staged interpretation of fine-tuning cell selection.

First, operating-condition coverage should be used as a prior split-quality rule. Its continuous relationship with transfer improvement is weak, but it can identify obviously risky splits where some target-test cells are poorly supported in condition space. This is why `d_NN,max_cond` is more useful than `TFSI_cond` at the condition level.

Second, feature-space support explains transfer behavior more directly. The 10-feature tail distance is strongly associated with MAE/MAPE improvement. This indicates that the effect of fine-tuning cell selection is mediated by degradation-feature similarity, not only by nominal cycling conditions.

Third, the active input-space index is currently validated only for the final selected Stage3 input columns. A complete candidate-level third step would require final target-test evaluation for each candidate feature space. Stage2 fine-tuning validation metrics are not sufficient for this purpose, because validation performance does not prove that the final held-out target-test MAE/MAPE improves.

## Recommended Thesis Statement

The following wording can be used in the thesis:

> To quantify whether the selected target fine-tuning cells geometrically support the target-test cells, a target fine-tuning support index was defined as $\mathrm{TFSI}=1/(1+d_{\mathrm{centroid}}+d_{\mathrm{NN,mean}}+d_{\mathrm{NN,p90}})$. The index combines global distribution alignment through the centroid gap with local and tail coverage through nearest-neighbour distances. The same formula was evaluated at three representation levels: operating conditions, the full 10-feature degradation representation, and the active model-input feature subset.

An additional interpretation sentence is:

> The operating-condition index was useful mainly as a prior split-quality constraint, whereas the full 10-feature support index and its tail-distance component showed substantially stronger associations with target-test MAE/MAPE improvement. This suggests that fine-tuning cell selection affects transfer performance primarily through feature-space support of the target-test degradation patterns.

For the candidate input-space status, the recommended wording is:

> The active input-space support index was evaluated for the final selected Stage3 input columns and compared with final target-test improvement. However, a complete candidate-level validation would require running final target-test evaluation for each Stage1-derived candidate feature space. Since the available Stage2 candidate runs provide fine-tuning validation metrics rather than target-test metrics, the candidate-space index is reported as a proposed extension rather than as a validated feature-subset selection rule.

## Final Recommendation for Cell Selection

For future fine-tuning cell selection, use the following rule hierarchy:

1. Before feature extraction, avoid splits with large `d_NN,max_cond` and missing target-test operating-condition groups.
2. After features are available, rank candidate fine-tuning sets by larger `TFSI_10f` and smaller `d_NN,p90_10f`.
3. During feature-subset selection, do not claim that `TFSI_candidate` is validated unless each candidate feature space has final target-test results.
4. After model-input candidates are selected, verify that the active input-space `TFSI_input` is not low and that `d_NN,p90_input` is not large.

This gives a consistent index framework while respecting the different information available at each stage of the experimental workflow.

## Additional Practical Interpretation

An important implication of the results is that the feature space used to train the prediction model does not need to be the same as the feature space used to guide target fine-tuning cell selection.

The Optuna results show that the best predictive model may use only a small subset of features. This is reasonable because sparse feature subsets can reduce redundancy, reduce overfitting risk, and improve model stability under limited target fine-tuning data. Therefore, the fact that a model trained with fewer features performs better does not contradict the observation that the 10-feature space gives a stronger support-index explanation.

The two feature spaces serve different purposes:

| Purpose | Recommended representation | Reason |
|---|---|---|
| Model training | Optuna-selected `x_cols` | Compact input space can improve prediction stability and reduce overfitting |
| Fine-tuning cell selection | Richer degradation-feature space, such as the 10-feature space | More complete description of whether fine-tuning cells support target-test degradation patterns |
| Early split design | Operating-condition space | Available before feature extraction and useful as a coarse split-quality constraint |

Thus, the support index should be interpreted as a target-domain selection descriptor rather than as a restriction on the final model input. A richer degradation-feature representation can be used to decide which target cells are useful for fine-tuning, while the final prediction model can still use a smaller feature subset selected by validation or hyperparameter optimization.

This distinction is useful in practice. It means that fine-tuning cell selection does not require knowing the final optimal model input space in advance. Instead, as long as a reasonably rich set of early-life degradation descriptors can be extracted from the target domain, these descriptors can be used to evaluate target-test support and guide the selection of target fine-tuning cells.

Recommended thesis wording:

> The representation used for fine-tuning cell selection is not necessarily identical to the representation used for final model training. While sparse feature subsets may be preferable for prediction because they reduce redundancy and overfitting, a richer degradation-feature space can better characterize whether the selected target fine-tuning cells support the target-test degradation distribution. Therefore, the proposed support index should be computed in a descriptive target-domain feature space for cell selection, whereas the final prediction model can still use a compact feature subset selected through validation and hyperparameter optimization.
