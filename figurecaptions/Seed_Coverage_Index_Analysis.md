# Seed Coverage Index Analysis

This note summarizes a quantitative way to describe how the distribution of target fine-tuning cells covers the target-test cells in the target-domain condition space for the random-seed sensitivity analysis.

## Purpose

The goal is to quantify when a random seed gives a better fine-tuning result relative to the benchmark, based on how the target fine-tuning cells are distributed with respect to the target-test cells.

The condition space is defined by:

- `chg_c_rate`
- `dchg_c_rate`
- `dod_pct`

These three variables are standardized before distance calculation.

## Standardized Condition Vector

For each battery cell, define the condition vector as:

```text
x = (z_chg, z_dchg, z_DoD)
```

where each coordinate is standardized by:

```text
z_j = (x_j - mu_j) / sigma_j
```

Here, `mu_j` and `sigma_j` are the mean and standard deviation of that condition variable across the target-domain condition points considered in the seed analysis.

## Sets

Let:

- `F = {f_1, ..., f_m}`: target fine-tuning cells
- `T = {t_1, ..., t_n}`: target-test cells

## Quantitative Coverage Measures

### 1. Centroid Gap

```text
d_centroid = || mean(F) - mean(T) ||_2

            = || (1/m) * sum_i(f_i) - (1/n) * sum_k(t_k) ||_2
```

Meaning:

- measures how far the center of the target fine-tuning cells is from the center of the target-test cells
- smaller values indicate that the overall fine-tuning distribution is more aligned with the target-test distribution

### 2. Mean Nearest-Neighbour Gap

```text
d_NN_mean = (1/n) * sum_k min_{f in F} || t_k - f ||_2
```

Meaning:

- for each target-test cell, find the closest target fine-tuning cell
- then average these nearest-neighbour distances
- smaller values indicate better coverage of target-test cells by the target fine-tuning cells

### 3. 90th-Percentile Nearest-Neighbour Gap

```text
d_NN_p90 = P90({ min_{f in F} || t_k - f ||_2 } for k = 1,...,n)
```

Meaning:

- measures the upper-tail coverage difficulty
- smaller values indicate that even the harder-to-cover target-test cells remain relatively close to some target fine-tuning cell

## Proposed Index

Define the Target Fine-Tuning Coverage Index as:

```text
TFCI = 1 / (1 + d_centroid + d_NN_mean)
```

Interpretation:

- `0 < TFCI <= 1`
- larger values indicate better geometric coverage of the target-test cells by the target fine-tuning cells
- this is a distribution-coverage index, not a performance metric itself

## Alternative Coverage Definitions

The current TFCI is intentionally simple: it combines global alignment through the centroid gap and local coverage through the mean nearest-neighbour gap. Other definitions can be used depending on which aspect of coverage should be emphasized.

### 1. Tail-Aware Coverage Index

```text
TFCI_p90 = 1 / (1 + d_centroid + d_NN_mean + d_NN_p90)
```

This version penalizes seeds where most target-test cells are close to the fine-tuning set but a few difficult target-test cells are poorly covered.

### 2. Worst-Case Nearest-Neighbour Coverage

```text
d_NN_max = max_k min_{f in F} || t_k - f ||_2
```

This is useful as a conservative diagnostic. It asks whether any target-test cell is far from all target fine-tuning cells.

### 3. Radius-Based Coverage Ratio

```text
Coverage(r) = (1/n) * sum_k I[min_{f in F} || t_k - f ||_2 <= r]
```

This reports the fraction of target-test cells covered within a fixed standardized distance `r`. It is easy to interpret, but the radius must be chosen and justified.

### 4. Group-Level Coverage

Instead of using distances only, define coverage by operating-condition groups:

```text
GroupCoverage = |G_F intersect G_T| / |G_T|
```

where `G_F` and `G_T` are the sets of groups represented in the fine-tuning and target-test sets. This is simple and interpretable, but it ignores within-group geometric distances.

### 5. Distribution-Distance Measures

More formal alternatives include Maximum Mean Discrepancy (MMD), Wasserstein distance, or energy distance between the fine-tuning and target-test condition distributions. These compare the two distributions more globally, but they are less transparent than centroid and nearest-neighbour distances for a thesis figure.

The original TFCI remains useful as a transparent operating-condition coverage descriptor. However, the all-seed results below show that it should not be used as the main indicator for whether transfer learning improves or degrades the target-test result.

## Positive-Improvement Seeds

The following seeds satisfy both:

- positive MAE improvement relative to the benchmark
- positive MAPE improvement relative to the benchmark

| Seed | MAE improve % | MAPE improve % | `d_centroid` | `d_NN_mean` | `d_NN_p90` | TFCI |
|---|---:|---:|---:|---:|---:|---:|
| seed009 | 16.1672 | 14.1811 | 0.204599 | 0.194484 | 0.787739 | 0.714754 |
| seed000 | 15.2350 | 12.8547 | 0.453634 | 0.111883 | 0.749584 | 0.638767 |
| seed017 | 11.7752 | 8.1381 | 0.366430 | 0.207856 | 0.787739 | 0.635209 |
| seed016 | 10.8437 | 11.5339 | 0.086309 | 0.000000 | 0.000000 | 0.920548 |
| seed008 | 9.2719 | 12.1419 | 0.362995 | 0.000000 | 0.000000 | 0.733678 |
| seed003 | 6.4646 | 6.7597 | 0.169749 | 0.000000 | 0.000000 | 0.854884 |
| seed013 | 5.5202 | 5.4059 | 0.089959 | 0.000000 | 0.000000 | 0.917466 |
| seed011 | 4.5626 | 14.2542 | 0.243516 | 0.107338 | 0.757215 | 0.740272 |
| seed012 | 1.1152 | 2.4497 | 0.345925 | 0.000000 | 0.000000 | 0.742983 |

## All Seeds

The table below uses the same coverage definition and standardization convention as the positive-seed table above. The standardization is based on the unique target-domain condition points appearing across the seed analysis.

Status labels are based only on MAE and MAPE:

- `positive`: both MAE improvement and MAPE improvement are positive
- `negative`: at least one of MAE improvement or MAPE improvement is not positive

| Seed | MAE improve % | MAPE improve % | `d_centroid` | `d_NN_mean` | `d_NN_p90` | TFCI | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| seed000 | 15.2350 | 12.8547 | 0.453634 | 0.111883 | 0.749584 | 0.638767 | positive |
| seed001 | -32.8608 | -36.9238 | 0.380283 | 0.176829 | 0.787739 | 0.642214 | negative |
| seed002 | -4.2194 | -1.6118 | 0.703172 | 0.000000 | 0.000000 | 0.587140 | negative |
| seed003 | 6.4646 | 6.7597 | 0.169749 | 0.000000 | 0.000000 | 0.854884 | positive |
| seed004 | -22.0193 | -30.1941 | 0.328806 | 0.102348 | 0.177764 | 0.698737 | negative |
| seed005 | 4.3588 | -4.6527 | 0.229316 | 0.272483 | 1.073355 | 0.665868 | negative |
| seed006 | -0.1311 | -6.5079 | 0.407898 | 0.104682 | 0.202385 | 0.661122 | negative |
| seed007 | -10.9316 | -3.9826 | 0.625862 | 0.484319 | 1.350035 | 0.473893 | negative |
| seed008 | 9.2719 | 12.1419 | 0.362995 | 0.000000 | 0.000000 | 0.733678 | positive |
| seed009 | 16.1672 | 14.1811 | 0.204599 | 0.194484 | 0.787739 | 0.714754 | positive |
| seed010 | -0.5848 | -4.0879 | 0.150999 | 0.218269 | 0.973344 | 0.730318 | negative |
| seed011 | 4.5626 | 14.2542 | 0.243516 | 0.107338 | 0.757215 | 0.740272 | positive |
| seed012 | 1.1152 | 2.4497 | 0.345925 | 0.000000 | 0.000000 | 0.742983 | positive |
| seed013 | 5.5202 | 5.4059 | 0.089959 | 0.000000 | 0.000000 | 0.917466 | positive |
| seed014 | -8.5477 | -4.2408 | 0.278402 | 0.000000 | 0.000000 | 0.782227 | negative |
| seed015 | -1.3302 | -7.4057 | 0.228047 | 0.083194 | 0.149917 | 0.762636 | negative |
| seed016 | 10.8437 | 11.5339 | 0.086309 | 0.000000 | 0.000000 | 0.920548 | positive |
| seed017 | 11.7752 | 8.1381 | 0.366430 | 0.207856 | 0.787739 | 0.635209 | positive |
| seed018 | -5.4500 | -5.7552 | 0.173192 | 0.000000 | 0.000000 | 0.852375 | negative |
| seed019 | 1.6394 | -1.3990 | 0.143675 | 0.000000 | 0.000000 | 0.874374 | negative |

Across all 20 seeds, 9 seeds are positive and 11 seeds are negative under the MAE+MAPE definition. The mean MAE improvement is only 0.0439%, but the range is wide, from -32.8608% to 16.1672%. The mean MAPE improvement is -0.9521%, with a range from -36.9238% to 14.2542%. This confirms that the fine-tuning cell selection can materially change whether transfer learning appears beneficial.

The Pearson correlation between TFCI and MAE improvement is 0.2690, and the correlation between TFCI and MAPE improvement is 0.2375. Therefore, TFCI should be treated as a coverage descriptor rather than a useful predictor of MAE/MAPE improvement.

## Search for a Better Index

Because the condition-space TFCI does not reliably distinguish positive from negative transfer, additional distribution indices were tested using the same 20 random seeds.

The key change is to move from the three operating-condition variables to the actual model-input feature space:

```text
u = (z_f1, z_f2, ..., z_f10)
```

where `f1, ..., f10` are standardized early-life input features. The same centroid and nearest-neighbour definitions can then be applied in this 10-dimensional feature space.

### Recommended Replacement Index

The best non-label index found in this search is a feature-space tail-aware coverage index:

```text
F-TFCI_p90 = 1 / (1 + d_centroid_feature + d_NN_mean_feature + d_NN_p90_feature)
```

where:

- `d_centroid_feature` is the centroid gap between target fine-tuning and target-test cells in the standardized feature space
- `d_NN_mean_feature` is the mean nearest-neighbour distance from target-test cells to the target fine-tuning set in feature space
- `d_NN_p90_feature` is the 90th-percentile nearest-neighbour distance in feature space

This index directly measures whether target-test cells are supported by nearby fine-tuning cells in the representation actually used by the model. The p90 term is important because poor coverage of a small tail of target-test cells can affect aggregate test error even when average coverage looks acceptable.

### Candidate Comparison

| Candidate metric | Space | Uses target-test label? | Corr. with MAE improvement | Corr. with MAPE improvement | Oriented AUC for both-positive sign | Best threshold accuracy |
|---|---|---:|---:|---:|---:|---:|
| Original TFCI | operating conditions | no | 0.2690 | 0.2375 | 0.6364 | 0.65 |
| feature-space `d_NN_p90` | input features | no | -0.7588 | -0.7700 | 0.6667 | 0.70 |
| `F-TFCI_p90` | input features | no | 0.6552 | 0.6380 | 0.6869 | 0.75 |
| Joint feature+RUL tail index | input features + RUL | yes | 0.6921 | 0.6885 | 0.7374 | 0.70 |
| RUL-only index | RUL label | yes | -0.1497 | -0.1997 | 0.5051 | 0.70 |

The feature-space p90 nearest-neighbour distance has the strongest linear relationship with both MAE and MAPE improvement among the non-label metrics. Its negative correlation means that larger tail distance corresponds to worse transfer performance. The feature-space tail-aware index $\mathrm{FTFCI}_{p90}$ gives the best positive index form among the non-label candidates, but its both-positive classification accuracy is only 0.75.

The joint feature+RUL metric is slightly stronger than the feature-only index, but it uses the target-test RUL labels. Therefore, it is useful only as a post-hoc diagnostic and should not be presented as a practical selection criterion.

### All-Seed Feature-Space Tail Coverage Results

| Seed | MAE improve % | MAPE improve % | Original TFCI | feature `d_NN_p90` | `F-TFCI_p90` | MAE/MAPE status |
|---|---:|---:|---:|---:|---:|---|
| seed000 | 15.2350 | 12.8547 | 0.6388 | 1.7631 | 0.1975 | positive |
| seed001 | -32.8608 | -36.9238 | 0.6422 | 4.5456 | 0.1203 | negative |
| seed002 | -4.2194 | -1.6118 | 0.5871 | 1.4495 | 0.2255 | negative |
| seed003 | 6.4646 | 6.7597 | 0.8549 | 1.9311 | 0.1878 | positive |
| seed004 | -22.0193 | -30.1941 | 0.6987 | 4.6999 | 0.1227 | negative |
| seed005 | 4.3588 | -4.6527 | 0.6659 | 1.9344 | 0.2105 | negative |
| seed006 | -0.1311 | -6.5079 | 0.6611 | 1.5221 | 0.2226 | negative |
| seed007 | -10.9316 | -3.9826 | 0.4739 | 2.3470 | 0.1737 | negative |
| seed008 | 9.2719 | 12.1419 | 0.7337 | 1.6794 | 0.2140 | positive |
| seed009 | 16.1672 | 14.1811 | 0.7148 | 2.0409 | 0.2076 | positive |
| seed010 | -0.5848 | -4.0879 | 0.7303 | 2.9099 | 0.1644 | negative |
| seed011 | 4.5626 | 14.2542 | 0.7403 | 2.5171 | 0.1784 | positive |
| seed012 | 1.1152 | 2.4497 | 0.7430 | 1.5702 | 0.2328 | positive |
| seed013 | 5.5202 | 5.4059 | 0.9175 | 1.6401 | 0.2446 | positive |
| seed014 | -8.5477 | -4.2408 | 0.7822 | 2.3951 | 0.1937 | negative |
| seed015 | -1.3302 | -7.4057 | 0.7626 | 2.6363 | 0.1757 | negative |
| seed016 | 10.8437 | 11.5339 | 0.9205 | 2.3496 | 0.1898 | positive |
| seed017 | 11.7752 | 8.1381 | 0.6352 | 2.0669 | 0.1975 | positive |
| seed018 | -5.4500 | -5.7552 | 0.8524 | 3.0581 | 0.1625 | negative |
| seed019 | 1.6394 | -1.3990 | 0.8744 | 1.6016 | 0.2367 | negative |

Using a simple threshold around $\mathrm{FTFCI}_{p90}=0.177$, 15 of 20 seeds are classified correctly under the stricter MAE+MAPE both-positive definition. The misclassified seeds are seed002, seed005, seed006, seed014, and seed019. This is weaker than the MAE-only classification, but still better than the original condition-space TFCI.

## Operating-Condition Prior Index Search

The feature-space indices above are useful diagnostics, but they are not ideal for guiding the initial domain split because feature extraction occurs after the source / target-finetune / target-test split. Therefore, a separate search was performed using only prior operating-condition information:

- `chg_c_rate`
- `dchg_c_rate`
- `dod_pct`
- operating-condition group identity

The goal of this search was to find an index that could guide how target fine-tuning cells should be selected before model training.

### Prior Indices Tested

Several condition-space prior indices were tested:

```text
condition d_NN_p90:
P90 nearest-neighbour distance from each target-test cell
to the target fine-tuning set in standardized condition space.
```

```text
condition d_NN_max:
maximum nearest-neighbour distance from target-test cells
to the target fine-tuning set in standardized condition space.
```

```text
condition range_extrap_p90:
P90 distance by which target-test cells fall outside the min-max
condition range covered by target fine-tuning cells.
```

```text
missing_group_ratio:
fraction of target-test operating-condition groups absent from
the target fine-tuning set.
```

```text
group d_NN_p90:
P90 nearest-neighbour distance between target-test group centers
and target fine-tuning group centers in condition space.
```

Several composite indices were also tested, including:

```text
TCSI_tail_group = 1 / (1 + condition d_NN_p90 + missing_group_ratio)
```

```text
TCSI_tail_range_group =
1 / (1 + condition d_NN_p90 + condition range_extrap_p90 + missing_group_ratio)
```

```text
TCSI_all_prior =
1 / (1 + condition d_NN_p90 + condition range_extrap_p90
         + missing_group_ratio + group d_NN_p90)
```

### Prior Index Results

| Prior metric | Uses only split-stage condition info? | Corr. with MAE improvement | Corr. with MAPE improvement | Best threshold accuracy |
|---|---:|---:|---:|---:|
| Original TFCI | yes | 0.2690 | 0.2375 | 0.65 |
| condition `d_NN_p90` | yes | -0.0395 | -0.0494 | 0.60 |
| condition `d_NN_max` | yes | -0.2934 | -0.4071 | 0.75 |
| condition tail-max index | yes | 0.2201 | 0.3422 | 0.75 |
| condition `range_extrap_p90` | yes | -0.1389 | -0.0035 | 0.55 |
| `missing_group_ratio` | yes | -0.1694 | -0.2271 | 0.60 |
| group-center `d_NN_p90` | yes | -0.2038 | -0.2369 | 0.65 |
| `TCSI_tail_group` | yes | 0.0601 | 0.1125 | 0.60 |
| `TCSI_tail_range_group` | yes | 0.0606 | 0.1116 | 0.60 |
| `TCSI_all_prior` | yes | 0.1110 | 0.1791 | 0.65 |
| stratified condition-tail index | yes | 0.0923 | 0.1029 | 0.60 |

Among the strictly prior condition-space metrics, the strongest continuous relationship was obtained by the worst-case condition support metric:

```text
condition d_NN_max
```

Its correlations were:

```text
MAE improvement correlation  = -0.2934
MAPE improvement correlation = -0.4071
```

This means that larger worst-case condition distance is associated with worse transfer performance. However, the association remains much weaker than the feature-space tail metrics.

### Interpretation

The operating-condition prior search shows that condition-space information alone is not sufficient to strongly explain seed-dependent transfer performance. Even more targeted prior indices, such as tail coverage, range extrapolation, group coverage, and their combinations, do not produce strong correlations with MAE/MAPE improvement.

This result is important for the domain-split design:

- operating-condition coverage is still useful as a practical split-design constraint
- target fine-tuning cells should avoid leaving target-test conditions completely unsupported
- however, condition-space coverage alone cannot reliably predict whether transfer will improve MAE and MAPE

The best practical use of the prior condition index is therefore not as a performance predictor, but as a split-quality rule:

```text
When selecting target fine-tuning cells, minimize the worst-case
condition-space distance from target-test cells to the fine-tuning set,
and avoid missing target-test operating-condition groups.
```

In other words, condition-space prior indices can help avoid obviously poor splits, but they cannot fully determine transfer quality. The stronger feature-space results suggest that the effect of fine-tuning selection is mediated by extracted degradation features, not by operating conditions alone.

### Updated Conclusion

The original condition-space TFCI is not sufficient for explaining seed-dependent transfer performance. It measures whether the selected fine-tuning cells cover the target-test cells in operating-condition space, but the model does not learn from operating conditions alone. It learns from the extracted early-life features `f1, ..., f10`.

The strongest diagnostic replacement is `F-TFCI_p90`, a feature-space tail-aware coverage index. Compared with the original TFCI, it has a stronger relationship with both MAE improvement and MAPE improvement. However, because feature extraction occurs after domain splitting and feature subsets may differ across experiments, this feature-space index should be interpreted as a diagnostic explanation rather than the main prior rule for selecting fine-tuning cells.

For practical target fine-tuning cell selection, the index must be based on split-stage operating-condition information. Among the prior condition-space metrics tested, `condition d_NN_max` is the most informative, but its correlations with MAE/MAPE improvement remain modest. Therefore, the prior condition index should be used as a split-quality constraint rather than as a standalone predictor: target fine-tuning cells should be selected to minimize worst-case condition-space distance and avoid missing target-test operating-condition groups.

## Interpretation Notes

1. A smaller centroid gap means the target fine-tuning cells are located closer to the overall center of the target-test cells in condition space.
2. A smaller mean nearest-neighbour gap means the target-test cells are better covered by nearby target fine-tuning cells.
3. A larger TFCI usually corresponds to better target fine-tuning coverage, although it does not guarantee the best MAE improvement by itself.
4. When `d_NN,mean = 0`, it means that, in the standardized condition space used here, the target fine-tuning cells include condition points that coincide with the target-test condition points.

## Suggested Thesis Wording

One possible formal sentence for the baseline condition-space index is:

> A target fine-tuning coverage index was defined as $\mathrm{TFCI}=1/(1+d_{\mathrm{centroid}}+d_{\mathrm{NN,mean}})$, where $d_{\mathrm{centroid}}$ measures the centroid gap between target fine-tuning and target-test cells, and $d_{\mathrm{NN,mean}}$ measures the average nearest-neighbour distance from each target-test cell to the target fine-tuning set in the standardized condition space.

For the revised feature-space index, the recommended wording is:

> Because the condition-space TFCI did not reliably distinguish positive and negative transfer seeds, a feature-space tail-aware coverage index, $\mathrm{FTFCI}_{p90}=1/(1+d^{(f)}_{\mathrm{centroid}}+d^{(f)}_{\mathrm{NN,mean}}+d^{(f)}_{\mathrm{NN,p90}})$, was evaluated using the standardized model-input features. This index better captured whether the target-test feature distribution was supported by nearby target fine-tuning cells, especially in the poorly covered tail of the target-test set.

A shorter interpretation sentence is:

> Larger $\mathrm{FTFCI}_{p90}$ values indicate better feature-space support of the target-test cells by the selected target fine-tuning cells, but the index remains an explanatory diagnostic rather than a deterministic performance predictor.
