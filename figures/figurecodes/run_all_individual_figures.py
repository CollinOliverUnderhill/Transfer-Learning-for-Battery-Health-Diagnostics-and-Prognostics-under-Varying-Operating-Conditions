#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run all individual figure scripts."""

import importlib

MODULES = [
    "figure01_transfer_learning_strategies",
    "figure03_eol70_rul_label",
    "figure04_dataset_sample_pipeline",
    "figure05_week_based_sample_construction",
    "figure06_soh_feature_correlation_ranking",
    "figure07_selected_soh_feature_combination",
    "figure08_rul_input_feature_correlation_matrix",
    "figure09_main_rerun400_input_features",
    "figure10_eol70_domain_split",
    "figure11_lifetime_distribution_partitions",
    "figure12_baseline_protocols",
    "figure13_fine_tuning_pipeline",
    "figure14_staged_hpo_workflow",
    "figure15_within_cell_soh_summary",
    "figure16_within_cell_soh_pred_vs_true",
    "figure17_soh_generalization_summary",
    "figure18_single_cell_to_multi_cell_soh",
    "figure19_subset_to_subset_soh",
    "figure20_main_rerun400_target_test_metrics",
    "figure21_main_rerun400_pred_vs_true",
    "figure22_main_rerun400_group_error",
    "figure23_main_rerun400_cell_error",
    "figure24_valid_sample_availability_weeks",
    "figure27_target_finetune_coverage_index",
    "figure28_lowcapacity_domain_split",
    "figure29_lowcapacity_target_test_metrics",
    "figure30_lowcapacity_pred_vs_true",
    "figure31_lowcapacity_group_error",
]


def main() -> None:
    for mod_name in MODULES:
        mod = importlib.import_module(mod_name)
        mod.main()


if __name__ == "__main__":
    main()
