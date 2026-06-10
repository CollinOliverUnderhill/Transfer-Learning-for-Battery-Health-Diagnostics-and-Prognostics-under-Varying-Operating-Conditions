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
