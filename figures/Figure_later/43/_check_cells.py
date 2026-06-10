import csv, json
from pathlib import Path

IVAS = Path(r"E:\Datasets\IVAS")
EXP  = IVAS / "week_based/Final/EOL70/3step/outputs_400/BasicModel/stage3_final_rerun_400"
WEEK = IVAS / "week_based/Final/EOL70/3step/outputs_400/protocol_w6_10_from_stage3_final_rerun_400_legacy400"

def rc(p):
    with open(p, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

# W5 ref
bm_tr = rc(EXP/"benchmark"/"source_all_train_overall_metrics.csv")[0]
bm_te = rc(EXP/"benchmark"/"test_overall_metrics.csv")[0]
tm_tr = rc(EXP/"transfer_model"/"source_all_train_overall_metrics.csv")[0]
tm_ft = rc(EXP/"transfer_model"/"target_finetune_overall_metrics.csv")[0]
tm_te = rc(EXP/"transfer_model"/"test_overall_metrics.csv")[0]

print(f"{'Week':<6} {'BM src_train':>13} {'BM test':>8} {'TM src_train':>13} {'TM FT':>6} {'TM test':>8}")
print("-"*60)
print(f"{'5':<6} {bm_tr['n_cells']:>13} {bm_te['n_cells']:>8} {tm_tr['n_cells']:>13} {tm_ft['n_cells']:>6} {tm_te['n_cells']:>8}")

rows = rc(WEEK/"protocol_w6_10_from_stage3_final_rerun_400_legacy400_benchmark_transfer_summary_all_splits.csv")
for r in rows:
    w = r["stage3_dir"].split("/")[0].replace("week","")
    print(f"{w:<6} {r['bench_source_inner_train_n_cells']:>13} {r['bench_test_n_cells']:>8} {r['transfer_source_inner_train_n_cells']:>13} {r['transfer_target_finetune_n_cells']:>6} {r['transfer_test_n_cells']:>8}")

# Also show total unique cells per week
print()
print("Total unique cells per week (BM: inner_train + val + test  |  all source groups):")
for r in rows:
    w = r["stage3_dir"].split("/")[0].replace("week","")
    src = int(r["bench_source_inner_train_n_cells"])
    val = int(r["bench_source_val_n_cells"]) if "bench_source_val_n_cells" in r else 0
    te  = int(r["bench_test_n_cells"])
    n_grp = int(r["bench_source_inner_train_n_groups"])
    print(f"  W{w}: src_inner_train={src}, src_val={val}, test={te}, total~{src+val+te}, src_groups={n_grp}")

# W5
src5 = int(bm_tr["n_cells"])
val5_csv = rc(EXP/"benchmark"/"source_val_overall_metrics.csv")[0]
val5 = int(val5_csv["n_cells"])
te5 = int(bm_te["n_cells"])
print(f"  W5:  src_inner_train={src5-val5}(inner)+{val5}(val)={src5}(all), test={te5}, total~{src5+te5}")
