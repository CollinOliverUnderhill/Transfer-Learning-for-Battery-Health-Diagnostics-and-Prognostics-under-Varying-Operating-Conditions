import csv
from pathlib import Path
from collections import defaultdict

EXP = Path(r"E:\Datasets\IVAS\week_based\Final\EOL70\3step\outputs_400\BasicModel\stage3_final_rerun_400")

def load_cells(p):
    with open(p, encoding="utf-8-sig") as f:
        return [(r["cell"], r["group_num"]) for r in csv.DictReader(f)]

# BM
bm_train = load_cells(EXP/"benchmark/train_cells.csv")
bm_test  = load_cells(EXP/"benchmark/test_cells.csv" if (EXP/"benchmark/test_cells.csv").exists() else EXP/"transfer_model/test_cells.csv")

# TM
tm_src_train = load_cells(EXP/"transfer_model/source_train_inner_cells.csv")
tm_src_val   = load_cells(EXP/"transfer_model/source_val_cells.csv")
tm_ft_inner  = load_cells(EXP/"transfer_model/target_finetune_inner_train_cells.csv")
tm_ft_val    = load_cells(EXP/"transfer_model/target_finetune_val_cells.csv")
tm_ft_all    = load_cells(EXP/"transfer_model/target_finetune_cells.csv")
tm_test      = load_cells(EXP/"transfer_model/test_cells.csv")

# Get all groups
all_groups = sorted(set(g for _, g in bm_train + bm_test + tm_ft_all + tm_test), key=lambda x: int(x))

# Map: group -> which split
def cell_set(lst):
    return set(c for c, g in lst)

def group_set(lst):
    return set(g for c, g in lst)

print("="*80)
print("CELL SPLITTING ANALYSIS (W5, stage3_final_rerun_400)")
print("="*80)

# Key question: which groups are in target (FT+test) vs source
target_groups = group_set(tm_ft_all + tm_test)
source_groups = group_set(tm_src_train + tm_src_val)
both = target_groups & source_groups

print(f"\nTarget groups (FT + Test): {sorted(target_groups, key=int)}")
print(f"  Count: {len(target_groups)}")
print(f"\nSource groups (Train): {sorted(source_groups, key=int)}")
print(f"  Count: {len(source_groups)}")
print(f"\nOverlap: {sorted(both, key=int)}")
print(f"  Count: {len(both)}")

# Check: within target groups, how are cells split?
print(f"\n{'='*80}")
print("TARGET GROUP DETAIL: cells in FT (inner+val) vs Test")
print(f"{'Group':<8} {'FT inner':>10} {'FT val':>10} {'Test':>10} {'Total':>8}")
print("-"*50)
for g in sorted(target_groups, key=int):
    ft_i = [c for c, gg in tm_ft_inner if gg == g]
    ft_v = [c for c, gg in tm_ft_val if gg == g]
    te   = [c for c, gg in tm_test if gg == g]
    print(f"G{g:<7} {','.join(ft_i):>10} {','.join(ft_v):>10} {','.join(te):>10} {len(ft_i)+len(ft_v)+len(te):>8}")

# BM train vs TM source
bm_cells = cell_set(bm_train)
tm_src_cells = cell_set(tm_src_train + tm_src_val)
tm_ft_cells = cell_set(tm_ft_all)
tm_test_cells = cell_set(tm_test)

print(f"\n{'='*80}")
print("CELL COUNT SUMMARY")
print(f"  BM train (all):         {len(bm_cells)}")
print(f"  TM source (inner+val):  {len(tm_src_cells)}")
print(f"  TM target FT (all):     {len(tm_ft_cells)}")
print(f"  TM test:                {len(tm_test_cells)}")
print(f"  BM train = TM source + TM FT?  {bm_cells == tm_src_cells | tm_ft_cells}")
print(f"  BM train extra vs TM:   {bm_cells - tm_src_cells - tm_ft_cells}")
print(f"  TM FT ∩ TM src:         {tm_ft_cells & tm_src_cells}")

# The key: support_cells from report
import json
with open(EXP/"stage3_final_report.json") as f:
    rpt = json.load(f)
support = set(rpt["selected_stage1_summary"]["support_cells"])
print(f"\n  Report support_cells:    {len(support)} -> {sorted(support)}")
print(f"  FT cells:               {sorted(tm_ft_cells)}")
print(f"  Support == FT?           {support == tm_ft_cells}")
