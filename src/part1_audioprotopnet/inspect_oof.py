import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from pathlib import Path

BASE = Path(r"D:\Python\BirdClef")
OOF = BASE / "outputs" / "oof"

print("="*70)
print("  ARRAY SHAPES / DTYPES / RANGES")
print("="*70)
arrs = {}
for name in ["preds_stage1","targets_stage1","logits_stage1",
             "preds_stage2","targets_stage2","logits_stage2"]:
    a = np.load(OOF/f"{name}.npy")
    arrs[name] = a
    print(f"  {name:18s} {str(a.shape):14s} {a.dtype}  "
          f"min={float(np.nanmin(a)):+.4f} max={float(np.nanmax(a)):+.4f} "
          f"mean={float(np.nanmean(a)):+.4f}")

p1, t1 = arrs["preds_stage1"], arrs["targets_stage1"]
p2, t2 = arrs["preds_stage2"], arrs["targets_stage2"]

print("\n" + "="*70)
print("  ROW ALIGNMENT / DISTINCTNESS CHECKS")
print("="*70)
print(f"  targets_stage1 == targets_stage2 (row-aligned?) : {np.array_equal(t1,t2)}")
print(f"  preds_stage1 == preds_stage2 (would be phantom!) : {np.array_equal(p1,p2)}")
print(f"  mean |preds1 - preds2|                           : {float(np.abs(p1-p2).mean()):.5f}")
print(f"  total positive labels stage1 / stage2           : {int(t1.sum())} / {int(t2.sum())}")


for s in ["1","2"]:
    mp = OOF/f"meta_stage{s}.csv"
    if mp.exists():
        m = pd.read_csv(mp)
        print(f"  meta_stage{s}: {len(m)} rows, cols={list(m.columns)}")
m1 = pd.read_csv(OOF/"meta_stage1.csv"); m2 = pd.read_csv(OOF/"meta_stage2.csv")
print(f"  meta_stage1.equals(meta_stage2)                 : {m1.equals(m2)}")
print(f"  meta_stage1 head:\n{m1.head(3).to_string()}")


species_order = pd.read_csv(OOF/"species_order.csv", header=None)[0].astype(str).tolist()
print(f"\n  species_order: n={len(species_order)} first3={species_order[:3]}")
tmp = OOF/"taxon_map.csv"
if tmp.exists():
    tm = pd.read_csv(tmp)
    print(f"  taxon_map.csv cols={list(tm.columns)} head:\n{tm.head(3).to_string()}")

tax = pd.read_csv(BASE/"data"/"taxonomy.csv")
lab2class = dict(zip(tax.primary_label.astype(str), tax.class_name))
classes = [lab2class.get(sp,"?") for sp in species_order]
from collections import Counter
print(f"  taxon counts in species_order: {dict(Counter(classes))}")

def per_species_auc(preds, targets):
    n = preds.shape[1]
    aucs = np.full(n, np.nan)
    for i in range(n):
        yt = targets[:,i]
        s = yt.sum()
        if 0 < s < len(yt):
            aucs[i] = roc_auc_score(yt, preds[:,i])
    return aucs

a_d  = per_species_auc(p1, t1)
a_nd = per_species_auc(p2, t2)

print("\n" + "="*70)
print("  PER-TAXON AUC  (stage1=distill, stage2=nodistill)  [VERIFY vs S10]")
print("  S10 anchors (fold0): Insecta distill .904 < nodistill .929 ;")
print("                       Aves    distill .972 > nodistill .967")
print("="*70)
print(f"  {'taxon':10s} {'n_sc':>5s}  {'distill':>8s} {'nodist':>8s} {'d(nd-d)':>8s}")
for tx in ["Aves","Amphibia","Insecta","Mammalia","Reptilia"]:
    idx = [i for i,c in enumerate(classes)
           if c==tx and not np.isnan(a_d[i]) and not np.isnan(a_nd[i])]
    if idx:
        d  = float(np.mean([a_d[i]  for i in idx]))
        nd = float(np.mean([a_nd[i] for i in idx]))
        print(f"  {tx:10s} {len(idx):5d}  {d:8.4f} {nd:8.4f} {nd-d:+8.4f}")
print(f"  {'MACRO':10s} {int((~np.isnan(a_d)).sum()):5d}  "
      f"{np.nanmean(a_d):8.4f} {np.nanmean(a_nd):8.4f} {np.nanmean(a_nd)-np.nanmean(a_d):+8.4f}")
print("\nDone (read-only).")
