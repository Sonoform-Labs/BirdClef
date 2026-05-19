import torch, numpy as np
from pathlib import Path
BASE = Path(r"D:\Python\BirdClef")/"outputs"
TAXA = ["Aves","Amphibia","Insecta","Mammalia","Reptilia"]

def taxon_auc(rel):
    c = torch.load(BASE/rel, map_location="cpu", weights_only=False)
    return c.get("taxon_auc", {}), c.get("macro_auc", float("nan"))

rows = {tx: {"d":[], "n":[]} for tx in TAXA}
macro = {"d":[], "n":[]}
print("Per-fold distill(d) vs nodistill(n) taxon_auc:\n")
hdr = f"{'fold':4s} {'cond':9s} {'macro':>7s} " + " ".join(f"{tx[:4]:>7s}" for tx in TAXA)
print(hdr); print("-"*len(hdr))
for k in range(5):
    td, md = taxon_auc(f"fold{k}/best.pt")
    tn, mn = taxon_auc(f"fold{k}_nodistill/best.pt")
    macro["d"].append(md); macro["n"].append(mn)
    print(f"{k:<4d} {'distill':9s} {md:7.4f} " + " ".join(f"{td.get(tx,float('nan')):7.4f}" for tx in TAXA))
    print(f"{k:<4d} {'nodistill':9s} {mn:7.4f} " + " ".join(f"{tn.get(tx,float('nan')):7.4f}" for tx in TAXA))
    for tx in TAXA:
        rows[tx]["d"].append(td.get(tx, np.nan)); rows[tx]["n"].append(tn.get(tx, np.nan))
    print()

print("="*72)
print("  FOLD-AVERAGED EFFECT  delta = nodistill - distill  (delta>0 => distill HURTS)")
print("  (positive macro-delta would mean distill hurts overall)")
print("="*72)
def summ(d, n):
    d=np.array(d,float); n=np.array(n,float)
    delta=n-d; m=np.nanmean(delta); sd=np.nanstd(delta,ddof=1); se=sd/np.sqrt(np.sum(~np.isnan(delta)))
    signs=np.sign(delta[~np.isnan(delta)])
    return m, se, delta, int((signs>0).sum()), int((signs<0).sum())
md_,mse_,_,_,_ = summ(macro["d"],macro["n"])
print(f"  MACRO     mean_d(nd-d)={md_:+.4f} +/- {mse_:.4f}  "
      f"(distill macro mean={np.mean(macro['d']):.4f}, nodistill={np.mean(macro['n']):.4f})")
print(f"\n  {'taxon':10s} {'mean d':>8s} {'SE':>7s}  {'folds(hurt/help)':>16s}  per-fold deltas")
for tx in TAXA:
    m, se, delta, hurt, help_ = summ(rows[tx]["d"], rows[tx]["n"])
    ci_lo, ci_hi = m-1.96*se, m+1.96*se
    spans0 = "  <-- CI spans 0" if ci_lo < 0 < ci_hi else ""
    pf = " ".join(f"{x:+.3f}" for x in delta)
    print(f"  {tx:10s} {m:+8.4f} {se:7.4f}  {hurt:>6d}/{help_:<9d}  [{pf}]{spans0}")
print("\nNOTE: delta>0 = distill HURTS that taxon. Insecta is the paper's lead claim.")
