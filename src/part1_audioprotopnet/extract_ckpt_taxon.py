import torch, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from pathlib import Path

BASE = Path(r"D:\Python\BirdClef")
OOF = BASE/"outputs"/"oof"

tax = pd.read_csv(BASE/"data"/"taxonomy.csv")
lab2class = dict(zip(tax.primary_label.astype(str), tax.class_name))
species_order = pd.read_csv(OOF/"species_order.csv", header=None)[0].astype(str).tolist()
classes = np.array([lab2class.get(sp,"?") for sp in species_order])

def load_ckpt(rel):
    c = torch.load(BASE/rel, map_location="cpu", weights_only=False)
    return c

dk = load_ckpt("outputs/fold0/best.pt")
nk = load_ckpt("outputs/fold0_nodistill/best.pt")

print("="*70); print("  CKPT-STORED taxon_auc (the SOURCE of S10)"); print("="*70)
print("  taxon          DISTILL   NODISTILL   d(nd-d)")
for tx in ["Aves","Amphibia","Insecta","Mammalia","Reptilia"]:
    d = dk["taxon_auc"].get(tx, float("nan")); n = nk["taxon_auc"].get(tx, float("nan"))
    print(f"  {tx:12s}  {d:.4f}    {n:.4f}    {n-d:+.4f}")
print(f"  macro_auc      {dk['macro_auc']:.4f}    {nk['macro_auc']:.4f}")


pc_d = dk["per_class_auc"]; pc_n = nk["per_class_auc"]
print(f"\n  per_class_auc type: {type(pc_d).__name__}", end="")
if isinstance(pc_d, dict):
    print(f"  len={len(pc_d)}  sample_keys={list(pc_d.keys())[:3]}")
else:
    print(f"  shape={np.asarray(pc_d).shape}")

def pc_to_arr(pc):
    arr = np.full(len(species_order), np.nan)
    if isinstance(pc, dict):

        keys = list(pc.keys())
        if all(isinstance(k,(int,np.integer)) for k in keys):
            for k,v in pc.items():
                if 0 <= int(k) < len(arr): arr[int(k)] = v
        else:
            idx = {sp:i for i,sp in enumerate(species_order)}
            for k,v in pc.items():
                if str(k) in idx: arr[idx[str(k)]] = v
    else:
        a = np.asarray(pc, dtype=float)
        arr[:len(a)] = a[:len(arr)]
    return arr

ck_d = pc_to_arr(pc_d); ck_n = pc_to_arr(pc_n)


p1 = np.load(OOF/"preds_stage1.npy"); t1 = np.load(OOF/"targets_stage1.npy")
p2 = np.load(OOF/"preds_stage2.npy"); t2 = np.load(OOF/"targets_stage2.npy")
meta = pd.read_csv(OOF/"meta_stage1.csv")
print(f"\n  meta source counts: {meta['source'].value_counts().to_dict()}")

def oof_auc(preds, targets):
    arr = np.full(preds.shape[1], np.nan); npos = np.zeros(preds.shape[1], int)
    for i in range(preds.shape[1]):
        yt = targets[:,i]; s = int(yt.sum()); npos[i]=s
        if 0 < s < len(yt): arr[i] = roc_auc_score(yt, preds[:,i])
    return arr, npos
oo_d, npos_d = oof_auc(p1,t1); oo_n, npos_n = oof_auc(p2,t2)

print("\n" + "="*70)
print("  INSECTA species: CKPT per_class vs OOF-recompute (distill / nodistill)")
print("="*70)
print(f"  {'species':10s} {'npos':>5s} | {'ck_dis':>7s} {'oof_dis':>7s} | {'ck_nod':>7s} {'oof_nod':>7s}")
ins_idx = [i for i,c in enumerate(classes) if c=="Insecta"]
for i in ins_idx:
    sp = species_order[i]
    f=lambda x: f"{x:.4f}" if not np.isnan(x) else "  nan "
    print(f"  {sp:10s} {npos_n[i]:5d} | {f(ck_d[i])} {f(oo_d[i])} | {f(ck_n[i])} {f(oo_n[i])}")

def taxmean(arr, tx):
    idx=[i for i,c in enumerate(classes) if c==tx and not np.isnan(arr[i])]
    return (np.mean([arr[i] for i in idx]), len(idx)) if idx else (float("nan"),0)

print("\n  Insecta taxon-mean reconciliation:")
for tag,cd,cn in [("CKPT per_class", ck_d, ck_n), ("OOF recompute", oo_d, oo_n)]:
    md,nd = taxmean(cd,"Insecta"); mn,nn = taxmean(cn,"Insecta")
    print(f"    {tag:16s} distill={md:.4f}(n={nd})  nodistill={mn:.4f}(n={nn})  d(nd-d)={mn-md:+.4f}")
