import sys
sys.path.insert(0, r"D:\Python\BirdClef")
import numpy as np
from pathlib import Path
OUT = Path(r"D:\Python\BirdClef\paper_analysis\joint_faithful")
TAXA = ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]

def vec_auc(p, y):
    N, C = p.shape
    order = np.argsort(p, axis=0, kind="mergesort")
    ranks = np.empty((N, C)); rk = np.arange(1, N+1, dtype=float)[:, None]
    np.put_along_axis(ranks, order, np.broadcast_to(rk, (N, C)).copy(), axis=0)
    npos = y.sum(0); nneg = N - npos
    with np.errstate(invalid="ignore", divide="ignore"):
        auc = ((ranks*y).sum(0) - npos*(npos+1)/2) / (npos*nneg)
    auc[(npos == 0) | (nneg == 0)] = np.nan
    return auc

def rank_norm_cols(S):
    N = S.shape[0]
    order = np.argsort(S, axis=0, kind="mergesort")
    ranks = np.empty_like(S, dtype=np.float64)
    ar = (np.arange(1, N+1, dtype=np.float64)/N)[:, None]
    np.put_along_axis(ranks, order, np.broadcast_to(ar, S.shape).copy(), axis=0)
    return ranks

def taxon_means(a, cls):
    d = {"MACRO": float(np.nanmean(a))}
    for tx in TAXA:
        m = cls == tx
        if np.any(m & ~np.isnan(a)): d[tx] = float(np.nanmean(a[m]))
    return d

from generate_oof import load_taxonomy
l2i, i2l, tm = load_taxonomy()
species = np.array([i2l[i] for i in range(len(i2l))])
cls = np.array([tm.get(s, "?") for s in species])

folds = [k for k in range(5) if (OUT/f"oof_faithful_proto_fold{k}.npy").exists()]
PP, PS, TG = [], [], []
perfold = {tx: [] for tx in ["MACRO"]+TAXA}
print("== (1) per-fold AUC invariance check (raw vs rank-normed must match) + (2) proto score scales ==")
print(f"  {'fold':4s} {'macroΔ_raw':>10s} {'macroΔ_rankN':>12s} | {'proto p50/p95':>16s} {'sig p50/p95':>14s}")
for k in folds:
    pp = np.load(OUT/f"oof_faithful_proto_fold{k}.npy"); ps = np.load(OUT/f"oof_faithful_sigmoid_fold{k}.npy")
    tg = np.load(OUT/f"val_tgt_fold{k}.npy").astype(np.float32)
    PP.append(pp); PS.append(ps); TG.append(tg)
    d_raw = taxon_means(vec_auc(pp, tg), cls)["MACRO"] - taxon_means(vec_auc(ps, tg), cls)["MACRO"]
    d_rn = taxon_means(vec_auc(rank_norm_cols(pp), tg), cls)["MACRO"] - taxon_means(vec_auc(rank_norm_cols(ps), tg), cls)["MACRO"]
    dd = taxon_means(vec_auc(pp, tg), cls); ds = taxon_means(vec_auc(ps, tg), cls)
    for tx in ["MACRO"]+TAXA:
        if tx in dd and tx in ds: perfold[tx].append(dd[tx]-ds[tx])
    print(f"  {k:<4d} {d_raw:>10.4f} {d_rn:>12.4f} | proto {np.percentile(pp,50):.3f}/{np.percentile(pp,95):.3f}"
          f"   sig {np.percentile(ps,50):.3f}/{np.percentile(ps,95):.3f}")

PPc = np.concatenate(PP); PSc = np.concatenate(PS); TGc = np.concatenate(TG)

praw = taxon_means(vec_auc(PPc, TGc), cls); sraw = taxon_means(vec_auc(PSc, TGc), cls)

PPrn = np.concatenate([rank_norm_cols(p) for p in PP]); PSrn = np.concatenate([rank_norm_cols(s) for s in PS])
prn = taxon_means(vec_auc(PPrn, TGc), cls); srn = taxon_means(vec_auc(PSrn, TGc), cls)

print("\n== (3) DECISIVE: delta (proto - sigmoid) by aggregation method ==")
print(f"  {'scope':10s} {'per-fold mean±SE':>20s} {'signs':>7s} {'raw-pooled':>11s} {'rankN-pooled':>13s}")
for tx in ["MACRO"]+TAXA:
    pf = np.array(perfold.get(tx, []))
    if len(pf) == 0: continue
    mean = pf.mean(); se = pf.std(ddof=1)/np.sqrt(len(pf)) if len(pf) > 1 else float("nan")
    signs = f"{int((pf>0).sum())}+/{int((pf<0).sum())}-"
    rp = (praw.get(tx, np.nan) - sraw.get(tx, np.nan))
    rn = (prn.get(tx, np.nan) - srn.get(tx, np.nan))
    print(f"  {tx:10s} {mean:+.4f}±{se:.4f}      {signs:>7s} {rp:>+11.4f} {rn:>+13.4f}")
print("\nREAD: if per-fold AUC_raw==AUC_rankN (col 2==3 above) -> per-fold is scale-invariant (trustworthy).")
print("      if rankN-pooled ~ per-fold-mean while raw-pooled is the outlier -> raw-pooled = cross-fold-scale ARTIFACT.")


import csv
with open(OUT/"gate1_faithful_FINAL.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["scope", "perfold_mean", "perfold_se", "n_pos_folds", "n_neg_folds", "raw_pooled_SCALE_SENSITIVE", "rankN_pooled"])
    for tx in ["MACRO"] + TAXA:
        pf = np.array(perfold.get(tx, []))
        if len(pf) == 0: continue
        mean = pf.mean(); se = pf.std(ddof=1)/np.sqrt(len(pf)) if len(pf) > 1 else float("nan")
        rp = praw.get(tx, float("nan")) - sraw.get(tx, float("nan"))
        rn = prn.get(tx, float("nan")) - srn.get(tx, float("nan"))
        w.writerow([tx, round(mean, 4), round(se, 4), int((pf > 0).sum()), int((pf < 0).sum()), round(rp, 4), round(rn, 4)])
print(f"wrote {OUT/'gate1_faithful_FINAL.csv'} (per-fold = primary metric)")
