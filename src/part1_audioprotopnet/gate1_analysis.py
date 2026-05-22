import sys
sys.path.insert(0, r"D:\Python\BirdClef")
import numpy as np
from pathlib import Path

FE = Path(r"D:\Python\BirdClef\paper_analysis\feats")
SEEDS = [0, 1, 2]
TAXA = ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]

def vec_auc(p, y):
    N, C = p.shape
    order = np.argsort(p, axis=0, kind="mergesort")
    ranks = np.empty((N, C), dtype=np.float64)
    rk = np.arange(1, N + 1, dtype=np.float64)[:, None]
    np.put_along_axis(ranks, order, np.broadcast_to(rk, (N, C)).copy(), axis=0)
    npos = y.sum(0); nneg = N - npos
    srp = (ranks * y).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        auc = (srp - npos * (npos + 1) / 2) / (npos * nneg)
    auc[(npos == 0) | (nneg == 0)] = np.nan
    return auc

def load_taxon():
    from generate_oof import load_taxonomy
    l2i, i2l, tm = load_taxonomy()
    species = [i2l[i] for i in range(len(i2l))]
    return np.array(species), tm

def main():
    tgt = np.load(FE/"val_tgt.npy").astype(np.float32)
    species, tm = load_taxon()
    cls = np.array([tm.get(s, "?") for s in species])
    P = {}
    for head in ["proto", "sigmoid"]:
        arrs = [np.load(FE/f"oof_{head}_s{s}.npy") for s in SEEDS]
        P[head] = np.mean(arrs, axis=0); P[head + "_seeds"] = arrs

    def taxon_means(a):
        return np.nanmean(a), {tx: float(np.nanmean(a[cls == tx])) for tx in TAXA}
    ap = vec_auc(P["proto"], tgt); as_ = vec_auc(P["sigmoid"], tgt)
    mp, txp = taxon_means(ap); ms, txs = taxon_means(as_)


    rng = np.random.default_rng(0); B = 2000; N = len(tgt)
    dm = []; dt = {tx: [] for tx in TAXA}
    for _ in range(B):
        idx = rng.integers(0, N, N)
        a1 = vec_auc(P["proto"][idx], tgt[idx]); a0 = vec_auc(P["sigmoid"][idx], tgt[idx])
        dm.append(np.nanmean(a1) - np.nanmean(a0))
        for tx in TAXA:
            dt[tx].append(np.nanmean(a1[cls == tx]) - np.nanmean(a0[cls == tx]))
    def ci(v):
        v = np.array(v); return float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


    seed_dm = []
    for k in range(len(SEEDS)):
        seed_dm.append(float(np.nanmean(vec_auc(P["proto_seeds"][k], tgt)) -
                             np.nanmean(vec_auc(P["sigmoid_seeds"][k], tgt))))

    n_per_tax = {tx: int(np.sum((cls == tx) & ~np.isnan(ap) & ~np.isnan(as_))) for tx in TAXA}

    lines = []
    lines.append("="*72)
    lines.append("  GATE 1 (PRELIMINARY, frozen BirdSet-Base features, fold 0) - proto vs sigmoid")
    lines.append("="*72)
    lines.append(f"  seed-avg MACRO AUC: proto={mp:.4f}  sigmoid={ms:.4f}")
    md, lo, hi = ci(dm)
    sig = "" if (lo < 0 < hi) else "  *"
    lines.append(f"  MACRO delta(proto-sigmoid) = {md:+.4f}  95%CI[{lo:+.4f},{hi:+.4f}]{sig}")
    lines.append(f"  per-seed macro deltas = {['%+.4f'%x for x in seed_dm]}  (seed std {np.std(seed_dm):.4f})")
    lines.append("")
    lines.append(f"  {'taxon':10s} {'n_sp':>4s} {'proto':>7s} {'sigmoid':>7s} {'delta':>8s}  {'95% CI':>20s}")
    for tx in TAXA:
        m_, lo_, hi_ = ci(dt[tx])
        flag = "" if (lo_ < 0 < hi_) else " *"
        lines.append(f"  {tx:10s} {n_per_tax[tx]:>4d} {txp[tx]:7.4f} {txs[tx]:7.4f} {m_:+8.4f}  "
                     f"[{lo_:+.4f},{hi_:+.4f}]{flag}")
    lines.append("")
    lines.append("  delta>0 => prototype head HELPS. '*' = 95% CI excludes 0.")
    lines.append("  CAVEAT: frozen backbone, no prototype projection/joint training -> PRELIMINARY")
    lines.append("  (closer to a 'ProtoSSM' scoring head than full AudioProtoPNet). Joint-trained = next.")
    out = "\n".join(lines)
    print(out)


    import csv
    with open(FE/"gate1_results.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["scope", "proto", "sigmoid", "delta", "ci_lo", "ci_hi", "n_sp"])
        w.writerow(["macro", mp, ms, md, lo, hi, int(np.sum(~np.isnan(ap)))])
        for tx in TAXA:
            m_, lo_, hi_ = ci(dt[tx]); w.writerow([tx, txp[tx], txs[tx], m_, lo_, hi_, n_per_tax[tx]])
    log = Path(r"D:\Python\BirdClef\OVERNIGHT_LOG.md")
    if log.exists():
        try:
            log.write_text(log.read_text(encoding="utf-8") + "\n\n## P5 RESULTS\n```\n" + out + "\n```\n",
                           encoding="utf-8")
        except Exception as e:
            print("log append skipped:", e)
    print("\nP5 done. (csv: feats/gate1_results.csv)")

if __name__ == "__main__":
    main()
