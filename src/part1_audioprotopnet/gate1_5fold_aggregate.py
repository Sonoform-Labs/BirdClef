import sys
sys.path.insert(0, r"D:\Python\BirdClef")
import numpy as np
from pathlib import Path

OUT = Path(r"D:\Python\BirdClef\paper_analysis\joint")
LOG = Path(r"D:\Python\BirdClef\OVERNIGHT_LOG.md")
TAXA = ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]
QUAR = {"Reptilia", "Mammalia"}

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

def taxon_delta(pp, ps, tg, cls):
    ap = vec_auc(pp, tg); a_s = vec_auc(ps, tg)
    d = {"MACRO": float(np.nanmean(ap) - np.nanmean(a_s))}
    for tx in TAXA:
        msk = cls == tx
        if np.any(msk & ~np.isnan(ap) & ~np.isnan(a_s)):
            d[tx] = float(np.nanmean(ap[msk]) - np.nanmean(a_s[msk]))
    return d, ap, a_s

def main():
    from generate_oof import load_taxonomy
    l2i, i2l, tm = load_taxonomy()
    species = np.array([i2l[i] for i in range(len(i2l))])
    cls = np.array([tm.get(s, "?") for s in species])

    folds = [k for k in range(5) if (OUT/f"oof_joint_proto_fold{k}.npy").exists()
             and (OUT/f"oof_joint_sigmoid_fold{k}.npy").exists()]
    if not folds:
        print("no completed folds yet"); return
    per_fold = {}
    PP, PS, TG = [], [], []
    for k in folds:
        pp = np.load(OUT/f"oof_joint_proto_fold{k}.npy")
        ps = np.load(OUT/f"oof_joint_sigmoid_fold{k}.npy")
        tg = np.load(OUT/f"val_tgt_fold{k}.npy").astype(np.float32)
        per_fold[k] = taxon_delta(pp, ps, tg, cls)[0]
        PP.append(pp); PS.append(ps); TG.append(tg)
    PP = np.concatenate(PP); PS = np.concatenate(PS); TG = np.concatenate(TG)


    rng = np.random.default_rng(0); B = 2000; N = len(TG)
    keys = ["MACRO"] + [t for t in TAXA]
    boot = {k: [] for k in keys}
    for _ in range(B):
        idx = rng.integers(0, N, N)
        ap = vec_auc(PP[idx], TG[idx]); a_s = vec_auc(PS[idx], TG[idx])
        boot["MACRO"].append(np.nanmean(ap) - np.nanmean(a_s))
        for tx in TAXA:
            msk = cls == tx
            boot[tx].append(np.nanmean(ap[msk]) - np.nanmean(a_s[msk]))
    pooled, _, _ = taxon_delta(PP, PS, TG, cls)

    def ci(v):
        v = np.array(v); v = v[~np.isnan(v)]
        return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

    lines = []
    lines.append("=" * 78)
    lines.append(f"  GATE 1 - 5-FOLD JOINT (ConvNeXt-Base fine-tune)  folds present: {folds}")
    lines.append("  delta = proto - sigmoid (pooled OOF). 95% CI = paired instance bootstrap.")
    lines.append("=" * 78)
    hdr = f"  {'scope':10s} {'pooled d':>9s} {'95% CI':>20s}  {'per-fold deltas':>32s}  flags"
    lines.append(hdr); lines.append("  " + "-" * (len(hdr)-2))
    for key in keys:
        if key not in pooled:
            continue
        lo, hi = ci(boot[key])
        pfd = [per_fold[k].get(key, np.nan) for k in folds]
        signs = [np.sign(x) for x in pfd if not np.isnan(x)]
        flip = len(set(signs)) > 1
        spans0 = lo < 0 < hi
        flags = []
        if key in QUAR: flags.append("QUARANTINE(small-n; no-sig)")
        if flip: flags.append("SIGN-FLIPS-ACROSS-FOLDS")
        if spans0: flags.append("CI-SPANS-0")
        pfd_s = " ".join(f"{x:+.3f}" if not np.isnan(x) else " nan " for x in pfd)
        lines.append(f"  {key:10s} {pooled[key]:+9.4f} [{lo:+.4f},{hi:+.4f}]  {pfd_s:>32s}  {';'.join(flags)}")
    lines.append("")
    lines.append("  READING: a per-taxon delta is only a candidate finding if it does NOT sign-flip")
    lines.append("  across folds AND its CI excludes 0. Reptilia/Mammalia = labeled obs only (quarantined).")
    out = "\n".join(lines)
    print(out)

    import csv
    with open(OUT/"gate1_5fold_results.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["scope", "pooled_delta", "ci_lo", "ci_hi"] + [f"fold{k}" for k in folds] + ["sign_flip", "ci_spans0", "quarantine"])
        for key in keys:
            if key not in pooled: continue
            lo, hi = ci(boot[key]); pfd = [per_fold[k].get(key, "") for k in folds]
            signs = [np.sign(x) for x in pfd if x != "" and not np.isnan(x)]
            w.writerow([key, pooled[key], lo, hi] + pfd + [len(set(signs)) > 1, lo < 0 < hi, key in QUAR])
    try:
        prev = LOG.read_text(encoding="utf-8") if LOG.exists() else ""
        LOG.write_text(prev + "\n\n## GATE-1 5-FOLD AGGREGATE\n```\n" + out + "\n```\n", encoding="utf-8")
    except Exception as e:
        print("log append failed:", e)
    print("\naggregate done. csv: joint/gate1_5fold_results.csv")

if __name__ == "__main__":
    main()
