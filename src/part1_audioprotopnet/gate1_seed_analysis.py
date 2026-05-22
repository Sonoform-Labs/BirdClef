import sys
sys.path.insert(0, r"D:\Python\BirdClef")
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
OUT = Path(r"D:\Python\BirdClef\paper_analysis\joint")
SEEDS = [101, 202, 303]

def macro(probs, tgt):
    a = []
    for i in range(probs.shape[1]):
        y = tgt[:, i]; p = probs[:, i]; m = np.isfinite(p); yy, pp = y[m], p[m]
        if 0 < yy.sum() < len(yy):
            try: a.append(roc_auc_score(yy, pp))
            except ValueError: pass
    return float(np.mean(a)) if a else float("nan")

def delta(suffix):
    ps = np.load(OUT/f"oof_joint_sigmoid_fold0{suffix}.npy")
    pp = np.load(OUT/f"oof_joint_proto_fold0{suffix}.npy")
    tg = np.load(OUT/f"val_tgt_fold0{suffix}.npy").astype(np.float32)
    return macro(pp, tg) - macro(ps, tg)

rows = []
for tag, sfx in [("orig(5fold)", "")] + [(f"seed{s}", f"_s{s}") for s in SEEDS]:
    try: rows.append((tag, delta(sfx)))
    except Exception as e: print(f"{tag} missing: {e}")
ds = [d for _, d in rows]
print("fold-0 macro delta (proto - sigmoid) by seed:")
for t, d in rows: print(f"  {t:12s} {d:+.4f}")
if len(ds) > 1:
    print(f"  mean={np.mean(ds):+.4f}  seed-std={np.std(ds, ddof=1):.4f}  n={len(ds)}")
log = Path(r"D:\Python\BirdClef\OVERNIGHT_LOG.md")
try:
    txt = "\n## GATE-1 SEED VARIANCE (fold 0)\n" + "\n".join(f"  {t}: {d:+.4f}" for t, d in rows)
    if len(ds) > 1: txt += f"\n  mean={np.mean(ds):+.4f}  seed-std={np.std(ds, ddof=1):.4f}\n"
    log.write_text(log.read_text(encoding="utf-8") + txt + "\n", encoding="utf-8")
except Exception as e:
    print("log append failed:", e)
print("seed analysis done.")
