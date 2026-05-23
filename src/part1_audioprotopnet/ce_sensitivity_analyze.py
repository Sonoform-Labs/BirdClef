import sys
sys.path.insert(0, r"D:\Python\BirdClef")
import numpy as np
from pathlib import Path
OUT = Path(r"D:\Python\BirdClef\paper_analysis\joint_faithful")
RES = Path(r"D:\Python\BirdClef\GATE1_RESULTS.md")
SCALES = [(0.5, "_ce05"), (1.0, "_ce10"), (2.0, "_ce20")]

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

from generate_oof import load_taxonomy
l2i, i2l, tm = load_taxonomy()
species = np.array([i2l[i] for i in range(len(i2l))])
cls = np.array([tm.get(s, "?") for s in species])

rows = []
for scale, sfx in SCALES:
    try:
        pp = np.load(OUT/f"oof_faithful_proto_fold0{sfx}.npy")
        ps = np.load(OUT/f"oof_faithful_sigmoid_fold0{sfx}.npy")
        tg = np.load(OUT/f"val_tgt_fold0{sfx}.npy").astype(np.float32)
    except Exception as e:
        rows.append([scale, None, None, None, f"MISSING ({type(e).__name__})"]); continue
    ap = vec_auc(pp, tg); a_s = vec_auc(ps, tg)
    macro_d = float(np.nanmean(ap) - np.nanmean(a_s))
    av = cls == "Aves"
    aves_d = float(np.nanmean(ap[av]) - np.nanmean(a_s[av]))
    proto_macro = float(np.nanmean(ap))
    note = "proto~chance(aborted?)" if proto_macro < 0.7 else ""
    rows.append([scale, macro_d, aves_d, proto_macro, note])

present = [r for r in rows if r[1] is not None]
allpresent = len(present) == 3
macro_in = all(-0.02 <= r[1] <= 0.01 for r in present) if present else False
aves_neg = all(r[2] < 0 for r in present) if present else False
if not allpresent:
    verdict = "INCOMPLETE - a scale is missing/failed; STOP and inspect"
elif macro_in and aves_neg:
    verdict = "HOLDS - headline stable across CE scale (macro in [-0.02,+0.01], Aves negative all 3)"
else:
    why = []
    if not macro_in: why.append("macro out of [-0.02,+0.01] for >=1 scale")
    if not aves_neg: why.append("Aves not negative for >=1 scale")
    verdict = "CE-DEPENDENT - " + "; ".join(why) + " -> headline needs reframing; do NOT build Gate 2"

L = ["", "## CE-RESCALE SENSITIVITY (fold 0; 0.5x / 1x / 2x per-sample-sum CE; seed 42)",
     f"  {'CE scale':>9s} {'macro Δ':>9s} {'Aves Δ':>9s} {'proto macro':>11s}  note",
     "  " + "-"*52]
for r in rows:
    if r[1] is None:
        L.append(f"  {r[0]:>9.1f} {'--':>9s} {'--':>9s} {'--':>11s}  {r[4]}")
    else:
        L.append(f"  {r[0]:>9.1f} {r[1]:>+9.4f} {r[2]:>+9.4f} {r[3]:>11.4f}  {r[4]}")
L.append("")
L.append("  PRE-REGISTERED VERDICT: " + verdict)
L.append("  (criteria fixed before run: HOLDS = all 3 scales macro∈[-0.02,+0.01] AND Aves Δ<0.)")
out = "\n".join(L)
print(out)
try:
    RES.write_text(RES.read_text(encoding="utf-8") + "\n" + out + "\n", encoding="utf-8")
    print(f"\nappended to {RES}")
except Exception as e:
    print("append failed:", e)
