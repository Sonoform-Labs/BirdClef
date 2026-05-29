import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np, pandas as pd
from scipy.stats import rankdata
B = r"D:\Python\BirdClef"
z = np.load(B + r"\_perch_meta\full_perch_arrays.npz")
emb = z["emb_full"].astype(np.float64); sc = z["scores_full_raw"].astype(np.float64)
d2 = np.load(B + r"\d2_weights.npz"); P = d2["P"].astype(np.float64); temp = float(d2["temp"]); bias = d2["bias"].astype(np.float64)
gw = d2["g_weight"].astype(np.float64); gb = d2["g_bias"].astype(np.float64)
cov = np.load(B + r"\covered_mask.npz")["covered"]
xn = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)
proto = (xn @ pn.T) * np.log1p(np.exp(temp)) + bias
gate = 1.0 / (1.0 + np.exp(-(emb @ gw.T + gb)))
D2 = gate * proto + (1.0 - gate) * sc

def spearman_col(a, b):
    return np.corrcoef(rankdata(a), rankdata(b))[0, 1]

def summarize(mask, name):
    g = gate[:, mask].mean()

    rs = [spearman_col(D2[:, c], sc[:, c]) for c in np.where(mask)[0] if np.std(sc[:, c]) > 0 and np.std(D2[:, c]) > 0]

    sr = (D2[:, mask].std(0) / (sc[:, mask].std(0) + 1e-9))
    print("  %-8s n=%-3d  mean gate=%.3f  median rho(D2,Perch)=%.3f  median std(D2)/std(Perch)=%.3f"
          % (name, mask.sum(), g, float(np.median(rs)) if rs else float('nan'), float(np.median(sr))))

print("=== D2 vs raw Perch on the 708-window bank ===")
print("(high gate + low rho on ABSENT species = D2 overwrites Perch's signal there -> LB hurt)")
summarize(cov, "covered")
summarize(~cov, "absent")

print("\nInterpretation: if absent-species gate is high (~0.7) and rho(D2,Perch) is low,")
print("D2 replaces Perch's discriminative ranking with a flat prototype on the 163 species")
print("it never trained positives for -> exactly the 234-species macro-AUC drag we see on the LB.")
