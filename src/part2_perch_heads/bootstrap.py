import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
B = r"D:\Python\BirdClef"
z = np.load(B + r"\_perch_meta\full_perch_arrays.npz"); perch = z["scores_full_raw"]
meta = pd.read_parquet(B + r"\_perch_meta\full_perch_meta.parquet")
species = [c for c in pd.read_csv(B + r"\_q\submission_protossm.csv", nrows=1).columns if c != "row_id"]
sp_idx = {s: i for i, s in enumerate(species)}; N, C = perch.shape
lab = pd.read_csv(B + r"\data\train_soundscapes_labels.csv"); lab["stem"] = lab["filename"].astype(str).str.replace(r"\.(ogg|wav)$", "", regex=True)
def _sec(t):
    p = [float(x) for x in str(t).split(":")]; return p[0]*3600 + p[1]*60 + p[2] if len(p) == 3 else p[0]*60 + p[1]
present = {}
for _, r in lab.iterrows():
    end = int(_sec(r["end"])); sps = [s for s in str(r["primary_label"]).split(";") if s in sp_idx]
    if sps: present[(r["stem"], end)] = set(sps)
Y = np.zeros((N, C))
for j, rid in enumerate(meta["row_id"].values):
    stem, end = str(rid).rsplit("_", 1)
    for s in present.get((stem, int(end)), ()): Y[j, sp_idx[s]] = 1
base = np.load(B + r"\_proto_ckpt\perch_cache\full_oof_meta_features.npz")["oof_base"]
models = {"D2_gated": np.load(B + r"\_oof_D2_gated_fusion.npy"),
          "D1_pure": np.load(B + r"\_oof_D1_direct_multiproto.npy"),
          "probe": np.load(B + r"\_oof_REF_linear_probe.npy"),
          "ProtoSSMv2": base, "Perch": perch}
sc = [c for c in range(C) if Y[:, c].sum() > 0]
def per_sp(P): return np.array([roc_auc_score(Y[:, c], P[:, c]) for c in sc])
A = {k: per_sp(v) for k, v in models.items()}
rng = np.random.default_rng(0); n = len(sc)
print("macro-AUC [95%% CI] over %d species:" % n)
for k in ["D2_gated", "ProtoSSMv2", "probe", "Perch", "D1_pure"]:
    bs = [A[k][rng.integers(0, n, n)].mean() for _ in range(4000)]
    print("  %-12s %.4f [%.4f-%.4f]" % (k, A[k].mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)))
print("\npaired gaps (bootstrap of per-species difference):")
for a, b in [("D2_gated", "ProtoSSMv2"), ("D2_gated", "probe"), ("D2_gated", "Perch")]:
    d = A[a] - A[b]; bs = [d[rng.integers(0, n, n)].mean() for _ in range(4000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print("  %s - %s = %+.4f [%+.4f, %+.4f]  %s" % (a, b, d.mean(), lo, hi, "SIG>0" if lo > 0 else "n.s."))
