import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
B = r"D:\Python\BirdClef"

z = np.load(B + r"\_perch_meta\full_perch_arrays.npz")
emb = torch.tensor(z["emb_full"], dtype=torch.float32)
meta = pd.read_parquet(B + r"\_perch_meta\full_perch_meta.parquet")
species = [c for c in pd.read_csv(B + r"\_q\submission_protossm.csv", nrows=1).columns if c != "row_id"]
sp_idx = {s: i for i, s in enumerate(species)}
N, C = emb.shape[0], len(species)
lab = pd.read_csv(B + r"\data\train_soundscapes_labels.csv")
lab["stem"] = lab["filename"].astype(str).str.replace(r"\.(ogg|wav)$", "", regex=True)
def _sec(t):
    p = [float(x) for x in str(t).split(":")]; return p[0]*3600 + p[1]*60 + p[2] if len(p) == 3 else p[0]*60 + p[1]
present = {}
for _, r in lab.iterrows():
    end = int(_sec(r["end"])); sps = [s for s in str(r["primary_label"]).split(";") if s in sp_idx]
    if sps: present[(r["stem"], end)] = set(sps)
Y = np.zeros((N, C), dtype=bool)
for j, rid in enumerate(meta["row_id"].values):
    stem, end = str(rid).rsplit("_", 1)
    for s in present.get((stem, int(end)), ()): Y[j, sp_idx[s]] = True
tax = pd.read_csv(B + r"\data\taxonomy.csv")
tmap = dict(zip(tax["primary_label"].astype(str), tax["class_name"].astype(str)))
taxon = np.array([tmap.get(s, "UNK") for s in species])
measurable = Y.sum(0) > 0
hn = F.normalize(emb, dim=1).numpy()

print("=== prototype purity, new heads (raw 1536-d Perch space, %d measurable species) ===" % measurable.sum())
print("ref: ProtoSSMv2 = 0.789 [0.69-0.87] (post-SSM space)\n")
for name in ["D1_direct_multiproto", "D2_gated_fusion", "D3_interp_clustersep"]:
    try:
        sd = torch.load(B + r"\_heads_%s_fold0.pt" % name, map_location="cpu")
    except Exception as e:
        print("%-22s : no checkpoint (%s)" % (name, e)); continue
    P = sd["P"].numpy()
    if P.ndim == 2: P = P[:, None, :]
    Cc, K, D = P.shape
    pn = P / (np.linalg.norm(P, axis=2, keepdims=True) + 1e-8)

    pure_perproto = np.zeros((Cc, K), dtype=bool)
    for k in range(K):
        sims = pn[:, k, :] @ hn.T
        nn = sims.argmax(1)
        pure_perproto[:, k] = Y[nn, np.arange(Cc)]
    pure_class = pure_perproto.any(1)
    null = np.array([Y[(pn[c, 0] @ hn.T).argmax()].sum() / C for c in range(Cc)])
    m = measurable
    ov = pure_class[m].mean()
    per = {}
    for tx in ["Insecta", "Amphibia", "Aves", "Mammalia"]:
        idx = m & (taxon == tx)
        if idx.sum(): per[tx] = (int(idx.sum()), round(pure_class[idx].mean(), 3))
    print("%-22s K=%d : purity(any-proto)=%.3f  null=%.3f  per-taxon=%s"
          % (name, K, ov, null[m].mean(), per))
