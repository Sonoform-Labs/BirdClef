import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np, pandas as pd
B = r"D:\Python\BirdClef"
z = np.load(B + r"\_perch_meta\full_perch_arrays.npz"); emb = z["emb_full"]
meta = pd.read_parquet(B + r"\_perch_meta\full_perch_meta.parquet")
species = [c for c in pd.read_csv(B + r"\_q\submission_protossm.csv", nrows=1).columns if c != "row_id"]
sp_idx = {s: i for i, s in enumerate(species)}; C = len(species)
lab = pd.read_csv(B + r"\data\train_soundscapes_labels.csv"); lab["stem"] = lab["filename"].astype(str).str.replace(r"\.(ogg|wav)$", "", regex=True)
def _sec(t):
    p = [float(x) for x in str(t).split(":")]; return p[0]*3600 + p[1]*60 + p[2] if len(p) == 3 else p[0]*60 + p[1]
present = {}
for _, r in lab.iterrows():
    end = int(_sec(r["end"])); sps = [s for s in str(r["primary_label"]).split(";") if s in sp_idx]
    if sps: present[(r["stem"], end)] = set(sps)
Y = np.zeros((len(emb), C), np.float32)
for j, rid in enumerate(meta["row_id"].values):
    stem, end = str(rid).rsplit("_", 1)
    for s in present.get((stem, int(end)), ()): Y[j, sp_idx[s]] = 1.0
tax = pd.read_csv(B + r"\data\taxonomy.csv"); tmap = dict(zip(tax["primary_label"].astype(str), tax["class_name"].astype(str)))
taxon = np.array([tmap.get(s, "UNK") for s in species]); meas = Y.sum(0) > 0
hn = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)

def purity_by_taxon(P):
    pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)
    nn = (pn @ hn.T).argmax(1)
    pure = Y[nn, np.arange(C)] > 0
    out = {}
    for t in ["Aves", "Amphibia", "Insecta", "Mammalia"]:
        cols = [i for i in range(C) if taxon[i] == t and meas[i]]
        out[t] = (round(float(pure[cols].mean()), 3), len(cols)) if cols else ("--", 0)
    out["all"] = (round(float(pure[meas].mean()), 3), int(meas.sum()))
    return out

for tag, f in [("D2 (gated, purity 0.44)", "d2_weights.npz"), ("D1 (pure proto, 0.96)", "d1_weights.npz")]:
    P = np.load(B + "\\" + f)["P"]
    o = purity_by_taxon(P)
    print("%-26s | %s" % (tag, "  ".join("%s=%s(n=%d)" % (t, v[0], v[1]) for t, v in o.items())))
print("\n(ProtoSSMv2 deployed, from the projection: Insecta 0.96 > Amphibia 0.88 > Aves 0.64)")
