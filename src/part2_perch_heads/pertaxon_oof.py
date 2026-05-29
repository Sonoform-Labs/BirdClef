import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
B = r"D:\Python\BirdClef"
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
N = len(meta); Y = np.zeros((N, C), np.float32)
for j, rid in enumerate(meta["row_id"].values):
    stem, end = str(rid).rsplit("_", 1)
    for s in present.get((stem, int(end)), ()): Y[j, sp_idx[s]] = 1.0
tax = pd.read_csv(B + r"\data\taxonomy.csv").set_index("primary_label")["class_name"].astype(str).to_dict()
sp_tax = np.array([tax.get(s, "?") for s in species])

heads = {"linear probe": "_oof_REF_linear_probe.npy", "D2 gated": "_oof_D2_gated_fusion.npy",
         "D1 pure proto": "_oof_D1_direct_multiproto.npy"}
oof = {k: np.load(B + "\\" + v) for k, v in heads.items()}

def taxon_auc(P, cols):
    a = [roc_auc_score(Y[:, c], P[:, c]) for c in cols if Y[:, c].min() != Y[:, c].max()]
    return (float(np.mean(a)), len(a)) if a else (float("nan"), 0)

print("Per-taxon OOF macro-AUC (covered species only; n = #species scored)")
print("%-13s | %s" % ("taxon", "  ".join("%-14s" % h for h in heads)))
order = ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]
for tx in order:
    cols = [i for i in range(C) if sp_tax[i] == tx and Y[:, i].sum() > 0]
    if not cols: continue
    cells = []
    for h in heads:
        a, n = taxon_auc(oof[h], cols)
        cells.append("%.3f (n=%d)" % (a, n) if n else "-- (n=0)")
    flag = "  [QUARANTINE small-n]" if len(cols) < 5 else ""
    print("%-13s | %s%s" % (tx, "  ".join("%-14s" % c for c in cells), flag))

allc = [i for i in range(C) if Y[:, i].sum() > 0]
print("%-13s | %s" % ("ALL covered", "  ".join("%-14s" % ("%.3f (n=%d)" % taxon_auc(oof[h], allc)) for h in heads)))
