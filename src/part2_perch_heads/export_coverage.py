import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np, pandas as pd
B = r"D:\Python\BirdClef"
meta = pd.read_parquet(B + r"\_perch_meta\full_perch_meta.parquet")
species = [c for c in pd.read_csv(B + r"\_q\submission_protossm.csv", nrows=1).columns if c != "row_id"]
sp_idx = {s: i for i, s in enumerate(species)}; C = len(species)
lab = pd.read_csv(B + r"\data\train_soundscapes_labels.csv")
lab["stem"] = lab["filename"].astype(str).str.replace(r"\.(ogg|wav)$", "", regex=True)
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
covered = (Y.sum(0) > 0)
np.savez(B + r"\covered_mask.npz", covered=covered, species=np.array(species))
print("covered species: %d / %d (the rest get raw Perch in the coverage-gated head)" % (covered.sum(), C))
print("aligned to submission_protossm.csv column order:", species[:3], "...")
