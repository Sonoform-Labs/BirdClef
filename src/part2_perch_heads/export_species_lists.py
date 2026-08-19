import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
B = r"D:\Python\BirdClef"
OUT = r"D:\Python\BirdClef_final\results"

species = [c for c in pd.read_csv(B + r"\_q\submission_protossm.csv", nrows=1).columns if c != "row_id"]
assert len(species) == 234
sp_idx = {s: i for i, s in enumerate(species)}

meta = pd.read_parquet(B + r"\_perch_meta\full_perch_meta.parquet")
lab = pd.read_csv(B + r"\data\train_soundscapes_labels.csv")
lab["stem"] = lab["filename"].astype(str).str.replace(r"\.(ogg|wav)$", "", regex=True)
def _sec(t):
    p = [float(x) for x in str(t).split(":")]
    return p[0]*3600 + p[1]*60 + p[2] if len(p) == 3 else (p[0]*60 + p[1] if len(p) == 2 else p[0])
present = {}
for _, r in lab.iterrows():
    end = int(_sec(r["end"]))
    sps = [s for s in str(r["primary_label"]).split(";") if s in sp_idx]
    if sps:
        present[(r["stem"], end)] = set(sps)
Y = np.zeros((708, 234), dtype=bool)
for j, rid in enumerate(meta["row_id"].values):
    stem, end = str(rid).rsplit("_", 1)
    for spc in present.get((stem, int(end)), ()):
        Y[j, sp_idx[spc]] = True
pos = Y.sum(0)
covered = pos > 0

tax = pd.read_csv(B + r"\data\taxonomy.csv")
tmap = dict(zip(tax["primary_label"].astype(str), tax["class_name"].astype(str)))
taxon = [tmap.get(s, "UNK") for s in species]

try:
    pm = np.load(B + r"\perch_map.npz")
    perch_mapped = (pm["bc_index"] >= 0)
    perch_corr = np.round(pm["best_corr"].astype(float), 4)
except Exception:
    perch_mapped = np.array([""] * 234, dtype=object)
    perch_corr = np.array([""] * 234, dtype=object)

df = pd.DataFrame({
    "species_index": np.arange(234),
    "species_code": species,
    "taxon": taxon,
    "n_positive_windows": pos.astype(int),
    "perch_logit_mapped": perch_mapped,
    "perch_match_corr": perch_corr,
})
df[covered].reset_index(drop=True).to_csv(OUT + r"\covered_species_71.csv", index=False)
df[~covered].reset_index(drop=True).to_csv(OUT + r"\absent_species_163.csv", index=False)
nd = int(np.sum(perch_mapped)) if perch_mapped.dtype != object else "n/a"
print("covered=%d  absent=%d  sum=%d  | perch_logit_mapped(all 234)=%s"
      % (int(covered.sum()), int((~covered).sum()), 234, nd))
