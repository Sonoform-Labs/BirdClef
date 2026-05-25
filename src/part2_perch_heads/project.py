import sys, io, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, r"D:\Python\BirdClef")
import torch, numpy as np, pandas as pd
import torch.nn.functional as F
import protossm_model as M

B = r"D:\Python\BirdClef"

model = M.ProtoSSMv2(d_input=1536, d_model=320, d_state=32, n_ssm_layers=4, n_classes=234,
                     n_windows=12, n_sites=20, meta_dim=24, use_cross_attn=True, cross_attn_heads=8)
sd = torch.load(B + r"\_proto_ckpt\train_proto_ssm_single\models\proto_ssm_best.pt", map_location="cpu")
model.load_state_dict(sd, strict=False); model.eval()
proto = model.prototypes.detach()


z = np.load(B + r"\_perch_meta\full_perch_arrays.npz")
emb = z["emb_full"].astype(np.float32)
meta = pd.read_parquet(B + r"\_perch_meta\full_perch_meta.parquet")
files = meta["filename"].values
cnt = collections.Counter(files)
assert all(v == 12 for v in cnt.values()) and len(emb) == 708, cnt


H = np.zeros((708, 320), dtype=np.float32)
with torch.no_grad():
    for f in pd.unique(files):
        idx = np.where(files == f)[0]
        x = torch.from_numpy(emb[idx][None])
        _, _, h = model(x, perch_logits=None, site_ids=None, hours=None)
        H[idx] = h[0].numpy()


species = [c for c in pd.read_csv(B + r"\_q\submission_protossm.csv", nrows=1).columns if c != "row_id"]
assert len(species) == 234
sp_idx = {s: i for i, s in enumerate(species)}


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


hn = F.normalize(H_t := torch.from_numpy(H), dim=1)
pn = F.normalize(proto, dim=1)
sims = (pn @ hn.T).numpy()
nearest = sims.argmax(1)
nsim = sims.max(1)

pos_per_class = Y.sum(0)
measurable = pos_per_class > 0
tax = pd.read_csv(B + r"\data\taxonomy.csv")
tmap = dict(zip(tax["primary_label"].astype(str), tax["class_name"].astype(str)))
taxon = np.array([tmap.get(s, "UNK") for s in species])

pure = np.array([Y[nearest[c], c] for c in range(234)])
null = np.array([Y[nearest[c]].sum() / 234.0 for c in range(234)])
empty = np.array([Y[nearest[c]].sum() == 0 for c in range(234)])

df = pd.DataFrame({"taxon": taxon, "pure": pure, "null": null, "empty": empty,
                   "measurable": measurable, "npos": pos_per_class, "nsim": nsim})
sub = df[df.measurable]
print("=== PROTOTYPE PROJECTION (708 soundscape windows, 320-d h-space) ===")
print("measurable species (>=1 positive window in pool):", int(measurable.sum()), "/ 234")
print("\nPURITY (nearest window CONTAINS prototype's class) over measurable species:")
print("  OVERALL: obs %.3f vs null %.3f  (n=%d)" % (sub.pure.mean(), sub.null.mean(), len(sub)))
g = sub.groupby("taxon").agg(n=("pure", "size"), purity=("pure", "mean"),
                             null=("null", "mean"), mean_npos=("npos", "mean"),
                             empty_shortcut=("empty", "mean")).round(3)
print(g.to_string())
print("\nSHORTCUT proxy (landed on a window with NO annotated species): overall %.3f" % sub["empty"].mean())
print("mean nearest-cosine: %.3f" % sub["nsim"].mean())
print("total positive (window,species) pairs in Y:", int(Y.sum()), "| positive windows:", int(Y.any(1).sum()), "/ 708")


rng = np.random.default_rng(0)
meas = np.where(measurable)[0]
pure_m = pure[meas].astype(float)
nulls = []
for _ in range(1000):
    rc = rng.integers(0, 234, size=len(meas))
    nulls.append(np.mean([Y[nearest[meas[k]], rc[k]] for k in range(len(meas))]))
null_mu, null_sd = float(np.mean(nulls)), float(np.std(nulls))
boots = [pure_m[rng.integers(0, len(pure_m), len(pure_m))].mean() for _ in range(3000)]
ci = np.percentile(boots, [2.5, 97.5])
print("\n=== RIGOR ===")
print("OVERALL purity %.3f  [95%% CI %.3f-%.3f]  vs perm-null %.3f +/- %.3f  (%.1f sigma above null)"
      % (pure_m.mean(), ci[0], ci[1], null_mu, null_sd, (pure_m.mean()-null_mu)/null_sd))
for tx in ["Insecta", "Amphibia", "Aves", "Mammalia", "Reptilia"]:
    mask = (taxon[meas] == tx)
    if mask.sum() == 0: continue
    pv = pure_m[mask]
    if len(pv) >= 3:
        bt = [pv[rng.integers(0, len(pv), len(pv))].mean() for _ in range(3000)]
        lo, hi = np.percentile(bt, [2.5, 97.5])
        print("  %-9s n=%2d  purity %.3f [%.3f-%.3f]" % (tx, len(pv), pv.mean(), lo, hi))
    else:
        print("  %-9s n=%2d  purity %.3f  (too few for CI; quarantine)" % (tx, len(pv), pv.mean()))


out = df.copy()
out["species"] = species
out["nearest_window"] = meta["row_id"].values[nearest]
out["pure"] = pure; out["nsim"] = nsim
out[out.measurable].to_csv(B + r"\GATE_PROTOSSM_projection.csv", index=False)
print("\nsaved per-prototype table -> GATE_PROTOSSM_projection.csv")
