import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
B = r"D:\Python\BirdClef"; torch.manual_seed(0); np.random.seed(0)

z = np.load(B + r"\_perch_meta\full_perch_arrays.npz")
emb = torch.tensor(z["emb_full"], dtype=torch.float32); perch = torch.tensor(z["scores_full_raw"], dtype=torch.float32)
meta = pd.read_parquet(B + r"\_perch_meta\full_perch_meta.parquet")
fold_id = np.load(B + r"\_proto_ckpt\perch_cache\full_oof_meta_features.npz")["fold_id"]
species = [c for c in pd.read_csv(B + r"\_q\submission_protossm.csv", nrows=1).columns if c != "row_id"]
sp_idx = {s: i for i, s in enumerate(species)}; N, D, C = emb.shape[0], emb.shape[1], len(species)
lab = pd.read_csv(B + r"\data\train_soundscapes_labels.csv"); lab["stem"] = lab["filename"].astype(str).str.replace(r"\.(ogg|wav)$", "", regex=True)
def _sec(t):
    p = [float(x) for x in str(t).split(":")]; return p[0]*3600 + p[1]*60 + p[2] if len(p) == 3 else p[0]*60 + p[1]
present = {}
for _, r in lab.iterrows():
    end = int(_sec(r["end"])); sps = [s for s in str(r["primary_label"]).split(";") if s in sp_idx]
    if sps: present[(r["stem"], end)] = set(sps)
Y = np.zeros((N, C), dtype=np.float32)
for j, rid in enumerate(meta["row_id"].values):
    stem, end = str(rid).rsplit("_", 1)
    for s in present.get((stem, int(end)), ()): Y[j, sp_idx[s]] = 1.0
Yt = torch.tensor(Y); tax = pd.read_csv(B + r"\data\taxonomy.csv")
tmap = dict(zip(tax["primary_label"].astype(str), tax["class_name"].astype(str))); taxon = np.array([tmap.get(s, "UNK") for s in species])
scor = [c for c in range(C) if Y[:, c].sum() > 0]; meas = np.array([Y[:, c].sum() > 0 for c in range(C)])
hn_emb = F.normalize(emb, dim=1).numpy()
def focal(o, t, g=2.0):
    ce = F.binary_cross_entropy_with_logits(o, t, reduction='none'); p = torch.sigmoid(o); pt = p*t + (1-p)*(1-t)
    return (ce*(1-pt)**g).mean()
def m_auc(P, idx):
    a = [roc_auc_score(Y[:, c], P[:, c]) for c in idx if Y[:, c].min() != Y[:, c].max()]
    return float(np.mean(a)) if a else float('nan')

class Head(nn.Module):
    def __init__(s, K=1, fusion="none"):
        super().__init__(); s.fusion = fusion
        s.P = nn.Parameter(torch.randn(C, K, D)*0.02); s.t = nn.Parameter(torch.tensor(5.0)); s.b = nn.Parameter(torch.zeros(C))
        if fusion == "static": s.a = nn.Parameter(torch.zeros(C))
        if fusion == "input": s.g = nn.Linear(D, C)
        if fusion == "probe": s.lin = nn.Linear(D, C)
    def ps(s, x):
        sims = torch.einsum('nd,ckd->nck', F.normalize(x, dim=-1), F.normalize(s.P, dim=-1)).max(-1).values
        return sims*F.softplus(s.t) + s.b
    def forward(s, x, pl):
        if s.fusion == "probe": return s.lin(x)
        p = s.ps(x)
        if s.fusion == "none": return p
        if s.fusion == "static": g = torch.sigmoid(s.a)[None, :]; return g*p + (1-g)*pl
        if s.fusion == "input": g = torch.sigmoid(s.g(x)); return g*p + (1-g)*pl
    def cl(s, x, y):
        xn = F.normalize(x, dim=-1); pn = F.normalize(s.P, dim=-1); L = 0.; n = 0
        for c in range(C):
            m = y[:, c] > .5
            if m.any(): L = L + (1 - (xn[m] @ pn[c].T)).min(-1).values.mean(); n += 1
        return L/max(n, 1)
    def sep(s, x, y):
        xn = F.normalize(x, dim=-1); pn = F.normalize(s.P, dim=-1); L = 0.; n = 0
        for c in range(C):
            m = y[:, c] < .5
            if m.any(): L = L + F.relu((xn[m] @ pn[c].T).max(-1).values - .1).mean(); n += 1
        return L/max(n, 1)


CFGS = [
    ("1_linear_probe",        1, "probe",  0, 0),
    ("2_proto1_noloss",       1, "none",   0, 0),
    ("3_proto1_cluster",      1, "none",   1, 0),
    ("4_proto1_clustersep",   1, "none",   1, 1),
    ("5_proto1_static_fuse",  1, "static", 1, 1),
    ("6_proto1_INPUT_fuse=D2better", 1, "input", 0, 0),
    ("7_proto3_clustersep_PUREinterp", 3, "none", 1, 1),
]
def purity(P):
    if P.ndim == 2: P = P[:, None, :]
    pn = P/(np.linalg.norm(P, axis=2, keepdims=True)+1e-8); K = pn.shape[1]
    pure = np.zeros((C, K), bool)
    for k in range(K): pure[:, k] = Y[(pn[:, k, :] @ hn_emb.T).argmax(1), np.arange(C)] > 0
    return pure.any(1)
folds = sorted(set(fold_id.tolist())); dev = "cuda" if torch.cuda.is_available() else "cpu"
rows = []
for name, K, fus, uc, us in CFGS:
    oof = np.zeros((N, C), np.float32); Pfinal = None
    for f in folds:
        tr = torch.tensor(fold_id != f); va = torch.tensor(fold_id == f)
        m = Head(K, fus).to(dev); opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 60)
        Xt, Yt2, Pt = emb[tr].to(dev), Yt[tr].to(dev), perch[tr].to(dev)
        for ep in range(60):
            m.train(); opt.zero_grad(); out = m(Xt, Pt); loss = focal(out, Yt2)
            if uc: loss = loss + 0.8*m.cl(Xt, Yt2)
            if us: loss = loss + 0.08*m.sep(Xt, Yt2)
            if not torch.isfinite(loss): opt.zero_grad(); continue
            loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step()
        m.eval()
        with torch.no_grad(): oof[va.numpy()] = torch.sigmoid(m(emb[va].to(dev), perch[va].to(dev))).cpu().numpy()
        if f == folds[0] and fus != "probe": Pfinal = m.P.detach().cpu().numpy()
    auc = m_auc(oof, scor)
    pt = {tx: round(m_auc(oof, [c for c in scor if taxon[c] == tx]), 3) for tx in ["Insecta", "Amphibia", "Aves", "Mammalia"]}
    pur = round(purity(Pfinal)[meas].mean(), 3) if Pfinal is not None else None
    rows.append((name, auc, pur, pt)); np.save(B + r"\_abl_oof_%s.npy" % name, oof)
    print("%-34s AUC=%.4f  purity=%s  %s" % (name, auc, pur, pt))

print("\n--- references (same 708 OOF, %d species) ---" % len(scor))
print("perch_logits_alone                 AUC=%.4f" % m_auc(perch.numpy(), scor))
print("ProtoSSMv2_deployed(oof_base)       AUC=%.4f" % m_auc(np.load(B + r"\_proto_ckpt\perch_cache\full_oof_meta_features.npz")["oof_base"], scor))
print("\n=== STEPWISE BUILD-UP TABLE (AUC | purity) ===")
for name, auc, pur, pt in rows: print("  %-34s %.3f  | %s" % (name, auc, pur))
