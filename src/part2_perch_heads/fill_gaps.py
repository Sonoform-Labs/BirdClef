import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
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
Y = np.zeros((N, C), np.float32)
for j, rid in enumerate(meta["row_id"].values):
    stem, end = str(rid).rsplit("_", 1)
    for s in present.get((stem, int(end)), ()): Y[j, sp_idx[s]] = 1.0
Yt = torch.tensor(Y); nsp = torch.tensor(Y.sum(1), dtype=torch.float32)
tax = pd.read_csv(B + r"\data\taxonomy.csv"); tmap = dict(zip(tax["primary_label"].astype(str), tax["class_name"].astype(str)))
taxon = np.array([tmap.get(s, "UNK") for s in species])
scor = [c for c in range(C) if Y[:, c].sum() > 0]; meas = np.array([Y[:, c].sum() > 0 for c in range(C)])
hn_emb = F.normalize(emb, dim=1).numpy(); folds = sorted(set(fold_id.tolist())); dev = "cuda" if torch.cuda.is_available() else "cpu"
def focal(o, t, g=2.0):
    ce = F.binary_cross_entropy_with_logits(o, t, reduction='none'); p = torch.sigmoid(o); pt = p*t + (1-p)*(1-t)
    return (ce*(1-pt)**g).mean()
def m_auc(P, idx):
    a = [roc_auc_score(Y[:, c], P[:, c]) for c in idx if Y[:, c].min() != Y[:, c].max()]; return float(np.mean(a)) if a else float('nan')

class Head(nn.Module):
    def __init__(s, K=1, fusion="none", proj=None):
        super().__init__(); s.fusion = fusion; s.proj = proj; pd = D
        if proj: s.pj = nn.Linear(D, proj); pd = proj
        s.P = nn.Parameter(torch.randn(C, K, pd)*0.02); s.t = nn.Parameter(torch.tensor(5.0)); s.b = nn.Parameter(torch.zeros(C))
        if fusion == "static": s.a = nn.Parameter(torch.zeros(C))
        if fusion == "scalar": s.a = nn.Parameter(torch.zeros(1))
        if fusion == "input": s.g = nn.Linear(D, C)
    def embed(s, x): return s.pj(x) if s.proj else x
    def ps(s, x):
        h = s.embed(x); sims = torch.einsum('nd,ckd->nck', F.normalize(h, dim=-1), F.normalize(s.P, dim=-1)).max(-1).values
        return sims*F.softplus(s.t) + s.b
    def forward(s, x, pl):
        p = s.ps(x)
        if s.fusion == "none": return p
        if s.fusion == "static": g = torch.sigmoid(s.a)[None, :]; return g*p + (1-g)*pl
        if s.fusion == "scalar": g = torch.sigmoid(s.a); return g*p + (1-g)*pl
        g = torch.sigmoid(s.g(x)); return g*p + (1-g)*pl
    def cl(s, x, y):
        h = F.normalize(s.embed(x), dim=-1); pn = F.normalize(s.P, dim=-1); L = 0.; n = 0
        for c in range(C):
            m = y[:, c] > .5
            if m.any(): L = L + (1 - (h[m] @ pn[c].T)).min(-1).values.mean(); n += 1
        return L/max(n, 1)
    def sep(s, x, y, w=None):
        h = F.normalize(s.embed(x), dim=-1); pn = F.normalize(s.P, dim=-1); L = 0.; n = 0
        for c in range(C):
            m = y[:, c] < .5
            if m.any():
                pen = F.relu((h[m] @ pn[c].T).max(-1).values - .1)
                if w is not None: pen = (pen * w[m]) / (w[m].mean() + 1e-8)
                L = L + pen.mean(); n += 1
        return L/max(n, 1)

def run(K=1, fusion="none", proj=None, uc=0, us=0, ml=False):
    oof = np.zeros((N, C), np.float32); P0 = None
    wgt = (1.0 / (1.0 + nsp)) if ml else None
    for f in folds:
        tr = torch.tensor(fold_id != f); va = torch.tensor(fold_id == f)
        m = Head(K, fusion, proj).to(dev); opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 60)
        Xt, Yt2, Pt = emb[tr].to(dev), Yt[tr].to(dev), perch[tr].to(dev); wt = wgt[tr].to(dev) if wgt is not None else None
        for ep in range(60):
            m.train(); opt.zero_grad(); loss = focal(m(Xt, Pt), Yt2)
            if uc: loss = loss + 0.8*m.cl(Xt, Yt2)
            if us: loss = loss + 0.08*m.sep(Xt, Yt2, wt)
            if torch.isfinite(loss): loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step()
            else: opt.zero_grad()
        m.eval()
        with torch.no_grad(): oof[va.numpy()] = torch.sigmoid(m(emb[va].to(dev), perch[va].to(dev))).cpu().numpy()
        if f == folds[0]: P0 = (m.P.detach().cpu().numpy(), m.pj.weight.detach().cpu().numpy() if proj else None)

    if proj:
        W = P0[1]; H = F.normalize(torch.tensor(emb.numpy() @ W.T), dim=-1).numpy()
    else:
        H = hn_emb
    P = P0[0]; pn = P/(np.linalg.norm(P, axis=2, keepdims=True)+1e-8)
    pure = np.zeros((C, pn.shape[1]), bool)
    for k in range(pn.shape[1]): pure[:, k] = Y[(pn[:, k, :] @ H.T).argmax(1), np.arange(C)] > 0
    return m_auc(oof, scor), float(pure.any(1)[meas].mean())

print("=== (A) FULL-SPEC D2: input gate WITH cluster+sep (vs D2-as-built no-loss 0.830/0.437) ===")
a, p = run(K=1, fusion="input", uc=1, us=1); print("  input + cluster + sep:        AUC=%.3f  purity=%.3f" % (a, p))
print("\n=== (B) K-sweep on the GATED head (input fusion, no loss; D2-as-built family) ===")
for K in [1, 3, 5]:
    a, p = run(K=K, fusion="input", uc=0, us=0); print("  D2 K=%d:                       AUC=%.3f  purity=%.3f" % (K, a, p))
print("\n=== (C) STATIC SCALAR fusion (single global beta) + cluster+sep (vs static per-class 0.742/0.958) ===")
a, p = run(K=1, fusion="scalar", uc=1, us=1); print("  scalar-gate + cluster + sep:  AUC=%.3f  purity=%.3f" % (a, p))
print("\n=== (D) MULTI-LABEL-AWARE separation (vs standard, both pure head + cluster) ===")
a, p = run(K=1, fusion="none", uc=1, us=1, ml=False); print("  standard sep (= build-up r4): AUC=%.3f  purity=%.3f" % (a, p))
a, p = run(K=1, fusion="none", uc=1, us=1, ml=True);  print("  multi-label-aware sep:        AUC=%.3f  purity=%.3f" % (a, p))
print("\n=== (E) D3 = projected (linear-320) pure prototype, distinct from D1 (raw-1536) ===")
a, p = run(K=1, fusion="none", proj=320, uc=1, us=1); print("  D3 (proj-320 pure proto):     AUC=%.3f  purity=%.3f" % (a, p))
a, p = run(K=1, fusion="none", uc=1, us=1);           print("  D1 (raw-1536 pure proto):     AUC=%.3f  purity=%.3f" % (a, p))

print("\n=== (F) PER-TAXON for EVERY build-up row (recomputed from saved OOF) ===")
txs = ["Aves", "Amphibia", "Insecta", "Mammalia"]
def taxauc(P): return {t: round(m_auc(P, [c for c in scor if taxon[c] == t]), 3) for t in txs}
import glob
for fp in sorted(glob.glob(B + r"\_abl_oof_*.npy")):
    nm = fp.split("_abl_oof_")[-1].replace(".npy", ""); P = np.load(fp)
    print("  %-30s all=%.3f  %s" % (nm, m_auc(P, scor), taxauc(P)))
print("DONE")
