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
Yt = torch.tensor(Y); scor = [c for c in range(C) if Y[:, c].sum() > 0]; meas = Y.sum(0) > 0
hn_raw = F.normalize(emb, dim=1).numpy()
folds = sorted(set(fold_id.tolist())); dev = "cuda" if torch.cuda.is_available() else "cpu"
def focal(o, t, g=2.0):
    ce = F.binary_cross_entropy_with_logits(o, t, reduction='none'); p = torch.sigmoid(o); pt = p*t + (1-p)*(1-t)
    return (ce*(1-pt)**g).mean()
def m_auc(P, idx=None):
    idx = idx if idx is not None else scor
    a = [roc_auc_score(Y[:, c], P[:, c]) for c in idx if Y[:, c].min() != Y[:, c].max()]
    return float(np.mean(a)) if a else float('nan')

class Head(nn.Module):
    def __init__(s, K=1, fusion="none", proj=None):
        super().__init__(); s.fusion = fusion; s.proj = proj; pdim = D
        if proj:
            kind, d = proj
            s.pj = nn.Linear(D, d) if kind == "linear" else nn.Sequential(nn.Linear(D, d), nn.GELU(), nn.Linear(d, d)); pdim = d
        s.P = nn.Parameter(torch.randn(C, K, pdim)*0.02); s.t = nn.Parameter(torch.tensor(5.0)); s.b = nn.Parameter(torch.zeros(C))
        if fusion == "static": s.a = nn.Parameter(torch.zeros(C))
        if fusion == "input": s.g = nn.Linear(D, C)
    def embed(s, x): return s.pj(x) if s.proj else x
    def ps(s, x):
        h = s.embed(x); return torch.einsum('nd,ckd->nck', F.normalize(h, dim=-1), F.normalize(s.P, dim=-1)).max(-1).values*F.softplus(s.t) + s.b
    def forward(s, x, pl):
        p = s.ps(x)
        if s.fusion == "none": return p
        if s.fusion == "static": g = torch.sigmoid(s.a)[None, :]; return g*p + (1-g)*pl
        g = torch.sigmoid(s.g(x)); return g*p + (1-g)*pl
    def cl(s, x, y):
        h = F.normalize(s.embed(x), dim=-1); pn = F.normalize(s.P, dim=-1); L = 0.; n = 0
        for c in range(C):
            m = y[:, c] > .5
            if m.any(): L = L + (1 - (h[m] @ pn[c].T)).min(-1).values.mean(); n += 1
        return L/max(n, 1)
    def sep(s, x, y):
        h = F.normalize(s.embed(x), dim=-1); pn = F.normalize(s.P, dim=-1); L = 0.; n = 0
        for c in range(C):
            m = y[:, c] < .5
            if m.any(): L = L + F.relu((h[m] @ pn[c].T).max(-1).values - .1).mean(); n += 1
        return L/max(n, 1)

def run(K=1, fusion="none", proj=None, uc=0, us=0, epochs=60, mixup=0.0):
    oof = np.zeros((N, C), np.float32); m0 = None
    for f in folds:
        tr = torch.tensor(fold_id != f); va = torch.tensor(fold_id == f)
        m = Head(K, fusion, proj).to(dev); opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        Xt, Yt2, Pt = emb[tr].to(dev), Yt[tr].to(dev), perch[tr].to(dev)
        for ep in range(epochs):
            m.train(); opt.zero_grad()
            X_, Y_ = Xt, Yt2
            if mixup > 0:
                lam = float(np.random.beta(mixup, mixup)); idx = torch.randperm(Xt.shape[0], device=dev)
                X_ = lam*Xt + (1-lam)*Xt[idx]; Y_ = torch.maximum(Yt2, Yt2[idx])
            out = m(X_, Pt); loss = focal(out, Y_)
            if uc: loss = loss + 0.8*m.cl(Xt, Yt2)
            if us: loss = loss + 0.08*m.sep(Xt, Yt2)
            if not torch.isfinite(loss): opt.zero_grad(); continue
            loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step()
        m.eval()
        with torch.no_grad(): oof[va.numpy()] = torch.sigmoid(m(emb[va].to(dev), perch[va].to(dev))).cpu().numpy()
        if f == folds[0]: m0 = m

    with torch.no_grad(): H = F.normalize(m0.embed(emb.to(dev)), dim=-1).cpu().numpy()
    P = m0.P.detach().cpu().numpy(); K_ = P.shape[1]
    pn = P/(np.linalg.norm(P, axis=2, keepdims=True)+1e-8)
    perproto = np.zeros((C, K_), bool)
    for k in range(K_): perproto[:, k] = Y[(pn[:, k, :] @ H.T).argmax(1), np.arange(C)] > 0
    return m_auc(oof), perproto[meas].any(1).mean(), perproto[meas].mean(), oof

print("=== CAT 3: prototype count (pure head, cluster+sep) ===")
for K in [1, 2, 3, 5, 10]:
    a, pa, pm, _ = run(K=K, fusion="none", uc=1, us=1)
    print("  K=%-2d  AUC=%.3f  purity(any)=%.3f  purity(mean-per-proto)=%.3f" % (K, a, pa, pm))

print("\n=== CAT 6: embedding space (pure head K=1, cluster+sep) ===")
for proj, nm in [(None, "direct-1536"), (("linear", 768), "linear-768"), (("linear", 320), "linear-320"), (("mlp", 320), "mlp-320")]:
    a, pa, pm, _ = run(K=1, fusion="none", proj=proj, uc=1, us=1)
    print("  %-12s AUC=%.3f  purity=%.3f" % (nm, a, pa))

print("\n=== CAT 7: training length + embedding-mixup (D2 = input gate) ===")
for ep in [10, 25, 40, 60]:
    a, _, _, _ = run(K=1, fusion="input", epochs=ep); print("  epochs=%-3d AUC=%.3f" % (ep, a))
for mx in [0.0, 0.4]:
    a, _, _, _ = run(K=1, fusion="input", mixup=mx); print("  mixup=%.1f   AUC=%.3f" % (mx, a))

print("\n=== CAT 5: temporal smoothing (post-hoc on D2 OOF, per-file pooling) ===")
d2 = np.load(B + r"\_oof_D2_gated_fusion.npy")
files = meta["filename"].values
for mode in ["none", "mean", "max"]:
    P = d2.copy()
    if mode != "none":
        for f in pd.unique(files):
            ix = np.where(files == f)[0]
            P[ix] = (d2[ix].mean(0) if mode == "mean" else d2[ix].max(0))[None, :]
    print("  smooth=%-5s AUC=%.3f" % (mode, m_auc(P)))
print("\nDONE")
