import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
B = r"D:\Python\BirdClef"; torch.manual_seed(0)
z = np.load(B + r"\_perch_meta\full_perch_arrays.npz")
emb = torch.tensor(z["emb_full"], dtype=torch.float32); perch = torch.tensor(z["scores_full_raw"], dtype=torch.float32)
meta = pd.read_parquet(B + r"\_perch_meta\full_perch_meta.parquet")
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
Yt = torch.tensor(Y)
class GatedProto(nn.Module):
    def __init__(s):
        super().__init__(); s.P = nn.Parameter(torch.randn(C, D)*0.02); s.t = nn.Parameter(torch.tensor(5.0))
        s.b = nn.Parameter(torch.zeros(C)); s.g = nn.Linear(D, C)
    def forward(s, x, pl):
        proto = (F.normalize(x, dim=-1) @ F.normalize(s.P, dim=-1).T)*F.softplus(s.t) + s.b
        gamma = torch.sigmoid(s.g(x)); return gamma*proto + (1-gamma)*pl
def focal(o, t, g=2.0):
    ce = F.binary_cross_entropy_with_logits(o, t, reduction='none'); p = torch.sigmoid(o); pt = p*t + (1-p)*(1-t)
    return (ce*(1-pt)**g).mean()
m = GatedProto(); opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 60)
for ep in range(60):
    m.train(); opt.zero_grad(); loss = focal(m(emb, perch), Yt)
    loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step()
print("final train loss %.4f" % float(loss))
sd = m.state_dict()
np.savez(B + r"\d2_weights.npz",
         P=sd["P"].numpy(), temp=np.array(float(sd["t"])), bias=sd["b"].numpy(),
         g_weight=sd["g.weight"].numpy(), g_bias=sd["g.bias"].numpy(),
         species=np.array(species))

def np_d2(x, pl):
    xn = x/(np.linalg.norm(x, axis=1, keepdims=True)+1e-8); pn = sd["P"].numpy()/(np.linalg.norm(sd["P"].numpy(), axis=1, keepdims=True)+1e-8)
    proto = (xn @ pn.T)*np.log1p(np.exp(float(sd["t"]))) + sd["b"].numpy()
    gate = 1/(1+np.exp(-(x @ sd["g.weight"].numpy().T + sd["g.bias"].numpy())))
    return gate*proto + (1-gate)*pl
with torch.no_grad(): torch_out = m(emb, perch).numpy()
np_out = np_d2(emb.numpy(), perch.numpy())
print("numpy-vs-torch forward max abs diff: %.2e (must be ~0 for the notebook reimpl)" % np.abs(torch_out - np_out).max())
print("saved d2_weights.npz: P(234,1536), temp, bias(234), g_weight(234,1536), g_bias(234)")
