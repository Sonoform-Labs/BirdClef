import os, sys, argparse
sys.path.insert(0, r"D:\Python\BirdClef")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import roc_auc_score

FE = Path(r"D:\Python\BirdClef\paper_analysis\feats")

class SigHead(nn.Module):
    def __init__(self, C=1024, n=234):
        super().__init__(); self.fc = nn.Linear(C, n)
    def forward(self, pool, spat): return self.fc(pool)

class ProtoHead(nn.Module):
    def __init__(self, C=1024, n=234, ppc=5):
        super().__init__()
        self.P = n * ppc
        self.proto = nn.Parameter(torch.randn(self.P, C, 1, 1) * 0.02)
        self.fc = nn.Linear(self.P, n)
    def forward(self, pool, spat):
        f = F.normalize(spat, dim=1); p = F.normalize(self.proto, dim=1)
        sim = F.relu(F.conv2d(f, p))
        return self.fc(sim.amax(dim=(2, 3)))

def macro_auc(probs, tgt):
    aucs = []
    for i in range(probs.shape[1]):
        y = tgt[:, i]
        if 0 < y.sum() < len(y):
            try: aucs.append(roc_auc_score(y, probs[:, i]))
            except ValueError: pass
    return float(np.mean(aucs)), len(aucs)

def train_one(head, seed, tr, va, args, dev):
    torch.manual_seed(seed); np.random.seed(seed)
    pool_tr, spat_tr, tgt_tr = tr
    pool_va, spat_va, tgt_va = va
    model = (ProtoHead(ppc=args.ppc) if head == "proto" else SigHead()).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    N = len(tgt_tr); idx = np.arange(N)
    for ep in range(args.epochs):
        model.train(); np.random.default_rng(seed * 1000 + ep).shuffle(idx)
        for s in range(0, N, args.batch):
            b = idx[s:s + args.batch]
            pl = torch.from_numpy(pool_tr[b]).float().to(dev)
            sp = torch.from_numpy(spat_tr[b]).float().to(dev)
            y = torch.from_numpy(tgt_tr[b]).float().to(dev)
            opt.zero_grad(); loss = lossf(model(pl, sp), y); loss.backward(); opt.step()
    model.eval(); out = []
    with torch.no_grad():
        for s in range(0, len(tgt_va), args.batch):
            pl = torch.from_numpy(pool_va[s:s + args.batch]).float().to(dev)
            sp = torch.from_numpy(spat_va[s:s + args.batch]).float().to(dev)
            out.append(torch.sigmoid(model(pl, sp)).cpu().numpy())
    return np.concatenate(out).astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--ppc", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--smoke", action="store_true")
    args, _ = ap.parse_known_args()
    if args.smoke:
        args.epochs, args.seeds = 2, [0]

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr = (np.load(FE/"train_pool.npy"), np.load(FE/"train_spat.npy"), np.load(FE/"train_tgt.npy").astype(np.float32))
    va = (np.load(FE/"val_pool.npy"), np.load(FE/"val_spat.npy"), np.load(FE/"val_tgt.npy").astype(np.float32))
    print(f"device={dev} train={tr[0].shape[0]} val={va[0].shape[0]} epochs={args.epochs} seeds={args.seeds}", flush=True)

    for head in ["sigmoid", "proto"]:
        for seed in args.seeds:
            probs = train_one(head, seed, tr, va, args, dev)
            m, ns = macro_auc(probs, va[2])
            tag = f"smoke_{head}_s{seed}" if args.smoke else f"oof_{head}_s{seed}"
            np.save(FE/f"{tag}.npy", probs)
            print(f"  {head:7s} seed{seed}: macroAUC={m:.4f} ({ns} sp)  saved {tag}.npy", flush=True)
    if not args.smoke:
        (FE/"DONE_heads.txt").write_text("done")
    print("P4 done.", flush=True)

if __name__ == "__main__":
    main()
