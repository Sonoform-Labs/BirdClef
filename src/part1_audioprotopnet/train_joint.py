import os, sys, argparse, time
os.environ.setdefault("HF_HOME", r"D:\hf_cache")
sys.path.insert(0, r"D:\Python\BirdClef")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torchaudio
from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import roc_auc_score

OUT = Path(r"D:\Python\BirdClef\paper_analysis\joint")
LOG = Path(r"D:\Python\BirdClef\OVERNIGHT_LOG.md")
REPO = "DBD-research-group/ConvNeXT-Base-BirdSet-XCL"
TAXA = ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]
QUARANTINE = {"Reptilia", "Mammalia"}
AMP_DTYPE = torch.bfloat16

class Mel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=32000, n_fft=1024,
                    hop_length=320, n_mels=128, f_min=20, f_max=16000, power=2.0)
        self.db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
        self.fm = torchaudio.transforms.FrequencyMasking(16)
        self.tm = torchaudio.transforms.TimeMasking(32)
    def forward(self, w, training=False):
        m = self.db(self.mel(w))
        B = m.size(0); f = m.reshape(B, -1)
        mean = f.mean(1).reshape(B,1,1,1); std = f.std(1).clamp(min=1e-6).reshape(B,1,1,1)
        m = (m - mean) / std
        return self.tm(self.fm(m)) if training else m

class JointModel(nn.Module):
    def __init__(self, head="proto", n=234, ppc=5):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(REPO, trust_remote_code=True)
        self.head_type = head; C = 1024
        if head == "proto":
            self.P = n * ppc
            self.proto = nn.Parameter(torch.randn(self.P, C, 1, 1) * 0.02)
            self.fc = nn.Linear(self.P, n)
        else:
            self.fc = nn.Linear(C, n)
    def forward(self, mel):
        out = self.backbone(pixel_values=mel)
        if self.head_type == "proto":
            f = F.normalize(out.last_hidden_state, dim=1); p = F.normalize(self.proto, dim=1)
            return self.fc(F.relu(F.conv2d(f, p)).amax(dim=(2, 3)))
        return self.fc(out.pooler_output)

def per_species_auc(probs, tgt):
    a = np.full(probs.shape[1], np.nan)
    for i in range(probs.shape[1]):
        y = tgt[:, i]; p = probs[:, i]; m = np.isfinite(p)
        yy, pp = y[m], p[m]
        if 0 < yy.sum() < len(yy):
            try: a[i] = roc_auc_score(yy, pp)
            except ValueError: pass
    return a

def taxon_means(a, cls):
    macro = float(np.nanmean(a)) if np.any(~np.isnan(a)) else float("nan")
    return macro, {tx: float(np.nanmean(a[cls == tx])) for tx in TAXA if np.any((cls == tx) & ~np.isnan(a))}

def build_loaders(args, fold):
    from generate_oof import load_taxonomy, load_train_data, load_soundscape_data
    from src.dataset import FocalDataset, SoundscapeDataset, collate_fn
    l2i, i2l, tm = load_taxonomy()
    species = np.array([i2l[i] for i in range(len(i2l))])
    cls = np.array([tm.get(s, "?") for s in species])
    tdf = load_train_data(l2i); scm, Ysc = load_soundscape_data(l2i)
    ftr = tdf[tdf.fold != fold].reset_index(drop=True)
    m = (scm.fold.values != fold) & (Ysc.sum(1) > 0)
    tr = ConcatDataset([FocalDataset(ftr, l2i, aug=True),
                        SoundscapeDataset(scm[m].reset_index(drop=True), Ysc[m], aug=True)])
    fva = tdf[tdf.fold == fold].reset_index(drop=True)
    mv = scm.fold.values == fold
    va = ConcatDataset([FocalDataset(fva, l2i, aug=False),
                        SoundscapeDataset(scm[mv].reset_index(drop=True), Ysc[mv], aug=False)])
    trl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                     collate_fn=collate_fn, pin_memory=True, drop_last=True)
    val = DataLoader(va, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                     collate_fn=collate_fn, pin_memory=True)
    return trl, val, species, cls

def run_head(head, args, trl, val, species, cls, dev, fold):
    model = JointModel(head=head, ppc=args.ppc).to(dev)
    melt = Mel().to(dev)
    bb = [p for n, p in model.named_parameters() if n.startswith("backbone.")]
    hd = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    opt = torch.optim.AdamW([{"params": bb, "lr": args.lr_bb}, {"params": hd, "lr": args.lr_head}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr_bb, args.lr_head],
                total_steps=max(len(trl) * args.epochs, 1), pct_start=0.1)
    lossf = nn.BCEWithLogitsLoss()
    best, best_probs, valtgt = -1.0, None, None
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); run = 0.0; nb = 0
        for wavs, targets, _w, _m, _s in trl:
            wavs = wavs.to(dev, non_blocking=True); y = targets.to(dev, non_blocking=True)
            with torch.no_grad(): mel = melt(wavs, training=True)
            opt.zero_grad()
            with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
                loss = lossf(model(mel), y)
            loss.backward(); opt.step(); sched.step()
            run += loss.item(); nb += 1
            if args.smoke and nb >= 3: break
        model.eval(); probs, tg = [], []
        with torch.no_grad():
            for wavs, targets, _w, _m, _s in val:
                wavs = wavs.to(dev, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
                    out = model(melt(wavs, training=False))
                probs.append(torch.sigmoid(out.float()).cpu().numpy()); tg.append(targets.numpy())
        probs = np.concatenate(probs); valtgt = np.concatenate(tg)
        nanr = int(np.isnan(probs).any(1).sum())
        macro, tax = taxon_means(per_species_auc(probs, valtgt), cls)
        print(f"  [f{fold} {head}] ep{ep+1}/{args.epochs} loss={run/max(nb,1):.4f} valAUC={macro:.4f} "
              f"nan_rows={nanr} {time.time()-t0:.0f}s " + " ".join(f"{k[:4]}:{v:.3f}" for k, v in tax.items()), flush=True)
        if macro > best:
            best, best_probs = macro, probs
            if not args.smoke:
                OUT.mkdir(parents=True, exist_ok=True)
                sfx = args.save_suffix
                np.save(OUT/f"oof_joint_{head}_fold{fold}{sfx}.npy", best_probs)
                np.save(OUT/f"val_tgt_fold{fold}{sfx}.npy", valtgt)
                torch.save(model.state_dict(), OUT/f"model_{head}_fold{fold}{sfx}.pt")
        if args.smoke: break
    del model; torch.cuda.empty_cache()
    print(f"  [f{fold} {head}] BEST valAUC={best:.4f}", flush=True)
    return best

def append_log(text):
    try:
        prev = LOG.read_text(encoding="utf-8") if LOG.exists() else ""
        LOG.write_text(prev + text, encoding="utf-8")
    except Exception as e:
        print("log append failed:", e, flush=True)

def summarize_fold(fold, species, cls):
    try:
        ps = np.load(OUT/f"oof_joint_sigmoid_fold{fold}.npy")
        pp = np.load(OUT/f"oof_joint_proto_fold{fold}.npy")
        tg = np.load(OUT/f"val_tgt_fold{fold}.npy").astype(np.float32)
    except Exception as e:
        append_log(f"\n  [fold {fold}] summary skipped: {e}\n"); return
    a_s = per_species_auc(ps, tg); a_p = per_species_auc(pp, tg)
    ms, txs = taxon_means(a_s, cls); mp, txp = taxon_means(a_p, cls)
    lines = [f"\n### joint fold {fold} (proto vs sigmoid)  delta=proto-sigmoid",
             f"  MACRO  proto={mp:.4f} sigmoid={ms:.4f} delta={mp-ms:+.4f}"]
    for tx in TAXA:
        if tx in txp and tx in txs:
            q = "  [QUARANTINE n<=8]" if tx in QUARANTINE else ""
            lines.append(f"  {tx:9s} proto={txp[tx]:.4f} sigmoid={txs[tx]:.4f} delta={txp[tx]-txs[tx]:+.4f}{q}")
    append_log("\n".join(lines) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--heads", nargs="+", default=["sigmoid", "proto"])
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr-bb", type=float, default=1e-4)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--ppc", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--save-suffix", type=str, default="")
    ap.add_argument("--smoke", action="store_true")
    args, _ = ap.parse_known_args()
    if args.seed is not None:
        import random as _r
        torch.manual_seed(args.seed); np.random.seed(args.seed); _r.seed(args.seed)
    if args.smoke: args.epochs, args.folds = 1, [0]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev} dtype={AMP_DTYPE} folds={args.folds} heads={args.heads} "
          f"epochs={args.epochs} batch={args.batch} smoke={args.smoke}", flush=True)
    if not args.smoke:
        append_log(f"\n## P6 5-FOLD JOINT RUN START (bf16, epochs={args.epochs} batch={args.batch}, folds={args.folds})\n")
    for fold in args.folds:
        trl, val, species, cls = build_loaders(args, fold)
        print(f"[fold {fold}] train_batches={len(trl)} val_batches={len(val)}", flush=True)
        for head in args.heads:
            run_head(head, args, trl, val, species, cls, dev, fold)
        if not args.smoke:
            if not args.save_suffix:
                summarize_fold(fold, species, cls)
            (OUT/f"DONE_fold{fold}{args.save_suffix}.txt").write_text("done")
    if not args.smoke:
        (OUT/"DONE_joint.txt").write_text("done")
    print("P6 done.", flush=True)

if __name__ == "__main__":
    main()
