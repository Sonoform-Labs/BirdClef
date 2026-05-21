import os, sys, argparse, time
os.environ.setdefault("HF_HOME", r"D:\hf_cache")
sys.path.insert(0, r"D:\Python\BirdClef")
sys.path.insert(0, r"D:\Python\BirdClef\paper_analysis")
import numpy as np, torch, torch.nn as nn
from pathlib import Path
from faithful_head import AudioProtoNetClassificationHead, AsymmetricLossMultiLabel, Cfg
from train_joint import build_loaders, Mel, per_species_auc, taxon_means, TAXA

FOUT = Path(r"D:\Python\BirdClef\paper_analysis\joint_faithful")
LOG = Path(r"D:\Python\BirdClef\OVERNIGHT_LOG.md")
REPO = "DBD-research-group/ConvNeXT-Base-BirdSet-XCL"
QUAR = {"Reptilia", "Mammalia"}
AMP = torch.bfloat16
COEF = {"ce": 1.0, "cluster": -0.25, "sep": 0.05, "orth": 1.0}

class FaithfulModel(nn.Module):
    def __init__(self, head_type, n=234, ppc=5):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(REPO, trust_remote_code=True)
        self.head_type = head_type
        if head_type == "proto":
            self.head = AudioProtoNetClassificationHead(Cfg(num_classes=n, prototypes_per_class=ppc, channels=1024))
            self.head.set_last_layer_incorrect_connection(None)
        else:
            self.fc = nn.Linear(1024, n)
    def forward(self, mel):
        out = self.backbone(pixel_values=mel)
        if self.head_type == "proto":
            return self.head(out.last_hidden_state)
        return self.fc(out.pooler_output), None

def proto_cluster_sep(info, labels, pci, num_classes):
    max_acts = info[0].float()
    labels = labels.float()
    poc = labels[:, pci]
    powc = 1.0 - poc
    cca = max_acts * poc
    relevant = poc > 0
    cidx = pci[None, :].expand(max_acts.size(0), -1)
    mapc = torch.full((max_acts.size(0), num_classes), -float("inf"), device=max_acts.device)
    mapc.scatter_reduce_(1, cidx, cca, reduce="amax", include_self=False)
    filt = torch.where(relevant, mapc.gather(1, cidx), torch.tensor(float("inf"), device=max_acts.device))
    minrel, _ = torch.min(filt, dim=1)
    fin = minrel[torch.isfinite(minrel)]
    cluster = fin.mean() if fin.numel() > 0 else torch.zeros((), device=max_acts.device)
    sep = torch.max(max_acts * powc, dim=1)[0].mean()
    return cluster, sep

def run_head(head, args, trl, val, species, cls, dev, fold):
    model = FaithfulModel(head).to(dev)
    melt = Mel().to(dev)
    pci = model.head.prototype_class_identity.to(dev) if head == "proto" else None
    bb = [p for n, p in model.named_parameters() if n.startswith("backbone.")]
    hd = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    opt = torch.optim.AdamW([{"params": bb, "lr": args.lr_bb}, {"params": hd, "lr": args.lr_head}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr_bb, args.lr_head],
                total_steps=max(len(trl) * args.epochs, 1), pct_start=0.1)
    asym = AsymmetricLossMultiLabel(reduction="none")
    best, best_probs, valtgt = -1.0, None, None
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); run = 0.0; nb = 0
        ce_s = clu_s = sep_s = orth_s = 0.0
        for wavs, targets, _w, _m, _s in trl:
            wavs = wavs.to(dev, non_blocking=True); y = targets.to(dev, non_blocking=True)
            with torch.no_grad(): mel = melt(wavs, training=True)
            opt.zero_grad()
            with torch.amp.autocast("cuda", dtype=AMP):
                logits, info = model(mel)
            logits = logits.float()
            ce = asym(logits, y).sum(dim=1).mean() * args.ce_scale; loss = COEF["ce"] * ce; ce_s += ce.item()
            if head == "proto":
                cluster, sep = proto_cluster_sep(info, y, pci, 234)
                orth = model.head.get_prototype_orthogonalities()
                orthl = torch.norm(orth) / orth.numel()
                loss = loss + COEF["cluster"] * cluster + COEF["sep"] * sep + COEF["orth"] * orthl
                clu_s += cluster.item(); sep_s += sep.item(); orth_s += orthl.item()
            loss.backward(); opt.step(); sched.step()
            run += loss.item(); nb += 1
            if args.smoke and nb >= 3: break
        model.eval(); probs, tg = [], []
        with torch.no_grad():
            for wavs, targets, _w, _m, _s in val:
                wavs = wavs.to(dev, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=AMP):
                    logits, _ = model(melt(wavs, training=False))
                probs.append(torch.sigmoid(logits.float()).cpu().numpy()); tg.append(targets.numpy())
        probs = np.concatenate(probs); valtgt = np.concatenate(tg)
        nanr = int(np.isnan(probs).any(1).sum())
        macro, tax = taxon_means(per_species_auc(probs, valtgt), cls)
        comp = (f"[CE={ce_s/max(nb,1):.4f} clu={clu_s/max(nb,1):.3f} sep={sep_s/max(nb,1):.3f} "
                f"orth={orth_s/max(nb,1):.4f}] " if head == "proto" else "")
        print(f"  [f{fold} {head}] ep{ep+1}/{args.epochs} loss={run/max(nb,1):.4f} valAUC={macro:.4f} "
              f"nan={nanr} {time.time()-t0:.0f}s {comp}" + " ".join(f"{k[:4]}:{v:.3f}" for k, v in tax.items()), flush=True)
        if macro > best:
            best, best_probs = macro, probs
            if not args.smoke:
                FOUT.mkdir(parents=True, exist_ok=True); sfx = args.save_suffix
                np.save(FOUT/f"oof_faithful_{head}_fold{fold}{sfx}.npy", best_probs)
                np.save(FOUT/f"val_tgt_fold{fold}{sfx}.npy", valtgt)
                torch.save(model.state_dict(), FOUT/f"model_faithful_{head}_fold{fold}{sfx}.pt")
        if head == "proto" and ep == 0 and not args.smoke and macro < args.proto_ep1_min:
            print(f"  [f{fold} proto] ABORT @ep1 valAUC={macro:.4f} < {args.proto_ep1_min} (CE-scale not learning)", flush=True)
            break
        if args.smoke: break
    del model; torch.cuda.empty_cache()
    print(f"  [f{fold} {head}] BEST valAUC={best:.4f}", flush=True)
    return best

def summarize_fold(fold, species, cls):
    try:
        ps = np.load(FOUT/f"oof_faithful_sigmoid_fold{fold}.npy")
        pp = np.load(FOUT/f"oof_faithful_proto_fold{fold}.npy")
        tg = np.load(FOUT/f"val_tgt_fold{fold}.npy").astype(np.float32)
    except Exception as e:
        return
    a_s = per_species_auc(ps, tg); a_p = per_species_auc(pp, tg)
    ms, txs = taxon_means(a_s, cls); mp, txp = taxon_means(a_p, cls)
    lines = [f"\n### FAITHFUL fold {fold} (proto vs sigmoid)  delta=proto-sigmoid",
             f"  MACRO  proto={mp:.4f} sigmoid={ms:.4f} delta={mp-ms:+.4f}"]
    for tx in TAXA:
        if tx in txp and tx in txs:
            q = "  [QUARANTINE]" if tx in QUAR else ""
            lines.append(f"  {tx:9s} proto={txp[tx]:.4f} sigmoid={txs[tx]:.4f} delta={txp[tx]-txs[tx]:+.4f}{q}")
    try:
        LOG.write_text(LOG.read_text(encoding="utf-8") + "\n".join(lines) + "\n", encoding="utf-8")
    except Exception as e:
        print("log append failed:", e)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--heads", nargs="+", default=["sigmoid", "proto"])
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr-bb", type=float, default=1e-4)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--save-suffix", type=str, default="")
    ap.add_argument("--ce-only", action="store_true")
    ap.add_argument("--ce-scale", type=float, default=1.0)
    ap.add_argument("--proto-ep1-min", type=float, default=0.0)
    ap.add_argument("--smoke", action="store_true")
    args, _ = ap.parse_known_args()
    if args.ce_only:
        COEF["cluster"] = 0.0; COEF["sep"] = 0.0; COEF["orth"] = 0.0
        print("CE-ONLY diagnostic: cluster/sep/orth coefs zeroed", flush=True)
    if args.seed is not None:
        import random as _r; torch.manual_seed(args.seed); np.random.seed(args.seed); _r.seed(args.seed)
    if args.smoke: args.epochs, args.folds = 1, [0]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev} dtype={AMP} folds={args.folds} heads={args.heads} epochs={args.epochs} "
          f"batch={args.batch} coefs={COEF} smoke={args.smoke}", flush=True)
    for fold in args.folds:
        trl, val, species, cls = build_loaders(args, fold)
        print(f"[fold {fold}] train_batches={len(trl)} val_batches={len(val)}", flush=True)
        for head in args.heads:
            run_head(head, args, trl, val, species, cls, dev, fold)
        if not args.smoke:
            summarize_fold(fold, species, cls)
            FOUT.mkdir(parents=True, exist_ok=True); (FOUT/f"DONE_fold{fold}{args.save_suffix}.txt").write_text("done")
    if not args.smoke:
        (FOUT/"DONE_faithful.txt").write_text("done")
    print("FAITHFUL done.", flush=True)

if __name__ == "__main__":
    main()
