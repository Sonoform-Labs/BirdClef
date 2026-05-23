import os, sys, argparse, time
os.environ.setdefault("HF_HOME", r"D:\hf_cache")
sys.path.insert(0, r"D:\Python\BirdClef"); sys.path.insert(0, r"D:\Python\BirdClef\paper_analysis")
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from collections import Counter
from torch.utils.data import DataLoader, ConcatDataset
from train_faithful import FaithfulModel, Mel
from src.dataset import FocalDataset, SoundscapeDataset, collate_fn
from generate_oof import load_taxonomy, load_soundscape_data

BASE = Path(r"D:\Python\BirdClef")
FOUT = BASE/"paper_analysis"/"joint_faithful"
RES = BASE/"GATE2_RESULTS.md"
TAXA = ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]
QUAR = {"Mammalia", "Reptilia"}
MIN_CLIPS = 5
MIN_SPECIES = 10
PCTS = [25, 33, 50]
rng = np.random.default_rng(42)

def build_pool(label2idx, per_class):
    tr = pd.read_csv(BASE/"data"/"train.csv"); tr["primary_label"] = tr.primary_label.astype(str)
    tr = tr[tr.primary_label.isin(label2idx)]
    foc_rows = []
    for sp, g in tr.groupby("primary_label"):
        if len(g) >= MIN_CLIPS:
            foc_rows.append(g.sample(n=min(per_class, len(g)), random_state=42))
    foc = pd.concat(foc_rows).reset_index(drop=True) if foc_rows else tr.iloc[:0]

    scm, Ysc = load_soundscape_data(label2idx)
    single_mask = (Ysc.sum(1) == 1)
    scs = scm[single_mask].reset_index(drop=True); Ys = Ysc[single_mask]
    sccls = Ys.argmax(1)
    keep = []
    for c in np.unique(sccls):
        idx = np.where(sccls == c)[0]
        if len(idx) >= MIN_CLIPS:
            keep.extend(rng.choice(idx, size=min(per_class, len(idx)), replace=False).tolist())
    scs = scs.iloc[keep].reset_index(drop=True); Ys = Ys[keep]
    fds = FocalDataset(foc, label2idx, aug=False) if len(foc) else None
    sds = SoundscapeDataset(scs, Ys, aug=False) if len(scs) else None
    ds = ConcatDataset([d for d in [fds, sds] if d is not None])

    foc_cls = [label2idx[s] for s in foc.primary_label] if len(foc) else []
    sc_cls = sccls[keep].tolist() if len(scs) else []
    pool_cls = np.array(foc_cls + sc_cls)
    return ds, pool_cls

def run():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); args, _ = ap.parse_known_args()
    per_class = 6 if args.smoke else 15
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    l2i, i2l, tm = load_taxonomy()
    species = np.array([i2l[i] for i in range(len(i2l))]); cls_of = np.array([tm.get(s, "?") for s in species])
    pci = (torch.arange(1170) // 5).numpy()
    ptax = np.array([cls_of[c] for c in pci])

    ds, pool_cls = build_pool(l2i, per_class)
    print(f"pool: {len(pool_cls)} clips; per-taxon clips={dict(Counter(cls_of[c] for c in pool_cls))}", flush=True)
    covered = Counter(cls_of[c] for c in np.unique(pool_cls))
    cov_species = Counter()
    for c in np.unique(pool_cls):
        cov_species[cls_of[c]] += 1
    print(f"covered species/taxon: {dict(cov_species)}", flush=True)

    melt = Mel().to(dev).eval()
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4, collate_fn=collate_fn, pin_memory=True)
    folds = [k for k in range(5) if (FOUT/f"model_faithful_proto_fold{k}.pt").exists()]
    if args.smoke: folds = folds[:1]
    print(f"fold-models: {folds}", flush=True)


    per_fold = []
    for k in folds:
        model = FaithfulModel("proto").to(dev).eval()
        sd = torch.load(FOUT/f"model_faithful_proto_fold{k}.pt", map_location=dev, weights_only=False)
        model.load_state_dict(sd);

        E, C, EP = [], [], []
        ic = 0
        with torch.no_grad():
            for wavs, _t, _w, _m, _s in loader:
                bs = wavs.size(0); wavs = wavs.to(dev, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    mel = melt(wavs, training=False)
                    feat = model.backbone(pixel_values=mel).last_hidden_state
                Hh, Ww = feat.shape[2], feat.shape[3]
                cell_e = F.adaptive_avg_pool2d(mel.float(), (Hh, Ww)).reshape(bs, Hh*Ww)

                order = cell_e.argsort(1).argsort(1).float() / (Hh*Ww - 1) * 100
                fe = feat.float().reshape(bs, 1024, Hh*Ww).permute(0, 2, 1).reshape(-1, 1024)
                E.append(fe.cpu()); EP.append(order.reshape(-1).cpu())
                C.append(np.repeat(pool_cls[ic:ic+bs], Hh*Ww)); ic += bs
        E = torch.cat(E); EP = torch.cat(EP).numpy(); C = np.concatenate(C)
        En = F.normalize(E, dim=1).to(dev)
        def project(protos):
            pn = F.normalize(protos.reshape(protos.shape[0], -1), dim=1).to(dev)
            near_cls = np.empty(1170, dtype=int); near_ep = np.empty(1170)
            for s in range(0, 1170, 130):
                sim = pn[s:s+130] @ En.T
                idx = sim.argmax(1).cpu().numpy()
                near_cls[s:s+130] = C[idx]; near_ep[s:s+130] = EP[idx]
            return near_cls, near_ep
        ncls, nep = project(model.head.prototype_vectors.detach())
        rcls, rep = project(torch.randn(1170, 1024, 1, 1))
        per_fold.append({"pur": (ncls == pci), "ep": nep, "pur_r": (rcls == pci), "ep_r": rep})
        pe = float((ncls == pci).mean())
        print(f"  fold{k}: overall purity={pe:.3f} (rand {float((rcls==pci).mean()):.3f})  "
              f"shortcut@33={float((nep<33).mean()):.3f}", flush=True)
        del model, En, E; torch.cuda.empty_cache()


    def agg(key_obs, mask):
        vals = [pf[key_obs][mask].mean() for pf in per_fold if mask.sum() > 0]
        return (float(np.mean(vals)), float(np.std(vals, ddof=1)/np.sqrt(len(vals))) if len(vals) > 1 else float("nan"))
    lines = ["", "## GATE 2 - PROTOTYPE VALIDITY (faithful models; inference-only; per GATE2_PREREG.md)",
             f"pool {len(pool_cls)} clips; folds {folds}; cross-fold mean±SE.",
             f"{'scope':10s} {'purity':>14s} {'rand':>6s} | shortcut@25/33/50 (null .25/.33/.50)  cover/flag"]
    scopes = [("ALL", np.ones(1170, bool))] + [(t, ptax == t) for t in TAXA]
    for name, m in scopes:
        if m.sum() == 0: continue
        pur_m, pur_se = agg("pur", m); rnd_m, _ = agg("pur_r", m)
        sc = []
        for p in PCTS:
            vals = [(pf["ep"][m] < p).mean() for pf in per_fold]
            sc.append(float(np.mean(vals)))
        ncov = cov_species.get(name, 0) if name != "ALL" else sum(cov_species.values())
        flag = ""
        if name in QUAR: flag = "QUARANTINE(n)"
        elif name != "ALL" and ncov < MIN_SPECIES: flag = f"PURITY coverage-limited ({ncov}sp)"
        lines.append(f"{name:10s} {pur_m:>7.3f}±{pur_se:<6.3f} {rnd_m:>6.3f} | "
                     f"{sc[0]:.3f}/{sc[1]:.3f}/{sc[2]:.3f}                 {flag}")
    out = "\n".join(lines); print("\n"+out, flush=True)
    if not args.smoke:
        try:
            prev = RES.read_text(encoding="utf-8") if RES.exists() else "# Gate-2 results\n"
            RES.write_text(prev + "\n" + out + "\n", encoding="utf-8")
        except Exception as e: print("append failed:", e)
        (FOUT/"GATE2_DONE.txt").write_text("done")
    print("gate2 done.", flush=True)

if __name__ == "__main__":
    run()
