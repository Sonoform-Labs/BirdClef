import os, sys, argparse
os.environ.setdefault("HF_HOME", r"D:\hf_cache")
sys.path.insert(0, r"D:\Python\BirdClef")
import numpy as np, torch, torch.nn as nn, torchaudio
from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset

OUT = Path(r"D:\Python\BirdClef\paper_analysis\feats")
REPO = "DBD-research-group/ConvNeXT-Base-BirdSet-XCL"

class Mel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=32000, n_fft=1024, hop_length=320, n_mels=128,
            f_min=20, f_max=16000, power=2.0)
        self.db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
    def forward(self, w):
        m = self.db(self.mel(w))
        B = m.size(0); f = m.reshape(B, -1)
        mean = f.mean(1).reshape(B,1,1,1); std = f.std(1).clamp(min=1e-6).reshape(B,1,1,1)
        return (m - mean) / std

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    args, _ = ap.parse_known_args()
    OUT.mkdir(parents=True, exist_ok=True)

    from generate_oof import load_taxonomy, load_train_data, load_soundscape_data
    from src.dataset import FocalDataset, SoundscapeDataset, collate_fn

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev} smoke={args.smoke} fold={args.fold}", flush=True)
    label2idx, idx2label, taxon_map = load_taxonomy()
    train_df = load_train_data(label2idx)
    sc_meta, Y_sc = load_soundscape_data(label2idx)

    mel = Mel().to(dev).eval()
    from transformers import AutoModel
    bb = AutoModel.from_pretrained(REPO, trust_remote_code=True).to(dev).eval()
    for p in bb.parameters(): p.requires_grad = False

    def build(split):
        if split == "val":
            focal = train_df[train_df.fold == args.fold].reset_index(drop=True)
            mfold = sc_meta.fold.values == args.fold
            scm = sc_meta[mfold].reset_index(drop=True); Ys = Y_sc[mfold]
        else:
            focal = train_df[train_df.fold != args.fold].reset_index(drop=True)
            m = (sc_meta.fold.values != args.fold) & (Y_sc.sum(1) > 0)
            scm = sc_meta[m].reset_index(drop=True); Ys = Y_sc[m]
        return ConcatDataset([FocalDataset(focal, label2idx, aug=False),
                              SoundscapeDataset(scm, Ys, aug=False)]), len(focal), len(scm)

    workers = 0 if args.smoke else args.workers
    splits = ["val"] if args.smoke else ["val", "train"]
    for split in splits:
        ds, nf, ns = build(split)
        dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=workers,
                        collate_fn=collate_fn, pin_memory=True)
        pool, spat, tgt, srcs, n = [], [], [], [], 0
        for bi, (wavs, targets, _w, _m, sources) in enumerate(dl):
            wavs = wavs.to(dev, non_blocking=True)
            with torch.no_grad():
                m = mel(wavs)
                with torch.amp.autocast("cuda", enabled=(dev.type == "cuda")):
                    out = bb(pixel_values=m)
            pool.append(out.pooler_output.float().cpu().numpy().astype(np.float16))
            spat.append(out.last_hidden_state.float().cpu().numpy().astype(np.float16))
            tgt.append(targets.numpy().astype(np.float16)); srcs.extend(sources); n += len(sources)
            if args.smoke:
                print(f"  smoke batch {bi}: mel{tuple(m.shape)} pool{tuple(out.pooler_output.shape)} "
                      f"spat{tuple(out.last_hidden_state.shape)}", flush=True)
                if bi >= 1: break
            elif n % 1600 < args.batch:
                print(f"  {split}: {n} done", flush=True)
        pool = np.concatenate(pool); spat = np.concatenate(spat); tgt = np.concatenate(tgt)
        if args.smoke:
            print(f"SMOKE OK {split}: pool{pool.shape} spat{spat.shape} tgt{tgt.shape}", flush=True)
        else:
            np.save(OUT/f"{split}_pool.npy", pool); np.save(OUT/f"{split}_spat.npy", spat)
            np.save(OUT/f"{split}_tgt.npy", tgt); np.save(OUT/f"{split}_src.npy", np.array(srcs))
            print(f"SAVED {split}: pool{pool.shape} spat{spat.shape} tgt{tgt.shape} "
                  f"(focal {nf}, sc {ns})", flush=True)
    if not args.smoke:
        (OUT/f"DONE_extract_fold{args.fold}.txt").write_text("done")
    print("P3 done.", flush=True)

if __name__ == "__main__":
    main()
