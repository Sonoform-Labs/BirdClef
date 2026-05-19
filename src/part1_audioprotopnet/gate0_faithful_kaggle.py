import argparse, os, time, tempfile, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HOME", r"D:\hf_cache")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

REPO = "DBD-research-group/ConvNeXT-Base-BirdSet-XCL"
SR, NFFT, HOP, NMELS, WIN_S = 32000, 1024, 320, 128, 5
torch.manual_seed(0)

class Mel(nn.Module):
    def __init__(self):
        super().__init__()
        import torchaudio
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=SR, n_fft=NFFT, hop_length=HOP,
                    n_mels=NMELS, f_min=20, f_max=16000, power=2.0)
        self.db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
    def forward(self, w):
        m = self.db(self.mel(w))
        B = m.size(0); f = m.reshape(B, -1)
        mean = f.mean(1).reshape(B, 1, 1, 1); std = f.std(1).clamp(min=1e-6).reshape(B, 1, 1, 1)
        return (m - mean) / std

class ProtoHead(nn.Module):
    def __init__(self, C=1024, n_class=234, ppc=5):
        super().__init__()
        self.proto = nn.Parameter(torch.randn(n_class * ppc, C, 1, 1) * 0.02)
        self.last = nn.Linear(n_class * ppc, n_class)
    def forward(self, feat):
        f = F.normalize(feat, dim=1); p = F.normalize(self.proto, dim=1)
        sim = F.relu(F.conv2d(f, p)).amax(dim=(2, 3))
        return torch.sigmoid(self.last(sim))

class Faithful(nn.Module):
    def __init__(self, backbone):
        super().__init__(); self.bb = backbone; self.head = ProtoHead()
    def forward(self, mel):
        return self.head(self.bb(pixel_values=mel).last_hidden_state)

def build_model():
    from transformers import AutoModel
    bb = AutoModel.from_pretrained(REPO, trust_remote_code=True)
    m = Faithful(bb).eval()
    npar = sum(p.numel() for p in m.parameters())
    return m, npar

def _per_win(ts, batch):
    return float(np.median(np.array(ts)) / batch)

def time_torch_full(model, melt, wav, iters, warmup):
    batch = wav.size(0)
    with torch.no_grad():
        for _ in range(warmup): model(melt(wav))
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter(); model(melt(wav)); ts.append(time.perf_counter() - t0)
    return _per_win(ts, batch)

def time_mel(melt, wav, iters, warmup):
    batch = wav.size(0)
    with torch.no_grad():
        for _ in range(warmup): melt(wav)
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter(); melt(wav); ts.append(time.perf_counter() - t0)
    return _per_win(ts, batch)

def export_onnx(model, mel, path):
    with torch.no_grad():
        torch.onnx.export(model, mel, path, input_names=["mel"], output_names=["y"],
                          opset_version=17, dynamo=False,
                          dynamic_axes={"mel": {0: "b"}, "y": {0: "b"}})

def time_ort(path, mel_np, iters, warmup):
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = os.cpu_count() or 4
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    for _ in range(warmup): sess.run(None, {"mel": mel_np})
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter(); sess.run(None, {"mel": mel_np}); ts.append(time.perf_counter() - t0)
    return _per_win(ts, mel_np.shape[0])

def time_ov(path, mel_np, iters, warmup):
    import openvino as ov
    cm = ov.Core().compile_model(ov.Core().read_model(path), "CPU")
    for _ in range(warmup): cm(mel_np)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter(); cm(mel_np); ts.append(time.perf_counter() - t0)
    return _per_win(ts, mel_np.shape[0])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--threshold-min", type=float, default=70.0)
    args, _ = ap.parse_known_args()
    if args.smoke:
        args.batch, args.iters, args.warmup = 2, 3, 1

    torch.set_num_threads(os.cpu_count() or 4)
    on_kaggle = os.path.exists("/kaggle")
    print(f"env: cpu_count={os.cpu_count()} torch_threads={torch.get_num_threads()} kaggle={on_kaggle} "
          f"batch={args.batch} iters={args.iters} smoke={args.smoke}")
    if not on_kaggle:
        print("WARNING: not on Kaggle -> these timings are NOT the Gate-0 numbers (different CPU).")

    have = {"onnxruntime": False, "openvino": False}
    for k in have:
        try: __import__(k); have[k] = True
        except Exception: pass
    print(f"engines: torch=True onnxruntime={have['onnxruntime']} openvino={have['openvino']}")

    print("building real BirdSet ConvNeXt-Base + prototype head ...", flush=True)
    model, npar = build_model()
    melt = Mel().eval()
    wav = torch.randn(args.batch, 1, SR * WIN_S)
    with torch.no_grad():
        mel = melt(wav)
    print(f"params={npar/1e6:.1f}M   mel shape={tuple(mel.shape)}   (5s @ {SR}Hz)\n", flush=True)

    mel_pw = time_mel(melt, wav, args.iters, args.warmup)
    NWIN = {"600f": 600 * 12, "700f": 700 * 12}
    header = (f"{'engine':>6s} {'model ms/win':>12s} {'+mel ms/win':>12s} "
              f"{'min@600f':>9s} {'min@700f':>9s} {'verdict':>8s}")
    print(f"mel-extraction overhead: {mel_pw*1000:.2f} ms/window (added to every engine)\n")
    print(header); print("-" * len(header))

    rows = [("torch", lambda: time_torch_full(model, melt, wav, args.iters, args.warmup) - mel_pw)]
    onnx_path = os.path.join(tempfile.mkdtemp(), "faithful.onnx"); onnx_ok = False
    try:
        export_onnx(model, mel, onnx_path); onnx_ok = True
    except Exception as e:
        print(f"(onnx export failed: {type(e).__name__}: {e} -> report torch row)")
    mel_np = mel.detach().numpy().astype(np.float32)
    if onnx_ok and have["onnxruntime"]:
        rows.append(("ort", lambda: time_ort(onnx_path, mel_np, args.iters, args.warmup)))
    if onnx_ok and have["openvino"]:
        rows.append(("ov", lambda: time_ov(onnx_path, mel_np, args.iters, args.warmup)))

    for ename, fn in rows:
        try:
            model_pw = fn(); tot = model_pw + mel_pw
            row = f"{ename:>6s} {model_pw*1000:12.2f} {tot*1000:12.2f}"
            for key in ("600f", "700f"):
                row += f" {tot*NWIN[key]/60.0:9.1f}"
            verdict = "PASS" if (tot * NWIN["700f"] / 60.0) <= args.threshold_min else "FAIL"
            print(row + f" {verdict:>8s}")
        except Exception as e:
            print(f"{ename:>6s}  ERR {type(e).__name__}: {e}")

    print(f"\nPASS/FAIL uses (model+mel) x 8400 windows / 60 <= {args.threshold_min:.0f} min. "
          f"batch={args.batch}, all CPU threads.")
    print("Report the TORCH row as the conservative Gate-0 number; ort/ov only improve it.")
    if not on_kaggle:
        print("These are NOT the Kaggle figures - re-run on a Kaggle CPU notebook for the paper number.")

if __name__ == "__main__":
    main()
