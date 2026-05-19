import argparse, time, os, sys, tempfile, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

torch.manual_seed(0)


class ProtoNet(nn.Module):
    def __init__(self, backbone, C, n_class=234, ppc=5):
        super().__init__()
        self.backbone = backbone
        self.P = n_class * ppc
        self.prototypes = nn.Parameter(torch.randn(self.P, C, 1, 1) * 0.02)
        self.last = nn.Linear(self.P, n_class)
    def forward(self, x):
        feat = self.backbone.forward_features(x)
        f = F.normalize(feat, dim=1)
        p = F.normalize(self.prototypes, dim=1)
        sim = F.relu(F.conv2d(f, p))
        pooled = sim.amax(dim=(2, 3))
        return torch.sigmoid(self.last(pooled))

def build_model(backbone_name, n_class=234, ppc=5):
    import timm
    bb = timm.create_model(backbone_name, pretrained=False, in_chans=1, num_classes=0)
    bb.eval()
    with torch.no_grad():
        feat = bb.forward_features(torch.zeros(1, 1, 128, 128))
    assert feat.dim() == 4, f"{backbone_name} forward_features is {feat.shape}, expected 4D"
    C = feat.shape[1]
    m = ProtoNet(bb, C, n_class, ppc).eval()
    n_params = sum(p.numel() for p in m.parameters())
    return m, C, n_params


def _stats(ts, batch):
    ts = np.array(ts)
    return float(np.median(ts) / batch)

def time_torch(model, shape, batch, iters, warmup):
    x = torch.randn(batch, *shape)
    with torch.no_grad():
        for _ in range(warmup): model(x)
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter(); model(x); ts.append(time.perf_counter() - t0)
    return _stats(ts, batch)

def export_onnx(model, shape, batch, path):
    x = torch.randn(batch, *shape)
    with torch.no_grad():
        torch.onnx.export(model, x, path, input_names=["x"], output_names=["y"],
                          opset_version=17, dynamo=False)

def time_ort(path, shape, batch, iters, warmup):
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = os.cpu_count() or 4
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    x = np.random.randn(batch, *shape).astype(np.float32)
    for _ in range(warmup): sess.run(None, {"x": x})
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter(); sess.run(None, {"x": x}); ts.append(time.perf_counter() - t0)
    return _stats(ts, batch)

def time_ov(path, shape, batch, iters, warmup):
    import openvino as ov
    core = ov.Core()
    cm = core.compile_model(core.read_model(path), "CPU")
    x = np.random.randn(batch, *shape).astype(np.float32)
    for _ in range(warmup): cm(x)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter(); cm(x); ts.append(time.perf_counter() - t0)
    return _stats(ts, batch)

def time_mel(batch, iters, warmup, n_mels=256, n_fft=2048, hop=256):
    import torchaudio
    mel = torchaudio.transforms.MelSpectrogram(sample_rate=32000, n_fft=n_fft,
                                               hop_length=hop, n_mels=n_mels)
    a2db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
    wav = torch.randn(batch, 32000 * 5)
    with torch.no_grad():
        for _ in range(warmup): a2db(mel(wav))
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter(); a2db(mel(wav)); ts.append(time.perf_counter() - t0)
    return _stats(ts, batch)


FULL_CONFIGS = [
    ("tiny  / 224sq ", "convnext_tiny",  224, 224),
    ("tiny  / 128mel", "convnext_tiny",  128, 500),
    ("tiny  / 256mel", "convnext_tiny",  256, 625),
    ("nano  / 256mel", "convnext_nano",  256, 625),
    ("small / 256mel", "convnext_small", 256, 625),
    ("base  / 224sq ", "convnext_base",  224, 224),
    ("base  / 128mel", "convnext_base",  128, 500),
    ("base  / 256mel", "convnext_base",  256, 625),
]
SMOKE_CONFIGS = [
    ("nano  / 128mel", "convnext_nano",  128, 256),
    ("tiny  / 128mel", "convnext_tiny",  128, 256),
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny local correctness run")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--threshold-min", type=float, default=70.0)
    args, _ = ap.parse_known_args()

    if args.smoke:
        args.batch, args.iters, args.warmup = 2, 3, 1

    torch.set_num_threads(os.cpu_count() or 4)
    on_kaggle = os.path.exists("/kaggle")
    print(f"env: cpu_count={os.cpu_count()} torch_threads={torch.get_num_threads()} "
          f"kaggle={on_kaggle} batch={args.batch} iters={args.iters} smoke={args.smoke}")
    if not args.smoke and not on_kaggle:
        print("WARNING: not on Kaggle -> these timings are NOT the Gate-0 numbers (different CPU).")

    have = {"onnxruntime": False, "openvino": False}
    for k in have:
        try: __import__(k); have[k] = True
        except Exception: pass
    print(f"engines available: torch=True onnxruntime={have['onnxruntime']} openvino={have['openvino']}")

    mel_pw = time_mel(args.batch, args.iters, args.warmup)
    print(f"mel-extraction overhead: {mel_pw*1000:.2f} ms/window (added to every model time)\n")

    configs = SMOKE_CONFIGS if args.smoke else FULL_CONFIGS
    NWIN = {"600f": 600 * 12, "700f": 700 * 12}

    header = (f"{'config':16s} {'params':>8s} {'engine':>5s} {'ms/win':>8s} "
              f"{'min@600f':>9s} {'min@700f':>9s} {'verdict':>8s}")
    print(header); print("-" * len(header))
    tmpdir = tempfile.mkdtemp()
    for label, bb, H, W in configs:
        try:
            model, C, npar = build_model(bb)
        except Exception as e:
            print(f"{label:16s}  BUILD FAIL: {type(e).__name__}: {e}"); continue
        shape = (1, H, W)
        engines = [("torch", lambda: time_torch(model, shape, args.batch, args.iters, args.warmup))]
        onnx_path = os.path.join(tmpdir, f"{bb}_{H}x{W}.onnx")
        onnx_ok = False
        try:
            export_onnx(model, shape, args.batch, onnx_path); onnx_ok = True
        except Exception as e:
            print(f"{label:16s}  onnx export fail: {type(e).__name__}: {e}")
        if onnx_ok and have["onnxruntime"]:
            engines.append(("ort", lambda p=onnx_path: time_ort(p, shape, args.batch, args.iters, args.warmup)))
        if onnx_ok and have["openvino"]:
            engines.append(("ov", lambda p=onnx_path: time_ov(p, shape, args.batch, args.iters, args.warmup)))
        for ename, fn in engines:
            try:
                pw = fn() + mel_pw
                row = f"{label:16s} {npar/1e6:7.1f}M {ename:>5s} {pw*1000:8.2f}"
                for key in ("600f", "700f"):
                    mins = pw * NWIN[key] / 60.0
                    row += f" {mins:9.1f}"
                verdict = "PASS" if (pw * NWIN["700f"] / 60.0) <= args.threshold_min else "FAIL"
                print(row + f" {verdict:>8s}")
            except Exception as e:
                print(f"{label:16s} {npar/1e6:7.1f}M {ename:>5s}  ERR {type(e).__name__}: {e}")
    print("\nNote: PASS/FAIL uses min@700f <= %.0f. Single model, batch=%d, all CPU threads."
          % (args.threshold_min, args.batch))
    print("Real Gate-0 numbers require running this on a Kaggle CPU notebook.")

if __name__ == "__main__":
    main()
