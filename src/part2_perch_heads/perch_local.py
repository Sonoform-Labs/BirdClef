import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np, pandas as pd, soundfile as sf, onnxruntime as ort
B = r"D:\Python\BirdClef"
SR = 32000; WIN = 160000; FILE_SAMPLES = 60 * SR; N_WINDOWS = 12
SS_DIR = B + r"\data\train_soundscapes"

z = np.load(B + r"\_perch_meta\full_perch_arrays.npz")
emb_full = z["emb_full"]; scores_full = z["scores_full_raw"]
meta = pd.read_parquet(B + r"\_perch_meta\full_perch_meta.parquet")
species = [c for c in pd.read_csv(B + r"\_q\submission_protossm.csv", nrows=1).columns if c != "row_id"]
C = len(species)

files = list(dict.fromkeys(meta["filename"].tolist()))
print("files=%d  expected windows=%d  bank=%d" % (len(files), len(files) * N_WINDOWS, len(emb_full)))

def read_60s(path):
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim == 2: y = y.mean(axis=1)
    assert sr == SR, "soundscape not 32k: %s" % path
    if len(y) < FILE_SAMPLES: y = np.pad(y, (0, FILE_SAMPLES - len(y)))
    elif len(y) > FILE_SAMPLES: y = y[:FILE_SAMPLES]
    return y

so = ort.SessionOptions(); so.intra_op_num_threads = 0
sess = ort.InferenceSession(B + r"\models\perch\perch_v2_no_dft.onnx", sess_options=so, providers=["CPUExecutionProvider"])
inm = sess.get_inputs()[0].name
omap = {o.name: i for i, o in enumerate(sess.get_outputs())}
print("input", inm, "outputs", list(omap))

emb_loc = np.zeros((len(files) * N_WINDOWS, 1536), np.float32)
log_loc = np.zeros((len(files) * N_WINDOWS, 14795), np.float32)
t0 = time.time(); w = 0
for fi, fn in enumerate(files):
    y = read_60s(SS_DIR + "\\" + fn).reshape(N_WINDOWS, WIN)
    outs = sess.run(None, {inm: y})
    emb_loc[w:w + N_WINDOWS] = outs[omap["embedding"]]
    log_loc[w:w + N_WINDOWS] = outs[omap["label"]]
    w += N_WINDOWS
    if fi == 0 or (fi + 1) % 10 == 0: print("  %d/%d  %.1fs" % (fi + 1, len(files), time.time() - t0))


d = np.abs(emb_loc - emb_full)
cc = np.corrcoef(emb_loc.ravel(), emb_full.ravel())[0, 1]
print("\n=== VERIFY emb_local vs emb_full ===")
print("  max abs diff %.4e | mean %.4e | corr %.6f" % (d.max(), d.mean(), cc))
ok = cc > 0.999 and d.mean() < 1e-2
print("  -> preprocessing %s" % ("MATCHES (mapping recovery is exact)" if ok else "DIFFERS - investigate before trusting focal embeddings"))


bc_index = np.full(C, -1, np.int64); best_corr = np.zeros(C, np.float32)

L = log_loc - log_loc.mean(0, keepdims=True); Ln = L / (np.linalg.norm(L, axis=0, keepdims=True) + 1e-9)
for p in range(C):
    s = scores_full[:, p]
    if s.std() < 1e-9:
        continue
    sc = (s - s.mean()); sc = sc / (np.linalg.norm(sc) + 1e-9)
    corr = Ln.T @ sc
    k = int(np.argmax(corr)); bc_index[p] = k; best_corr[p] = float(corr[k])
mapped = bc_index >= 0
exact = (best_corr > 0.999) & mapped
print("\n=== RECOVER 14795->234 mapping ===")
print("  species with a logit match: %d / %d  | exact(>0.999): %d  | proxy/weak(<0.999): %d" %
      (mapped.sum(), C, exact.sum(), (mapped & ~exact).sum()))
print("  all-zero (unmapped) species: %d" % (~mapped).sum())
np.savez(B + r"\perch_map.npz", bc_index=bc_index, best_corr=best_corr, species=np.array(species),
         emb_match_corr=cc, emb_max_diff=float(d.max()))
print("saved perch_map.npz")
