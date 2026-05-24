import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np, pandas as pd, librosa, onnxruntime as ort
B = r"D:\Python\BirdClef"
SR = 32000; WIN = 160000; N_WIN_MAX = 12
N_FILES_PER_SP = 15; TOPK_WIN = 2
AUDIO = B + r"\data\train_audio"

species = [c for c in pd.read_csv(B + r"\_q\submission_protossm.csv", nrows=1).columns if c != "row_id"]
sp_idx = {s: i for i, s in enumerate(species)}; C = len(species)
pm = np.load(B + r"\perch_map.npz"); bc_index = pm["bc_index"]; mapped = bc_index >= 0
df = pd.read_csv(B + r"\data\train.csv")
df["primary_label"] = df["primary_label"].astype(str)
if "rating" not in df.columns: df["rating"] = 0.0
df = df[df["primary_label"].isin(sp_idx)]

so = ort.SessionOptions(); so.intra_op_num_threads = 0
sess = ort.InferenceSession(B + r"\models\perch\perch_v2_no_dft.onnx", sess_options=so, providers=["CPUExecutionProvider"])
inm = sess.get_inputs()[0].name
omap = {o.name: i for i, o in enumerate(sess.get_outputs())}

def windows(path):
    try:
        y, _ = librosa.load(path, sr=SR, mono=True)
    except Exception:
        return None
    if len(y) < WIN: y = np.pad(y, (0, WIN - len(y)))
    n = min(N_WIN_MAX, len(y) // WIN) or 1
    return y[:n * WIN].reshape(n, WIN).astype(np.float32)

emb_all, sco_all, y_all = [], [], []
t0 = time.time(); done = 0
groups = df.groupby("primary_label")
for si, sp in enumerate(species):
    if sp not in groups.groups:
        continue
    g = groups.get_group(sp).sort_values("rating", ascending=False).head(N_FILES_PER_SP)
    bc = int(bc_index[sp_idx[sp]]) if mapped[sp_idx[sp]] else -1
    for _, r in g.iterrows():
        w = windows(AUDIO + "\\" + r["filename"].replace("/", "\\"))
        if w is None: continue
        outs = sess.run(None, {inm: w})
        emb = outs[omap["embedding"]]; logit = outs[omap["label"]]

        sc = np.zeros((len(w), C), np.float32)
        sc[:, mapped] = logit[:, bc_index[mapped]]

        if bc >= 0:
            order = np.argsort(-logit[:, bc])[:TOPK_WIN]
        else:
            order = np.arange(min(TOPK_WIN, len(w)))
        for k in order:
            emb_all.append(emb[k]); sco_all.append(sc[k]); y_all.append(sp_idx[sp])
        done += 1
    if (si + 1) % 25 == 0:
        print("  %d/%d species  files=%d  windows=%d  %.0fs" % (si + 1, C, done, len(y_all), time.time() - t0))

emb_all = np.stack(emb_all).astype(np.float32); sco_all = np.stack(sco_all).astype(np.float32)
y_all = np.array(y_all, np.int64)
covered_sp = len(np.unique(y_all))
np.savez(B + r"\focal_bank.npz", emb=emb_all, scores=sco_all, y=y_all, species=np.array(species))
print("\nDONE focal_bank.npz: emb%s scores%s | %d windows over %d species (vs 71 in soundscape bank)"
      % (emb_all.shape, sco_all.shape, len(y_all), covered_sp))
print("elapsed %.0fs" % (time.time() - t0))
