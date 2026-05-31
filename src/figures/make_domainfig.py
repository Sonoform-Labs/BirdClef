import sys, io, os, glob
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np, pandas as pd, librosa
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
B = r"D:\Python\BirdClef"; SR = 32000
def _sec(t):
    p = [float(x) for x in str(t).split(":")]; return int(p[0]*3600+p[1]*60+p[2]) if len(p)==3 else int(p[0]*60+p[1])
lab = pd.read_csv(B + r"\data\train_soundscapes_labels.csv")
lab["stem"] = lab["filename"].astype(str).str.replace(r"\.(ogg|wav)$","",regex=True)
tax = pd.read_csv(B + r"\data\taxonomy.csv"); tname = dict(zip(tax["primary_label"].astype(str), tax["common_name"].astype(str)))

counts = {}
for _, r in lab.iterrows():
    for s in str(r["primary_label"]).split(";"):
        s = s.strip()
        if s and os.path.isdir(B + r"\data\train_audio\\" + s): counts[s] = counts.get(s, 0) + 1
sp = max(counts, key=counts.get)
print("species:", sp, tname.get(sp, ""), "| soundscape windows:", counts[sp])

ff = sorted(glob.glob(B + r"\data\train_audio\\" + sp + r"\*.ogg"))[0]
yf, _ = librosa.load(ff, sr=SR, mono=True)
if len(yf) > 5*SR:
    e = np.array([np.sum(yf[i:i+5*SR]**2) for i in range(0, max(1,len(yf)-5*SR), SR)]); st = int(np.argmax(e)*SR); yf = yf[st:st+5*SR]
yf = np.pad(yf, (0, max(0, 5*SR-len(yf))))[:5*SR]

row = None; best = 0
for _, r in lab.iterrows():
    sps = [x.strip() for x in str(r["primary_label"]).split(";")]
    if sp in sps and len(sps) >= best:
        f = B + r"\data\train_soundscapes\\" + str(r["filename"])
        if os.path.exists(f): row = r; best = len(sps)
endsec = _sec(row["end"]); f = B + r"\data\train_soundscapes\\" + str(row["filename"])
ys, _ = librosa.load(f, sr=SR, mono=True, offset=max(0, endsec-5), duration=5)
ys = np.pad(ys, (0, max(0, 5*SR-len(ys))))[:5*SR]
cooc = [x.strip() for x in str(row["primary_label"]).split(";")]
print("soundscape window:", row["filename"], "end", endsec, "| species in window:", len(cooc))
def mel(y):
    m = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=2048, hop_length=512, n_mels=128, fmin=50, fmax=16000, power=2.0)
    return librosa.power_to_db(m, ref=np.max)
fig, ax = plt.subplots(2, 1, figsize=(5.2, 4.6), sharex=True)
for a, (M, ttl) in zip(ax, [(mel(yf), "FOCAL: clean, single-species (%s)" % tname.get(sp, sp)),
                            (mel(ys), "SOUNDSCAPE: %d overlapping species + noise" % len(cooc))]):
    im = a.imshow(M, origin="lower", aspect="auto", cmap="magma", extent=[0,5,50,16000], vmin=-80, vmax=0)
    a.set_ylabel("Hz", fontsize=9); a.set_title(ttl, fontsize=9.5)
ax[1].set_xlabel("Time (s)", fontsize=9.5)
fig.colorbar(im, ax=ax, label="dB", fraction=0.046, pad=0.02)
fig.suptitle("Focal $\\rightarrow$ soundscape domain shift (same species)", fontsize=11, y=0.99)
fig.savefig(B + r"\paper\fig_domain_shift.pdf", bbox_inches="tight")
fig.savefig(B + r"\paper\fig_domain_shift.png", dpi=200, bbox_inches="tight")
print("saved fig_domain_shift")
