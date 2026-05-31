import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
OUT = r"D:\Python\BirdClef\paper"
def save(fig, name):
    fig.savefig(OUT+"\\"+name+".pdf", bbox_inches="tight")
    fig.savefig(OUT+"\\"+name+".png", dpi=200, bbox_inches="tight")
    plt.close(fig); print("saved", name)

z = np.load("focal_bank.npz", allow_pickle=True)
emb, y, species = z["emb"], z["y"], z["species"]
counts = np.bincount(y, minlength=len(species))


N = 12
top = np.argsort(counts)[::-1][:N]
print("top-%d species codes:" % N, [str(species[i]) for i in top])

idx = []
rng = np.random.RandomState(0)
for s in top:
    si = np.where(y == s)[0]
    if len(si) > 300: si = rng.choice(si, 300, replace=False)
    idx.append(si)
idx = np.concatenate(idx)
E, Y = emb[idx], y[idx]
Ep = PCA(n_components=50, random_state=0).fit_transform(E)
T = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=42).fit_transform(Ep)
fig, ax = plt.subplots(figsize=(7.0, 5.2))
palette = plt.cm.tab20(np.linspace(0, 1, N))
for c, s in zip(palette, top):
    m = Y == s
    ax.scatter(T[m, 0], T[m, 1], s=9, color=c, alpha=0.6, linewidths=0, label=str(species[s]))
ax.set_title("Perch embedding space: focal recordings cluster by species", fontsize=12, fontweight="bold")
ax.set_xticklabels([]); ax.set_yticklabels([])
ax.tick_params(length=0)
ax.grid(True, ls="--", alpha=0.3)
leg = ax.legend(title="Species (eBird/iNat code)", fontsize=7.5, title_fontsize=8.5,
                loc="center left", bbox_to_anchor=(1.01, 0.5), markerscale=1.8, framealpha=0.95)
save(fig, "fig_embeddings")


H = [
    ("ProtoSSMv2 (deployed)", 0.803, 0.94218, "deployed",  "*", "#9467bd", (0.803, 0.94120, "center")),
    ("drop (raw Perch)",      0.739, 0.93920, "raw Perch", "o", "#7f7f7f", (0.739, 0.93975, "center")),
    ("D1 pure",               0.606, 0.93810, "author",    "o", "#1f77b4", (0.628, 0.93810, "left")),
    ("D2 input-gated",        0.830, 0.93743, "author",    "s", "#d62728", (0.830, 0.93690, "center")),
    ("D1 cov-gated",          0.606, 0.93964, "rescue",    "^", "#2ca02c", None),
    ("ProtoSSMv2 cov-gated",  0.803, 0.93954, "rescue",    "^", "#2ca02c", None),
    ("D2 re-weighted",        0.830, 0.93928, "rescue",    "^", "#2ca02c", None),
    ("D2 cov-gated",          0.830, 0.93777, "rescue",    "^", "#2ca02c", None),
]
fig, ax = plt.subplots(figsize=(5.6, 4.1))
for l, x, yv, cat, m, c, ann in H:
    ax.scatter(x, yv, s=(340 if m == "*" else 130), marker=m, c=c, edgecolors="k", linewidths=0.7, zorder=5)
    if ann:
        ax.annotate(l, (x, yv), (ann[0], ann[1]), fontsize=8.3, ha=ann[2], va="center",
                    arrowprops=dict(arrowstyle="-", lw=0.6, color="#999", shrinkA=2, shrinkB=4))
ax.annotate("", xy=(0.830, 0.93743), xytext=(0.606, 0.93810), arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2, alpha=0.5))
ax.text(0.700, 0.93715, "more offline accuracy ->\nlower deployment score", fontsize=7.8, color="#d62728", style="italic", ha="center")
ax.text(0.585, 0.94320, "deployed head wins despite\nmiddling offline AUC", fontsize=7.8, color="#9467bd", style="italic", ha="left")
leg = [Line2D([0],[0],marker="*",color="w",mfc="#9467bd",mec="k",ms=13,label="deployed ProtoSSMv2"),
       Line2D([0],[0],marker="o",color="w",mfc="#7f7f7f",mec="k",ms=9,label="raw Perch (drop)"),
       Line2D([0],[0],marker="o",color="w",mfc="#1f77b4",mec="k",ms=9,label="author head (pure/gated)"),
       Line2D([0],[0],marker="^",color="w",mfc="#2ca02c",mec="k",ms=9,label="rescue (cov-gated / re-weighted)")]
ax.legend(handles=leg, fontsize=7.6, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, framealpha=0.95)
ax.set_xlabel("Offline OOF AUC (71-species bank)", fontsize=10)
ax.set_ylabel("Private leaderboard (234-species test)", fontsize=10)
ax.set_xlim(0.57, 0.875); ax.set_ylim(0.9358, 0.9437); ax.grid(True, ls=":", alpha=0.35)
save(fig, "fig_oof_vs_lb_dense")
print("DONE")
