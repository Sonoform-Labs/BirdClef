import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch
import numpy as np
import os
OUT = r"D:\Python\BirdClef\paper"; os.makedirs(OUT, exist_ok=True)
def save(fig, name):
    fig.savefig(os.path.join(OUT, name+".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name+".png"), dpi=200, bbox_inches="tight")
    plt.close(fig); print("saved", name)

def lab(ax, text, xy, xytext, ha, fs=8.5, color="k"):
    ax.annotate(text, xy=xy, xytext=xytext, fontsize=fs, ha=ha, va="center", color=color,
                arrowprops=dict(arrowstyle="-", lw=0.6, color="#aaa", shrinkA=2, shrinkB=5))


pts = [
    (0.69,  0.762, "1-proto (no loss)", "o", "#7f7f7f", (0.45, 0.787), "right"),
    (0.437, 0.830, "+input gate (D2)",  "s", "#d62728", (0.40, 0.806), "right"),
    (0.958, 0.742, "+static gate",      "s", "#2ca02c", (0.86, 0.722), "right"),
    (0.944, 0.604, "+cluster",          "o", "#1f77b4", (0.83, 0.584), "right"),
    (0.958, 0.606, "D1 (pure)",         "D", "#1f77b4", (1.00, 0.628), "left"),
    (0.79,  0.803, "ProtoSSMv2",        "*", "#9467bd", (0.79, 0.781), "center"),
]
fig, ax = plt.subplots(figsize=(6.0, 4.4))
ax.axhline(0.739, ls=":",  lw=1.1, color="#666"); ax.text(0.02, 0.7425, "Perch logits = 0.739", fontsize=7.5, color="#666")
ax.axhline(0.766, ls="--", lw=1.1, color="#333"); ax.text(0.02, 0.7695, "linear probe = 0.766", fontsize=7.5, color="#333")
for x, y, l, m, c, lxy, ha in pts:
    ax.scatter(x, y, s=(360 if m == "*" else 135), marker=m, c=c, edgecolors="k", linewidths=0.7, zorder=5)
for x, y, l, m, c, lxy, ha in pts:
    lab(ax, l, (x, y), lxy, ha, fs=8.6)

ax.text(0.47, 0.861, "input gate: +AUC, -purity", fontsize=8.0, color="#d62728", style="italic", ha="left")
ax.text(0.63, 0.668, "cluster loss:\n+purity, -AUC", fontsize=8.0, color="#1f77b4", style="italic", ha="left")
ax.set_xlabel("Prototype purity (presence-purity)", fontsize=10.5)
ax.set_ylabel("Macro ROC-AUC (71 species, OOF)", fontsize=10.5)
ax.set_xlim(0, 1.13); ax.set_ylim(0.575, 0.875); ax.grid(True, ls=":", alpha=0.35)
save(fig, "fig_frontier")


rows = [("ProtoSSMv2 (deployed)", 0.94218, "#9467bd"), ("D1 cov-gated", 0.93964, "#2ca02c"),
        ("ProtoSSMv2 cov-gated", 0.93954, "#2ca02c"), ("D2 re-weighted", 0.93928, "#2ca02c"),
        ("drop (raw Perch)", 0.93920, "#7f7f7f"), ("D1 pure", 0.93810, "#1f77b4"),
        ("D2 cov-gated", 0.93777, "#2ca02c"), ("D2 (gated)", 0.93743, "#1f77b4"),
        ("D2-FOCAL", 0.93522, "#d62728")]
fig, ax = plt.subplots(figsize=(7.6, 4.6)); x = range(len(rows))
ax.bar(x, [r[1] for r in rows], color=[r[2] for r in rows], edgecolor="k", lw=0.7, width=0.72)
ax.axhline(0.94218, ls="--", lw=1.1, color="#9467bd", alpha=0.7)
for xi, r in zip(x, rows):
    ax.text(xi, r[1]+0.00006, "%.5f" % r[1], ha="center", va="bottom", fontsize=7.3)
ax.set_xticks(list(x)); ax.set_xticklabels([r[0] for r in rows], rotation=28, ha="right", fontsize=8.2)
ax.set_ylim(0.9340, 0.9432)
ax.set_ylabel("Private leaderboard (234-species macro-AUC)", fontsize=10)
ax.grid(True, axis="y", ls=":", alpha=0.35)
leg = [Patch(facecolor="#9467bd", ec="k", label="deployed (soundscape + SSM)"),
       Patch(facecolor="#7f7f7f", ec="k", label="raw Perch (no head)"),
       Patch(facecolor="#1f77b4", ec="k", label="author head"),
       Patch(facecolor="#2ca02c", ec="k", label="coverage-gated / re-weighted"),
       Patch(facecolor="#d62728", ec="k", label="broad focal training")]
ax.legend(handles=leg, fontsize=7.6, loc="upper right", framealpha=0.95)
ax.annotate("broad training\nbackfires (domain gap)", xy=(8, 0.93540), xytext=(6.9, 0.93915),
            fontsize=7.8, color="#d62728", ha="center",
            arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.1))
save(fig, "fig_lb_bars")


taxa = ["Aves", "Amphibia", "Insecta"]; apn = [0.951, 0.562, 0.107]; pssm = [0.64, 0.88, 0.96]
fig, ax = plt.subplots(figsize=(5.4, 4.1)); w = 0.38; xi = np.arange(len(taxa))
ax.bar(xi-w/2, apn, w, label="AudioProtoPNet (ConvNeXt, bird-pretrained)", color="#1f77b4", edgecolor="k", lw=0.7)
ax.bar(xi+w/2, pssm, w, label="ProtoSSMv2 (Perch, soundscape-trained)", color="#d62728", edgecolor="k", lw=0.7)
for i, (a, p) in enumerate(zip(apn, pssm)):
    ax.text(i-w/2, a+0.015, "%.2f" % a, ha="center", fontsize=7.8)
    ax.text(i+w/2, p+0.015, "%.2f" % p, ha="center", fontsize=7.8)
ax.text(2-w/2, 0.107+0.07, "(cvg-\nlimited)", ha="center", fontsize=6.5, color="#1f77b4", style="italic")
ax.set_xticks(xi); ax.set_xticklabels(taxa, fontsize=9.5); ax.set_ylabel("Prototype purity", fontsize=10)
ax.set_ylim(0, 1.10)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=1, fontsize=8, frameon=True)
ax.grid(True, axis="y", ls=":", alpha=0.35)
ax.set_title("Inverted purity gradients by training domain", fontsize=10.5)
save(fig, "fig_taxon_purity")


fig, ax = plt.subplots(figsize=(8.4, 3.6)); ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 5)
def box(x, y, w, h, txt, fc, fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04", fc=fc, ec="k", lw=1.0))
    ax.text(x+w/2, y+h/2, txt, ha="center", va="center", fontsize=fs)
def arrow(x1, y1, x2, y2, c="k"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, lw=1.1, color=c))
box(0.15, 2.05, 1.7, 1.0, "Soundscape\naudio (5s win)", "#eeeeee", 8)
box(2.35, 2.05, 1.8, 1.0, "Perch v2\nembeddings\n(1536-d)", "#dbe7ff", 8)
arrow(1.85, 2.55, 2.35, 2.55)
box(5.0, 3.55, 3.1, 0.9, "ProtoSSMv2  (52.4%)\nproto + gate + SSM", "#e7d9f5", 8)
box(5.0, 2.45, 3.1, 0.72, "Perch-distilled SED  (34.9%)", "#d9f5e1", 8)
box(5.0, 1.55, 3.1, 0.62, "EfficientNet B0/B2  (12.6%)", "#fff0d9", 8)
for yy in (4.0, 2.81, 1.86): arrow(4.15, 2.55, 5.0, yy)
box(8.7, 2.4, 1.7, 1.2, "rank\nfusion +\neco. priors", "#ffe0e0", 8.5)
for yy in (4.0, 2.81, 1.86): arrow(8.1, yy, 8.7, 3.0)
box(10.8, 2.4, 1.05, 1.2, "234-sp\nscores", "#eeeeee", 8)
arrow(10.4, 3.0, 10.8, 3.0)
ax.add_patch(FancyBboxPatch((4.92, 3.47), 3.26, 1.06, boxstyle="round,pad=0.04", fc="none", ec="#d62728", lw=1.8, ls="--"))
ax.text(6.0, 0.62, "Part II: only the ProtoSSMv2 member (dashed red) is replaced with author-built\nprototype heads, then re-scored on the hidden-test leaderboard.",
        ha="center", va="center", fontsize=7.6, color="#d62728")
ax.set_title("Competition system (Part II varies only the prototype member)", fontsize=11)
ax.set_ylim(0.1, 5.0)
save(fig, "fig_system")
print("DONE _make_figs")
