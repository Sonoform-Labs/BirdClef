import json, io, sys, os, shutil
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
B = r"D:\Python\BirdClef"
variant = sys.argv[1] if len(sys.argv) > 1 else "d1"
SRC_NB = B + r"\kaggle_push_v52\birdclef-2026-v52-exp.ipynb"
SRC_META = B + r"\kaggle_push_v52\kernel-metadata.json"

CFG = {
    "d1": dict(
        kid="bobmyersthesecond/birdclef-2026-d1-swap", title="BirdCLEF 2026 D1 swap",
        nbfile="birdclef-2026-d1swap.ipynb", outdir=B + r"\kaggle_push_d1swap",
        needs_d2ds=True,
        snippet=[
            "# === D1 author-built head swap (pure prototype, Model_7) ===",
            "import glob as _glob, os as _os",
            "_c=[p for p in ['/kaggle/input/datasets/bobmyersthesecond/d2-head-weights/d1_weights.npz','/kaggle/input/d2-head-weights/d1_weights.npz'] if _os.path.exists(p)]",
            "_c=_c or _glob.glob('/kaggle/input/*/d1_weights.npz') or _glob.glob('/kaggle/input/*/*/d1_weights.npz') or _glob.glob('/kaggle/input/*/*/*/d1_weights.npz')",
            "assert _c, 'd1_weights.npz not found (fast paths)'",
            "_d1 = np.load(_c[0]); print('D1 weights from', _c[0])",
            "_P=_d1['P']; _temp=float(_d1['temp']); _bias=_d1['bias']",
            "_xn = emb_te/(np.linalg.norm(emb_te,axis=1,keepdims=True)+1e-8)",
            "_pn = _P/(np.linalg.norm(_P,axis=1,keepdims=True)+1e-8)",
            "proto_scores_flat = ((_xn@_pn.T)*np.log1p(np.exp(_temp)) + _bias).astype(np.float32)",
            "print('>>> D1 SWAP ACTIVE (Model_7): pure prototype', proto_scores_flat.shape)",
            "open('/kaggle/working/D1_RAN.txt','w').write('D1 swap ran '+str(proto_scores_flat.shape))",
        ],
    ),
    "drop": dict(
        kid="bobmyersthesecond/birdclef-2026-drop-proto", title="BirdCLEF 2026 drop proto",
        nbfile="birdclef-2026-dropproto.ipynb", outdir=B + r"\kaggle_push_dropproto",
        needs_d2ds=False,
        snippet=[
            "# === DROP proto member: replace with raw Perch (Model_7) ===",
            "proto_scores_flat = sc_te.astype(np.float32)",
            "print('>>> PROTO DROPPED (Model_7): proto member = raw Perch', proto_scores_flat.shape)",
            "open('/kaggle/working/DROP_RAN.txt','w').write('proto dropped '+str(proto_scores_flat.shape))",
        ],
    ),
}
c = CFG[variant]
os.makedirs(c["outdir"], exist_ok=True)
TARGET = "proto_scores_flat = proto_out.reshape(-1, N_CLASSES).astype(np.float32)"

nb = json.load(open(SRC_NB, encoding="utf-8"))
done = False
for ci, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
    if TARGET not in src or "if 'Model_7' in _ensemble_models" not in src or "to_csv(_file_name_submission" not in src:
        continue
    lines = src.splitlines(keepends=True)
    for k, ln in enumerate(lines):
        if TARGET in ln:
            indent = ln[:len(ln) - len(ln.lstrip())]
            lines.insert(k + 1, "".join(indent + s + "\n" for s in c["snippet"]))
            done = True
            print("Injected %s into cell idx %d after line %d" % (variant, ci, k))
            break
    cell["source"] = "".join(lines)
    break
assert done, "Model_7 target line not found!"
json.dump(nb, open(os.path.join(c["outdir"], c["nbfile"]), "w", encoding="utf-8"))

meta = json.load(open(SRC_META, encoding="utf-8"))
meta["id"] = c["kid"]; meta["title"] = c["title"]; meta["code_file"] = c["nbfile"]
if c["needs_d2ds"] and "d2-head-weights" not in " ".join(meta.get("dataset_sources", [])):
    meta.setdefault("dataset_sources", []).append("bobmyersthesecond/d2-head-weights")
json.dump(meta, open(os.path.join(c["outdir"], "kernel-metadata.json"), "w"), indent=2)
print("built", c["kid"], "| nb:", c["nbfile"], "| dataset needed:", c["needs_d2ds"])
