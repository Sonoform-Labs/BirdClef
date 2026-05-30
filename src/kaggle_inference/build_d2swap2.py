import json, io, sys, os, shutil
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
B = r"D:\Python\BirdClef"
SRC_NB = B + r"\kaggle_push_v52\birdclef-2026-v52-exp.ipynb"
SRC_META = B + r"\kaggle_push_v52\kernel-metadata.json"
OUT = B + r"\kaggle_push_d2swap"; os.makedirs(OUT, exist_ok=True)
os.makedirs(B + r"\d2_ds", exist_ok=True)
shutil.copy(B + r"\d2_weights.npz", B + r"\d2_ds\d2_weights.npz")

TARGET = "proto_scores_flat = proto_out.reshape(-1, N_CLASSES).astype(np.float32)"
SNIPPET = [
    "# === D2 author-built head swap (Model_7 proto member) ===",
    "import glob as _glob, os as _os",
    "_c=[p for p in ['/kaggle/input/datasets/bobmyersthesecond/d2-head-weights/d2_weights.npz','/kaggle/input/d2-head-weights/d2_weights.npz'] if _os.path.exists(p)]",
    "_c=_c or _glob.glob('/kaggle/input/*/d2_weights.npz') or _glob.glob('/kaggle/input/*/*/d2_weights.npz') or _glob.glob('/kaggle/input/*/*/*/d2_weights.npz')",
    "assert _c, 'd2_weights.npz not found (fast paths)'",
    "_d2 = np.load(_c[0]); print('D2 weights from', _c[0])",
    "_P=_d2['P']; _temp=float(_d2['temp']); _bias=_d2['bias']; _gw=_d2['g_weight']; _gb=_d2['g_bias']",
    "_xn = emb_te/(np.linalg.norm(emb_te,axis=1,keepdims=True)+1e-8)",
    "_pn = _P/(np.linalg.norm(_P,axis=1,keepdims=True)+1e-8)",
    "_proto_d2 = (_xn@_pn.T)*np.log1p(np.exp(_temp)) + _bias",
    "_gate_d2 = 1.0/(1.0+np.exp(-(emb_te@_gw.T+_gb)))",
    "proto_scores_flat = (_gate_d2*_proto_d2 + (1.0-_gate_d2)*sc_te).astype(np.float32)",
    "print('>>> D2 SWAP ACTIVE (Model_7): proto_scores_flat = D2 head', proto_scores_flat.shape, 'gate mean', float(_gate_d2.mean()))",
    "open('/kaggle/working/D2_RAN.txt','w').write('D2 swap ran; proto_scores_flat '+str(proto_scores_flat.shape)+' gate_mean '+str(float(_gate_d2.mean())))",
]

nb = json.load(open(SRC_NB, encoding="utf-8"))
done = False
for ci, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
    if TARGET not in src:
        continue

    if "if 'Model_7' in _ensemble_models" not in src:
        print("  (skipping non-Model_7 cell %d that also has the target line)" % ci)
        continue
    if "to_csv(_file_name_submission" not in src:
        print("WARN: Model_7 cell %d does not write _file_name_submission - skipping" % ci)
        continue
    lines = src.splitlines(keepends=True)
    for k, ln in enumerate(lines):
        if TARGET in ln:
            indent = ln[:len(ln) - len(ln.lstrip())]
            snippet = "".join(indent + s + "\n" for s in SNIPPET)
            lines.insert(k + 1, snippet)
            done = True
            print("Injected into cell idx %d after line %d (indent=%d spaces)" % (ci, k, len(indent)))
            break
    c["source"] = "".join(lines)
    break
assert done, "target line not found in a Model_7 cell!"
json.dump(nb, open(OUT + r"\birdclef-2026-d2swap.ipynb", "w", encoding="utf-8"))

meta = json.load(open(SRC_META, encoding="utf-8"))
meta["id"] = "bobmyersthesecond/birdclef-2026-d2-swap"
meta["title"] = "BirdCLEF 2026 D2 swap"
meta["code_file"] = "birdclef-2026-d2swap.ipynb"
if "d2-head-weights" not in " ".join(meta.get("dataset_sources", [])):
    meta.setdefault("dataset_sources", []).append("bobmyersthesecond/d2-head-weights")
json.dump(meta, open(OUT + r"\kernel-metadata.json", "w"), indent=2)


nb2 = json.load(open(OUT + r"\birdclef-2026-d2swap.ipynb", encoding="utf-8"))
nhits = 0
for ci, c in enumerate(nb2["cells"]):
    s = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
    if "D2 SWAP ACTIVE" in s:
        nhits += 1
        has_runperch = "run_perch(test_paths" in s
        has_writesub = "sub.to_csv(\"submission_protossm.csv\"" in s or "submission_protossm.csv" in s
        has_finalsub = "_file_name_submission" in s
        print("  injection in cell idx %d | run_perch(test)=%s writes_protossm=%s writes_final_submission=%s"
              % (ci, has_runperch, has_writesub, has_finalsub))
print("total injections:", nhits, "| meta id:", meta["id"], "| datasets:", meta.get("dataset_sources"))
assert nhits == 1, "expected exactly 1 injection"
print("OK clean build")
