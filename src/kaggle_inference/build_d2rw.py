import json, os
B = r"D:\Python\BirdClef"
SRC_NB = B + r"\kaggle_push_v52\birdclef-2026-v52-exp.ipynb"
SRC_META = B + r"\kaggle_push_v52\kernel-metadata.json"
OUT = B + r"\kaggle_push_d2rw"; os.makedirs(OUT, exist_ok=True)

PROTO_TARGET = "proto_scores_flat = proto_out.reshape(-1, N_CLASSES).astype(np.float32)"
CNN_TARGET = '_exp_result = _run_cnn_inference(_EXP_DIR, "B0-Focal-4th")'
WEIGHT_TARGET = 'PROTO_W, SED_W = [float(v) for v in _this_model.get("xSED"'
CNN_DISABLE = ["# === FAST CORE: skip aux CNN members ===", "_run_cnn_inference = lambda *a, **k: None",
               "print('>>> FAST CORE: B0/B2 disabled')"]
D2_SWAP = [
    "# === D2 gated-fusion head swap ===", "import glob as _glob, os as _os",
    "_c=[p for p in ['/kaggle/input/datasets/bobmyersthesecond/d2-head-weights/d2_weights.npz','/kaggle/input/d2-head-weights/d2_weights.npz'] if _os.path.exists(p)]",
    "_c=_c or _glob.glob('/kaggle/input/*/d2_weights.npz') or _glob.glob('/kaggle/input/*/*/d2_weights.npz') or _glob.glob('/kaggle/input/*/*/*/d2_weights.npz')",
    "assert _c, 'd2_weights.npz not found'",
    "_d2 = np.load(_c[0]); _P=_d2['P']; _temp=float(_d2['temp']); _bias=_d2['bias']; _gw=_d2['g_weight']; _gb=_d2['g_bias']",
    "_xn = emb_te/(np.linalg.norm(emb_te,axis=1,keepdims=True)+1e-8)",
    "_pn = _P/(np.linalg.norm(_P,axis=1,keepdims=True)+1e-8)",
    "_proto_d2 = (_xn@_pn.T)*np.log1p(np.exp(_temp)) + _bias",
    "_gate_d2 = 1.0/(1.0+np.exp(-(emb_te@_gw.T+_gb)))",
    "proto_scores_flat = (_gate_d2*_proto_d2 + (1.0-_gate_d2)*sc_te).astype(np.float32)",
    "print('>>> D2 swap active', proto_scores_flat.shape)",
]
WEIGHT_OVR = ["PROTO_W, SED_W = 0.40, 0.60  # RE-WEIGHT: trust the worse D2 proto member less",
              "print('>>> D2-REWEIGHT: PROTO_W=0.40 SED_W=0.60 (was 0.60/0.40)')"]

nb = json.load(open(SRC_NB, encoding="utf-8"))
cnn = proto = wt = False
for cell in nb["cells"]:
    if cell["cell_type"] != "code": continue
    src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
    if "if 'Model_7' in _ensemble_models" not in src or "to_csv(_file_name_submission" not in src: continue
    out = []
    for ln in src.splitlines(keepends=True):
        if CNN_TARGET in ln and not cnn:
            ind = ln[:len(ln)-len(ln.lstrip())]; out.append("".join(ind+s+"\n" for s in CNN_DISABLE)); cnn = True
        out.append(ln)
        if PROTO_TARGET in ln and not proto:
            ind = ln[:len(ln)-len(ln.lstrip())]; out.append("".join(ind+s+"\n" for s in D2_SWAP)); proto = True
        if WEIGHT_TARGET in ln and not wt:
            ind = ln[:len(ln)-len(ln.lstrip())]; out.append("".join(ind+s+"\n" for s in WEIGHT_OVR)); wt = True
    cell["source"] = "".join(out); break
assert cnn and proto and wt, "injections: cnn=%s proto=%s weight=%s" % (cnn, proto, wt)
json.dump(nb, open(OUT + r"\birdclef-2026-d2rw.ipynb", "w", encoding="utf-8"))
meta = json.load(open(SRC_META, encoding="utf-8"))
meta["id"] = "bobmyersthesecond/birdclef-2026-d2-reweight"; meta["title"] = "BirdCLEF 2026 d2 reweight"
meta["code_file"] = "birdclef-2026-d2rw.ipynb"
if "d2-head-weights" not in " ".join(meta.get("dataset_sources", [])):
    meta.setdefault("dataset_sources", []).append("bobmyersthesecond/d2-head-weights")
json.dump(meta, open(OUT + r"\kernel-metadata.json", "w"), indent=2)
print("built d2rw: cnn=%s proto=%s weight=%s | id=%s" % (cnn, proto, wt, meta["id"]))
