import json, io, sys, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
B = r"D:\Python\BirdClef"
variant = sys.argv[1] if len(sys.argv) > 1 else "baseline"
SRC_NB = B + r"\kaggle_push_v52\birdclef-2026-v52-exp.ipynb"
SRC_META = B + r"\kaggle_push_v52\kernel-metadata.json"

PROTO_TARGET = "proto_scores_flat = proto_out.reshape(-1, N_CLASSES).astype(np.float32)"
CNN_TARGET = '_exp_result = _run_cnn_inference(_EXP_DIR, "B0-Focal-4th")'
CNN_DISABLE = [
    "# === FAST CORE: skip aux CNN members (B0 8% + B2 5%); blend auto-falls-back to Proto+SED ===",
    "_run_cnn_inference = lambda *a, **k: None",
    "print('>>> FAST CORE: B0/B2 CNN inference disabled -> Proto+SED 2-way blend')",
]
PROTO_SNIPPETS = {
    "baseline": None,
    "basecov": [
        "# === BASELINE coverage-gated control: ProtoSSMv2 on covered, raw Perch on absent ===",
        "import glob as _glob, os as _os",
        "_cm=[p for p in ['/kaggle/input/datasets/bobmyersthesecond/d2-head-weights/covered_mask.npz','/kaggle/input/d2-head-weights/covered_mask.npz'] if _os.path.exists(p)]",
        "_cm=_cm or _glob.glob('/kaggle/input/*/covered_mask.npz') or _glob.glob('/kaggle/input/*/*/covered_mask.npz') or _glob.glob('/kaggle/input/*/*/*/covered_mask.npz')",
        "assert _cm, 'covered_mask not found'",
        "_cov = np.load(_cm[0])['covered']",
        "proto_scores_flat = np.where(_cov[None,:], proto_scores_flat, sc_te).astype(np.float32)",
        "print('>>> BASE-COV: ProtoSSMv2 on', int(_cov.sum()), 'covered, raw Perch on absent', proto_scores_flat.shape)",
    ],
    "drop": [
        "# === DROP proto member: replace with raw Perch ===",
        "proto_scores_flat = sc_te.astype(np.float32)",
        "print('>>> DROP (fast core): proto member = raw Perch', proto_scores_flat.shape)",
    ],
    "d2": [
        "# === D2 gated-fusion head swap ===",
        "import glob as _glob, os as _os",
        "_c=[p for p in ['/kaggle/input/datasets/bobmyersthesecond/d2-head-weights/d2_weights.npz','/kaggle/input/d2-head-weights/d2_weights.npz'] if _os.path.exists(p)]",
        "_c=_c or _glob.glob('/kaggle/input/*/d2_weights.npz') or _glob.glob('/kaggle/input/*/*/d2_weights.npz') or _glob.glob('/kaggle/input/*/*/*/d2_weights.npz')",
        "assert _c, 'd2_weights.npz not found'",
        "_d2 = np.load(_c[0]); _P=_d2['P']; _temp=float(_d2['temp']); _bias=_d2['bias']; _gw=_d2['g_weight']; _gb=_d2['g_bias']",
        "_xn = emb_te/(np.linalg.norm(emb_te,axis=1,keepdims=True)+1e-8)",
        "_pn = _P/(np.linalg.norm(_P,axis=1,keepdims=True)+1e-8)",
        "_proto_d2 = (_xn@_pn.T)*np.log1p(np.exp(_temp)) + _bias",
        "_gate_d2 = 1.0/(1.0+np.exp(-(emb_te@_gw.T+_gb)))",
        "proto_scores_flat = (_gate_d2*_proto_d2 + (1.0-_gate_d2)*sc_te).astype(np.float32)",
        "print('>>> D2 (fast core): proto = gated-fusion head', proto_scores_flat.shape)",
    ],
    "d1": [
        "# === D1 pure-prototype head swap ===",
        "import glob as _glob, os as _os",
        "_c=[p for p in ['/kaggle/input/datasets/bobmyersthesecond/d2-head-weights/d1_weights.npz','/kaggle/input/d2-head-weights/d1_weights.npz'] if _os.path.exists(p)]",
        "_c=_c or _glob.glob('/kaggle/input/*/d1_weights.npz') or _glob.glob('/kaggle/input/*/*/d1_weights.npz') or _glob.glob('/kaggle/input/*/*/*/d1_weights.npz')",
        "assert _c, 'd1_weights.npz not found'",
        "_d1 = np.load(_c[0]); _P=_d1['P']; _temp=float(_d1['temp']); _bias=_d1['bias']",
        "_xn = emb_te/(np.linalg.norm(emb_te,axis=1,keepdims=True)+1e-8)",
        "_pn = _P/(np.linalg.norm(_P,axis=1,keepdims=True)+1e-8)",
        "proto_scores_flat = ((_xn@_pn.T)*np.log1p(np.exp(_temp)) + _bias).astype(np.float32)",
        "print('>>> D1 (fast core): proto = pure prototype', proto_scores_flat.shape)",
    ],
    "d2focal": [
        "# === D2-FOCAL: gated-fusion head trained on combined soundscape+focal (broad label space) ===",
        "import glob as _glob, os as _os",
        "_c=[p for p in ['/kaggle/input/datasets/bobmyersthesecond/d2-head-weights/d2focal_weights.npz','/kaggle/input/d2-head-weights/d2focal_weights.npz'] if _os.path.exists(p)]",
        "_c=_c or _glob.glob('/kaggle/input/*/d2focal_weights.npz') or _glob.glob('/kaggle/input/*/*/d2focal_weights.npz') or _glob.glob('/kaggle/input/*/*/*/d2focal_weights.npz')",
        "assert _c, 'd2focal_weights.npz not found'",
        "_d2 = np.load(_c[0]); _P=_d2['P']; _temp=float(_d2['temp']); _bias=_d2['bias']; _gw=_d2['g_weight']; _gb=_d2['g_bias']",
        "_xn = emb_te/(np.linalg.norm(emb_te,axis=1,keepdims=True)+1e-8)",
        "_pn = _P/(np.linalg.norm(_P,axis=1,keepdims=True)+1e-8)",
        "_proto_d2 = (_xn@_pn.T)*np.log1p(np.exp(_temp)) + _bias",
        "_gate_d2 = 1.0/(1.0+np.exp(-(emb_te@_gw.T+_gb)))",
        "proto_scores_flat = (_gate_d2*_proto_d2 + (1.0-_gate_d2)*sc_te).astype(np.float32)",
        "print('>>> D2-FOCAL (fast core): broad-trained gated head', proto_scores_flat.shape)",
    ],
    "d2cov": [
        "# === D2 COVERAGE-GATED: gated-fusion head on in-bank species, raw Perch on the rest ===",
        "import glob as _glob, os as _os",
        "_c=[p for p in ['/kaggle/input/datasets/bobmyersthesecond/d2-head-weights/d2_weights.npz','/kaggle/input/d2-head-weights/d2_weights.npz'] if _os.path.exists(p)]",
        "_c=_c or _glob.glob('/kaggle/input/*/d2_weights.npz') or _glob.glob('/kaggle/input/*/*/d2_weights.npz') or _glob.glob('/kaggle/input/*/*/*/d2_weights.npz')",
        "_cm=[p for p in ['/kaggle/input/datasets/bobmyersthesecond/d2-head-weights/covered_mask.npz','/kaggle/input/d2-head-weights/covered_mask.npz'] if _os.path.exists(p)]",
        "_cm=_cm or _glob.glob('/kaggle/input/*/covered_mask.npz') or _glob.glob('/kaggle/input/*/*/covered_mask.npz') or _glob.glob('/kaggle/input/*/*/*/covered_mask.npz')",
        "assert _c and _cm, 'd2/covered not found'",
        "_d2 = np.load(_c[0]); _P=_d2['P']; _temp=float(_d2['temp']); _bias=_d2['bias']; _gw=_d2['g_weight']; _gb=_d2['g_bias']",
        "_cov = np.load(_cm[0])['covered']",
        "_xn = emb_te/(np.linalg.norm(emb_te,axis=1,keepdims=True)+1e-8)",
        "_pn = _P/(np.linalg.norm(_P,axis=1,keepdims=True)+1e-8)",
        "_proto_d2 = (_xn@_pn.T)*np.log1p(np.exp(_temp)) + _bias",
        "_gate_d2 = 1.0/(1.0+np.exp(-(emb_te@_gw.T+_gb)))",
        "_d2full = (_gate_d2*_proto_d2 + (1.0-_gate_d2)*sc_te)",
        "proto_scores_flat = np.where(_cov[None,:], _d2full, sc_te).astype(np.float32)",
        "print('>>> D2-COV (fast core): D2 on', int(_cov.sum()), 'covered species, raw Perch on rest', proto_scores_flat.shape)",
    ],
    "d1cov": [
        "# === D1 COVERAGE-GATED: pure prototype on in-bank species, raw Perch on the rest ===",
        "import glob as _glob, os as _os",
        "_c=[p for p in ['/kaggle/input/datasets/bobmyersthesecond/d2-head-weights/d1_weights.npz','/kaggle/input/d2-head-weights/d1_weights.npz'] if _os.path.exists(p)]",
        "_c=_c or _glob.glob('/kaggle/input/*/d1_weights.npz') or _glob.glob('/kaggle/input/*/*/d1_weights.npz') or _glob.glob('/kaggle/input/*/*/*/d1_weights.npz')",
        "_cm=[p for p in ['/kaggle/input/datasets/bobmyersthesecond/d2-head-weights/covered_mask.npz','/kaggle/input/d2-head-weights/covered_mask.npz'] if _os.path.exists(p)]",
        "_cm=_cm or _glob.glob('/kaggle/input/*/covered_mask.npz') or _glob.glob('/kaggle/input/*/*/covered_mask.npz') or _glob.glob('/kaggle/input/*/*/*/covered_mask.npz')",
        "assert _c and _cm, 'd1/covered not found'",
        "_d1 = np.load(_c[0]); _P=_d1['P']; _temp=float(_d1['temp']); _bias=_d1['bias']",
        "_cov = np.load(_cm[0])['covered']",
        "_xn = emb_te/(np.linalg.norm(emb_te,axis=1,keepdims=True)+1e-8)",
        "_pn = _P/(np.linalg.norm(_P,axis=1,keepdims=True)+1e-8)",
        "_d1full = ((_xn@_pn.T)*np.log1p(np.exp(_temp)) + _bias)",
        "proto_scores_flat = np.where(_cov[None,:], _d1full, sc_te).astype(np.float32)",
        "print('>>> D1-COV (fast core): D1 on covered, raw Perch on rest', proto_scores_flat.shape)",
    ],
}
KID = {
    "baseline": ("bobmyersthesecond/birdclef-2026-v52", "BirdCLEF 2026 v52", "birdclef-2026-v52-exp.ipynb"),
    "d2": ("bobmyersthesecond/birdclef-2026-d2-swap", "BirdCLEF 2026 D2 swap", "birdclef-2026-d2swap.ipynb"),
    "d1": ("bobmyersthesecond/birdclef-2026-d1-swap", "BirdCLEF 2026 D1 swap", "birdclef-2026-d1swap.ipynb"),
    "drop": ("bobmyersthesecond/birdclef-2026-drop-proto", "BirdCLEF 2026 drop proto", "birdclef-2026-dropproto.ipynb"),
    "d2cov": ("bobmyersthesecond/birdclef-2026-d2-coverage-gated", "BirdCLEF 2026 d2 coverage gated", "birdclef-2026-d2cov.ipynb"),
    "d1cov": ("bobmyersthesecond/birdclef-2026-d1-coverage-gated", "BirdCLEF 2026 d1 coverage gated", "birdclef-2026-d1cov.ipynb"),
    "basecov": ("bobmyersthesecond/birdclef-2026-base-coverage-gated", "BirdCLEF 2026 base coverage gated", "birdclef-2026-basecov.ipynb"),
    "d2focal": ("bobmyersthesecond/birdclef-2026-d2-focal", "BirdCLEF 2026 d2 focal", "birdclef-2026-d2focal.ipynb"),
}
OUTDIR = {"baseline": B + r"\kaggle_push_v52", "d2": B + r"\kaggle_push_d2swap",
          "d1": B + r"\kaggle_push_d1swap", "drop": B + r"\kaggle_push_dropproto",
          "d2cov": B + r"\kaggle_push_d2cov", "d1cov": B + r"\kaggle_push_d1cov",
          "basecov": B + r"\kaggle_push_basecov", "d2focal": B + r"\kaggle_push_d2focal"}[variant]

nb = json.load(open(SRC_NB, encoding="utf-8"))
cnn_done = proto_done = False
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
    if "if 'Model_7' in _ensemble_models" not in src or "to_csv(_file_name_submission" not in src:
        continue
    lines = src.splitlines(keepends=True)
    out = []
    for ln in lines:

        if CNN_TARGET in ln and not cnn_done:
            ind = ln[:len(ln) - len(ln.lstrip())]
            out.append("".join(ind + s + "\n" for s in CNN_DISABLE))
            cnn_done = True
        out.append(ln)

        if PROTO_TARGET in ln and not proto_done and PROTO_SNIPPETS[variant] is not None:
            ind = ln[:len(ln) - len(ln.lstrip())]
            out.append("".join(ind + s + "\n" for s in PROTO_SNIPPETS[variant]))
            proto_done = True
    cell["source"] = "".join(out)
    break
assert cnn_done, "CNN target not found!"
assert proto_done or PROTO_SNIPPETS[variant] is None, "proto target not found!"
kid, title, nbfile = KID[variant]
os.makedirs(OUTDIR, exist_ok=True)
json.dump(nb, open(os.path.join(OUTDIR, nbfile), "w", encoding="utf-8"))
meta = json.load(open(SRC_META, encoding="utf-8"))
meta["id"] = kid; meta["title"] = title; meta["code_file"] = nbfile
if variant in ("d2", "d1", "d2cov", "d1cov", "basecov", "d2focal") and "d2-head-weights" not in " ".join(meta.get("dataset_sources", [])):
    meta.setdefault("dataset_sources", []).append("bobmyersthesecond/d2-head-weights")
json.dump(meta, open(os.path.join(OUTDIR, "kernel-metadata.json"), "w"), indent=2)
print("built FAST %-8s -> %s (cnn_disable=%s, proto_swap=%s)" % (variant, kid, cnn_done, proto_done))
