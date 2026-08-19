import json, io, sys, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
B = r"D:\Python\BirdClef"
SRC_NB = B + r"\kaggle_push_v52\birdclef-2026-v52-exp.ipynb"
SRC_META = B + r"\kaggle_push_v52\kernel-metadata.json"
OUTDIR = B + r"\kaggle_push_nossm"
KID, TITLE, NBFILE = "bobmyersthesecond/birdclef-2026-nossm-ablation", "BirdCLEF 2026 noSSM ablation", "birdclef-2026-nossm.ipynb"

CNN_TARGET = '_exp_result = _run_cnn_inference(_EXP_DIR, "B0-Focal-4th")'
CNN_DISABLE = [
"# === FAST CORE: skip aux CNN members; blend falls back to Proto+SED ===",
"_run_cnn_inference = lambda *a, **k: None",
"print('>>> FAST CORE: B0/B2 CNN inference disabled -> Proto+SED 2-way blend')",
]
PROTO_RUN_PREFIX = "proto_out = run_tta_proto("
SSM_BYPASS = [
"# === noSSM ABLATION: bypass bidirectional SSM + per-layer cross-attention on the submission proto model ===",
"_pm_abl = getattr(proto_model, 'module', proto_model)  # unwrap SWA AveragedModel -> the underlying LightProtoSSM",
"_pm_abl.ssm_fwd = type(_pm_abl.ssm_fwd)()  # empty ModuleList -> the zip(...) SSM/cross-attn loop runs 0 iters",
"_pm_abl.use_cross_attn = False",
"print('>>> noSSM ABLATION applied: ssm_fwd len =', len(_pm_abl.ssm_fwd), '| use_cross_attn =', _pm_abl.use_cross_attn)",
]

nb = json.load(open(SRC_NB, encoding="utf-8"))
cnn_done = ssm_done = False
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
    if not ("if 'Model_7' in _ensemble_models" in src and "to_csv(_file_name_submission" in src
            and PROTO_RUN_PREFIX in src):
        continue
    out = []
    for ln in src.splitlines(keepends=True):
        stripped = ln.lstrip()
        ind = ln[:len(ln) - len(stripped)]
        if CNN_TARGET in ln and not cnn_done:
            out.append("".join(ind + s + "\n" for s in CNN_DISABLE)); cnn_done = True
        if stripped.startswith(PROTO_RUN_PREFIX) and not ssm_done:
            out.append("".join(ind + s + "\n" for s in SSM_BYPASS)); ssm_done = True
        out.append(ln)
    cell["source"] = "".join(out)
    break
assert cnn_done, "CNN target not found"
assert ssm_done, "'proto_out = run_tta_proto(' not found in submission cell"

os.makedirs(OUTDIR, exist_ok=True)
json.dump(nb, open(os.path.join(OUTDIR, NBFILE), "w", encoding="utf-8"))
meta = json.load(open(SRC_META, encoding="utf-8"))
meta["id"], meta["title"], meta["code_file"] = KID, TITLE, NBFILE
json.dump(meta, open(os.path.join(OUTDIR, "kernel-metadata.json"), "w"), indent=2)
json.load(open(os.path.join(OUTDIR, NBFILE), encoding="utf-8"))
print("rebuilt noSSM kernel (CORRECTED): cnn_disable=%s, ssm_bypass=%s" % (cnn_done, ssm_done))
