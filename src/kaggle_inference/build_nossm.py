# -*- coding: utf-8 -*-
"""Build the ProtoSSMv2 no-SSM temporal-ablation submission notebook.

This script produced the `ProtoSSMv2 no-SSM` leaderboard row
(public 0.94805 / private 0.94089; see ../../results/leaderboard_rows.csv).

What it ablates
---------------
It takes the fixed deployed submission notebook and, in the submission cell, neutralizes the
temporal module of the deployed prototype member immediately before it is run:
  * the deployed proto model is an SWA ``AveragedModel``; we unwrap it to the underlying
    LightProtoSSM via ``getattr(proto_model, 'module', proto_model)``;
  * ``ssm_fwd`` (the bidirectional selective-SSM ModuleList) is replaced with an empty
    ModuleList, so the ``zip(self.ssm_fwd, ...)`` loop runs 0 iterations -> the SSM is bypassed;
  * ``use_cross_attn`` is set to False -> the per-layer temporal cross-attention is disabled.
With both off, ``h = input_proj + pos_enc + metadata`` flows straight to the prototype cosine.

What is kept unchanged
----------------------
The prototypes, the fusion gate (alpha / class_bias), the site/hour metadata, the Perch-distilled
SED member, the rank fusion, the ecological priors, and all post-processing are untouched: they
live in the base notebook, which this script does not modify beyond the two SSM-bypass lines.
The auxiliary CNN members are disabled with the same FAST CORE 2-way Proto+SED fallback used for
every other fast-core leaderboard row, so the ablation is compared on equal footing.

Note on a corrected earlier version: a first attempt edited the ``ProtoSSMv2`` class in an
OOF/dev cell, which does NOT feed the submission, so it scored bit-identical to the baseline
(0.94908 / 0.94218). This corrected version targets the real deployed ``proto_model`` instance
right before ``run_tta_proto``.

Usage: ``python build_nossm.py`` writes the ablated notebook + kernel-metadata.json to OUTDIR,
which is then pushed/submitted via the Kaggle kernel flow (kernel id below). Paths use the
original working directory (``B``); adjust before running. The source deployment notebook and
head weights are not redistributed (see docs/data_statement.md)."""
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
PROTO_RUN_PREFIX = "proto_out = run_tta_proto("   # the submission inference (NOT proto_tr_out)
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
json.load(open(os.path.join(OUTDIR, NBFILE), encoding="utf-8"))  # valid JSON
print("rebuilt noSSM kernel (CORRECTED): cnn_disable=%s, ssm_bypass=%s" % (cnn_done, ssm_done))
