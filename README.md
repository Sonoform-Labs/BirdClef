# Partial-Label Evaluation of Prototype Networks with Temporal Aggregation in BirdCLEF+ 2026

Working note and supporting code for our BirdCLEF+ 2026 (LifeCLEF 2026) entry.

**Finding:** offline rankings of an interpretable model component do not survive deployment. We use a public BirdCLEF+ 2026 pipeline as a fixed testbed and replace only its prototype member with author-built heads. Offline, on a bank covering 71 of 234 species, the heads look strong; swapped into the full 234-species system and scored on the hidden test, the ranking **reverses**: the best-offline head scores worst. The cause is the partial-label evaluation, the heads damage the 163 species the offline bank omits. The gaps are small, so we frame this as a documented failure mode, not a law.

> Entry: 163rd of ~4,000, private macro-AUC 0.942.

## Layout

- `paper/` — the working note (`main.tex`, `references.bib`, `ceurart.cls`) and `figures/`. Compiles on Overleaf.
- `results/` — `tables/` (CSV metrics) and `reports/` (write-ups). Start at [`results/README.md`](results/README.md).
- `src/part1_audioprotopnet/` — matched-control study: a faithful AudioProtoPNet head vs a sigmoid head on a shared ConvNeXt-Base backbone (5-fold).
- `src/part2_perch_heads/` — the author-built prototype heads on frozen Perch embeddings: offline frontier, deployed-head audit, mechanism diagnostics, exports.
- `src/kaggle_inference/` — builders that swap each head into the deployed notebook for hidden-test scoring.
- `src/figures/` — figure scripts.

## Reproducing

Heavy inputs (audio, the Perch ONNX model, the 708-window embedding bank, head weights) are git-ignored; get the data from Kaggle and Perch v2 (see the paper). Scripts use absolute paths, adjust the `B = r"..."` line at the top of each. Hidden-test scores are recorded Kaggle outcomes, not recomputable (labels are private).

Rough order: **Part I** `train_faithful.py` → `gate{0,1,2}_*.py`; **Part II** `train_ablation.py` → `train_sweeps.py` → `fill_gaps.py` → `export_*.py` → `train_d2_focal.py` → `mechanism_diag.py`; then `kaggle_inference/build_*.py`; then `figures/`. Details in [`docs/reproducibility.md`](docs/reproducibility.md) and [`docs/data_statement.md`](docs/data_statement.md).

## Attribution

The pipeline (Perch + ProtoSSMv2 + SED rank-fusion) is adopted unchanged from public Kaggle notebooks (notably hideyukizushi's ProtoSSMv2) and is not claimed as novel. Author-built: the Part II prototype heads, the full study, the mechanism decomposition, and all analysis. The Part I AudioProtoPNet head is from Heinrich et al. (2025) / BirdSet, used under their terms. This repository ships only first-party code, results, and the paper.

## Citation & license

Cite the preprint ([`CITATION.cff`](CITATION.cff)). Code under MIT ([`LICENSE`](LICENSE)); paper text and figures under CC BY 4.0.
