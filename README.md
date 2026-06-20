# Partial-Label Evaluation of Prototype Networks with Temporal Aggregation in BirdCLEF+ 2026

Working note and supporting code for our **BirdCLEF+ 2026** (LifeCLEF 2026) submission.

**Headline finding:** offline rankings of an interpretable component do **not** reliably survive deployment. We use a public BirdCLEF+ 2026 pipeline as a fixed *deployment testbed* and replace only its prototype member with author-built heads. Offline (on a bank covering just 71 of 234 species) the heads look strong; swapped into the full 234-species system and scored on the hidden test, the ranking **reverses** (the highest-offline head scores *worst*). The cause is the partial-label evaluation: the heads damage the 163 species the offline bank omits. The gaps are small, so we frame this as a documented failure mode, not a universal law.

> Competition entry: BirdCLEF+ 2026, placed 163rd, private macro-AUC 0.942.

## Repository layout
- **`paper/`**: the working note (`main.tex`, `references.bib`, `ceurart.cls`) plus `figures/` (7 PDFs). Compiles on Overleaf with the bundled `ceurart.cls`.
- **`results/`**: results and write-ups.
  - `RESULTS.md`: single-file summary of everything (start here).
  - `GATE_PROTOSSM_RESULTS.md`: Part II detail (frontier, leaderboard reversal, mechanism).
  - `GATE1_RESULTS.md`, `GATE2_RESULTS.md`: Part I detail; `PAPER_PREREGISTRATION.md`; `drafts/` (prose).
- **`src/`**
  - `part1_audioprotopnet/`: faithful AudioProtoPNet vs matched-sigmoid study (Gates 0–2), 5-fold, ConvNeXt-Base.
  - `part2_perch_heads/`: author-built prototype heads on frozen Perch embeddings: the offline frontier, the deployed-ProtoSSMv2 audit, mechanism diagnostics, head export, and focal re-training.
  - `kaggle_inference/`: builders that swap each head into the deployed inference notebook for hidden-test leaderboard evaluation.
  - `figures/`: figure-generation scripts.

## What the paper shows
The paper uses the public pipeline as a fixed **deployment testbed** and replaces only its prototype member with author-built heads (D1/D2/static-gate) on frozen Perch embeddings. Offline, on a 708-window soundscape bank covering 71 of 234 species, the heads trace a clean interpretability↔accuracy frontier and one even exceeds the deployed head. Swapped into the full 234-species system and scored on the hidden test, the offline ranking **reverses**: every author head underperforms raw Perch, and the highest-offline head is worst. The mechanism is the partial-label gap: the offline bank omits 163 species, and the heads damage exactly those uncovered species in the full macro-AUC system, while the deployed head generalizes to them.

A **supporting matched-control study** (faithful AudioProtoPNet vs a matched sigmoid head on a shared ConvNeXt backbone; in the appendix) confirms prototype interpretability can be near-free and call-focused in a clean setting, which sharpens the deployment failure.

## Reproducing
- **Heavy inputs are git-ignored** (audio, the Perch ONNX model, the 708-window embedding bank, trained head weights), as they are large and/or derived. Sources: Kaggle BirdCLEF+ 2026 data, Perch v2, and the public ProtoSSMv2 notebook (see the paper).
- Scripts use **absolute paths** to the original working directory; adjust the `B = r"..."` line at the top of each before running.
- Rough order: **Part I** `train_faithful.py` → `gate0/gate1/gate2_*.py`; **Part II** `train_ablation.py` → `train_sweeps.py` → `fill_gaps.py` → `export_*.py` → `perch_local.py`/`focal_embed.py`/`train_d2_focal.py` → `mechanism_diag.py`; then `kaggle_inference/build_*.py` for the leaderboard swaps; then the `figures/` scripts.

## Attribution
The competition **pipeline** (Perch + ProtoSSM + SED rank-fusion) is adopted unchanged from public Kaggle notebooks (notably hideyukizushi's ProtoSSMv2) and is **not** claimed as novel, as stated in the paper. The **author-built** contributions are the Part II prototype heads, the full experimental study, the mechanism decomposition, and all analysis. The AudioProtoPNet head (Part I) is from Heinrich et al. (2025) / the BirdSet repository, used under their terms. This repository deliberately excludes those upstream codebases; it contains only first-party code, results, and the paper.

## Reproducibility & data
See [`docs/reproducibility.md`](docs/reproducibility.md) and [`docs/data_statement.md`](docs/data_statement.md). In short: offline OOF metrics and prototype purity are reproducible given the BirdCLEF+ 2026 data and checkpoints; **hidden-test leaderboard scores cannot be independently recomputed** because the hidden labels are not public, so they are reported as descriptive Kaggle-submission outcomes. Summary numbers live in [`results/`](results/): `leaderboard_rows.csv`, `offline_oof_summary.csv`, `mechanism_diagnostics.csv`.

## Citation
If you use this repository, please cite the associated preprint (see [`CITATION.cff`](CITATION.cff)).

## License
Code is provided under the [MIT License](LICENSE). Paper text and figures are provided under CC BY 4.0, consistent with the paper's CEUR declaration.
