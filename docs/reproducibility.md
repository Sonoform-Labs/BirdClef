# Reproducibility

Three tiers, by how each number was produced.

**Fully reproducible** (given the BirdCLEF+ 2026 data + checkpoints): offline OOF macro-AUC per head (`results/tables/offline_oof_summary.csv`), prototype purity, the interpretability–accuracy frontier, the figures, and the matched-control AudioProtoPNet study (`src/part1_audioprotopnet/`, 5-fold ConvNeXt-Base). Code in `src/part2_perch_heads/` and `src/figures/`.

**Partially reproducible:** deployment submissions can be re-run inside Kaggle by swapping a head into the deployed notebook (`src/kaggle_inference/`); scores depend on the hidden test.

**Not independently reproducible:** hidden-test macro-AUC (`results/tables/leaderboard_rows.csv`). The labels are private, so every score is a recorded Kaggle submission outcome, not a recomputable metric.

Scripts use absolute paths, adjust the `B = r"..."` line at the top of each. Heavy inputs (audio, the Perch ONNX model, the 708-window bank, head weights) are git-ignored; see `data_statement.md`. Dependencies: `requirements.txt` (unpinned names).

## Artifact map

| Artifact | Path |
|---|---|
| Offline OOF training / eval | `src/part2_perch_heads/` (`train_ablation.py`, `train_sweeps.py`, `export_d1.py`, `export_d2.py`, `train_d2_focal.py`, `pertaxon_oof.py`, `fill_gaps.py`) |
| Leaderboard head-swaps | `src/kaggle_inference/` (`build_fast.py`, `build_swap.py`, `build_d2swap2.py`, `build_d2rw.py`, `build_nossm.py`, `sub_detail.py`) |
| Figures | `src/figures/` |
| Metric tables | `results/tables/` |
| Covered / absent species | `results/tables/covered_species_71.csv`, `absent_species_163.csv`; generator `src/part2_perch_heads/export_species_lists.py`; notes `results/reports/species_coverage.md` |
| D2-FOCAL provenance | `results/reports/d2_focal_provenance.md` (focal-bank builder `src/part2_perch_heads/focal_embed.py`) |
| Matched-control study | `src/part1_audioprotopnet/` |
| Paper | `paper/` |
