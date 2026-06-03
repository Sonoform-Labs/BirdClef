# Reproducibility

Three tiers, matching how each number was produced.

## Fully reproducible (given data + checkpoints)
- Offline out-of-fold (OOF) macro-AUC for each head (`results/offline_oof_summary.csv`), from the training and evaluation scripts in `src/part2_perch_heads/`.
- Prototype purity calculations.
- The interpretability--accuracy frontier and figures, from the CSV summaries via `src/figures/`.
- The matched-control AudioProtoPNet study (`src/part1_audioprotopnet/`): 5-fold on a ConvNeXt-Base backbone.

## Partially reproducible
- Deployment submissions can be re-run **inside the Kaggle environment** by swapping a head into the deployed inference notebook (`src/kaggle_inference/`). The resulting scores depend on the hidden test.

## Not independently reproducible
- Hidden-test public/private macro-AUC (`results/leaderboard_rows.csv`): the hidden-test labels are not public, so these cannot be recomputed outside Kaggle. They are descriptive submission outcomes, not recomputable metrics.

## Notes
- Scripts use absolute paths to the original working directory; adjust the `B = r"..."` line at the top of each before running.
- Heavy inputs (audio, the Perch ONNX model, the 708-window embedding bank, trained head weights) are git-ignored; see `docs/data_statement.md`.

## Artifact map
| Artifact | Path |
|---|---|
| Offline OOF training / evaluation | `src/part2_perch_heads/` (`train_ablation.py`, `train_sweeps.py`, `export_d1.py`, `export_d2.py`, `train_d2_focal.py`, `pertaxon_oof.py`, `fill_gaps.py`) |
| Leaderboard build (head swaps) | `src/kaggle_inference/` (`build_fast.py`, `build_swap.py`, `build_d2swap2.py`, `build_d2rw.py`, `sub_detail.py`) |
| no-SSM temporal ablation | `src/kaggle_inference/build_nossm.py` |
| Figure generation | `src/figures/` (`make_figs.py`, `make_figs2.py`, `make_domainfig.py`) |
| Result tables | `results/leaderboard_rows.csv`, `offline_oof_summary.csv`, `mechanism_diagnostics.csv`, `GATE_PROTOSSM_projection.csv` |
| Covered / absent species | `results/covered_species_71.csv`, `results/absent_species_163.csv`, `results/species_coverage_README.md`; generator `src/part2_perch_heads/export_species_lists.py` |
| D2-FOCAL provenance | `results/d2_focal_provenance.md` (focal-bank builder: `src/part2_perch_heads/focal_embed.py`) |
| Matched-control study (Appendix A) | `src/part1_audioprotopnet/` |
| Environment record | `environment_freeze.txt` (local versions + Kaggle runtime metadata); `requirements.txt` (unpinned names) |
| Paper source | `paper/main.tex`, `paper/references.bib`, `paper/ceurart.cls`, `paper/figures/` |

## What can and cannot be reproduced
- **Reproducible from this repo** (given the competition data + checkpoints): the offline OOF analyses, prototype-purity / interpretability metrics, the figures, the covered/absent species lists, the focal-bank provenance, and the build scripts that assemble each submitted notebook.
- **NOT independently reproducible:** the hidden-test public/private macro-AUC. BirdCLEF+ 2026 hidden-test labels are private, so leaderboard scores cannot be recomputed outside the Kaggle evaluation platform. Every score in `results/leaderboard_rows.csv` is **recorded from the Kaggle leaderboard output** of an actual submission, not a locally computed metric.
