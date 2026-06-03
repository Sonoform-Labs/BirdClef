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
