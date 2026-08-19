# Results

Metric tables in `tables/`, write-ups in `reports/`. Start with `reports/results.md` (the consolidated record). Hidden-test scores are recorded Kaggle outcomes, not recomputable; see `../docs/reproducibility.md`.

## tables/

| File | Contents |
|---|---|
| `leaderboard_rows.csv` | Hidden-test public/private macro-AUC per head-swap (the reversal). |
| `offline_oof_summary.csv` | Offline OOF macro-AUC per head. |
| `mechanism_diagnostics.csv` | Gate / correlation / variance diagnostics, covered vs absent species. |
| `gate_protossm_projection.csv` | Per-prototype projection data from the ProtoSSMv2 audit. |
| `covered_species_71.csv` / `absent_species_163.csv` | The 71 covered / 163 absent target species. |

## reports/

| File | Contents |
|---|---|
| `results.md` | Consolidated results record (Parts I + II). Start here. |
| `part2_protossm.md` | Part II detail: frontier, leaderboard reversal, mechanism. |
| `part1_gate1.md` / `part1_gate2.md` | Part I detail (accuracy cost; prototype validity). |
| `preregistration.md` | Pre-registered predictions (which the leaderboard outcome failed). |
| `species_coverage.md` | How the covered/absent split is built. |
| `d2_focal_provenance.md` | D2-FOCAL focal-bank construction. |
