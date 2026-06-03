# D2-FOCAL provenance

D2-FOCAL is the broad-training sensitivity variant: the gated-fusion (D2) head trained on the
708 soundscape windows **plus** focal recordings, to test whether broader species coverage during
training improves deployment. It scored worst of all variants.

## Focal-window bank (`focal_bank.npz`)
- **Source script:** `src/part2_perch_heads/focal_embed.py`.
- **Selection rule (exactly as implemented):**
  - for each of the 234 target species, take its `train.csv` recordings, sort by `rating`
    descending, keep the **top 15 files** (`N_FILES_PER_SP = 15`);
  - load each file as up to twelve 5-second windows (`WIN = 160000` samples @ 32 kHz,
    `N_WIN_MAX = 12`);
  - run Perch, then select the **top-2 windows per FILE** by that species' Perch logit
    (`TOPK_WIN = 2`; `order = np.argsort(-logit[:, bc])[:2]`). If the species is Perch-unmapped,
    fall back to the first <= 2 windows of the file.
  - **This is top-2 per FILE, not top-2 per species** (each species contributes up to
    15 files x 2 windows).
- **Total focal windows:** **5,090**  (`focal_bank.npz['emb'].shape == (5090, 1536)`).
- **Focal-covered species:** **206**  (`len(np.unique(focal_bank.npz['y'])) == 206`).

## Soundscape windows
- **708** windows — the standard offline bank (`full_perch_arrays.npz`), 59 files x 12 windows.

## Mixing / weighting (`src/part2_perch_heads/train_d2_focal.py`)
- Focal and soundscape examples are **concatenated and weighted equally** (`np.concatenate`);
  soundscape windows are multi-label, focal windows are single one-hot.
- **No class or taxon balancing**, no batch-composition control; full-batch training.
- Hyperparameters identical to D2: 60 epochs, AdamW lr 1e-3 / weight_decay 1e-4,
  CosineAnnealingLR(T_max=60), FocalBCE gamma=2.0 (no cluster/separation loss), `torch.manual_seed(0)`.

## Evaluation and export
- **No offline OOF/AUC is computed for D2-FOCAL** (the trainer prints only a final train loss and a
  numpy-vs-torch sanity check). It is evaluated by leaderboard only.
- Exported as `d2focal_weights.npz`; submitted via `src/kaggle_inference/build_fast.py d2focal`
  (Kaggle kernel `bobmyersthesecond/birdclef-2026-d2-focal`), same fast-core as the other rows
  (auxiliary CNN members disabled, Proto+SED 2-way blend).
- **Comparison protocol:** leaderboard public/private AUC against D1 / D2 / Drop / ProtoSSMv2,
  all built through the same harness differing only in the prototype-member swap.
- **Leaderboard score:** public **0.93872** / private **0.93522** (worst of all variants; see
  `results/leaderboard_rows.csv`).

## Note for the manuscript
The paper currently describes the focal selection as "top-2 per species"; the implemented rule is
**top-2 per file** (the 5,090 / 206 counts are consistent only with per-file selection). This file
records the implemented behaviour; the paper wording is flagged for the next manuscript pass.
