# Data statement

## Sources
- **Training and soundscape audio** come from the BirdCLEF+ 2026 (LifeCLEF 2026) competition dataset on Kaggle. We do not redistribute it; obtain it from the competition page under its license.
- **Perch v2** embeddings and logits are produced by the public Perch model.
- The **deployment pipeline** (Perch + ProtoSSMv2 + SED rank-fusion) is adopted unchanged from public Kaggle notebooks (notably hideyukizushi's ProtoSSMv2); see the paper. The prototype heads we swap in are the only author-built component.

## Offline labelled bank
Offline evaluation uses a 708-window labelled soundscape bank covering **71 of the 234** target species. Its construction (window selection, label alignment, GroupKFold splits) is described in the paper. The remaining **163 species have no positive examples** in this bank, which is central to the paper's finding.

## Hidden test
The BirdCLEF+ 2026 hidden-test labels are **not public**. Every leaderboard score in this repository is the outcome of a Kaggle submission and **cannot be independently recomputed** outside the competition platform. They are reported as descriptive system-level submission outcomes.

## Redistribution
Do not redistribute competition data or hidden labels. This repository contains only first-party code, result summaries, and the paper; heavy or derived inputs (audio, the Perch model, the embedding bank, trained head weights) are git-ignored.
