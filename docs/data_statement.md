# Data statement

**Sources.** Training and soundscape audio: the BirdCLEF+ 2026 (LifeCLEF 2026) Kaggle dataset, not redistributed here; obtain it from the competition under its license. Embeddings and logits: the public Perch v2 model. The deployment pipeline (Perch + ProtoSSMv2 + SED rank-fusion) is adopted unchanged from public Kaggle notebooks (notably hideyukizushi's ProtoSSMv2); the swapped-in prototype heads are the only author-built component.

**Offline bank.** Offline evaluation uses a 708-window labelled soundscape bank covering 71 of the 234 target species. The other 163 have no positives in it, which is central to the paper's finding.

**Hidden test.** BirdCLEF+ 2026 hidden-test labels are private. Every leaderboard score here is a Kaggle submission outcome and cannot be recomputed off-platform.

**Redistribution.** Do not redistribute competition data or hidden labels. This repository ships only first-party code, result summaries, and the paper; heavy or derived inputs (audio, the Perch model, the embedding bank, head weights) are git-ignored.
