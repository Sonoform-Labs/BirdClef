# Section draft: The interpretability–accuracy frontier of prototype heads (Part II)

*Standalone draft to be consolidated into `paper_working_note_draft.md`, mirroring the Gate-1/Gate-2 section drafts. Part I (Gates 1–2) asks whether a faithful AudioProtoPNet head on a shared ConvNeXt backbone costs accuracy (answer: macro ~free, small robust avian cost) and whether its prototypes are valid (answer: class-pure, call-focused, taxon-graded). Part II widens the lens from one head to the **design space** of prototype heads, on a different and cheaper substrate, frozen Perch v2 embeddings, and asks: across designs, what is the **shape** of the interpretability-vs-accuracy trade, and where does the head that was actually **deployed in the competition** sit on it?*

## X.1 Motivation and setup

Part I compares exactly two heads (faithful prototype vs matched sigmoid). It cannot say whether the small avian cost is *intrinsic to prototypes* or an artifact of one design point. Part II answers that by sweeping the design space and measuring both axes, discriminative accuracy and prototype validity, for every design under one fixed protocol.

**Substrate.** All Part-II heads sit on top of **frozen Perch v2** embeddings (1536-d), the same foundation model the competition pipeline uses. Working on frozen embeddings makes the head the *only* moving part (the backbone cannot absorb or mask the head's inductive bias) and makes the full sweep cheap (seconds per head, so we can run grids). Evaluation is on the **708 fully-labelled soundscape windows** (59 files × 12 windows) under **GroupKFold(5)** by file, scoring **macro ROC-AUC over the 71 species that have positives** in this set. ROC-AUC is within-fold rank-invariant (verified, as in Part I), so we report per-fold means.

**Prototype validity metric (presence-purity).** For each class prototype we take its nearest soundscape window in cosine space and ask whether that window's (multi-label) annotation contains the prototype's class. The fraction of measured classes for which it does is the head's *purity*. This is the soundscape analogue of Part I's strict single-species purity; it is weaker (multi-label windows) and we flag it as suggestive rather than head-to-head with Gate-2. A random-prototype null sits near the class base-rate (≪ observed).

**Heads.** A single class-prototype per class with a positive-temperature diagonal cosine read-out is the base; we then add, one at a time, the interpretability losses (cluster, separation), a Perch-fusion gate (static per-class, then input-dependent), prototype count *K*, and a learned projection before the cosine. Two reference points anchor the space: **Perch logits alone** (no head) and **ProtoSSMv2**, the prototype head that was actually deployed as the 52%-weight member of our competition blend.

## X.2 The deployed competition head is component-interpretable (ProtoSSMv2 projection)

Before building new heads we audited the one we shipped. ProtoSSMv2 carries exactly **one class-specific prototype per class** with a positive-temperature diagonal cosine read-out and a per-class fusion gate γ_c = σ(fusion_alpha) mixing the prototype score against Perch, an architecture that *does* support a scoped, component-level interpretability claim (we initially doubted this and were wrong; the per-class prototype geometry is real). Projecting its prototypes onto the 708-window bank:

- **Prototype purity = 0.789** [0.69–0.87 bootstrap], **42σ above the random-prototype null**: the deployed prototypes are genuinely class-aligned.
- **Taxon pattern *inverts* AudioProtoPNet's:** Insecta 0.96 > Amphibia 0.88 > Aves 0.64 (Mammalia n=3 / Reptilia n=1 quarantined). Part I's faithful head was *cleanest on birds*; the deployed head, trained on this soundscape bank, is cleanest on insects. **Prototype validity tracks the training domain, not the mechanism**: a result only visible because we measured the same property on two differently-trained prototype systems.
- The taxon effect **survives a frequency control** (Gate-3 discipline): purity correlates with log-positive-count (ρ=0.30), but a logistic regression with log-npos as a covariate still puts Insecta +1.40 / Aves −0.71 vs the Amphibia reference, insect prototypes are valid *even where rare*, avian prototypes only where frequent.
- The learned fusion gate γ_c is **flat at ≈0.50 across all taxa** (fusion_alpha ∈ [−0.027, 0.026]), an **honest negative**: the deployed head did *not* learn to defer to Perch differently for well- vs poorly-covered classes.

So the head we deployed is not a black box bolted to Perch; its per-class prototypes are measurably valid, with a taxon signature set by what it was trained on.

## X.3 Stepwise build-up: one knob per row (the main Part-II table)

Starting from Perch logits and changing exactly one thing per row (single prototype, raw 1536-d, GroupKFold(5)):

| # | config | macro-AUC | purity | marginal effect |
|---|---|---|---|---|
|, | Perch logits alone | 0.739 |, | foundation floor |
| 1 | linear probe | 0.766 |, | learned head, no prototypes |
| 2 | single prototype, no losses | 0.762 | 0.69 | a bare prototype is already a competent classifier |
| 3 | + cluster loss | 0.604 | 0.944 | **+0.25 purity, −0.158 AUC** |
| 4 | + separation loss | 0.605 | 0.958 | marginal (cluster is the driver) |
| 5 | + static-gate Perch fusion | 0.742 | 0.958 | **recovers Perch-level AUC, keeps purity** |
| 6 | + input-dependent gate (**D2**) | **0.830** | **0.437** | **+0.088 AUC over static, −0.52 purity** |
| 7 | 3-proto + cluster/sep (D1≡D3) | 0.606 | 0.958 | capacity is not the lever |
|, | ProtoSSMv2 (deployed) | 0.803 | 0.79 | sits in the middle of the frontier |

![The interpretability–accuracy frontier of prototype heads on frozen Perch v2 embeddings. Each point is a head from the build-up; the cluster loss drives heads down-right (high purity, low AUC) and the input-dependent gate drives them up-left (high AUC, low purity). The deployed ProtoSSMv2 head sits in the interior. Dashed/dotted lines are the no-prototype reference floors (linear probe 0.766; Perch logits 0.739).](paper/fig_frontier.png)

*Figure 1 (`paper/fig_frontier.{png,pdf}`, generated by `_fig_frontier.py`) plots this table as the frontier.*

**Read it as two opposing levers.** The *cluster loss* (row 2→3) is what buys interpretability, +0.25 purity, at a steep accuracy cost (−0.16). The *input-dependent gate* (row 5→6) is what buys accuracy, +0.088 over Perch-level, by hollowing the prototype out (−0.52 purity). Separation loss and prototype count barely move either axis. **Row 5 (static gate) is a genuine sweet spot:** 0.96 purity at Perch-level accuracy, interpretability essentially free, you just do not *beat* Perch. To beat it you must spend the prototype's validity (row 6). The deployed ProtoSSMv2 lands between rows 5 and 6, exactly where a system tuned for leaderboard accuracy under a soft interpretability prior should.

## X.4 The frontier is robust: four sweeps

We pushed on the four axes most likely to break the trade-off. None did.

- **Prototype count K∈{1,2,3,5,10}** (pure head): AUC flat at ~0.60; *presence*-purity stays 0.958 but **per-prototype** purity collapses 0.958→0.435 as K grows, extra prototypes are mostly dead/off-class. Capacity is not the missing lever, and more prototypes make the head *less* interpretable per unit.
- **Prototype embedding space** (pure head): a learned **linear projection to 320-d recovers +0.043 AUC** (0.604→0.647) while keeping purity ≥0.94, the *only* knob that improves the interpretable branch without spending purity. A nonlinear (MLP) projection does not beat linear. Still ~0.18 below the gated head.
- **D2 training length / mixup:** the input-gate head peaks at **25 epochs (0.848)** then overfits the 708-window bank; embedding-mixup hurts.
- **Post-hoc temporal smoothing on D2:** per-file mean/max pooling is neutral-to-negative (0.827→0.814/0.827), matching the competition-wide finding that temporal smoothing does not move this rank metric.

The trade-off survives every axis; the only positive levers are a low-rank metric space for the interpretable branch (+0.04) and the right training length for the accurate branch.

## X.5 Leaderboard confirmation: the frontier on the hidden test (Cat 8)

Every number above is the 708-window OOF proxy. To check the frontier is not an artifact of that small bank, we transplant each author-built head into the **deployed competition inference pipeline** and score it on the **hidden test** via late submission (which returns both public and private scores; standings are locked, so this is a measurement instrument, not a ranking attempt). The swap replaces only the proto member's forward, the head's prototype/gate computed on the test Perch embeddings, and leaves the entire downstream (per-class fusion, residual correction, rank post-processing, and the final four-member rank-blend) identical to the shipped system.

**A runtime caveat that became a constraint.** The full deployed pipeline (four audio passes over ~700 files: Perch, SED, and two auxiliary CNNs) runs *at* the 90-minute hidden-test limit, so on late re-runs it times out by variance, the unmodified baseline timed out too. To obtain a consistent, scoreable set we ran all four variants on the deployed blend's **Proto+SED two-way core** (the blend already gates the 8%- and 5%-weight CNN members on file existence and falls back to Proto+SED; we disable them uniformly, so only the proto member differs). Scores are therefore on this core, ~0.0025 below their full-pipeline equivalents.

| head as proto member | OOF AUC / purity | public LB | **private LB** | rank |
|---|---|---|---|---|
| **ProtoSSMv2 (deployed baseline)** | 0.803 / 0.79 | 0.94908 | **0.94218** | **1** |
| drop proto (raw Perch in the slot) | 0.739 /, | 0.94513 | **0.93920** | 2 |
| D1, pure prototype (≡D3) | 0.605 / 0.99 | 0.93767 | **0.93810** | 3 |
| D2, input-gated fusion | **0.830** / 0.44 | 0.94050 | **0.93743** | 4 |

**Pre-registered expectation** (written before reading the scores): if the OOF frontier transferred, the blend should order **D2 ≳ baseline > drop > D1**. **It did not.** What we observe instead, robustly on *both* public and private:

1. **The deployed ProtoSSMv2 head is best**: top on both splits, +0.003 private over dropping it. The shipped prototype member genuinely helps the production ensemble.
2. **Both author-built heads score *below* drop-proto**: swapping in either head is worse than putting raw Perch in the slot. The heads, despite spanning the OOF frontier, *actively hurt* the system.
3. **OOF accuracy anti-predicts the leaderboard.** D2, the highest-OOF head and the "accuracy winner" of the frontier, is last on private; the deployed head (lower OOF, 0.803) beats it by ~0.005. The offline ranking inverts under deployment.

The most plausible mechanism: the OOF frontier is measured over the 71 species present in the 708-window bank, whereas the leaderboard is macro-AUC over all 234. The author heads, trained only on that bank, produce flat non-discriminative scores for the ~163 absent species; fused at 60% proto weight these *dampen* the foundation's signal for those classes and drag the 234-species macro-AUC.

### X.5.1 We tested the mechanism directly: three controls, and it holds

We did not leave the mechanism as a hypothesis. (i) **Diagnostic:** on the 708-window bank, D2's input gate stays at ~0.77 *even on the 163 species it never trained positives for*: it does not learn to defer to the foundation there, so it overwrites Perch's discriminative scores with a flat prototype (only ~35% of Perch's per-species variance survives; rank-corr ρ≈0.28). (ii) **Coverage-gated heads**: prototype on the 71 covered species, raw Perch on the 163 absent, *helped both heads on both splits* (D1: 0.93810→0.93964; D2: 0.93743→0.93777), confirming the absent-species drag is real, but recovered only to the **Perch floor** (≈ drop 0.93920), not the deployed head. Coverage is therefore *necessary but not sufficient*. (iii) **Control, coverage-gating the deployed head itself** *hurt* it (0.94218→0.93954): decomposing ProtoSSMv2's +0.003 edge over raw Perch, only +0.0003 comes from covered species and **+0.0026 from the absent ones**. So ~90% of the deployed head's advantage is its ability to *generalize to species it never trained positives for*: the exact inverse of our linear heads.

This isolates the deployed head's edge to two things: on-domain training and an architecture (bidirectional + residual SSM over the 12-window sequence) that generalizes across the full label space. We tested whether the first alone, **broad training**: closes the gap, using a head we could only build because our local Perch *exactly* reproduces the deployed one (embedding correlation 1.000000 on the 708-window bank, which also let us recover the 14795→234 mapping by column-matching). We embedded focal recordings across 206 species and trained the gated head on the combined soundscape+focal bank covering **all 234 species** (no absent-species hole at all). **It scored worst of all (0.93522).** Broad training *backfired*: the focal recordings are off-domain relative to the soundscape test, and 5090 focal windows swamped the 708 on-domain ones, so the prototypes became focal-centric and matched the deployment *less* well, even on covered species. This is the **focal→soundscape domain gap**, the central challenge of the benchmark, surfacing inside the prototype head.

The full ladder (Figure 3): deployed 0.94218 > coverage-gated heads ≈ 0.9396 > raw Perch 0.93920 > ungated heads ≈ 0.9374 > broad-trained 0.93522. **Every intervention we built underperforms the shipped head**, and the offline interpretability–accuracy frontier predicts none of this ordering. The deployed head's advantage is *on-domain training plus architectural generalization*: neither label-coverage gating nor broad off-domain training substitutes for it. *(Figures: 1 = offline frontier; 2 = offline-vs-deployment inversion; 3 = the leaderboard ladder, `paper/fig_lb_bars.{png,pdf}`.)*

## X.6 What Part II adds over Part I

Part I shows prototype interpretability is *roughly free at one design point* (faithful AudioProtoPNet ≈ matched sigmoid). Part II adds two things. First, **offline**: a sharp interpretability–accuracy frontier across head designs, the cluster loss buys purity at an accuracy cost, an input-dependent gate buys accuracy by hollowing the prototype, a static gate sits free-but-capped at the foundation's ceiling, and the deployed ProtoSSMv2 lands in the interior. Second, and more pointed, **on deployment the frontier does not transfer**: scored inside the real competition ensemble on the hidden test, every author-built head, including the one that *won* the offline accuracy axis, underperforms both the shipped head and even removing the head entirely; the offline ranking inverts. The contribution is thus the frontier **and its falsification as a deployment predictor**: a component-level interpretability/accuracy characterization on a matched, narrow label space is necessary but *not sufficient*, because a head tuned to that bank degrades on the full label space and trades away the ensemble diversity the deployed head provides. This is a cautionary, honest result, not a new architecture, and not a tidy confirmation, but a measured gap between offline component metrics and system-level deployment that the BirdCLEF setting (locked standings, late-submission private scoring, a real shipped head) is unusually well suited to expose.

## Citations to add when porting to LaTeX (do NOT fabricate: verify each)
Part II is empirical, but when consolidated it should cite: **Perch** (the frozen embedding substrate; Ghani et al. 2023, *Global birdsong embeddings enable superior transfer learning*, Sci. Reports; verify the Perch v2 model card / exact ref the competition pipeline used) and **Mamba / selective SSM** (Gu & Dao 2023, arXiv:2312.00752) for ProtoSSMv2's bidirectional sequence model. Reuse existing keys `chen2019protopnet`, `heinrich2025audioprotopnet`, `nauta2020thislooks`, `ridnik2021asymmetric` for the prototype-head lineage and loss. The deployed head (ProtoSSMv2) is a public Kaggle notebook (hideyukizushi), not a paper, cite as a notebook URL, not a fabricated bib entry.

## X.7 Limitations specific to Part II

- Purity here is **presence-purity on multi-label soundscape windows**: suggestive, not the strict single-species purity of Gate-2; we never headline the pooled 0.79 without its taxon breakdown.
- Evaluation is **71 species / 708 windows**: AUC CIs are wide for individual cells, though the frontier (large, monotone effects) is robust; bootstrap CIs are reported for the anchor heads (D2 0.827 [0.78–0.87]; D2 beats the linear probe and Perch significantly but is **n.s. vs the deployed ProtoSSMv2**, +0.024 CI spans 0, "competitive with," not "beats").
- The two pure-prototype designs we specified (D1, D3) are **identical as coded**; we report one row and say so rather than padding the table.
- Heads are trained on **708 soundscape windows**, not focal recordings, the same domain as the deployed head, which is why the taxon-purity signature matches it and not Part I's bird-clean gradient.
