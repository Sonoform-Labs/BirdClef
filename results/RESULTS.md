# BirdCLEF+ 2026: Paper Results (consolidated)

*Single-file record of all experimental results. Two parts: Part I = faithful AudioProtoPNet interpretability study (offline, ConvNeXt); Part II = prototype heads on Perch embeddings + the leaderboard investigation. Figures referenced live in `paper/fig_*.png`.*

---

## TL;DR: the headline findings

1. **Interpretability is roughly free at one design point** (Part I): a faithful AudioProtoPNet head costs ~0 macro-AUC vs an identical sigmoid head (Δ −0.007, 95% CI spans 0); small robust avian cost (−0.011). Its prototypes are class-pure and call-focused (purity 0.774 ≫ null), strongest for birds.
2. **Across head *designs* there is a sharp interpretability↔accuracy frontier** (Part II, offline): a cluster loss buys prototype purity at a steep accuracy cost; an input-dependent gate buys accuracy by hollowing the prototype out. The deployed competition head sits in the interior.
3. **The offline frontier does NOT transfer to deployment** (Part II, leaderboard, the centerpiece): swapped into the real competition ensemble on the hidden test, the highest-offline-accuracy head scores *worst*; the deployed head wins; every head we built underperforms even raw Perch.
4. **Mechanism, fully decomposed:** our linear heads dampen the foundation on the ~163 species absent from the training bank; the deployed head's edge is ~90% its ability to *generalize to those untrained species* (architecture) plus on-domain training. Neither coverage-gating nor broad off-domain training recovers it, broad training actively *backfires* via the focal→soundscape domain gap.

---

# PART I: Faithful AudioProtoPNet vs a matched sigmoid head (offline, 5-fold)

Shared ConvNeXt-Base backbone; only the head differs (prototype vs sigmoid); identical init/data/folds/aug/epochs. Metric = per-fold macro ROC-AUC (rank-invariant within fold).

### Gate 0: CPU feasibility (measured on Kaggle)
ConvNeXt-Base + faithful head times at **torch 51.6 / ORT 49.6 / OpenVINO 62.7 min @ 700 files**: all under the 70-min budget. (OpenVINO did *not* speed up on 4 cores.)

### Gate 1: accuracy cost of the prototype head
- **Macro Δ(proto − sigmoid) = −0.007 ± 0.006, 95% CI spans 0** → accuracy-competitive (interpretability ~free on macro).
- **Aves −0.011 [−0.014, −0.008], robust (5/5 folds)**: the one solid effect, a small avian cost.
- No robust per-taxon effect elsewhere (Amphibia/Insecta sign-flip). Mammalia (n=8) / Reptilia (n=1) quarantined.
- **Methodology point:** naive cross-fold pooling gave a spurious −0.034 / insect −0.17 ("costly"); per-fold rank-normalization collapses it to +0.0003, a pooling artifact, not a truth. CE-scaling sensitivity (0.5×–2×): headline holds.

### Gate 2: prototype validity (inference on the 5 saved faithful models)
- **Purity (nearest-patch own-class) 0.774 vs random-proto null 0.004**; Aves **0.951**, Amphibia 0.562, Insecta 0.107 (coverage-limited, 3/28 clean species → not a claim).
- **Shortcut rate** (nearest patch in low-energy cell, 3 cutoffs 25/33/50): all below null → prototypes prefer call-energy, not background. Aves robust; Amphibia mixed (avoids silence but not call-peaked at 50%).
- **Verdict:** meaningful overall + for Aves; taxon-graded (amphibian partial, insect data-limited).

---

# PART II: Prototype heads on frozen Perch embeddings

Substrate: frozen Perch v2 (1536-d). Eval: 708 labelled-soundscape windows, GroupKFold(5), macro-AUC over the **71 species with positives**. Purity = nearest soundscape window contains the prototype's class (presence-purity).

## II.1: The deployed competition head (ProtoSSMv2) audit
1 class-specific prototype each, positive-temp diagonal cosine read-out, per-class fusion gate, bidirectional + residual SSM.
- **Fusion gate γ flat ≈ 0.50 across all taxa** (no learned proto-vs-Perch differentiation), honest negative.
- **Prototype purity 0.789 [0.69–0.87], 42σ above null.** Taxon: **Insecta 0.96 > Amphibia 0.88 > Aves 0.64**: *inverts* AudioProtoPNet's bird-clean gradient → validity tracks the **training domain**, not the mechanism.
- Taxon pattern **survives a frequency control** (purity↔log-npos ρ=0.30, but logistic with log-npos covariate keeps Insecta +1.40 / Aves −0.71 vs Amphibia).

## II.2: The offline frontier (`train_ablation.py`, `train_sweeps.py`)

**Stepwise build-up, one knob per row (macro-AUC / purity):**
| # | config | AUC | purity | effect |
|---|---|---|---|---|
|, | Perch logits alone | 0.739 |, | foundation floor |
| 1 | linear probe | 0.766 |, | learned head, no prototypes |
| 2 | single prototype, no losses | 0.762 | 0.69 | bare prototype already competent |
| 3 | + cluster loss | 0.604 | 0.944 | **+0.25 purity, −0.158 AUC** (the interpretability lever) |
| 4 | + separation loss | 0.605 | 0.958 | marginal, cluster is the driver |
| 5 | + static-gate Perch fusion | 0.742 | 0.958 | **recovers Perch-level AUC, keeps purity** (free-interp sweet spot) |
| 6 | + input-dependent gate (**D2**) | **0.830** | **0.437** | **+0.088 AUC, −0.52 purity** (accuracy by spending purity) |
| 7 | 3-proto + cluster/sep (D1≡D3) | 0.606 | 0.958 | capacity not the lever |
|, | ProtoSSMv2 (deployed) | 0.803 | 0.79 | mid-frontier |

**Sweeps:** prototype count K∈1..10 → AUC flat ~0.60 (per-proto purity collapses 0.96→0.44); embedding space → **linear-320 projection the only purity-preserving AUC gain (0.604→0.647)**; D2 training length peaks **25 ep (0.848)** then overfits; temporal smoothing neutral-to-negative.

**Per-taxon OOF accuracy (`_pertaxon_oof.py`, covered species):**
| taxon | linear probe | D2 (gated) | D1 (pure proto) |
|---|---|---|---|
| Aves (25) | 0.699 | **0.795** | 0.554 |
| Amphibia (17) | 0.799 | **0.884** | 0.689 |
| Insecta (25) | 0.835 | 0.828 | 0.620 |
| all (71) | 0.766 | 0.827 | 0.607 |

→ D2's accuracy gain is largest where the foundation is weakest (Aves/Amphibia) and ~0 on Insecta; D1's collapse is taxon-general. The accuracy-gain gradient is **orthogonal** to the purity gradient (Insecta-cleanest).

**Figure 1** = the offline frontier (purity vs AUC), `paper/fig_frontier.png`.

**Completeness, the remaining never-run rows (`_fill_gaps.py`):**
- **Full-spec D2 (input gate + cluster + sep) = 0.691 / 0.958** vs as-built D2 (no loss) 0.830 / 0.437. The purity collapse is a *direct trade*, not the gate hollowing a pure prototype: force purity back and accuracy collapses *below* the static gate. **Accuracy XOR purity, no gated config is both.** (Sharpens the frontier.)
- **K-sweep on the gated head:** K=1→5 → AUC 0.831→0.836, purity 0.423→0.606 (opposite of the pure head; K is a mild lever once a gate is present).
- **Static scalar β = static per-class gate** (both 0.742/0.958), per-class flexibility adds nothing (echoes ProtoSSMv2's flat γ).
- **Multi-label-aware separation = standard** (0.605 vs 0.603), *honest negative*; the proposed novel loss doesn't help.
- **D1 vs D3 distinct:** D1 raw-1536 = 0.606/0.958; **D3 linear-320-projected = 0.641/0.958** (projection = free-purity AUC lever).
- **Per-taxon for all 7 build-up rows** (Cat 9 complete): the static gate recovers Aves/Amphibia but leaves Insecta at 0.598; the *input* gate lifts insects to 0.827.

## II.3: Leaderboard: the frontier does NOT transfer (the centerpiece)

Each head swapped in as the proto member of the **deployed competition pipeline**, scored on the **hidden test** via late submission. (The full 4-member pipeline times out at the 90-min limit on late re-runs, so all variants run on the deployed blend's **Proto+SED 2-way core**: aux CNNs disabled uniformly; only the proto member differs.)

| head as proto member | offline AUC / purity | public LB | **private LB** | rank |
|---|---|---|---|---|
| **ProtoSSMv2 (deployed)** | 0.803 / 0.79 | 0.94908 | **0.94218** | **1** |
| D1 coverage-gated | 0.605 / 0.99 | 0.94412 | 0.93964 | 2 |
| ProtoSSMv2 coverage-gated (control) | 0.803 /, | 0.94642 | 0.93954 | 3 |
| D2 re-weighted blend (PROTO_W 0.40) | 0.830 / 0.44 | 0.94183 | 0.93928 | 4 |
| drop, raw Perch (no head) | 0.739 /, | 0.94513 | 0.93920 | 5 |
| D1, pure prototype | 0.605 / 0.99 | 0.93767 | 0.93810 | 6 |
| D2 coverage-gated | 0.830 / 0.44 | 0.94391 | 0.93777 | 7 |
| D2, input-gated fusion | **0.830** / 0.44 | 0.94050 | 0.93743 | 8 |
| **D2-FOCAL, broad-trained (234 sp)** |, | 0.93872 | **0.93522** | **9 (worst)** |

*(Full-config drop-proto, with the aux CNNs, scored 0.94779 / 0.94168, a separate full-pipeline anchor. Proto+SED-core scores run ~0.0025 below their full-config equivalents.)*

**Pre-registration (D2 ≳ baseline > drop > D1) FAILED. The robust findings (both splits):**
1. **The deployed ProtoSSMv2 head is best** (+0.003 priv over dropping it, the prototype member genuinely helps).
2. **Both author heads score *below* raw Perch**: swapping in either is worse than no head.
3. **Offline accuracy anti-predicts the leaderboard**: D2 (highest offline AUC) is *last*.

**Figure 2** = offline-vs-deployment inversion (`paper/fig_oof_vs_lb.png`); **Figure 3** = the leaderboard ladder (`paper/fig_lb_bars.png`).

## II.4: Mechanism, decomposed (three controls + a capstone)

**(i) Diagnostic** (`_mechanism_diag.py`, 708-window bank); D2 vs raw Perch:
| species | mean gate | ρ(D2, Perch) | std(D2)/std(Perch) |
|---|---|---|---|
| covered (71) | 0.779 | 0.319 | 0.485 |
| **absent (163)** | **0.767** | **0.276** | **0.347** |

The gate stays ~0.77 *even on the 163 species D2 never trained positives for* → it overwrites Perch's discriminative scores with a flat prototype there (only ~35% of Perch's variance survives). That is the 234-species macro-AUC drag.

**(ii) Coverage-gated heads** (prototype on covered species, raw Perch on absent): **helped both heads on both splits** (D1 0.93810→0.93964, D2 0.93743→0.93777) → the absent-species drag is real, but recovered only to the **Perch floor** (≈ drop 0.93920), *not* the deployed head. **Coverage is necessary but not sufficient.**

**(iii) Control, coverage-gating the deployed head itself *hurt* it** (0.94218→0.93954). Decomposing ProtoSSMv2's +0.003 edge over raw Perch: only **+0.0003 from covered species, +0.0026 from absent ones** → **~90% of the deployed head's advantage is generalizing to species it never trained positives for** (the exact inverse of our linear heads).

**Capstone; D2-FOCAL (broad training).** Built on the user's machine: local Perch verified to reproduce the deployed one **exactly** (embedding correlation 1.000000 on the 708-window bank; this also recovered the 14795→234 mapping by column-matching: 203 exact + 3 proxy + 28 unmapped). Embedded focal recordings across **206 species** (5090 windows, top-2 by species logit) and trained the gated head on the combined soundscape+focal bank covering **all 234 species** (no absent-species hole). **It scored worst of all (0.93522).** Broad training *backfired*: the focal recordings are off-domain vs the soundscape test, and 5090 focal windows swamped the 708 on-domain ones, so prototypes became focal-centric and matched the deployment *less* well, the **focal→soundscape domain gap** inside the prototype head.

---

## Conclusion

The deployed head's advantage is **on-domain (soundscape) training + an architecture (bidirectional/residual SSM) that generalizes across the full label space**. A simple linear prototype cannot match it: it dampens the foundation on untrained classes (coverage-gating only reaches the Perch floor), and broadening its training with off-domain focal data makes it worse. **A component-level interpretability/accuracy characterization on a matched, narrow label space is necessary but not sufficient; it does not predict, and here completely inverts, system-level deployment performance.** The BirdCLEF+ 2026 setting (locked standings, late-submission private scoring, a real shipped head) is unusually well-suited to expose this gap.

---

## Files
- **Results:** this file + `GATE_PROTOSSM_RESULTS.md` (Part II detail), `GATE1_RESULTS.md` / `GATE2_RESULTS.md` (Part I detail).
- **Paper prose:** `paper_frontier_draft.md` (Part II section), `paper_gate1_draft.md` / `paper_gate2_draft.md` (Part I), `paper/paper.tex` (working-note port).
- **Figures:** `paper/fig_frontier.{png,pdf}` (1), `paper/fig_oof_vs_lb.{png,pdf}` (2), `paper/fig_lb_bars.{png,pdf}` (3).
- **Reproducibility:** `train_ablation.py`, `train_sweeps.py`, `_pertaxon_oof.py`, `_mechanism_diag.py`, `_perch_local.py`, `_focal_embed.py`, `_train_d2_focal.py`, `_build_fast.py`, `_export_d{1,2}.py`, `_export_coverage.py`.
