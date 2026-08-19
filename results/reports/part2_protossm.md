# ProtoSSMv2 Prototype-Projection Results (competition Proto member): 2026-06-09

**What this is:** first-party projection/validity analysis of the *actual deployed* competition prototype head, `hideyukizushi`'s **ProtoSSMv2** (the 52%-weight "Proto" member of the v41 ensemble, private 0.943), NOT the faithful AudioProtoPNet of Gates 1–2. Run fully locally (no audio, no Kaggle, no crash risk) from the trained checkpoint + cached Perch embeddings.

## Architecture verified (from `hideyukizushi/bird26-reprod-perch-proto-residualssm` source)
- `self.prototypes`: **(234, 320)**: ONE class-specific prototype per class.
- Readout: `sim[...,c] = softplus(proto_temp)·cos(h_temporal, prototype_c) + class_bias_c`: positive-temp-scaled cosine to each class's **own** prototype (structurally diagonal; no cross-class leakage). `proto_temp`=5.02 (>0).
- Fusion: `species_logits = γ_c·sim + (1−γ_c)·perch_logits`, `γ_c = sigmoid(fusion_alpha)`.
- So **(a) class-specific prototypes ✓ and (b) transparent own-class positive readout ✓**: the scoped component-level interpretability claim is architecturally valid (cleaner than AudioProtoPNet's non-negative *linear* layer).
- Prototypes live in **320-d post-SSM `h_temporal` space** (NOT raw 1536-d Perch); init = per-class mean of encoded reps.

## RESULT 1: the learned fusion gate is FLAT (honest negative)
`fusion_alpha ∈ [−0.027, +0.026]` → **γ ≈ 0.50 for all 234 species** (global mean 0.501, std 0.003).
Per taxon: Amphibia 0.497 · Aves 0.501 · Insecta 0.503 · Mammalia 0.499 · Reptilia 0.501.
→ **The gate did NOT learn per-class proto-vs-Perch differentiation** (no "trusts prototypes where Perch can't help" effect; no γ-vs-coverage signal). The prototype pathway contributes a uniform ~50% of each class's score. *Kills the "γ is a finding" hypothesis; is itself a clean reportable negative.*

## RESULT 2: prototypes ARE class-valid (positive), with a taxon pattern that INVERTS AudioProtoPNet
**Method:** project each of the 234 prototypes to its nearest of 708 labeled-soundscape windows (59 files × 12), cosine in 320-d `h_temporal` space (Perch embeddings → encoded through ProtoSSMv2). Purity = nearest window **contains** the prototype's class (presence-purity, multi-label soundscape). 71 species measurable (≥1 positive window in pool).

| Taxon | n | purity [95% CI] | null |
|---|---|---|---|
| **OVERALL** | 71 | **0.789 [0.690–0.874]** | 0.024±0.018 (**42σ**) |
| Insecta | 25 | 0.960 [0.880–1.000] | 0.028 |
| Amphibia | 17 | 0.882 [0.706–1.000] | 0.026 |
| Aves | 25 | 0.640 [0.440–0.840] | 0.021 |
| Mammalia | 3 | 0.333 (quarantine) | 0.021 |
| Reptilia | 1 | 0.000 (quarantine) | 0.030 |

**Finding:** ProtoSSMv2's prototypes are strongly class-valid overall (0.79 ≫ 0.025 null, 42σ). The taxon pattern is the **opposite** of AudioProtoPNet's: the soundscape-trained model is cleanest for **acoustically-distinctive non-avian taxa** (insects 0.96, amphibians 0.88) and *lower* for the dense/overlapping **avian** taxon (0.64), whereas the bird-pretrained AudioProtoPNet (Gate 2) was cleanest for birds (0.95) and degraded for non-birds.

**Synthesis (paper-worthy, honest):** prototype validity appears to track the **training domain**, not the prototype mechanism per se, bird-pretrained → bird-clean; Pantanal-soundscape-trained → distinctive-non-avian-clean.

## RESULT 2b: the taxon pattern survives a frequency control (Gate-3 discipline)
Purity rises with sample count (Spearman(purity, log npos) = 0.30; npos-tercile purity 0.63 / 0.83 / 0.92), a real confound, since Aves species are rarer in the pool (mean 16 positive windows vs Insecta 23, Amphibia 122). **But the taxon effect is not just frequency:** logistic `pure ~ lognpos + taxon` (ref=Amphibia) gives **Insecta +1.40, Aves −0.71, lognpos +0.61**: taxon coefficients large and non-zero after controlling sample count. Frequency-matched (within npos terciles): **Insecta is clean at every frequency level (even low-npos → 1.00), while Aves is frequency-dependent (low-npos 0.47 → high-npos 1.00).**
→ **Sharper finding:** acoustically-distinctive insect prototypes are valid *regardless of how often the species is observed*; overlapping avian prototypes are valid *only when the species is frequent*. (Gate-3 regression: `_gate3.py`.)

## CAVEATS (state all in the paper)
1. **Metric is NOT identical to Gate 2.** This is *presence-purity on multi-label soundscape windows* (~4.4 species/window); Gate 2 was *strict single-species clean-clip purity*. So the AudioProtoPNet↔ProtoSSMv2 contrast is **suggestive, not a head-to-head**: say so.
2. **Encoded WITHOUT the optional site/hour metadata** (mapping uncertain). Since metadata is a per-file additive shift present in training, the no-metadata 0.789 is a **conservative lower bound**.
3. **Pool is densely active** (708/708 windows have ≥1 species) → no silent windows → the classic silence-shortcut test is not directly applicable here; an energy-tercile version needs the waveform cache (pending).
4. **Mammalia (n=3) / Reptilia (n=1) quarantined**: never write "across mammals/reptiles."
5. This is the **competition** Proto member (ProtoSSMv2), distinct from the faithful AudioProtoPNet study; keep them clearly separated, as the participant-note framing already does.

## Reproduce
- Model code: `_protossm_model.py` (3 classes lifted verbatim from the training kernel). Projection: `_project.py`.
- Inputs: `_proto_ckpt/train_proto_ssm_single/models/proto_ssm_best.pt` (trained weights), `_perch_meta/full_perch_arrays.npz` (`emb_full` 708×1536) + `full_perch_meta.parquet`, `data/train_soundscapes_labels.csv`.
- Per-prototype table: `../tables/gate_protossm_projection.csv`.

## Paper implication
The scoped component-level claim now has **first-party numbers**: the deployed prototype head is projection-valid (0.79, 42σ), its gate is flat (honest negative), and its taxon pattern contrasts with the faithful AudioProtoPNet, a genuinely new, defensible result that strengthens the interpretability section without overclaiming.

---

# ★★★ AUTHOR-BUILT PROTOTYPE HEADS: interpretability↔accuracy frontier (2026-06-09)
**This is the author-built SYSTEM the venue pattern wants.** Three prototype heads designed by the author, trained on the **same 708-window Perch bank + same GroupKFold(5)** as ProtoSSMv2 (per-window multi-label, frozen Perch). Scripts: `train_heads.py`, `_project_heads.py`, `_bootstrap.py`. OOF arrays `_oof_*.npy`, heads `_heads_*_fold0.pt`.

## The result table (OOF macro-AUC over 71 soundscape species + prototype purity)
| Model | macro-AUC [95% CI] | prototype purity | interpretable? |
|---|---|---|---|
| **D2** input-dependent gate + Perch fusion | **0.827 [0.781–0.868]** | **0.48** (Aves 0.12) | NO, prototype non-valid |
| ProtoSSMv2 (deployed, static γ) | 0.803 [0.749–0.854] | 0.79 | partial |
| linear probe (ref) | 0.766 [0.713–0.814] |, |, |
| Perch logits alone (ref) | 0.739 [0.690–0.789] |, |, |
| **D1/D3** pure multi-proto (+cluster/sep) | 0.607 [0.517–0.695] | **0.96** | YES, class-valid |

**Significance (paired per-species bootstrap):** D2 > Perch **+0.088 (sig)**, D2 > probe **+0.061 (sig)**, D2 vs ProtoSSMv2 **+0.024 [−0.020,+0.071] = n.s. (statistical tie, do NOT claim "beats the deployed model").** D1 ≪ all (sig).

## THE FINDING: a sharp interpretability↔accuracy frontier (the paper's centerpiece)
- **Force the prototype to carry the prediction** (D1/D3, no Perch fallback, cluster+separation losses) → prototypes become **class-pure (0.96)** but accuracy collapses to **0.61** (below even a linear probe).
- **Let an adaptive gate offload to Perch** (D2) → accuracy wins (**0.83**, sig > probe/Perch, tied with deployed) but the prototype becomes **non-valid (0.48, Aves 0.12)**: the model no longer needs it, so it stops being interpretable.
- The deployed **ProtoSSMv2 sits in the middle** (γ flat 0.50, purity 0.79, acc 0.80), partial reliance on both.
- **You cannot get high purity AND high accuracy on this data.** That frontier, measured across 5 first-party models with bootstrap CIs, is the contribution. The cluster/separation losses *work* (they buy purity 0.96) but cannot buy accuracy; the accuracy comes from the foundation logits, which the gate learns to lean on, hollowing out the prototype.

## Caveats (state all)
1. **Small eval:** 71 species / 708 OOF windows → inter-model AUC CIs are wide (hence D2-vs-ProtoSSMv2 is n.s.). The *frontier* (0.96↔0.61 purity↔acc) is large and robust; the inter-model *rankings* are softer.
2. **D1 ≡ D3 as coded** (the memo's two designs are architecturally identical, both multi-proto + cluster/sep). Collapse to one "pure prototype" design, OR differentiate (e.g., D1 = separation-only vs D3 = cluster+separation) for a loss ablation.
3. **D2's "gate" is a `Linear(1536→234)`** (a linear probe), so D2 is "interpretable prototype gated by a non-interpretable probe"; its accuracy win is genuine (sig > plain probe) but its interpretability is compromised (the point of the finding).
4. Purity spaces differ (D1/D2 raw 1536-d Perch; ProtoSSMv2 320-d post-SSM), each model's prototypes projected in their own space.
5. Per-window soundscape eval (no temporal pooling); macro over species-with-positives only.

## Paper framing
System section = the three author-built heads (D2 competitive with the deployed model). Results = the table + bootstrap. Finding = the interpretability↔accuracy frontier, with ProtoSSMv2 and AudioProtoPNet (Gates 1–2) as the comparison anchors. Author-built system ✓ + a real, rigorously-measured, honestly-hedged result ✓.

## ★ STEPWISE BUILD-UP (the main ablation table: one knob per row): `train_ablation.py`, `_abl_oof_*.npy`
Single-prototype, raw 1536-d Perch, GroupKFold(5), 708-window OOF, each row changes exactly ONE thing vs the row above.
| # | config | macro-AUC | purity | marginal effect |
|---|---|---|---|---|
|, | Perch logits alone | 0.739 |, | foundation floor |
| 1 | linear probe | 0.766 |, | learned head, no prototypes |
| 2 | single proto, no losses | 0.762 | 0.69 | bare prototype = competent classifier, moderate purity |
| 3 | + cluster loss | 0.604 | 0.944 | **interpretability loss: +0.25 purity, −0.158 AUC** |
| 4 | + separation loss | 0.605 | 0.958 | marginal (+0.014 purity, ~0 AUC), cluster is the driver |
| 5 | + static-gate Perch fusion | 0.742 | 0.958 | **recovers Perch-level AUC, keeps purity** (gate defers to Perch), the "free-interpretability" point |
| 6 | + input-dependent gate (**D2**) | **0.830** | **0.437** | **+0.088 AUC over static, −0.521 purity**: accuracy via abandoning the prototype |
| 7 | 3-proto + cluster/sep (D1/D3 pure) | 0.606 | 0.958 | multi-proto ≈ single-proto (capacity not the lever) |
|, | ProtoSSMv2 (deployed) | 0.803 | 0.79 | middle of the frontier |

**The mechanism, isolated:** the *cluster loss* (row 2→3) is what buys interpretability (+0.25 purity) at a steep accuracy cost (−0.16); the *input-dependent gate* (row 5→6) is what buys accuracy (+0.088 over Perch-level) by hollowing out the prototype (−0.52 purity). Separation loss and prototype count barely matter. **Row 5 is a genuine sweet spot** (static gate): 0.96 purity at Perch-level accuracy, interpretability essentially free, you just don't *beat* Perch. To beat it you must spend the prototype's validity (row 6). This is the build-up + frontier in one table, with single-knob attribution.
(Per-taxon AUC for every row saved in the run log; purity per-taxon in `gate_protossm_projection.csv`-style outputs.)

## ★ SWEEPS (Cat 3 / 5 / 6 / 7): `train_sweeps.py`, 708-window OOF, GroupKFold(5)
Four knob-grids that probe the frontier's edges. Headline: **none of them break the frontier**: they refine where each branch sits.

**Cat 3, prototype count K** (pure head, cluster+sep losses):
| K | macro-AUC | purity(any) | purity(mean-per-proto) |
|---|---|---|---|
| 1 | 0.603 | 0.958 | 0.958 |
| 2 | 0.604 | 0.958 | 0.796 |
| 3 | 0.604 | 0.958 | 0.671 |
| 5 | 0.608 | 0.958 | 0.566 |
| 10 | 0.604 | 0.958 | 0.435 |

AUC is **flat at ~0.60 across K=1→10**: extra prototypes buy nothing. `purity(any)` stays 0.958 (≥1 proto lands on-class) but `mean-per-proto` purity **collapses** (0.958→0.435): most of the added prototypes are off-class/dead. **Capacity is not the lever**, and more prototypes make the head *less* interpretable per-unit. Single-proto is both as accurate and the cleanest.

**Cat 6, prototype embedding space** (pure head K=1, cluster+sep):
| projection | macro-AUC | purity |
|---|---|---|
| direct-1536 (raw Perch) | 0.604 | 0.958 |
| linear→768 | 0.644 | 0.958 |
| linear→320 | **0.647** | 0.944 |
| mlp→320 | 0.633 | 0.958 |

A **learned linear metric space** before the cosine recovers **+0.043 AUC** (0.604→0.647) while keeping purity ≥0.94, the only knob that improves the *interpretable* branch without spending purity. Still ~0.18 below D2, and a nonlinear (mlp) projection does *not* beat linear. Modest, honest, on-frontier.

**Cat 7; D2 training length + embedding-mixup** (input-gate head):
| epochs | AUC | | mixup | AUC |
|---|---|---|---|---|
| 10 | 0.814 | | 0.0 | 0.832 |
| 25 | **0.848** | | 0.4 | 0.817 |
| 40 | 0.840 | | | |
| 60 | 0.826 | | | |

D2 **peaks at 25 epochs (0.848)** then overfits the 708-window bank (60ep→0.826); embedding-mixup hurts. So D2's best OOF is **~0.848**, a touch above the 60-epoch point used in the frontier table (the deployed `d2_weights.npz` is the 60-ep export ≈0.83; 25-ep is the optimum if re-exported).

**Cat 5, post-hoc temporal smoothing on D2 OOF** (per-file pooling):
none 0.827 · mean **0.814** · max 0.827 → per-file pooling is **neutral-to-negative**. Matches the competition finding that temporal smoothing doesn't move this rank metric, dead compute on top of D2.

**Takeaway for the paper:** the frontier is robust to all four axes. The only positive lever for the *interpretable* branch is a learned low-rank metric space (Cat 6, +0.04); the only positive lever for D2 is the right training length (Cat 7, 25ep). Neither closes the interpretability↔accuracy gap.

## Per-taxon structure of the frontier (`_pertaxon_oof.py`)
OOF accuracy decomposed by taxon (covered species; Mammalia n=3 / Reptilia n=1 quarantined):
| taxon | linear probe | D2 (gated) | D1 (pure proto) |
|---|---|---|---|
| Aves (25) | 0.699 | **0.795** | 0.554 |
| Amphibia (17) | 0.799 | **0.884** | 0.689 |
| Insecta (25) | 0.835 | 0.828 | 0.620 |
| all (71) | 0.766 | 0.827 | 0.607 |

Two findings: (1) **D2's accuracy gain over the probe is taxon-dependent, largest where the foundation is weakest** (Aves +0.10, Amphibia +0.09) and **negligible on Insecta** (0.828 vs 0.835, where the probe is already strongest). The input gate adds discrimination where there is headroom and stays out of the way where Perch already wins. (2) **D1's collapse is taxon-general**: the pure prototype is below the probe in *every* taxon, so the purity→accuracy cost is not a one-taxon artifact. Note the **accuracy-gain gradient (Aves/Amphibia) is orthogonal to the prototype-purity gradient (Insecta-cleanest, §RESULT 2)**: where prototypes are purest is *not* where the gated head helps accuracy most; purity and accuracy-utility have different taxon structure.

## ★ COMPLETENESS: the remaining ablation rows (`_fill_gaps.py`, 708-window OOF)
The genuinely never-run rows, now filled (the build-up + sweeps covered the rest; per-taxon below is recomputed from the 11 saved OOF arrays):

**(A) Full-spec D2, the gate is accuracy XOR purity (sharpens the frontier).** D2 in the build-up has *no* prototype losses; adding them:
| D2 variant | AUC | purity |
|---|---|---|
| input gate, no loss (as-built, build-up row 6) | 0.830 | 0.437 |
| input gate **+ cluster + sep** | **0.691** | **0.958** |

The purity collapse is **not** the gate hollowing out a pure prototype; it is a direct trade. Forcing purity back (cluster loss) recovers it to 0.958 but **collapses AUC to 0.691, below the static gate (0.742)**. Accuracy and purity are in genuine tension; the input gate buys accuracy *only* by abandoning the purity loss. (This strengthens the frontier claim; there is no gated config that is both accurate and pure.)

**(B) K-sweep on the GATED head** (input fusion, no loss), *opposite* of the pure head:
| K | AUC | purity |
|---|---|---|
| 1 | 0.831 | 0.423 |
| 3 | 0.833 | 0.549 |
| 5 | 0.836 | 0.606 |

More prototypes give the gated head a small AUC gain **and** recover purity (0.423→0.606), unlike the pure head, where K was AUC-flat and collapsed per-proto purity. So prototype count *is* a mild lever once a gate is present.

**(C) Static SCALAR fusion (single global β) + cluster + sep = 0.742 / 0.958**: *identical* to the static per-class gate (build-up row 5, 0.742/0.958). Per-class gate flexibility adds nothing; a single global mix suffices. Echoes ProtoSSMv2's flat γ≈0.50.

**(D) Multi-label-aware separation = standard** (0.605 vs 0.603, both purity 0.958), **honest negative.** Downweighting busy multi-species negatives does not help; the separation loss barely matters in either form (cluster is the driver).

**(E) D1 vs D3 now distinct.** D1 (raw-1536 pure prototype) = 0.606 / 0.958; **D3 (linear-320 projected pure prototype) = 0.641 / 0.958.** The projection is a *free-purity* AUC lever (+0.035 at identical purity), the two pure-prototype designs are no longer identical.

**(F) Per-taxon for EVERY build-up row** (Category 9 complete; AUC by taxon, covered species):
| row | all | Aves | Amphibia | Insecta |
|---|---|---|---|---|
| linear probe | 0.766 | 0.712 | 0.799 | 0.820 |
| single proto, no loss | 0.762 | 0.711 | 0.800 | 0.817 |
| + cluster | 0.604 | 0.548 | 0.686 | 0.617 |
| + cluster + sep | 0.605 | 0.549 | 0.695 | 0.615 |
| + static gate | 0.742 | 0.818 | 0.831 | **0.598** |
| + input gate (D2) | 0.830 | 0.804 | 0.884 | **0.827** |
| 3-proto pure | 0.606 | 0.552 | 0.687 | 0.619 |

Note: the static gate recovers Aves/Amphibia but leaves **Insecta low (0.598)**; the *input* gate is what lifts insects (0.598→0.827), the input-dependence specifically helps the taxon where Perch already led, consistent with "the gate adds where there is signal to route."

**Per-taxon PURITY of the heads (`_pertaxon_purity.py`, closes the last Cat-9 sub-item):**
| head | Aves | Amphibia | Insecta | overall |
|---|---|---|---|---|
| D1 (pure prototype) | 0.96 | 1.0 | 1.0 | 0.986 |
| ProtoSSMv2 (deployed) | 0.64 | 0.88 | 0.96 | 0.789 |
| D2 (gated) | **0.12** | 0.71 | 0.72 | 0.465 |

The gate's purity loss is **taxon-localized**: D2 hollows out the *Aves* prototype almost completely (0.96→0.12) while keeping insect/amphibian prototypes ~0.7, i.e. it spends bird-prototype purity (where Perch is strong, so it routes to Perch) to buy bird accuracy, and leaves the prototype intact where Perch is weaker. This links the purity-vs-accuracy trade to the per-taxon AUC pattern.

## ★ LEADERBOARD CONFIRMATION (Cat 8): head-swap on the hidden test
Every number above is the labeled-soundscape OOF proxy (708 windows). The Cat-8 rows transplant each author-built head into the **deployed competition pipeline** and score it on the **hidden test** via late submission, the real-test confirmation that the frontier is not a 708-window artifact.

**Method (validated end-to-end).** The deployed inference notebook (v52/v56) is *multi-solution*: each `Model_N` block is gated by `if 'Model_N' in _ensemble_models`. The active config is a **single solution, `Model_7`** (`solutions={'type_add':'single','Models':[{'Model':'Model_7','subm':'submission.csv',...}]}`). Only the `Model_7` cell runs and writes the scored `submission.csv` (a rank-blend of `submission_protossm.csv` + SED + two CNNs). So the swap must be injected into the **`Model_7` cell**, right after `proto_scores_flat = proto_out.reshape(-1, N_CLASSES)` (line ~1146), overwriting the ProtoSSM proto member with the head's forward on the test Perch embeddings (`emb_te`) + raw Perch scores (`sc_te`, both from `run_perch(test_paths)` at line ~1113). The swapped `proto_scores_flat` then flows through the *identical* downstream (per-class first-pass fusion → ResidualSSM correction → file-confidence/rank/Δ-smoothing → per-class thresholds → `submission_protossm.csv` → final rank-blend). Each head's tiny weights ride in as a Kaggle dataset; the notebook globs `/kaggle/input/**/<head>_weights.npz` (datasets mount at `/kaggle/input/datasets/<owner>/<slug>/` in this kernel env, not `/kaggle/input/<slug>/`). A `*_RAN.txt` marker + `>>> … SWAP ACTIVE` print confirm execution.

(Two earlier mis-injections, now fixed and worth recording: the swap first went into the `Model_3` then `Model_4` cell, both **inactive** (not in `_ensemble_models`), so it was dead code and `submission_protossm.csv` was unchanged; and the first path used `/kaggle/input/<slug>/` which `FileNotFoundError`'d. Lesson: in a multi-solution notebook, inject into the cell whose guard matches the active solution and that writes `_file_name_submission`.)

**Rows (kernels pushed + committing 2026-06-10).** ProtoSSMv2 = the deployed head (baseline, score from the v52/v56 finals). D3 ≡ D1 (the two pure-prototype specs are identical as coded) → one D1 row covers both.
| row | head as proto member | local OOF AUC / purity | public LB | private LB | tests |
|---|---|---|---|---|---|
| baseline | ProtoSSMv2 (deployed) | 0.803 / 0.79 | _re-running (v52)_ | _re-running_ | the shipped head |
| D2-swap | gated-fusion (D2) | 0.827 / 0.44 | _re-running (v4)_ | _re-running_ | does the accuracy-winner survive on hidden test? |
| D1-swap | pure prototype (D1≡D3) | 0.605 / **0.99** | _re-running (v2)_ | _re-running_ | cost of full interpretability in the real ensemble |
| **drop-proto** | raw Perch (no head) | 0.739 /, | **0.94779** | **0.94168** | the prototype slot's marginal contribution |

**First result in (2026-06-10): drop-proto = 0.94779 public / 0.94168 private.** Against the deployed pipeline (~0.949–0.950 pub / ~0.943 priv), replacing the entire prototype member with raw Perch costs roughly **−0.0015 public / −0.0013 private**: i.e. the ProtoSSMv2 prototype head contributes ~0.001 to the production ensemble over the bare foundation. Small but consistent across both splits (exact baseline re-running for a clean delta).

**⚠️ Runtime lesson (cost me a round):** D2-swap v3 and D1-swap v1 both **timed out (>90 min)** on the hidden test and scored null, *not* a logic bug. Cause: the weight loader used `glob.glob('/kaggle/input/**/*.npz', recursive=True)`, which recursively walks the entire input tree including the multi-GB waveform-cache dataset (tens of thousands of files); that walk added minutes to an already-near-90-min pipeline. drop-proto has no weight load → no timeout → scored fine. Fix = explicit dataset paths + bounded-depth globs (no `**`), re-pushed as D2 v4 / D1 v2. **General rule for this pipeline: any injected file I/O must avoid recursive globs, the pipeline has almost no runtime headroom.**

**Pre-registered expectation:** if the frontier is real on the hidden test, the blend score should order **D2-swap ≳ baseline > drop-proto > D1-swap** (D1's 0.605-AUC member drags the rank-blend; D2 ≈ or slightly above the deployed head; dropping the head ≈ Perch floor). Scores fill in on submission; late subs return public **and** private.

### ★★ RESULT (2026-06-10): the pre-registration FAILED, and that is the finding
The full deployed pipeline (4 audio passes: Perch+SED+B0+B2) sits *exactly* at the 90-min limit, so on late re-runs it **times out by variance**: even the unmodified baseline timed out; only drop-proto (no weight load) squeaked through once at full config (0.94779/0.94168). To get a consistent 4-point set I ran all four on the **Proto+SED 2-way core** (the deployed blend already gates the 8%+5% aux CNNs on file existence and falls back to Proto+SED; disabling them, `_run_cnn_inference=lambda:None`: halves the runtime and clears the wall, applied uniformly so only the proto member differs).

**Proto+SED core, all four on the identical config (private LB):**
| proto member | OOF AUC / purity | public | **private** | rank |
|---|---|---|---|---|
| **baseline; ProtoSSMv2 (deployed)** | 0.803 / 0.79 | 0.94908 | **0.94218** | **1** |
| drop, raw Perch (no head) | 0.739 /, | 0.94513 | **0.93920** | 2 |
| D1, pure prototype | 0.605 / **0.99** | 0.93767 | **0.93810** | 3 |
| D2, gated fusion | **0.830** / 0.44 | 0.94050 | **0.93743** | 4 |

**Robust findings (hold on BOTH public and private):**
1. **The deployed ProtoSSMv2 head is best** (top on both splits, +0.003 priv over drop). The prototype member genuinely helps the production ensemble, a bigger effect (~0.003) than the full-config drop suggested, because in the 2-way core the proto member carries 60% weight.
2. **Both author-built heads score BELOW drop-proto** (bottom two on both splits). Swapping in *either* author head is **worse than putting raw Perch in the slot**: the heads actively hurt the system.
3. **OOF accuracy anti-predicts LB.** D2, the highest-OOF head (0.830, the "accuracy winner" of the frontier), is **last on private**. The pre-registered "D2 ≳ baseline" failed hard; baseline (OOF 0.803) beats D2 (OOF 0.830) by +0.0048 priv. (D1-vs-D2 swaps between splits, so "D2 specifically worst" is not robust; "both author heads worst" is.)

**Interpretation (hypothesis, not proven).** The OOF frontier is computed over the **71 species with positives in the 708-window bank**; the LB is **macro-AUC over 234 species**. The author heads were trained only on that bank, so for the ~163 species absent from it they learn "predict ~absent", flat, non-discriminative scores that, fused at 60% proto weight, *dampen* Perch's real signal for those classes and drag the 234-species macro-AUC. The deployed ProtoSSMv2 (same 708-window training) is nonetheless more robust on the full label space (better untrained-class behaviour and/or more decorrelated from the Perch-distilled SED member → more ensemble diversity). **The lesson is the paper's sharpest point: a component-level frontier measured on a matched, narrow label space does not transfer to the full deployed system, the head that wins the offline metric loses in deployment, and naive head-swapping underperforms both the shipped head and even removing the head.** Note these Proto+SED-core numbers are ~0.0025 below their full-config equivalents (the aux CNNs add a little); the full-config drop-proto (0.94168) is a separate, full-pipeline anchor.

### Mechanism diagnostic (local, `_mechanism_diag.py`): D2 dampens Perch on absent species
Direct test of the hypothesis on the 708-window bank: compare D2's gate + how well it preserves Perch's per-species ranking, on the **71 covered** species vs the **163 absent** ones.
| species | mean gate | median ρ(D2, Perch) | median std(D2)/std(Perch) |
|---|---|---|---|
| covered (71) | 0.779 | 0.319 | 0.485 |
| **absent (163)** | **0.767** | **0.276** | **0.347** |

**The gate is ~0.77 even on absent species**: D2 never learns to defer to Perch for the classes it saw no positives for; it applies its prototype indiscriminately. On those species it therefore overwrites Perch's discriminative scores with its trained-to-be-flat prototype: only ~35% of Perch's per-species variance survives (vs 49% on covered) and the ranking is scrambled (ρ 0.28). This is precisely the failure mode that drags the 234-species macro-AUC, confirmed independently of the LB. (On covered species D2 also deviates from Perch, but there the deviation is an *informed* trained prototype; on absent species it is flat noise.)

**Coverage-gated result (2026-06-10, Proto+SED core)**: prototype on the 71 covered species, raw Perch on the 163 absent:
| variant | public | **private** | vs ungated |
|---|---|---|---|
| d1cov (D1 gated) | 0.94412 | **0.93964** | D1 0.93810 -> **+0.0015** |
| d2cov (D2 gated) | 0.94391 | **0.93777** | D2 0.93743 -> +0.0003 |
| _anchors_ | | baseline 0.94218 / drop 0.93920 | |

**Verdict: the mechanism is real but coverage is necessary-not-sufficient.** (1) Gating **helped both heads on both splits** (d1cov>D1, d2cov>D2) -> the absent-species dampening was a genuine drag, confirmed. (2) **d1cov reaches the raw-Perch floor** (0.93964 vs drop 0.93920) -> the pure prototype's covered-species signal is ~as good as Perch once it stops polluting absent species. (3) **Neither gated head reaches the deployed baseline** (0.94218) -> fixing coverage lifts the heads only to ~the Perch floor, not the shipped head; ProtoSSMv2's edge is more than label coverage (better prototypes / its ResidualSSM 2nd pass / TTA / decorrelation from the Perch-distilled SED). Notably d2cov<drop on both splits: even on *covered* species D2's gated output slightly underperforms raw Perch in the ensemble (a diversity cost), whereas D1's purer prototype does not. **Motivates the broad-training test (D2-FOCAL): does training on the full label space, not just gating, close the gap?**

**basecov control (2026-06-10): ProtoSSMv2 coverage-gated = 0.94642 pub / 0.93954 priv.** Gating the *deployed* head by coverage **HURT it** (baseline 0.94218 -> basecov 0.93954, -0.0026), i.e. ProtoSSMv2's predictions on the 163 absent species are *better than raw Perch*. Decompose its +0.003 priv edge over drop: **covered species contribute only +0.0003** (basecov-drop), **absent species contribute +0.0026** (baseline-basecov). **So ~90% of the deployed head's advantage is GENERALIZATION TO ABSENT SPECIES, the very classes it never trained positives for.** This is the exact inverse of my linear heads, which *inject noise* on absent species (gate ~0.77, dampen Perch). The mechanism is now fully decomposed: ProtoSSMv2's bidirectional-SSM + ResidualSSM + per-class gating extract better-than-Perch signal on untrained classes; a simple linear prototype cannot, and over-applies a flat prototype instead. The deployed head's edge is architectural generalization on the full label space, not label coverage, which sets up D2-FOCAL (broad training of a *linear* head): if it still can't beat drop, the gap is the architecture, not the data.

### ★★ D2-FOCAL capstone (2026-06-10): broad training BACKFIRES (the domain gap)
Built the full pipeline locally on the user's machine: verified our local Perch reproduces the deployed one EXACTLY (`_perch_local.py`, emb corr **1.000000**), recovered the 14795->234 mapping (`perch_map.npz`, 203 exact+3 proxy+28 unmapped), embedded focal audio (`_focal_embed.py`, **5090 windows / 206 species**, top-2 windows/file by species logit), and trained D2 on the **combined soundscape(708)+focal(5090) bank covering all 234 species** (`_train_d2_focal.py` -> `d2focal_weights.npz`). LB (Proto+SED core):
| head | public | **private** | rank (of 8) |
|---|---|---|---|
| D2-FOCAL (all 234 sp) | 0.93872 | **0.93522** | **8 / 8 (WORST)** |

**Broad training made it WORSE, not better** (D2 71-sp 0.93743 -> D2-FOCAL 0.93522, and below drop 0.93920). Covering the full label space did not help because the **focal recordings are off-domain** vs soundscape deployment, and the 5090 focal windows swamped the 708 on-domain soundscape ones -> the prototypes became focal-centric and matched the soundscape test *less* well, **even on covered species**. This is the **focal->soundscape domain gap** (the competition's central challenge) manifesting inside the prototype head. **Full resolution:** the deployed ProtoSSMv2's edge is BOTH (a) on-domain soundscape training and (b) its SSM/ResidualSSM architecture that generalizes to absent species. Neither coverage-gating (-> Perch floor) nor broad off-domain training (-> worse) recovers it, and the offline frontier predicted none of the LB ordering. Final LB ranking (priv, all 9 heads): baseline 0.94218 > d1cov 0.93964 > basecov 0.93954 > **d2rw 0.93928** > drop 0.93920 > D1 0.93810 > d2cov 0.93777 > D2 0.93743 > D2-FOCAL 0.93522.

**Last Cat-8 row; D2 with re-optimized blend weight (`_build_d2rw.py`, PROTO_W 0.60→0.40): 0.94183 pub / 0.93928 priv.** Down-weighting the harmful D2 proto member recovers it from full-weight D2 (0.93743) **to ≈ the drop/Perch floor** (0.93920), i.e. the best a blend re-weight can do for a head that hurts is *neutralize the harm* by trusting it less; it extracts no value and does not approach baseline. Confirms the proto-member's blend weight is not a lever that rescues a bad head (the optimum is ~0 weight = drop). **The entire 9-category plan + all Cat-8 leaderboard rows are now complete.**
