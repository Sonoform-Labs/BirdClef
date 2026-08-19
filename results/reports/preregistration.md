# Pre-Registration: Diversity–Distillation Tradeoff under a Shared Bioacoustic Foundation Teacher
**BirdCLEF+ 2026 / LifeCLEF working note. Locked BEFORE any new private/public scores are observed.**
Date locked: 2026-06-04 (fill exact). Author: bobmyersthesecond.
> Purpose: freeze the protocol so claims are confirmatory, not fitted-after-the-fact. Do NOT edit after the first new submission; append a dated "deviations" log instead.
>
> **⟢ FINAL DIRECTION (2026-06-04, supersedes the "predictive diversity diagnostic" framing throughout this doc):** after 5 collision-searches the novel-MECHANISM lane is confirmed CLOSED on BOTH sides; ML (Kuncheva 2003 diversity↔accuracy largely-negative; Yuan 2020 + Random Teachers 2023 KD-as-regularization; Progressive Ensemble Distillation = shared-teacher homogenization) and ecology (Ghani 2023 cross-taxa transfer; Perch 2.0; DeepForestSound 2026; Aug-2025 multi-taxa foundation-model review). PIVOT to a **rigorous per-taxon distillation APPLICATION note**: *"On BirdCLEF+ 2026, using the standard PUBLIC pipeline, when does distilling from a bird-tuned foundation teacher help vs hurt across the 5 Pantanal taxa?"* **Contribution = EXECUTION QUALITY** (paired bootstrap CIs, convergent private-LB + 59-soundscape validation, per-species correlation, honest negatives), NOT idea novelty. **Target = CEUR-WS ACCEPTANCE** (cleared by construction, 2026 notes unpublished); prize = low-prob upside via rigor only. Cite Ghani/Yuan/Kuncheva/overview as established foundation. The §6 "predictive" regression is DEMOTED to a DESCRIPTIVE correlation report (do NOT claim prediction; Kuncheva). The rigor/validation/scope/plan sections (§5,7,8,10,11) carry over unchanged; "the grid" is repurposed as the clean distill-vs-no-distill measurement.

---

## 1. Thesis (the unified claim)
The field reports *contradictory* verdicts on identical techniques (distillation kept/dropped 4 ways; 4 teams → 4 "best" losses; pseudo-depth helps/hurts; GeM vs mean split). **Hypothesis: the hidden variable governing whether a technique helps is its effect on inter-model rank-correlation relative to its single-model quality gain.** A technique that raises single-model AUC but also raises inter-model correlation degrades a rank-fused ensemble's *out-of-domain* generalization. Distillation from a **shared Perch teacher** is the cleanest instance, because it homogenizes every downstream member.

**What is novel (and what is NOT):** "distillation reduces ensemble diversity" is known in general ML (NeurIPS 2021). doc 46 *observed* the correlation rise (~0.4→0.5) as a footnote. Our contribution = (a) formalize it on the **rank-AUC** metric, (b) tie it to **focal→field domain-shift** generalization (the PAM-central version), (c) make it a **predictive per-taxon diagnostic**. Cite doc 46 + NeurIPS 2021 + Wood JMLR 2023 + WSDM 2021 (ranking BV) up front as "observed/general; we explain+specialize+predict."

## 2. Claim ladder: verb is selected by the eval world, decided at Step 0 BEFORE training
- **World A (late scoring returns PRIVATE):** verb = **"predicts."** Validate diagnostic against fresh private scores. Full paper as pitched.
- **World B (late scoring PUBLIC-ONLY, but we trust the 59 soundscapes):** verb = **"predicts (small-sample, field-validated)."** Validate against the 59 labeled soundscapes with bootstrap CIs. Weaker but legitimate; still the only field-audio link.
- **World C (no trustworthy target):** verb = **"explains," never "predicts."** Mechanism + decomposition + contradiction-resolution only. Explicitly state prediction is unvalidated.
> The seed already run (public→private travel, REAL private data): decorrelated configs climbed +20 ranks, AUC-on-metric configs sank, directionally confirms the mechanism (n=2/3, a sniff, not proof). ρ(proto,sed)=0.45 corroborates doc 46's 0.4→0.5.

## 3. STEP 0: the gate behind the gate (do FIRST, one throwaway submission)
Submit ONE config late. Confirm whether Kaggle returns a **private** score or **public-only**. This selects World A/B/C and therefore the headline verb. **Do not train the grid until this is known.** Record the result here: **RESOLVED 2026-06-04, late submissions return BOTH public AND private. → WORLD A. Headline verb = "predicts." Grid greenlit. Bonus: every grid config yields a (public,private) pair → feeds both the diversity diagnostic (validate vs private) and the public-non-predictiveness sub-finding (fresh points). Caveat retained: private = one ~66% draw → still dual-validate vs the 59 soundscapes.**

## 4. Pre-specified student grid (FROZEN: fill CONFIRM slots, then lock)
Factorial over factors the field contradicted on + that vary baseline diversity:
- **Distillation:** {Perch-distilled, non-distilled} , core factor (required).
- **Backbone:** MUST include ≥1 genuinely **non-Perch-lineage** member (ConvNeXt-tiny from ImageNet; NOT Perch-distilled, and/or BirdNET as the non-Perch anchor), alongside EfficientNet-B0/B2. **If every cell is Perch-flavored the grid cannot vary the thing the paper is about → the study is void.** CONFIRM available backbones incl. the non-Perch one: `[ ____ ]`.
- **Pooling:** {GeM, mean} , the doc46/doc48 contradiction.
- **(Secondary, optional) Loss:** {FocalBCE, SoftAUC} , the 4-losses contradiction. Include only if compute/time allow; mark secondary.
Base members = 2 × (#backbones) × 2 (× 2 if loss). With 3 backbones, pooling, no loss factor = **12 base members**. Ensembles = pre-specified rank-fusion of member subsets (see §6). CONFIRM grid size + seeds-per-cell: **≥3 seeds for the distill on/off contrast** (the claim the whole paper rests on; 0.001–0.002 is seed noise here), ≥2 elsewhere: `[ ____ ]`.

## 5. The diagnostic (FROZEN definition): UPDATED Day-1 2026-06-04
**PRIMARY = PER-SPECIES** (metric-appropriate, since macro-AUC averages per-species AUC). Day-1 finding: flattened-global Spearman runs ~0.2 HIGHER than per-species and captures irrelevant "which species are loud overall" agreement, do NOT use it as the headline number.
On a fixed evaluation split (OOF/full-soundscapes for diagnostic construction; never the validation target):
- **Inter-model correlation (PRIMARY)** R_c[i,j] = Spearman of members i,j on species column c (over rows), averaged over species where both members have variance. Report n_species per pair (BirdNET covers ~161/234, so its pairs use fewer columns, disclose this).
- **Secondary/diagnostic-only:** flattened (row×species) Spearman, reported once to show the definition-dependence, NOT used for claims.
- Day-1 dry-run values (20 files, NOISY, to be re-measured on 59 soundscapes): per-species mean corr-to-others = b0f4th 0.38, birdnet 0.37, proto 0.41, sed 0.44 → members only MODERATELY correlated; BirdNET only mildly diverse. NOTE: this snapshot canNOT test the core claim (distill vs no-distill), see §11 D1 real test.
- **Effective diversity** D = 1 − mean off-diagonal(R). Per-species D_c = 1 − mean off-diag(R_c).
- **Single-model quality** Q = macro-AUC (and per-species AUC Q_c) on the diagnostic split.
- **Distillation deltas** for each (backbone,pooling) pair: ΔQ = Q(distill) − Q(nodistill); Δcorr = mean correlation of the distilled member to the others − same for non-distilled.

## 6. The regression / predictive test (FROZEN spec)
For every pre-specified ensemble subset S:
- Predicted ensemble quality model: **EnsAUC(S) ≈ mean_{i∈S} Q_i + γ·D(S)** (rank-AUC analog of "quality + diversity"; γ fit on OOF only).
- Per-species/per-taxon: **ΔEnsAUC_c ~ β1·ΔQ_c + β2·Δcorr_c** (OLS, fit OOF).
- **Predictive claim test:** using OOF-fit coefficients, predict each held-out config's *travel* (sign and magnitude) on the validation target (§7). Report **out-of-fold R²** and **sign-accuracy** (fraction of configs whose gain/loss direction is correctly predicted).
- Pre-specified success: R² ≥ 0.3 AND sign-accuracy ≥ 0.7 → "predicts." Below → "diagnostic correlate" (verb downgrades to "explains," per §9).

## 7. Validation: DUAL, convergent (FROZEN)
Validate the diagnostic against **two imperfect, independent held-out signals** and report whether they agree:
1. **Private LB** via late submission (World A), primary if Step 0 returns private.
2. **59 labeled soundscapes** (`data/train_soundscapes_labels.csv`), real field audio, the target distribution; bootstrap CIs (it's tiny).
Convergent validation (both agree) is the credible claim and inoculates against "overfit one leaderboard subset." One leaderboard number alone is the dismissible version. Report both even in World A.

## 8. Submission protocol (anti-p-hacking: FROZEN)
- The grid (§4) and diagnostic+regression (§5–6) are fixed by this document BEFORE any new score is read.
- Submission budget: Kaggle late limit (CONFIRM, often ~5/day): `[ ____ ]`. Submit the **entire pre-specified grid**; do NOT add configs reactively to make the diagnostic "work."
- No reading private scores until the full grid is submitted. Any deviation → dated entry in the Deviations Log (bottom), and the affected claim is marked exploratory.
- **EXPLORATORY/CONFIRMATORY SPLIT (peeking is already on the record, own it):** v62's private climb was observed BEFORE this pre-reg → it is the **hypothesis-GENERATING exploratory** finding, NOT evidence. The grid is the **hypothesis-TESTING confirmatory** experiment, and its cells MUST be **fresh submissions whose private scores were not previously seen** (no re-scoring board configs). The paper must state this split explicitly; otherwise the pre-registration is theater. The 50-config public→private table is likewise exploratory/descriptive, not confirmatory of the diversity claim.

## 9. Pivot thresholds (decide in advance, per the research report)
- **KILL-THRESHOLDS (set 2026-06-04, BEFORE any grid result, do not soften after seeing numbers):**
  - **Mechanism test** = (distilled − non-distilled) mean per-species correlation-to-other-members, matched backbones, on the grid. **Survives if ≥ +0.05 AND consistent sign across ≥2 measurable taxa. KILLS Direction 1 if < +0.03 → pivot to Direction 2.** 0.03–0.05 = ambiguous → resolved by grid (more members, 3 seeds). (Per-species-mean SE ≈ 0.014 on 708 rows → +0.05 ≈ 3.5σ, +0.03 ≈ 2σ.)
  - **Predictive test** = OOF-fit regression predicting held-out (private + 59-soundscape) ensemble travel. **Survives if sign-accuracy ≥ 0.70 AND out-of-fold R² ≥ 0.30. KILLS the "predicts" verb if sign-acc ≤ 0.60 (≈chance)** → downgrade to "explains."
  - **HARD KILL for Direction 1 as pitched: mechanism < 0.03 OR predictive sign-acc ≤ 0.60.** Either one → stop defending the framing.
  - The DRAMATIC version ("severe homogenization") is ALREADY not visible (moderate 0.33–0.48 correlations, BirdNET not a standout). The paper's live form is the MODEST-BUT-PREDICTIVE version; a modest effect is fine, an UNpredictive one is fatal.
- **Day-4 check:** if distillation does NOT raise correlation per the mechanism threshold above → pivot to **Direction 2** (config-dependent efficacy; no correlation mechanism).
- **Measurement note:** per-species CORRELATION is computable on all ~200 species with prediction variance (not sparsity-limited); per-species AUC/quality needs positives (~11–15 species on 59 clips) → for the quality/predictive half, lean on private LB, and when reporting soundscape AUC state the measurable-species count explicitly.
- **World B/C** (Step 0) → verbs per §2.
- **Hedge section (always include):** Direction 3 self-contained, random/shuffled-embedding distillation control (is the transfer to non-avian taxa representation or regularization?). Cheap; de-risks the paper if the predictive result is weak.

## 10. Honest scope (LOCKED: do not let excitement erode these)
- **First-party proof only.** We prove the mechanism in OUR pipeline (we control OOF + hold the private reveal). We do NOT have other teams' OOF/ensemble compositions → cross-team contradictions are **motivation, not evidence.** State this explicitly.
- **Single-model help is fragile:** the dramatic distill gain (+0.148) is **Reptilia = 1 species (2026)**. The robust effect is distill **HURTS Insecta (n=28)**. Frame as *"narrow, fragile single-model help; broad, robust ensemble harm,"* leading with the harm side.
- **Domain-shift framing:** the claim is about **OOD (focal→field) generalization**, not in-domain fit, on public, the full Perch ensemble matched/beat decorrelated variants; the decorrelation benefit appeared on private/field. Validate on the field target accordingly.
- **Cite 2026-specific numbers** (234 species / Pantanal: 162 Aves, 35 Amphibia, 28 Insecta, 8 Mammalia, 1 Reptilia) from imageclef.org BirdCLEF2026 / Kaggle; NOT the 2025 overview (206-class Colombia task).
- **Do NOT touch closed areas:** generic calibration (Schwinger 2025), per-taxon difficulty description, taxonomy-bug fixes, rank-metric invariance, basic distill ablation, ecological priors, leaderboard-shakeup-as-topic.

## 11. Staged plan (2 weeks)
- **D1:** Step 0 gate. Reproduce distill/no-distill single-model + ensemble AUC from existing assets; build the OOF correlation matrix (extends this session's ρ=0.45). Confirm seed direction holds on the 59 soundscapes.
- **D2–3:** lock grid + diagnostic (this doc). Re-pull/compute per-model + per-species correlation and AUC.
- **D4–8:** train the grid (unlimited compute); record R, Q, ensemble AUC at every step; Day-4 pivot check.
- **D9–11:** fit decomposition + regression on OOF; run dual validation (private LB + 59 soundscapes); compute R²/sign-accuracy → select verb per §6/§9.
- **D12–14:** write (per-taxon tables, decision rule, doc-46 positioning), release code.

---
## Deviations Log (append-only, dated)
- (none yet)
