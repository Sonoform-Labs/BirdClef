# Gate 1: Does prototype interpretability cost accuracy? (matched control)
*Draft section for the BirdCLEF+ 2026 / LifeCLEF working note. Numbers locked 2026-06-05.*

## 1. Setup (the matched comparison)

We compare two classifiers that are **identical in every respect except the classification head**:

| component | both conditions |
|---|---|
| backbone | ConvNeXt-Base, BirdSet-XCL pretrained (`DBD-research-group/ConvNeXT-Base-BirdSet-XCL`) |
| input | 128-mel log-spectrogram, 5 s @ 32 kHz, per-sample z-norm |
| data | 234-species Pantanal focal recordings + labelled soundscapes |
| CV | 5 folds (StratifiedKFold on primary label; GroupKFold by soundscape file) |
| augmentation | gain/noise (waveform) + SpecAugment |
| budget | 6 epochs, batch 32, AdamW (bb 1e-4 / head 1e-3), OneCycle, bf16 AMP |
| loss | Asymmetric multi-label loss (Ridnik et al. 2021) |

- **Prototype condition:** the AudioProtoPNet head (Heinrich et al. 2025), class-specific prototypes (5/class = 1170), cosine-similarity activations, top-1 spatial pooling, and a non-negative correct-class-only last layer (`LinearLayerWithoutNegativeConnections`). The head, its non-negative layers, and the asymmetric loss were **lifted verbatim from the official repository**; only a lightweight config shim adapts it to our backbone (we report this so the "AudioProtoPNet" claim is the real architecture, not a re-implementation).
- **Baseline condition:** the same backbone with a global-pooled linear sigmoid head.

**Two documented deviations from the native AudioProtoPNet training recipe (stated up front):**
1. **Post-hoc projection.** We project prototypes onto nearest training patches after training (for Gate 2 analysis) rather than running the in-training "push" schedule.
2. **Cross-entropy rescaling.** At 234 classes the native coefficient balance leaves the prototype regularizers ≈ 12× the cross-entropy term, which collapses training to chance (macro AUC ≈ 0.51). Rescaling the asymmetric loss to a per-sample sum (CE ≈ 1.1, comparable to the regularizer magnitudes) restores normal training; the native regularizer coefficients (cluster −0.25, separation 0.05, orthogonality 1.0) are kept. *(Sensitivity of the result to this rescaling is an open robustness check, see §5.)*

## 2. Evaluation protocol

We report the **paired** macro-ROC-AUC delta (prototype − sigmoid; both heads scored on the same out-of-fold instances) **per fold**, summarized as a cross-fold mean ± standard error, and we flag any taxon whose per-fold sign is not consistent.

**Why per-fold and not pooled OOF.** ROC-AUC depends only on the *within-set rank order* of scores, so a per-fold AUC is invariant to any monotonic rescaling of that fold's scores. Pooling predictions from the five independently-trained fold models before scoring is **not** scale-invariant, and we show below it is materially inflated for the prototype head.

## 3. Result

| taxon | per-fold Δ (proto − sigmoid) | cross-fold mean ± SE | folds (+/−) | verdict |
|---|---|---|---|---|
| **MACRO** | −0.007 −0.015 −0.021 +0.012 −0.002 | **−0.007 ± 0.006** (CI spans 0) | 1+/4− | **accuracy-competitive** |
| **Aves** (161) | −0.011 −0.013 −0.014 −0.006 −0.011 | **−0.011 ± 0.001** | 0+/5− | **robust small cost** |
| Amphibia (27) | +0.013 −0.042 +0.015 +0.052 +0.019 | +0.011 ± 0.015 | 4+/1− | sign-flips → n.s. |
| Insecta (16*) | +0.026 −0.020 −0.186 +0.118 +0.118 | +0.011 ± 0.056 | 3+/2− | sign-flips → n.s. |
| Mammalia (8) |, | (quarantined) |, | labelled obs only |
| Reptilia (1) |, | (quarantined) |, | labelled obs only |

Absolute macro AUC: prototype ≈ 0.96, sigmoid ≈ 0.97. (*Insecta: ~16 of 28 species have positives per fold.)

**Reading:** The faithful AudioProtoPNet head is **accuracy-competitive** with a matched non-prototype head, the macro delta is small and changes sign across folds (one fold favours the prototype head). The **only robust effect is a small −0.011 AUC cost on Aves**, the data-rich avian taxon (negative in all five folds, and identical across every aggregation method). No other taxon shows a reliable effect.

## 4. Methodology note: naive pooled OOF over-states the cost

A naive cross-fold *pooled* estimate gives MACRO −0.034 (95% CI excludes 0) and Insecta −0.17, i.e. it would tell the opposite ("interpretability costs ~3.4 AUC points; prototypes are terrible on insects"). **This is a cross-fold score-scale artifact.** Per-fold within-column rank-normalization, a transform that provably cannot change any per-fold AUC (verified: per-fold deltas are bit-identical before/after), collapses the pooled estimate to MACRO **+0.0003** and Insecta **+0.02**, matching the per-fold values. The per-species *absolute* prototype-similarity scale is not comparable across independently-trained folds (within-fold AUCs are healthy, 0.96–0.97; only pooling leaks the scale). We therefore recommend **per-fold paired analysis** for cross-validated prototype models, and we report the pooled number only to document the pitfall.

## 5. Limitations / robustness
- **CE-rescaling, robustness confirmed.** We re-ran fold 0 at 0.5× / 1× / 2× the per-sample-sum CE (seed-matched, all else identical): macro Δ = −0.009 / −0.008 / −0.008 and Aves Δ = −0.012 / −0.011 / −0.012. The headline (competitive macro, robust small avian cost) is **stable across a 4× range of CE scaling** (pre-registered verdict: HOLDS), so the result does not depend on the specific rescaling chosen. *(Residual caveat: the rescaling was still needed for training to converge at 234 classes; we report that as an adaptation, not a free choice.)*
- **Post-hoc, not in-training, projection.**
- **6-epoch budget; one seed per fold.** A 3-seed check on a fold (simplified-head pilot) gave macro-delta seed-std 0.0008, far below the cross-fold spread, so seed noise is not the limiting uncertainty.

## 6. Pilot context (appendix material)
A simplified cosine-prototype head on frozen features gave a *robust* −0.0265 macro cost; the faithful class-specific architecture is more competitive, i.e. interpretability is closer to accuracy-free than a simplified probe implies. A frozen-feature probe also showed a striking +0.14 "insect rescue" by prototypes, which **did not survive joint fine-tuning** (it is fold-unstable, sign-flipping). We report this as an honest negative: it illustrates why per-fold, joint-trained analysis (not frozen-feature probing or pooled CIs) is required to avoid artifactual per-taxon stories.
