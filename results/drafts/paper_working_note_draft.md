# Does inherent prototype interpretability transfer beyond birds? A matched-control stress-test of AudioProtoPNet on the BirdCLEF+ 2026 multi-taxa soundscape task

*BirdCLEF+ 2026 / LifeCLEF 2026 working note, consolidated draft (Gates 0–2). Numbers locked 2026-06-05 (Gate 1) / 2026-06-07 (Gate 2). Author/affiliation TBD.*

> **Status:** assembly draft. Full per-gate detail lives in `paper_gate1_draft.md` and `paper_gate2_draft.md`; results cards in `GATE1_RESULTS.md` / `GATE2_RESULTS.md`; protocols in `GATE2_PREREG.md`. This document stitches the three gates into submission shape and fixes the framing.

---

## Abstract

We participated in the BirdCLEF+ 2026 multi-taxa soundscape task (**234 species across five taxonomic classes**: birds, amphibians, insects, mammals, reptiles; a focal→soundscape domain shift under a CPU-only inference budget). Our competition submission, a rank-blended ensemble of public-pipeline models (Perch and BirdNET predictions with EfficientNet classifiers), scored **0.943 macro-ROC-AUC on the private leaderboard**. As participants, we additionally conducted the controlled study reported here, which is the contribution of this note. Inherently-interpretable part-prototype models (the ProtoPNet family) promise explanations "for free," and **AudioProtoPNet** (Heinrich et al., 2025) brings them to bird sound; we ask whether that promise **transfers beyond birds** to this five-taxon, domain-shifted, CPU-constrained setting. **Our contribution is the question and the first-party measurements, not a new architecture**: to our knowledge, the first matched-control evaluation of inherent prototype interpretability beyond avian audio. We compare the **faithful AudioProtoPNet head, lifted verbatim from the official repository**: against a **matched non-prototype sigmoid head** that is identical in backbone, initialization, data, folds, augmentation, and training budget; this matched control is the spine that makes every claim identifiable. We find: **(Gate 0)** a single faithful model fits the CPU-only budget, **~52 min measured** on the competition's 4-core hardware, which *forces* (not merely permits) a single-model design; and, across five folds, **(Gate 1)** the prototype head is **accuracy-competitive**: macro-AUC Δ (prototype − sigmoid) = **−0.007, 95% CI spanning zero**: with the only robust effect a small **−0.011** cost on the data-rich avian taxon; **(Gate 2)** its prototypes are **valid and call-focused for birds** (own-class purity **0.95 vs a 0.006 random-prototype null**; below the background-energy null at every cutoff), **only partially so for amphibians** (pure-ish, but not call-peaked), and **call-focused-but-purity-unmeasurable for insects**: i.e. **interpretability is real but taxon-graded, weakening with distance from the bird-centric pre-training.** We additionally document a **methodological pitfall**: naively pooling prototype scores across cross-validation folds overstates the accuracy cost by roughly five-fold (−0.034 vs −0.007), a within-fold score-scale artifact that per-fold, rank-invariant analysis removes. All protocols were pre-registered with every outcome declared publishable in advance. We release the per-fold, per-taxon measurements.

---

## 1. Introduction

**Participation.** We took part in BirdCLEF+ 2026. Our competition submission was a rank-blended ensemble of public-pipeline models; Perch and BirdNET soundscape predictions combined with EfficientNet-B0/B2 classifiers, the now-standard public recipe, scoring **0.943** macro-ROC-AUC on the private leaderboard. That ensemble is not the contribution of this note; we report it to situate the controlled study below, in which we reuse the same task, data, and CPU-inference budget to ask a question the leaderboard cannot: does an *interpretable* prototype model pay for its interpretability, and are its explanations valid on this multi-taxa, domain-shifted audio?

Part-prototype networks classify by comparing an input to a set of learned, class-specific "this looks like that" prototypes, and are widely promoted as **inherently** interpretable, the explanation is the model, not a post-hoc approximation. AudioProtoPNet ports this to avian bioacoustics on a BirdSet-pretrained ConvNeXt backbone and reports state-of-the-art transfer. Two questions remain open and are the subject of this note:

1. **Does the interpretability come for free?** Inherent interpretability is only attractive if it does not cost accuracy. This requires a *matched* comparison, same backbone, data, and budget, only the head differs, which the original paper, focused on cross-dataset SOTA, does not isolate.
2. **Does it transfer beyond birds?** AudioProtoPNet is bird-only and its authors name non-avian taxa (insects, amphibians, mammals) as *unrealized future work*. BirdCLEF+ 2026's five-taxon Pantanal soundscapes are a natural first test of whether prototype validity survives acoustically divergent, data-poor taxa under domain shift.

We do **not** propose a new model. We contribute the **question**, a **matched-control protocol**, and **first-party per-fold/per-taxon measurements** on a task the original work did not cover. We organize the study as a sequence of **pre-registered gates**, each with kill-criteria fixed before looking and *both* outcomes declared publishable, a clean negative is a result; a messy positive is a reject.

## 2. Related work and scope of novelty

- **AudioProtoPNet** (Heinrich et al., 2025; *Ecological Informatics*): ConvNeXt-Base + class-specific cosine prototypes, non-negative own-class last layer, asymmetric loss, in-training prototype projection. **Bird-only**: trained and evaluated entirely within BirdSet's avian scope (a bird-only training set; all seven BirdSet test sets are bird soundscapes), the exact boundary our multi-taxa study probes; non-avian taxa are flagged as future work. We reuse its head verbatim.
- **Post-hoc audio explanation (e.g. APEX, 2026)** explains a trained model *after the fact* on BirdSet. We draw the **inherent-vs-post-hoc** line and the **BirdSet-vs-multitaxa-BirdCLEF** line: ours is a stress-test of the *inherent* method on a *broader taxon set*.
- **In-competition "prototype" scoring heads (e.g. ProtoSSM on Perch embeddings)** use prototype-shaped scoring with **no projection and no visualization**: not interpretable in the ProtoPNet sense; we cite to distinguish.
- **ProtoPNet validity critiques** (Hoffmann et al., 2021; Nauta et al., 2020; and recent comprehensive evaluations) show prototype explanations can be unfaithful or shortcut-driven. This literature *motivates Gate 2*: we test validity rather than assume it.

To our knowledge, this is the **first matched-control evaluation of inherent prototype interpretability beyond avian audio**. Scoped honestly, the novelty axis is **"inherent prototype interpretability, stress-tested beyond birds, with a matched control and pre-registered validity metrics"**: not a new architecture and not a leaderboard result.

## 3. Data and task

BirdCLEF+ 2026 targets **234 species** of the Brazilian Pantanal across **five classes** (Aves 162, Amphibia 35, Insecta 28, Mammalia 8, Reptilia 1), scored by macro-averaged ROC-AUC. Training is **focal recordings + labelled soundscapes**; the hidden test is **soundscapes** (≈600 one-minute files × 12 windows), so the core difficulty is a **focal→soundscape domain shift**. The labelled soundscapes are **richly multi-taxa** (≈4.2 species per positive window; 75 species with positives), which both enables and constrains the validity analysis (Gate 2): clean single-species call signal is abundant for amphibians and birds but **scarce for insects** (3 of 28 species), and **no bounding-box annotations exist anywhere**, so call *localization* is a documented data limitation rather than a metric.

**Species-count convention (stated once).** Three per-taxon counts recur; we mark which applies wherever it matters. **(i) Taxonomy**: the label space (Aves 162, Amphibia 35, Insecta 28, Mammalia 8, Reptilia 1), used in §3–§4 and as the Gate-1 row labels. **(ii) Clean-covered**: species with usable single-species projection patches, over which Gate-2 purity is computed (Aves 161, Amphibia 25, Insecta 3; these are the Gate-2 table counts). **(iii) With-positives**: species with ≥1 positive in a fold's validation set, over which per-taxon Gate-1 ROC-AUC is averaged (≤ taxonomy, fold-varying, smallest for insects). Counts (ii)–(iii) fall below taxonomy because rare species lack clean clips or fold positives, the 234-class label space is unchanged.

## 4. Method: the matched control

Both conditions are **identical except the classification head**:

| component | both conditions |
|---|---|
| backbone | ConvNeXt-Base, BirdSet-XCL pretrained (`DBD-research-group/ConvNeXT-Base-BirdSet-XCL`) |
| input | 128-mel log-spectrogram, 5 s @ 32 kHz, per-sample z-norm |
| CV | 5 folds (StratifiedKFold on primary label; GroupKFold by soundscape file) |
| augmentation | gain/noise (waveform) + SpecAugment |
| budget | 6 epochs, batch 32, AdamW (bb 1e-4 / head 1e-3), OneCycle, **bf16** AMP |
| loss | Asymmetric multi-label loss (Ridnik et al., 2021) |

- **Prototype condition:** the AudioProtoPNet head, class-specific prototypes (5/class = 1170), cosine activations, top-1 spatial pooling, non-negative correct-class last layer. **The head, its non-negative layers, and the asymmetric loss are lifted verbatim from the official repository**; only a config shim adapts it to our backbone, so "AudioProtoPNet" denotes the real architecture, not a re-implementation.
- **Baseline condition:** the same backbone with a global-pooled linear sigmoid head.

**Two deviations from the native recipe, stated up front and carried through every gate:** **(i) post-hoc projection** (we project prototypes to nearest patches *after* training, for Gate 2, rather than the in-training push); **(ii) cross-entropy rescaling**: at 234 classes the native coefficient balance leaves the prototype regularizers ≈12× the CE term and training collapses to chance (macro ≈0.51); rescaling the asymmetric loss to a per-sample sum restores normal training, keeping the native regularizer coefficients. The Gate-1 result is shown **robust across a 4× range** of this rescaling (§6).

*(Engineering note carried as lessons, not claims: ConvNeXt-Base requires **bf16**: fp16 autocast overflows to NaN, and a single NaN poisons a column's AUC; long runs use **per-fold process isolation** after a 5-fold monolith died silently mid-run.)*

## 5. Gate 0: CPU inference feasibility (measured)

Before committing to the architecture we verified that a **single** faithful AudioProtoPNet on ConvNeXt-Base meets the competition's **90-minute CPU-only** inference budget. We measured this **first-party on a Kaggle CPU notebook (4 cores, the competition's inference environment)**: the real BirdSet ConvNeXt-Base backbone + prototype head with our exact 128-mel front-end, timed across inference engines and projected to the full hidden test, taking **700 one-minute files × 12 = 8 400 windows as a conservative upper bound** (the actual test is ≈600 files, so the real margin is larger, and PASS at 700 implies PASS at 600).

| engine | per-window (incl. mel) | projected @700 files | verdict |
|---|---|---|---|
| PyTorch | 368 ms | **51.6 min** | PASS |
| ONNX Runtime | 355 ms | 49.6 min | PASS |
| OpenVINO | 448 ms | 62.7 min | PASS |

A single model finishes in **~52 min (PyTorch) / ~50 min (ONNX Runtime)**: inside the 90-min cap and our ≤70-min headroom target; even the slowest engine passes. This **matches the published 2025 DS@GT precedent (~51–60 min for ConvNeXt-Base) and confirms it on our own model.** (OpenVINO did *not* accelerate inference on this 4-core CPU; we report it for completeness rather than claiming a speedup.) Critically, the budget **forces** the single-model design rather than merely permitting it: at ~52 min/model a five-fold ensemble would need <18 min/model, unreachable for ConvNeXt-Base on CPU, so the 90-min cap admits exactly one model. We therefore study a single faithful model not as a concession but as the **budget-determined design point**, which Gate 0 confirms fits (an added SED head would likely break it too, not separately measured). *(This is a deployment-feasibility measurement, not a competition submission; the head timed is compute-equivalent to the faithful head, exact to <1%, since the backbone dominates FLOPs. The paper's scientific results are the offline five-fold gates of §6–§7.)*

## 6. Gate 1: does prototype interpretability cost accuracy?

We report the **paired** macro-ROC-AUC delta (prototype − sigmoid, scored on the same out-of-fold instances) **per fold**, summarized as cross-fold mean ± SE, flagging any taxon whose per-fold sign is inconsistent.

| taxon | cross-fold mean ± SE | folds (+/−) | verdict |
|---|---|---|---|
| **MACRO** | **−0.007 ± 0.006** (CI spans 0) | 1+/4− | **accuracy-competitive** |
| **Aves** (162) | **−0.011 ± 0.001** | 0+/5− | **robust small cost** |
| Amphibia (35) | +0.011 ± 0.015 | 4+/1− | sign-flips → n.s. |
| Insecta (28) | +0.011 ± 0.056 | 3+/2− | sign-flips → n.s. |
| Mammalia (8) / Reptilia (1) | (quarantined) |, | labelled obs only |

*(Row labels are taxonomy counts (§3 convention); per-taxon AUC is averaged over the species with ≥1 positive in each fold, which is ≤ the taxonomy count.)*

**Reading:** the faithful head is **accuracy-competitive**; the macro delta is small and changes sign across folds. The **only robust effect is a −0.011 cost on Aves** (negative in all five folds, identical across aggregations). **Robustness:** re-running fold 0 at 0.5×/1×/2× the rescaled CE gives macro Δ −0.009/−0.008/−0.008 and Aves −0.012/−0.011/−0.012, the headline is stable across a **4× CE range** (pre-registered verdict: holds).

**Methodology note, naive pooled OOF overstates the cost ~5×.** A cross-fold *pooled* estimate gives macro **−0.034** (CI excludes 0) and Insecta **−0.17**: the opposite story. **Three numbers are in play, and only one is our reported result:** the **per-fold paired** delta (**−0.007**, the table above) is what we report, the correct metric, since ROC-AUC depends only on within-fold rank order; the **naive raw-pooled** delta (**−0.034**) is the artifact; and the **rank-normalized pooled** delta (**+0.0003**) is an *independent* confirmation that the −0.034 came from pooling, not from the prototype head. The reported −0.007 and the confirmatory +0.0003 are both effectively zero, two different corrections of one artifact, not two competing "true" answers. The mechanism is a **within-fold score-scale artifact**: per-fold rank-normalization, a transform that *provably cannot change any per-fold AUC* (verified bit-identical), collapses the pooled estimate to macro **+0.0003** and Insecta **+0.02**. Independently-trained fold models produce per-species prototype-similarity scales that are **not comparable across folds**; only pooling leaks that scale. We recommend **per-fold paired analysis** for cross-validated prototype models and report the pooled number only to document the pitfall. *(We then took care not to repeat this error in Gate 2, see §7's owned bird-weighting.)*

## 7. Gate 2: are the prototypes valid, or shortcuts? (taxon-graded)

Inference-only on the five saved prototype fold-models (cannot affect Gate 1). For each fold we build a **clean single-species projection pool** (2 857 clips: focal ≥5/species + single-species soundscape windows), **project every prototype to its nearest pool patch**, and score: **purity** (nearest patch from the prototype's own class, vs a random-prototype null) and **shortcut-rate** (nearest patch in the bottom energy tercile, reported at 25/33/50% with required robustness across all three). All definitions, nulls, the per-taxon **quarantine** (Mammalia n=8, Reptilia n=1) and **coverage** rules, and the outcome interpretations were **pre-registered** (`GATE2_PREREG.md`) with **every outcome declared publishable in advance**.

We report **per-taxon first**; the pooled number is bird-dominated (161/234 classes) and the across-taxa gradient is the finding.

| taxon | purity (obs vs random null) | shortcut @25/33/50 (null .25/.33/.50) | reading |
|---|---|---|---|
| **Aves** (161) | **0.951 ± 0.004** vs 0.006 | 0.199 / 0.263 / 0.407, below null all 3 | **clean: pure *and* call-focused** |
| **Amphibia** (25) | **0.562 ± 0.020** vs 0.001 | 0.195 / 0.313 / **0.538**: below@25,33; at-chance@50 | **pure-ish; call-focus partial** |
| **Insecta** (3/28) | *coverage-limited, not assessable* | 0.056 / 0.177 / 0.384, below null all 3 | call-focused; purity is a data gap |
| *(pooled)* | *0.774 ± 0.003 vs 0.004, **bird-weighted**, reported as such* | *0.177 / 0.257 / 0.422* | *bird-dominated average* |
| Mammalia (8) / Reptilia (1) | (quarantined) | (quarantined) | small-n, labelled obs only |

*(Aves/Amphibia/Insecta counts are clean-covered species (§3 convention); Insecta shows clean-covered/taxonomy = 3/28. Mammalia/Reptilia show taxonomy n, quarantined.)*

**The gradient is the result:** validity is **cleanest for birds**, **partial for amphibians**, and **purity-unmeasurable (but shortcut-clean) for insects**: it **degrades with acoustic and data distance from the bird-centric pre-training.** Two honesty calibrations: **(i)** purity ≫ null is *partly expected* (the cluster loss and non-negative own-class layer push prototypes toward own-class features), so the informative content is the **taxon spread** (0.95 birds vs 0.56 amphibians), not the headline gap; **(ii)** the **three-cutoff check caught a real overclaim**: amphibian shortcut reads "call-aligned" at the 33% line (0.313) but is **at chance at 50%** (0.538), i.e. amphibian prototypes *avoid silence but are not call-peaked.* Insect purity is a **coverage limitation, not a null result** (3 of 28 species have clean clips).

## 8. Limitations

- **Two recipe deviations** (post-hoc projection; CE rescaling), disclosed; CE robustness quantified (§6).
- **Field call-localization** is a documented data gap (multi-taxa soundscapes, no bounding boxes); we measure purity/shortcut, not localization overlap.
- **Insect purity unmeasurable** (coverage), reported as such; insect *shortcut* is reported.
- **Gate 3 (taxon-stratified regression) deferred, not dropped.** With only two taxa reportable for purity (two more quarantined), a regression controlling for log sample-count and support is **underpowered**; we report per-taxon descriptives and state that a formal taxon regression is not supportable at this coverage (a journal-extension item). Forcing an n≈2 regression would invite the over-reading the pre-registration was written to prevent.
- **One seed per fold, 6-epoch budget.** A 3-seed check on the **simplified-head pilot** (fold 0, not the faithful head) gave macro-delta seed-std 0.0008, far below the cross-fold spread; the faithful head was not seed-replicated (a noted limitation, though the pilot suggests seed noise is small relative to the cross-fold variation).

## 9. Conclusion

A matched-control, pre-registered stress-test shows that **faithful AudioProtoPNet buys interpretability at near-zero accuracy cost** on a 234-species, five-taxon soundscape task (macro Δ −0.007, CI spanning zero; one robust −0.011 avian cost), and that its **interpretability is genuine but taxon-graded**: prototypes are valid and call-focused for birds, only partially for amphibians, and call-focused-but-purity-unmeasurable for insects, weakening with distance from the bird-centric pre-training. Along the way we document a cross-fold **score-scale pooling artifact** that would otherwise overstate the cost five-fold. The promise of inherent prototype interpretability **largely holds where the pre-training and data are dense, and visibly thins where they are not**: a concrete, honest answer to the "beyond birds" question, and a cautionary calibration for anyone deploying prototype models on long-tailed, multi-taxa bioacoustic data.
