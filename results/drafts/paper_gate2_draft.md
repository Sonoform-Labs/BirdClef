# Gate 2: Are the prototypes valid, or shortcuts? (taxon-graded)
*Draft section for the BirdCLEF+ 2026 / LifeCLEF working note. Numbers locked 2026-06-07.*

Gate 1 establishes that the prototype head is *accuracy*-competitive. Gate 2 asks the question that actually justifies an inherently-interpretable model: **are its learned prototypes valid**: do they correspond to real same-class acoustic events, and do they attend to call energy rather than to background shortcuts, **and does that hold beyond birds?**

## 1. Setup

This is an **inference-only** analysis on the **five saved fold-models from Gate 1's prototype condition**: no retraining, so it cannot influence the Gate-1 result. For each fold-model we:

1. Build a **clean projection pool** of single-species clips: focal recordings (≥5 clips/species) plus single-species labelled-soundscape windows, sampled ≤15/species (seed 42) → **2 857 clips**. Each grid cell of each clip is a *patch* tagged with (source species, within-clip energy percentile).
2. Run the pool through that fold's backbone and **project every prototype to its nearest pool patch** by cosine similarity (post-hoc projection).
3. Score two metrics per prototype, then aggregate per taxon (cross-fold mean ± SE; per-fold purity is stable at 0.765–0.785).

**Metric 1, purity** = fraction of prototypes whose nearest patch is from the prototype's **own class**, always reported against a **random-prototype null** (Gaussian vectors projected the same way ≈ class-prevalence chance).
**Metric 2, shortcut-rate** = fraction of prototypes whose nearest patch sits in the **bottom energy tercile** of its clip (low energy ⇒ background). We report it at the 25 / 33 / 50-percentile cutoffs (null = .25 / .33 / .50) and require the conclusion to hold across all three.

## 2. Pre-registration (locked before any computation)

The metrics, the random-prototype and energy-percentile nulls, the three-cutoff robustness requirement, the per-taxon quarantine (Mammalia n=8, Reptilia n=1) and **coverage** rules, and the outcome interpretations were all fixed in `GATE2_PREREG.md` **before** the models were touched. Crucially, **every outcome was declared publishable in advance**: "clean for all measurable taxa," "shortcut for all," and "clean for birds but degraded/unmeasurable for others" were each pre-committed as interesting, so the taxon-graded result below is not a post-hoc story.

The pre-registration also fixed a **coverage rule** that turns out to bind: purity needs own-class patches in the pool, so a taxon's purity is a claim only if ≥10 of its species have ≥5 clean clips. Only **Aves (161 species) and Amphibia (25)** clear it; **Insecta has just 3 of 28 species with clean single-species clips**, so insect *purity is not assessable* (flagged, not reported as a value). Shortcut-rate needs no class attribution, a prototype lands *somewhere* and we read that patch's energy, so **insect shortcut is measurable** even though insect purity is not.

## 3. Result: interpretability is taxon-graded

We report the **per-taxon** result first, because the pooled number is dominated by birds (161 of 234 classes are Aves) and the across-taxa gradient, not the average, is the finding.

| taxon | purity (obs vs random null) | shortcut-rate @25/33/50 (null .25/.33/.50) | reading |
|---|---|---|---|
| **Aves** (161 sp) | **0.951 ± 0.004** vs 0.006 | 0.199 / 0.263 / 0.407, **below null at all three** | **clean: pure *and* call-focused** |
| **Amphibia** (25 sp) | **0.562 ± 0.020** vs 0.001 | 0.195 / 0.313 / **0.538**: below@25/33, **at chance@50** | **pure-ish; call-focus only partial** |
| **Insecta** (3 of 28 sp) | *coverage-limited, not assessable* | 0.056 / 0.177 / 0.384, **below null at all three** | call-focused; purity is a data gap |
| *(pooled, bird-weighted)* | *0.774 ± 0.003 vs 0.004* | *0.177 / 0.257 / 0.422* | *reported only as a bird-dominated average* |
| Mammalia (8) / Reptilia (1) | (quarantined) | (quarantined) | small-n, labelled obs only |

**The gradient is the result.** Prototype validity is **cleanest for birds** (95% land on an own-class patch; call-focused at every cutoff), **partial for amphibians** (clearly above the purity null, but call-focus that holds at the strict cutoffs evaporates at the lenient one), and for **insects** purity is a documented coverage gap while their prototypes are, where measurable, call-focused. Validity **degrades with acoustic and data distance from the bird-centric pre-training**: which is exactly the "beyond birds" question the paper is built to ask, answered honestly: *partially, and it degrades.*

## 4. Reading: what is news and what is confirmation

We separate the two so the claims survive review:

- **Purity ≫ null is partly expected, and we say so.** The cluster loss and the non-negative own-class last layer *explicitly* push prototypes toward own-class features, so "0.77 vs 0.004" mainly confirms training did what it was built to do, a sanity check, not a discovery. The informative content *within* purity is the **taxon spread** (birds 0.95 vs amphibians 0.56): amphibian prototypes land on another class's patch ~44% of the time, so even purity is materially weaker off-birds.
- **Shortcut-rate is the genuine test.** Nothing in the objective forces a prototype onto *high-energy* patches, so a below-null shortcut-rate is a real interpretability finding: birds, insects, and the pooled set prefer call energy over background.
- **The three-cutoff check earned its place.** Amphibian shortcut-rate is 0.313 at the 33% line, under a single threshold we would have written "call-aligned." Seeing 0.538 at the 50% cutoff shows it is actually *at chance* there: amphibian prototypes **avoid silence but are not call-peaked.** Reporting all three cutoffs converted a clean overclaim into the correct, weaker statement, the same discipline that, in Gate 1, kept us from reporting a pooled artifact.

## 5. Limitations and an explicitly deferred test

- **Two deviations from the native recipe (as in Gate 1).** (i) **Post-hoc projection**: we project after training rather than running the in-training "push," so these are the validity properties of the learned prototype *vectors*, not of pushed prototypes. (ii) **CE rescaling**: required for convergence at 234 classes (Gate 1 §5; shown robust across a 4× CE range).
- **Insect purity is a coverage limitation, not a null result.** Only 3 of 28 insect species have clean single-species clips; we report insect *shortcut* and explicitly decline to report insect *purity*.
- **Projection pool is the full clean set, not fold-restricted.** Validity is a correspondence property, not a generalization metric, so train/val separation is not required here; we note it for completeness.
- **Gate 3 (taxon-stratified regression) is deferred, not dropped.** A formal regression of the validity metrics on taxon while controlling for log sample-count and support is **underpowered**: only two taxa (Aves, Amphibia) are reportable for purity, with two more quarantined. We therefore report the per-taxon descriptives above and state plainly that a taxon regression is not supportable at this coverage, a journal extension, not a working-note claim. Forcing an n≈2 regression would invite exactly the over-reading the pre-registration was written to prevent.

**Gate-2 headline.** *Faithful AudioProtoPNet prototypes are valid and call-focused for birds, only partially so for amphibians, and call-focused-but-purity-unmeasurable for insects: prototype interpretability is real but **taxon-graded**, strongest where the pre-training and the data are, and it weakens with distance from birds.* Combined with Gate 1, the model buys interpretability at **near-zero accuracy cost**, and that interpretability is **genuine, but not uniform across taxa.**
