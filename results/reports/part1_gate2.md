# Gate-2 results: prototype validity (faithful AudioProtoPNet, 5-fold, inference-only)
Pre-reg: `preregistration.md`. Pool: 2857 clean single-species clips. Cross-fold mean ± SE over 5 fold-models. Per-fold purity 0.765–0.785 (stable).

| scope | purity (obs vs random null) | shortcut@25/33/50 (null .25/.33/.50) | read |
|---|---|---|---|
| **ALL** | **0.774 ± 0.003** vs 0.004 | 0.177 / 0.257 / 0.422, **< null all 3** | **MEANINGFUL** |
| **Aves** | **0.951 ± 0.004** vs 0.006 | 0.199 / 0.263 / 0.407, **< null all 3** | **MEANINGFUL (clean)** |
| **Amphibia** | **0.562 ± 0.020** vs 0.001 | 0.195 / 0.313 / **0.538**: <null@25,33; >null@50 | **pure; call-focus PARTIAL** |
| Insecta | 0.107 (coverage-limited, 3/28 sp) | 0.056 / 0.177 / 0.384, < null all 3 | purity N/A (data); **shortcut call-aligned** |
| Mammalia (n=8) | 0.560 ± 0.038 | 0.095 / 0.175 / 0.340 | QUARANTINE |
| Reptilia (n=1) | 0.000 | 0.040 / 0.160 / 0.400 | QUARANTINE |

## Pre-registered verdict
- **MEANINGFUL (overall + Aves):** purity ≫ random-proto null (0.77 / 0.95 vs ~0) **AND** shortcut-rate below chance at all three cutoffs → prototypes are class-pure and call-focused, not background shortcuts.
- **Taxon-dependent (declared publishable in advance):** **Amphibia** prototypes are class-pure (0.56 ≫ null) but only **partially** call-focused, below chance at the 25/33% cutoffs (they avoid silence) yet ~chance at 50% (not strongly call-peaked). *(The 3-cutoff robustness check caught this; a single 33% threshold would have mislabeled it "call-aligned", exactly the threshold-shopping the pre-reg guards against.)* **Insecta** prototypes are call-aligned (shortcut < null all 3) but **purity is not assessable** (only 3/28 species have clean single-species clips, a documented data limitation, not a result).
- **Localization-to-ground-truth-call overlap:** not attempted (no bbox annotations), footnote.

## Headline
Faithful AudioProtoPNet prototypes are **class-pure and predominantly call-focused (not background shortcuts)** on multi-taxa, field-shifted audio, strongest for birds, present overall; amphibian call-focus is partial; insect purity is data-limited. Combined with Gate 1 (accuracy-competitive), this is **near-free, genuinely meaningful interpretability**: the result that lifts the paper above Gate-1-only.
