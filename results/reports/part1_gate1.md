# Gate-1: canonical results card (locked 2026-06-05)
Code: `src/part1_audioprotopnet/` (`faithful_head.py`, `train_faithful.py`, `gate1_faithful_diagnose.py`).

## Faithful AudioProtoPNet vs matched sigmoid head (5-fold, per-fold paired Δ = primary metric)

| scope | per-fold mean ± SE | folds (+/−) | raw-pooled ⚠ | rankN-pooled | verdict |
|---|---|---|---|---|---|
| **MACRO** | **−0.007 ± 0.006** | 1+/4− | −0.034 | +0.000 | **competitive (CI spans 0)** |
| **Aves** | **−0.011 ± 0.001** | 0+/5− | −0.011 | −0.012 | **robust small cost** |
| Amphibia | +0.011 ± 0.015 | 4+/1− | −0.031 | +0.041 | sign-flips → n.s. |
| Insecta | +0.011 ± 0.056 | 3+/2− | −0.170 | +0.023 | sign-flips → n.s. |
| Mammalia (n=8) | −0.029 ± 0.020 | 2+/3− | −0.071 | −0.038 | quarantined |
| Reptilia (n=1) | +0.045 ± 0.151 | 2+/2− | +0.252 | +0.198 | quarantined |

**Headline:** faithful AudioProtoPNet is **accuracy-competitive** with a matched plain head; the only robust effect is a small **−0.011 AUC cost on Aves**; no reliable per-taxon effects.

**`raw-pooled` is a proven cross-fold-scale artifact** (per-fold AUC is rank-invariant, verified bit-identical under within-fold rank-norm; rank-normalizing then pooling collapses −0.034→+0.000 and −0.17→+0.02). Report per-fold.

**Deviations (documented):** post-hoc projection (not in-training push); CE rescaled to per-sample-sum (native coefs drowned the 234-class CE → chance collapse; rescale → trains normally, native regularizer coefs kept).

**Open robustness item:** CE-rescale sensitivity not yet quantified.


## CE-RESCALE SENSITIVITY (fold 0; 0.5x / 1x / 2x per-sample-sum CE; seed 42)
   CE scale   macro Δ    Aves Δ proto macro  note
  ----------------------------------------------------
        0.5   -0.0087   -0.0123      0.9694  
        1.0   -0.0077   -0.0114      0.9705  
        2.0   -0.0076   -0.0118      0.9706  

  PRE-REGISTERED VERDICT: HOLDS, headline stable across CE scale (macro in [-0.02,+0.01], Aves negative all 3)
  (criteria fixed before run: HOLDS = all 3 scales macro∈[-0.02,+0.01] AND Aves Δ<0.)
