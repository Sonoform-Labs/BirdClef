# BirdCLEF+ 2026 working note: build & submission TODO

Source of truth for prose/numbers: `../paper_working_note_draft.md` (markdown). `paper.tex` is the
ceurart port of it. If a number changes, change it in BOTH or pick one as canonical.

## Compile
**Easiest; Overleaf:** New Project → search the **"CEURART"** template → replace its `main.tex`
with `paper.tex` and add `references.bib`. (The template supplies `ceurart.cls`.)
**Local:** needs `ceurart.cls` (from the CEUR-WS template zip) in this folder, then:
```
pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper
```
> We don't ship `ceurart.cls` (it's CEUR's file); Overleaf is the path of least resistance.

**Bibliography style:** `paper.tex` calls `\bibliography{references}` with NO explicit `\bibliographystyle`: matching the official ceurart template (the class sets the style internally). If the first BibTeX run errors with "no `\bibliographystyle`" or the references don't appear, add `\bibliographystyle{plainurl}` immediately before `\bibliography{references}` and recompile. Let the compile decide, don't add it preemptively.

**Declaration on Generative AI:** added before the bibliography (CEUR/CLEF requirement). It honestly discloses substantial AI assistance (code, experiments, analysis, drafting) under author direction. REVIEW it for accuracy and comfort, and confirm the exact required wording/format in the 2026 CfP.

## Citations
**Method/dataset refs VERIFIED 2026-06-08** (arXiv / DBLP / CVF / publisher), corrections applied:
Chen ProtoPNet (fixed "Daniel"→**Chaofan** Tao; order Barnett/Su/Rudin), Nauta (fixed authors →
**Nauta/Jutte/Provoost/Seifert**: was wrongly the *ProtoTree* author list; ECML PKDD 2021),
Hoffmann (ICML'21 XAI workshop, 2105.02968), Ridnik (ICCV'21, pp. 82–91), Liu ConvNeXt **V1**
CVPR'22 (matches BirdSet's `ConvNextModel` checkpoint, not V2), BirdSet (2403.10380, full 10-author
list), AudioProtoPNet (Ecol. Inf. 87:103081, 2025, Elsevier; arXiv 2404.10420).
- [ ] **`birdclef2026`**: BirdCLEF+ 2026 task overview. **STRUCTURAL REQUIREMENT, still TODO** , 
      doesn't exist until the proceedings; fill from the organizers' official "cite as." Do **not** invent it.
- [ ] **`lifeclef2026`**: LifeCLEF 2026 lab overview. **STRUCTURAL REQUIREMENT, still TODO.** Same.

## Author block (`paper.tex` frontmatter)
- [x] Author block FILLED: Tsz Chai Alan Liu, Independent Researcher, tc.alanliu@gmail.com (ORCID optional, omitted).
- [~] DBLP/affiliation email to organizers, user DEPRIORITIZED (their call, 2026-06-08). Re-raise only if eligibility actually surfaces at submission.
- [x] `\conference{...}`: FILLED & verified: CLEF 2026, September 21--24, 2026, Jena, Germany (Friedrich-Schiller-Universität Jena).

## Optional (after template + bib + author)
- [ ] Fig: taxon-gradient bar (Gate-2 purity + shortcut vs. nulls, Aves/Amphibia/Insecta).
- [ ] Fig: one prototype panel (a projected prototype + its nearest patch).
- [ ] Abstract whitespace (it's one dense paragraph; optional readability).
> Pre-registration chose numbers over panels, so figures are nice-to-have, not load-bearing.

## State
Science + prose done and advisor-reviewed (#1–#5 consistency/precision fixes applied). Four gates
measured/stress-tested; taxon-gradient finding + pooling-artifact methodology note. Remaining work
is: compile on Overleaf, fill the two overview citations at camera-ready, optional figures. Author
block + conference line + Gen-AI declaration are DONE; DBLP email deprioritized by user. Full
leaderboard submission PARKED (participant 0.943 + measured Gate-0 + complete study = on-pattern).
