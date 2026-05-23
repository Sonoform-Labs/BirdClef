import sys
sys.path.insert(0, r"D:\Python\BirdClef")
import pandas as pd, numpy as np
from pathlib import Path
from collections import Counter
BASE = Path(r"D:\Python\BirdClef")
df = pd.read_csv(BASE/"data"/"train_soundscapes_labels.csv")
tax = pd.read_csv(BASE/"data"/"taxonomy.csv")
l2c = dict(zip(tax.primary_label.astype(str), tax.class_name))
TAXA = ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]

present, single = Counter(), Counter()
nwin = len(df); nsingle = 0; nmulti = 0; nempty = 0; per_window_n = []
for s in df.primary_label.astype(str):
    if s == "nan" or not s.strip():
        nempty += 1; per_window_n.append(0); continue
    sps = set(x.strip() for x in s.split(";") if x.strip())
    per_window_n.append(len(sps))
    for sp in sps: present[sp] += 1
    if len(sps) == 1: nsingle += 1; single[next(iter(sps))] += 1
    else: nmulti += 1

print("="*70)
print("  GATE-2 MEASURABILITY (prototype localization signal)")
print("="*70)
print(f"  windows={nwin}  single-species={nsingle} ({100*nsingle/nwin:.0f}%)  "
      f"multi={nmulti} ({100*nmulti/nwin:.0f}%)  empty={nempty}")
print(f"  mean species/window = {np.mean(per_window_n):.2f}  (higher => more co-occurrence confound)")
print(f"  species present in soundscapes = {len(present)} ; species EVER single = {len(single)}")
print("\n  NOTE: no bbox/time-freq annotations exist -> localization uses single-species-window")
print("  (+ energy) proxy. A species needs enough SINGLE-species windows to evaluate cleanly.")

def brk(counter, thr):
    spp = [sp for sp, c in counter.items() if c >= thr]
    tc = Counter(l2c.get(sp, "?") for sp in spp)
    return len(spp), {t: tc.get(t, 0) for t in TAXA}

print("\n  # species by SINGLE-species-window count (clean field localization signal):")
print(f"  {'thresh':>8s} {'n_sp':>5s}  taxa(Aves/Amph/Inse/Mamm/Rept)")
for thr in [1, 5, 10, 20, 30, 50]:
    n, tc = brk(single, thr)
    print(f"  >={thr:>4} win {n:>5}  {tc['Aves']}/{tc['Amphibia']}/{tc['Insecta']}/{tc['Mammalia']}/{tc['Reptilia']}")


tr = pd.read_csv(BASE/"data"/"train.csv")
focal = Counter(tr.primary_label.astype(str))
fc = Counter(l2c.get(sp, "?") for sp in focal)
print(f"\n  focal recordings: {len(tr)} across {len(focal)} species (in-domain localization fallback)")
print(f"    by taxon: {dict(fc)}")

print("\n  top 20 species by single-species-window count (localization candidates):")
for sp, c in single.most_common(20):
    print(f"    {sp:10s} {l2c.get(sp,'?'):9s} single={c:4d}  present={present[sp]:4d}")


n20, tc20 = brk(single, 20)
nonav20 = tc20['Amphibia'] + tc20['Insecta'] + tc20['Mammalia'] + tc20['Reptilia']
print("\n  --- read ---")
print(f"  {n20} species have >=20 single-species field windows "
      f"({tc20['Aves']} Aves + {nonav20} non-avian).")
print("  If mostly ~Aves -> Gate 2 = footnote (lean Gate 1 + artifact story).")
print("  If many non-avian too -> Gate 2 = headline depth (cross-taxa localization).")
