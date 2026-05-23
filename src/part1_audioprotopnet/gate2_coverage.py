import pandas as pd
from collections import Counter
from pathlib import Path
BASE = Path(r"D:\Python\BirdClef")
tax = pd.read_csv(BASE/"data"/"taxonomy.csv")
l2c = dict(zip(tax.primary_label.astype(str), tax.class_name))
TAXA = ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]
print("taxonomy totals:", {t: int((tax.class_name == t).sum()) for t in TAXA})

tr = pd.read_csv(BASE/"data"/"train.csv")
focal = Counter(tr.primary_label.astype(str))
print("\nFOCAL clips/species -> #species with >= N clips, by taxon:")
for thr in [1, 5, 15]:
    spp = [s for s, c in focal.items() if c >= thr]
    tc = Counter(l2c.get(s, "?") for s in spp)
    print(f"  >= {thr:>2} clips: {len(spp):>3} species  " + " ".join(f"{t}:{tc.get(t,0)}" for t in TAXA))

df = pd.read_csv(BASE/"data"/"train_soundscapes_labels.csv")
single = Counter()
for s in df.primary_label.astype(str):
    sps = {x.strip() for x in s.split(";") if x.strip()} if s != "nan" else set()
    if len(sps) == 1:
        single[next(iter(sps))] += 1
print("\nSINGLE-SPECIES soundscape windows -> #species with >= N, by taxon:")
for thr in [1, 5]:
    spp = [s for s, c in single.items() if c >= thr]
    tc = Counter(l2c.get(s, "?") for s in spp)
    print(f"  >= {thr} win: {len(spp):>3} species  " + " ".join(f"{t}:{tc.get(t,0)}" for t in TAXA))


covered = set(s for s, c in focal.items() if c >= 5) | set(s for s, c in single.items() if c >= 5)
tc = Counter(l2c.get(s, "?") for s in covered)
print(f"\nCLEAN-COVERED species (focal>=5 OR single-ss>=5): {len(covered)}  " + " ".join(f"{t}:{tc.get(t,0)}" for t in TAXA))
print("=> purity reportable for taxa with enough covered species; others coverage-limited.")
