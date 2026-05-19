import pandas as pd
from pathlib import Path
from collections import Counter
BASE = Path(r"D:\Python\BirdClef")
df = pd.read_csv(BASE/"data"/"train_soundscapes_labels.csv")
tax = pd.read_csv(BASE/"data"/"taxonomy.csv")
lab2class = dict(zip(tax.primary_label.astype(str), tax.class_name))

print(f"segments(rows): {len(df)}   unique soundscape files: {df.filename.nunique()}")
sp_counter = Counter(); n_empty = 0
for s in df.primary_label.astype(str):
    if s == "nan" or not s.strip():
        n_empty += 1; continue
    for sp in {x.strip() for x in s.split(";") if x.strip()}:
        sp_counter[sp] += 1
print(f"segments with >=1 label: {len(df)-n_empty}   empty: {n_empty}")
present = list(sp_counter)
print(f"unique species present: {len(present)}")
print(f"taxon breakdown (present): {dict(Counter(lab2class.get(sp,'?') for sp in present))}")
for thr in [1, 5, 10, 20, 50]:
    spp = [sp for sp, c in sp_counter.items() if c >= thr]
    tc = Counter(lab2class.get(sp, "?") for sp in spp)
    print(f"  species with >={thr:>2d} positive segments: {len(spp):>3d}  taxa={dict(tc)}")
print(f"\ntaxonomy totals (all 234): {dict(Counter(tax.class_name))}")

print("\ntop 15 species by #positive segments (Gate-2 localization candidates):")
for sp, c in sp_counter.most_common(15):
    print(f"  {sp:10s} {lab2class.get(sp,'?'):10s} {c:4d} segments")
