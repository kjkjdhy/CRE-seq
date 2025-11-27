# Purpose: Find K562-related table files in the sharpr_mpra directory,
#          Try reading a file, print column names + first few rows to help us identify "sequence/activity" columns.

import os, re, pandas as pd

DATA_DIR = os.path.expanduser('~/CRE-seq/data/sharpr_mpra')

# 1) Collect files: name contains k562 and is a common table extension
files = [f for f in os.listdir(DATA_DIR)
         if re.search(r'k562', f, re.I) and f.endswith(('.txt', '.tsv', '.csv'))]
print(f'Found {len(files)} K562 files:')
for f in files[:12]:
    print('  -', f)

# 2) Try different separators in sequence, read first 5 rows for "probing" (don't load whole table, faster)
for f in files:
    path = os.path.join(DATA_DIR, f)
    for sep in ['\t', ',', ' ']:   # tab / comma / space
        try:
            df = pd.read_csv(path, sep=sep, nrows=5, comment='#')
            if df.shape[1] >= 2:
                print('\nPreview:', f, '| sep =', repr(sep))
                print('Columns:', list(df.columns))
                print(df.head(3))
                raise SystemExit  # Stop after finding one readable sample
        except Exception:
            pass

print("\nDidn't auto-preview. Tell me the filenames you see and we'll tune the reader.")
