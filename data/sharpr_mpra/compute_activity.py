"""
Goal:
  Calculate normalized activity for each sequence from k562_counts_wide.csv
  Steps:
    1) Divide mRNA by corresponding Plasmid counts -> ratio
    2) Take log2(ratio + 1e-6) to prevent division by zero
    3) If there are multiple replicate groups, take the average
    4) Save k562_sequence_activity.csv, containing Sequence + activity
"""

import os, re, numpy as np, pandas as pd

DATA_DIR = os.path.expanduser('~/CRE-seq/data/sharpr_mpra')
wide = pd.read_csv(os.path.join(DATA_DIR, 'k562_counts_wide.csv'))

# Group columns: mRNA vs Plasmid
mrna_cols = [c for c in wide.columns if re.search(r'mrna', c, re.I)]
plasmid_cols = [c for c in wide.columns if re.search(r'plasmid', c, re.I)]
print(f'Detected {len(mrna_cols)} mRNA columns, {len(plasmid_cols)} plasmid columns')

# Take average counts
wide['mRNA_mean'] = wide[mrna_cols].mean(axis=1)
wide['Plasmid_mean'] = wide[plasmid_cols].mean(axis=1)

# Calculate log2 fold-change
wide['activity'] = np.log2((wide['mRNA_mean'] + 1e-6) / (wide['Plasmid_mean'] + 1e-6))

# Output table: Sequence + activity
out_df = wide[['Sequence', 'activity']]
out_path = os.path.join(DATA_DIR, 'k562_sequence_activity.csv')
out_df.to_csv(out_path, index=False)

print(f"\nSaved: {out_path}")
print(out_df.head(5))
