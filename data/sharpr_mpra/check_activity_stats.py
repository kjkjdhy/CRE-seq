# Purpose: Check the distribution of activity in k562_train.csv to see if it's "almost all 0"
import os, numpy as np, pandas as pd

DATA_DIR = os.path.expanduser('~/CRE-seq/data/sharpr_mpra')
df = pd.read_csv(os.path.join(DATA_DIR, 'k562_train.csv'))

# Basic statistics
print("Train size:", len(df))
print(df['activity'].describe())

# Proportion of zeros (allow a small threshold to avoid floating point errors)
thr = 1e-9
zeros = np.sum(np.isclose(df['activity'].values, 0.0, atol=thr))
print(f"zeros: {zeros} ({zeros/len(df):.3%})")

# Sample obviously non-zero samples
nz = df[~np.isclose(df['activity'], 0.0, atol=1e-6)]
print("\nExamples |activity| > 1e-3:")
print(nz[np.abs(nz['activity']) > 1e-3].head(10))

# If all are 0, suggest next debugging step
if zeros == len(df):
    print("\n>> All activities are ~0. Likely cause: per-group mRNA and plasmid means are nearly identical after our current tally.")
    print(">> Next step will inspect one group's raw means (mRNA_mean vs Plasmid_mean) on a small subset.")
