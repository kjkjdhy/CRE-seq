"""
Purpose:
  1) Based on k562_counts_wide.csv, check each sequence's coverage across groups (design+promoter):
       - Coverage condition: mRNA_mean > 0 and plasmid_mean > 0 for that group
  2) Only keep sequences with "coverage >= 1 group", recalculate activity (average log2FC across covered groups)
  3) Output training set k562_train.csv (two columns: sequence, activity)
"""

import os, re, numpy as np, pandas as pd

DATA_DIR = os.path.expanduser('~/CRE-seq/data/sharpr_mpra')
wide = pd.read_csv(os.path.join(DATA_DIR, 'k562_counts_wide.csv'))

# -------- Parse column group / type (consistent with previous script) --------
def parse_col(col):
    name = col
    typ = 'mRNA' if re.search(r'mrna', name, re.I) else ('Plasmid' if re.search(r'plasmid', name, re.I) else None)
    design = None
    if re.search(r'PilotDesign', name, re.I):
        design = 'PilotDesign'
    elif re.search(r'ScaleUpDesign1', name, re.I):
        design = 'ScaleUpDesign1'
    elif re.search(r'ScaleUpDesign2', name, re.I):
        design = 'ScaleUpDesign2'
    promoter = 'SV40P' if re.search(r'SV40P', name, re.I) else ('minP' if re.search(r'minP', name, re.I) else None)
    if design is None and 'Plasmid' in (typ or ''):
        if re.search(r'ScaleUpDesign1', name, re.I): design = 'ScaleUpDesign1'
        if re.search(r'ScaleUpDesign2', name, re.I): design = 'ScaleUpDesign2'
        if re.search(r'PilotDesign',   name, re.I): design = 'PilotDesign'
    group = None
    if design and promoter:
        group = f'{design}_{promoter}'
    elif design:
        group = f'{design}_SV40P'
    return group, typ

value_cols = [c for c in wide.columns if c not in ['ID','Sequence']]
meta = []
for c in value_cols:
    g, t = parse_col(c)
    if t is not None and g is not None:
        meta.append({'col': c, 'group': g, 'type': t})
meta = pd.DataFrame(meta)
groups = sorted(meta['group'].unique())

print("Groups:", groups)

# -------- Compute mRNA_mean / Plasmid_mean for each group --------
eps = 1e-6
mrna_means = {}
plasmid_means = {}
for g in groups:
    cols_m = meta[(meta['group']==g) & (meta['type']=='mRNA')]['col'].tolist()
    cols_p = meta[(meta['group']==g) & (meta['type']=='Plasmid')]['col'].tolist()
    if len(cols_m)==0 or len(cols_p)==0:
        continue
    mrna_means[g]    = wide[cols_m].mean(axis=1)
    plasmid_means[g] = wide[cols_p].mean(axis=1)

# -------- Coverage matrix: whether this row has valid counts in that group --------
coverage = {}
for g in groups:
    if g in mrna_means and g in plasmid_means:
        cov = (mrna_means[g] > 0) & (plasmid_means[g] > 0)
        coverage[g] = cov
coverage_df = pd.DataFrame(coverage)
covered_counts = coverage_df.sum(axis=1).fillna(0).astype(int)
print("Sample coverage statistics (number of samples with coverage >= 1 group):", int((covered_counts>=1).sum()))

# -------- Calculate activity only for samples with coverage >= 1 (average log2FC across covered groups) --------
log2fcs = []
for g in groups:
    if g in mrna_means and g in plasmid_means:
        log2fcs.append(np.log2((mrna_means[g] + eps) / (plasmid_means[g] + eps)))
log2fcs = pd.concat(log2fcs, axis=1) if log2fcs else pd.DataFrame(index=wide.index)

# Only include cells with coverage=True in the average, set others to NaN
for g in groups:
    if g in coverage_df.columns and g in log2fcs.columns:
        log2fcs.loc[~coverage_df[g], g] = np.nan

activity = log2fcs.mean(axis=1, skipna=True)

# -------- Generate training set --------
train = pd.DataFrame({'sequence': wide['Sequence'], 'activity': activity})
train = train.dropna().reset_index(drop=True)

out_path = os.path.join(DATA_DIR, 'k562_train.csv')
train.to_csv(out_path, index=False)

print(f"Saved training set: {out_path}")
print("Train size:", len(train))
print(train.head(5))
