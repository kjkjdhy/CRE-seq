"""
按设计-启动子成对归一化，计算每条序列的 activity：
  对每个 group ∈ {ScaleUpDesign1_minP, ScaleUpDesign1_SV40P, ScaleUpDesign2_minP, ScaleUpDesign2_SV40P, PilotDesign_SV40P}：
    activity_g = log2( mean(mRNA_reps_g) / mean(plasmid_reps_g) )
  最终 activity = mean_g activity_g  （忽略该组缺失时）
"""

import os, re, numpy as np, pandas as pd

DATA_DIR = os.path.expanduser('~/CRE-seq/data/sharpr_mpra')
wide = pd.read_csv(os.path.join(DATA_DIR, 'k562_counts_wide.csv'))

# 1) 识别所有列中的 group / 类型（mRNA or Plasmid）
def parse_col(col):
    name = col
    typ = 'mRNA' if re.search(r'mrna', name, re.I) else ('Plasmid' if re.search(r'plasmid', name, re.I) else None)
    # 捕捉设计/启动子标签
    design = None
    if re.search(r'PilotDesign', name, re.I):
        design = 'PilotDesign'
    elif re.search(r'ScaleUpDesign1', name, re.I):
        design = 'ScaleUpDesign1'
    elif re.search(r'ScaleUpDesign2', name, re.I):
        design = 'ScaleUpDesign2'

    promoter = 'SV40P' if re.search(r'SV40P', name, re.I) else ('minP' if re.search(r'minP', name, re.I) else None)

    if design is None and 'Plasmid' in (typ or ''):
        # 纯 plasmid 文件名里没有 K562，但有设计/启动子
        if re.search(r'ScaleUpDesign1', name, re.I): design = 'ScaleUpDesign1'
        if re.search(r'ScaleUpDesign2', name, re.I): design = 'ScaleUpDesign2'
        if re.search(r'PilotDesign',   name, re.I): design = 'PilotDesign'

    group = None
    if design and promoter:
        group = f'{design}_{promoter}'
    elif design:  # PilotDesign 只有 SV40P
        group = f'{design}_SV40P'
    return group, typ

value_cols = [c for c in wide.columns if c not in ['ID','Sequence']]
meta = []
for c in value_cols:
    g, t = parse_col(c)
    if t is not None and g is not None:
        meta.append({'col': c, 'group': g, 'type': t})
meta = pd.DataFrame(meta)
assert not meta.empty, "没有识别到任何 mRNA/Plasmid 列，请发我列名，我来调整规则。"

print("Detected groups:")
for g in sorted(meta['group'].unique()):
    sub = meta[meta['group']==g]
    print(f"  - {g}: mRNA={sum(sub['type']=='mRNA')}, Plasmid={sum(sub['type']=='Plasmid')}")

# 2) 对每个 group 求每条序列的 log2FC
eps = 1e-6
per_group_acts = []
for g in sorted(meta['group'].unique()):
    cols_m = meta[(meta['group']==g) & (meta['type']=='mRNA')]['col'].tolist()
    cols_p = meta[(meta['group']==g) & (meta['type']=='Plasmid')]['col'].tolist()
    if len(cols_m)==0 or len(cols_p)==0:
        continue
    mrna_mean = wide[cols_m].mean(axis=1)
    plasmid_mean = wide[cols_p].mean(axis=1)
    act_g = np.log2((mrna_mean + eps) / (plasmid_mean + eps))
    per_group_acts.append(act_g.rename(g))

# 3) 跨组取平均（忽略缺失）
acts = pd.concat(per_group_acts, axis=1)
wide['activity'] = acts.mean(axis=1, skipna=True)

# 4) 导出 Sequence + activity
out = wide[['Sequence','activity']].copy()
out_path = os.path.join(DATA_DIR, 'k562_sequence_activity.csv')
out.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
print(out.head(5))
