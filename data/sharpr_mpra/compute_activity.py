"""
目标：
  从 k562_counts_wide.csv 计算每条序列的归一化活性（activity）
  步骤：
    1) 把 mRNA 与对应的 Plasmid 计数相除 -> ratio
    2) 取 log2(ratio + 1e-6) 防止除零
    3) 若有多组 replicate，取平均
    4) 保存 k562_sequence_activity.csv，包含 Sequence + activity
"""

import os, re, numpy as np, pandas as pd

DATA_DIR = os.path.expanduser('~/CRE-seq/data/sharpr_mpra')
wide = pd.read_csv(os.path.join(DATA_DIR, 'k562_counts_wide.csv'))

# 把列分组：mRNA vs Plasmid
mrna_cols = [c for c in wide.columns if re.search(r'mrna', c, re.I)]
plasmid_cols = [c for c in wide.columns if re.search(r'plasmid', c, re.I)]
print(f'检测到 {len(mrna_cols)} 个 mRNA 列, {len(plasmid_cols)} 个 plasmid 列')

# 取平均计数
wide['mRNA_mean'] = wide[mrna_cols].mean(axis=1)
wide['Plasmid_mean'] = wide[plasmid_cols].mean(axis=1)

# 计算 log2 fold-change
wide['activity'] = np.log2((wide['mRNA_mean'] + 1e-6) / (wide['Plasmid_mean'] + 1e-6))

# 输出表：Sequence + activity
out_df = wide[['Sequence', 'activity']]
out_path = os.path.join(DATA_DIR, 'k562_sequence_activity.csv')
out_df.to_csv(out_path, index=False)

print(f"\nSaved: {out_path}")
print(out_df.head(5))
