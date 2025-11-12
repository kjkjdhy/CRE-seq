# 作用：检查我们得到的 k562_train.csv 里 activity 的分布，看看是不是“几乎全为 0”
import os, numpy as np, pandas as pd

DATA_DIR = os.path.expanduser('~/CRE-seq/data/sharpr_mpra')
df = pd.read_csv(os.path.join(DATA_DIR, 'k562_train.csv'))

# 基本统计
print("Train size:", len(df))
print(df['activity'].describe())

# 0值比例（允许一个很小阈值避免浮点误差）
thr = 1e-9
zeros = np.sum(np.isclose(df['activity'].values, 0.0, atol=thr))
print(f"zeros: {zeros} ({zeros/len(df):.3%})")

# 抽查明显非零的样本
nz = df[~np.isclose(df['activity'], 0.0, atol=1e-6)]
print("\nExamples |activity| > 1e-3:")
print(nz[np.abs(nz['activity']) > 1e-3].head(10))

# 如果全是0，提示下一步该怎么查
if zeros == len(df):
    print("\n>> All activities are ~0. Likely cause: per-group mRNA and plasmid means are nearly identical after our current tally.")
    print(">> Next step will inspect one group’s raw means (mRNA_mean vs Plasmid_mean) on a small subset.")
