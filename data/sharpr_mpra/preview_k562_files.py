# 目的：在 sharpr_mpra 目录里找到 K562 相关表格文件，
#       试读一个文件，打印列名 + 前几行，方便我们确定“sequence/activity”列。

import os, re, pandas as pd

DATA_DIR = os.path.expanduser('~/CRE-seq/data/sharpr_mpra')

# 1) 收集文件：名字里包含 k562，且是常见表格扩展名
files = [f for f in os.listdir(DATA_DIR)
         if re.search(r'k562', f, re.I) and f.endswith(('.txt', '.tsv', '.csv'))]
print(f'Found {len(files)} K562 files:')
for f in files[:12]:
    print('  -', f)

# 2) 依次尝试不同分隔符，读前5行做“探测”（不整表加载，快）
for f in files:
    path = os.path.join(DATA_DIR, f)
    for sep in ['\t', ',', ' ']:   # 制表符 / 逗号 / 空格
        try:
            df = pd.read_csv(path, sep=sep, nrows=5, comment='#')
            if df.shape[1] >= 2:
                print('\nPreview:', f, '| sep =', repr(sep))
                print('Columns:', list(df.columns))
                print(df.head(3))
                raise SystemExit  # 找到一个能读的样本就先停
        except Exception:
            pass

print("\nDidn't auto-preview. Tell me the filenames you see and we'll tune the reader.")
