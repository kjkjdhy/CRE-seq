"""
目标：
  1) 扫描当前目录下所有 K562 的 *.counts.txt 文件（mRNA/Plasmid、各个replicate）
  2) 读取三列：ID、Sequence、Counts
  3) 把 'Counts' 这一列里逗号分隔的条形码计数求和 => total_counts
  4) 汇总成一个“宽表”：每行是一条序列，每列是一个文件（一个replicate），单元格是该序列在该库的总计数
  5) 保存为 k562_counts_wide.csv，后续我们再从中计算 activity（例如 RNA / Plasmid 归一化、取 log2 等）
"""

import os, re, pandas as pd

DATA_DIR = os.path.expanduser('~/CRE-seq/data/sharpr_mpra')

def list_count_files():
    """
    同时收集：
      - 含 K562 的 *.counts.txt（mRNA 计数）
      - 含 Plasmid 的 *.counts.txt（质粒对照），即使文件名里不含 K562
    """
    files = []
    for fn in os.listdir(DATA_DIR):
        if fn.endswith('.counts.txt') and (re.search(r'k562', fn, re.I) or re.search(r'plasmid', fn, re.I)):
            files.append(os.path.join(DATA_DIR, fn))
    files.sort()
    return files



def short_name(path):
    """
    压缩成好读的列名；确保能看出是 mRNA 还是 Plasmid、哪个设计/rep。
    例：
      GSM1831779_K562_PilotDesign_SV40P_mRNA_Rep1.counts.txt -> K562_PilotDesign_SV40P_mRNA_Rep1
      GSM1831773_ScaleUpDesign1_minP_Plasmid.counts.txt      -> ScaleUpDesign1_minP_Plasmid
    """
    base = os.path.basename(path).replace('.counts.txt', '')
    base = re.sub(r'^GSM\d+_', '', base)  # 去掉 GSM 编号
    parts = [p for p in base.split('_') if p]

    # 只保留关键信息的顺序
    keep_order = []
    keys = ['K562','PilotDesign','ScaleUpDesign1','ScaleUpDesign2','SV40P','minP','mRNA','Plasmid','Rep1','Rep2','Rep3']
    for k in keys:
        for p in parts:
            if p.lower() == k.lower():
                keep_order.append(k)
                break

    # 如果既没有 K562 也没有细胞名，就保持原样的前两个词，避免信息丢失
    if not any(k in keep_order for k in ['K562']):
        # 例如纯 Plasmid 文件：用设计名+启动子名做前缀
        prefix = []
        for p in parts:
            if p not in ['mRNA','Plasmid','Rep1','Rep2','Rep3']:
                prefix.append(p)
            if len(prefix) >= 2:
                break
        keep_order = prefix + [x for x in keep_order if x not in prefix]

    name = '_'.join(dict.fromkeys(keep_order))  # 去重保持顺序
    return name if name else base




def sum_counts_str(s):
    """
    把 '154,3,188,...' 这类字符串转成整数列表并求和。
    空字符串或异常返回 0。
    """
    if not isinstance(s, str) or s.strip() == '':
        return 0
    try:
        return sum(int(x) for x in s.split(',') if x != '')
    except Exception:
        return 0

def load_one(path):
    """读取一个 counts 文件，只取需要的列，并计算 total_counts"""
    # 大多数是制表符分隔
    df = pd.read_csv(path, sep='\t', comment='#')
    # 兼容不同大小写的列名
    cols = {c.lower(): c for c in df.columns}
    id_col = cols.get('id', 'ID')
    seq_col = cols.get('sequence', 'Sequence')
    cnt_col = cols.get('counts', 'Counts')
    df = df[[id_col, seq_col, cnt_col]].rename(columns={id_col:'ID', seq_col:'Sequence', cnt_col:'Counts'})
    df['total_counts'] = df['Counts'].apply(sum_counts_str)
    df['file'] = short_name(path)
    return df[['ID','Sequence','file','total_counts']]

def build_wide_table(paths):
    """把多个文件拼起来，透视成宽表"""
    frames = [load_one(p) for p in paths]
    tall = pd.concat(frames, ignore_index=True)
    # 行去重（有些文件可能同一 ID 出现一次以上，取和）
    tall = tall.groupby(['ID','Sequence','file'], as_index=False)['total_counts'].sum()
    # 透视：index=ID+Sequence，columns=file，values=total_counts
    wide = tall.pivot_table(index=['ID','Sequence'], columns='file', values='total_counts', fill_value=0, aggfunc='sum')
    wide = wide.reset_index()
    # 列名整理：去掉列索引名字
    wide.columns.name = None
    return wide

def main():
    paths = list_count_files()
    print(f'Found {len(paths)} K562 count files.')
    for p in paths:
        print('  -', os.path.basename(p))
    wide = build_wide_table(paths)
    out_csv = os.path.join(DATA_DIR, 'k562_counts_wide.csv')
    wide.to_csv(out_csv, index=False)
    print(f'\nSaved wide table: {out_csv}')
    print('Shape:', wide.shape)
    print('Columns (first 10):', list(wide.columns)[:10])
    print('\nHead:')
    print(wide.head(3))

if __name__ == '__main__':
    main()
