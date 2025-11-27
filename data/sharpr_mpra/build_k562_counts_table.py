"""
Goal:
  1) Scan all K562 *.counts.txt files in the current directory (mRNA/Plasmid, each replicate)
  2) Read three columns: ID, Sequence, Counts
  3) Sum the comma-separated barcode counts in the 'Counts' column => total_counts
  4) Aggregate into a "wide table": each row is a sequence, each column is a file (one replicate), cells are the total counts for that sequence in that library
  5) Save as k562_counts_wide.csv, then we'll calculate activity from it (e.g., RNA / Plasmid normalization, log2, etc.)
"""

import os, re, pandas as pd

DATA_DIR = os.path.expanduser('~/CRE-seq/data/sharpr_mpra')

def list_count_files():
    """
    Collect simultaneously:
      - *.counts.txt containing K562 (mRNA counts)
      - *.counts.txt containing Plasmid (plasmid controls), even if filename doesn't contain K562
    """
    files = []
    for fn in os.listdir(DATA_DIR):
        if fn.endswith('.counts.txt') and (re.search(r'k562', fn, re.I) or re.search(r'plasmid', fn, re.I)):
            files.append(os.path.join(DATA_DIR, fn))
    files.sort()
    return files



def short_name(path):
    """
    Compress into readable column names; ensure you can see whether it's mRNA or Plasmid, which design/rep.
    Examples:
      GSM1831779_K562_PilotDesign_SV40P_mRNA_Rep1.counts.txt -> K562_PilotDesign_SV40P_mRNA_Rep1
      GSM1831773_ScaleUpDesign1_minP_Plasmid.counts.txt      -> ScaleUpDesign1_minP_Plasmid
    """
    base = os.path.basename(path).replace('.counts.txt', '')
    base = re.sub(r'^GSM\d+_', '', base)  # Remove GSM number
    parts = [p for p in base.split('_') if p]

    # Keep only key information in order
    keep_order = []
    keys = ['K562','PilotDesign','ScaleUpDesign1','ScaleUpDesign2','SV40P','minP','mRNA','Plasmid','Rep1','Rep2','Rep3']
    for k in keys:
        for p in parts:
            if p.lower() == k.lower():
                keep_order.append(k)
                break

    # If there's neither K562 nor cell name, keep the first two words as-is to avoid information loss
    if not any(k in keep_order for k in ['K562']):
        # For example, pure Plasmid files: use design name + promoter name as prefix
        prefix = []
        for p in parts:
            if p not in ['mRNA','Plasmid','Rep1','Rep2','Rep3']:
                prefix.append(p)
            if len(prefix) >= 2:
                break
        keep_order = prefix + [x for x in keep_order if x not in prefix]

    name = '_'.join(dict.fromkeys(keep_order))  # Deduplicate while maintaining order
    return name if name else base




def sum_counts_str(s):
    """
    Convert strings like '154,3,188,...' to integer list and sum.
    Return 0 for empty strings or exceptions.
    """
    if not isinstance(s, str) or s.strip() == '':
        return 0
    try:
        return sum(int(x) for x in s.split(',') if x != '')
    except Exception:
        return 0

def load_one(path):
    """Read a counts file, take only needed columns, and calculate total_counts"""
    # Most are tab-separated
    df = pd.read_csv(path, sep='\t', comment='#')
    # Compatible with different case column names
    cols = {c.lower(): c for c in df.columns}
    id_col = cols.get('id', 'ID')
    seq_col = cols.get('sequence', 'Sequence')
    cnt_col = cols.get('counts', 'Counts')
    df = df[[id_col, seq_col, cnt_col]].rename(columns={id_col:'ID', seq_col:'Sequence', cnt_col:'Counts'})
    df['total_counts'] = df['Counts'].apply(sum_counts_str)
    df['file'] = short_name(path)
    return df[['ID','Sequence','file','total_counts']]

def build_wide_table(paths):
    """Combine multiple files and pivot into wide table"""
    frames = [load_one(p) for p in paths]
    tall = pd.concat(frames, ignore_index=True)
    # Deduplicate rows (some files may have the same ID appear multiple times, sum them)
    tall = tall.groupby(['ID','Sequence','file'], as_index=False)['total_counts'].sum()
    # Pivot: index=ID+Sequence, columns=file, values=total_counts
    wide = tall.pivot_table(index=['ID','Sequence'], columns='file', values='total_counts', fill_value=0, aggfunc='sum')
    wide = wide.reset_index()
    # Clean up column names: remove column index name
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
