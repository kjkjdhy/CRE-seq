import pandas as pd


IN_PATH = "scripts/CODA_all_sequences.txt"      
OUT_PATH = "CODA_k562_subset.tsv"       


USECOLS = [
    "sequence",
    "origin",
    "target_cell",
    "K562_l2fc",
    "K562_prediction",
]


chunks = pd.read_table(
    IN_PATH,
    sep="\t",         
    comment="#",
    chunksize=50000,  
)

selected_chunks = []

for chunk in chunks:
  
    mask = chunk["target_cell"].str.contains("k562", case=False, na=False)
    sub = chunk.loc[mask, USECOLS]
    selected_chunks.append(sub)

if selected_chunks:
    out = pd.concat(selected_chunks, ignore_index=True)
    out.to_csv(OUT_PATH, sep="\t", index=False)
    print(f"Wrote {out.shape[0]} rows to {OUT_PATH}")
else:
    print("No rows with target_cell containing 'k562' were found.")
