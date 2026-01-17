"""
Utility script used to score CODA sequences with PARM-K562
for activity scale reference in the manuscript.
Not part of the CRE-seq generation pipeline.
"""

# scripts/score_coda_k562.py
from __future__ import annotations
import pathlib

import numpy as np
import pandas as pd
import os
from pathlib import Path
from creseq.score_adapter import ParmScorer


def main() -> None:
    # Input / output paths
    in_path = pathlib.Path("CODA_k562_subset.tsv")
    out_path = pathlib.Path("CODA_k562_with_parm.tsv")

    print(f"Loading CODA K562 subset from: {in_path}")
    df = pd.read_table(in_path, sep="\t")

    # Basic sanity checks
    if "sequence" not in df.columns:
        raise ValueError("Input file must contain a 'sequence' column.")

    seqs = df["sequence"].astype(str).str.upper().tolist()
    if len(seqs) == 0:
        raise ValueError("No sequences found in input file.")

    Ls = {len(s) for s in seqs}
    if len(Ls) != 1:
        raise ValueError(f"Sequences have different lengths: {sorted(Ls)}")

    print(f"Scoring {len(seqs)} sequences of length {Ls.pop()} with PARM (K562)...")

    parm_model_dir = os.environ.get("PARM_MODEL_DIR", None)
if parm_model_dir is None:
    raise ValueError(
        "PARM_MODEL_DIR not set. "
        "Set environment variable PARM_MODEL_DIR to PARM model path."
    )

parm_model_dir = Path(parm_model_dir).expanduser().resolve()
if not parm_model_dir.exists():
    raise FileNotFoundError(f"PARM model dir not found: {parm_model_dir}")

scorer = ParmScorer(model_dir=str(parm_model_dir))


scores = scorer.score_batch(seqs)
scores = np.asarray(scores, dtype=np.float32)

if scores.shape[0] != len(seqs):
        raise RuntimeError(
            f"Expected {len(seqs)} scores, got shape {scores.shape}"
        )

        df["parm_k562_score"] = scores

print(
        f"PARM K562 score: min={scores.min():.4f}, "
        f"median={np.median(scores):.4f}, max={scores.max():.4f}"
    )

df.to_csv(out_path, sep="\t", index=False)
print(f"Wrote scored CODA K562 sequences to: {out_path}")

if __name__ == "__main__":
    main()
