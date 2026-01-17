# src/creseq/score_adapter.py append / overwrite
from __future__ import annotations
import numpy as np
import re
import subprocess, tempfile, os
from typing import List

class ParmScorer:
    """Use PARM CLI ('parm predict') as a black-box scorer."""
    def __init__(self, model_dir: str, exe: str = "parm"):
        # model_dir example: "/Users/heyangdong/software/PARM/pre_trained_models/K562/"
        self.model_dir = model_dir
        self.exe = exe

    def score_batch(self, seqs: List[str]) -> np.ndarray:
        with tempfile.TemporaryDirectory() as tmpdir:
            fasta_path = os.path.join(tmpdir, "input.fasta")
            out_path   = os.path.join(tmpdir, "out.txt")

            # Write temporary FASTA
            with open(fasta_path, "w") as f:
                for i, s in enumerate(seqs):
                    f.write(f">seq{i}\n{s}\n")

            # Call PARM
            cmd = [
                self.exe, "predict",
                "--input", fasta_path,
                "--output", out_path,
                "--model", self.model_dir,
            ]
            subprocess.run(cmd, check=True)

            # Read back scores (last column)
            scores = []
            with open(out_path) as f:
                header = f.readline().rstrip("\n").split("\t")
                score_idx = len(header) - 1
                for line in f:
                    if not line.strip():
                        continue
                    parts = line.rstrip("\n").split("\t")
                    scores.append(float(parts[score_idx]))

        return np.array(scores, dtype=float)
