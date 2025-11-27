# src/creseq/score_adapter.py append / overwrite
from __future__ import annotations
import numpy as np
import re

BASE_TO_IDX = {"A":0,"C":1,"G":2,"T":3}

def one_hot_encode(seqs: list[str]) -> np.ndarray:
    L = len(seqs[0])
    X = np.zeros((len(seqs), L, 4), dtype=np.float32)
    for i,s in enumerate(seqs):
        assert len(s)==L, "All sequences must have the same length"
        for j,ch in enumerate(s):
            X[i,j,BASE_TO_IDX[ch]] = 1.0
    return X

class DeepSTARRScorer:
    """
    Universal SavedModel loader:
    - Prioritizes using serving_default signature; if not available, directly calls the callable object
    - Only imports tensorflow when you pass --deepstarr
    """
    def __init__(self, savedmodel_dir: str):
        try:
            import tensorflow as tf
        except Exception as e:
            raise ImportError(
                "TensorFlow is required for DeepSTARRScorer. "
                "Install with: python -m pip install tensorflow"
            ) from e
        self.tf = tf
        self.model = tf.saved_model.load(savedmodel_dir)
        self.fn = None
        # Get signature (compatible with different export methods)
        if hasattr(self.model, "signatures") and self.model.signatures:
            self.fn = self.model.signatures.get("serving_default", None)

    def score_batch(self, seqs: list[str]) -> np.ndarray:
        tf = self.tf
        X = one_hot_encode(seqs)
        X = tf.constant(X)
        if self.fn is not None:
            out = self.fn(X)
            # Get the first output tensor
            y = next(iter(out.values())).numpy()
        else:
            # Directly callable (some SavedModel exports work this way)
            y = self.model(X).numpy()
        return y.squeeze().astype(np.float32)

# Keep existing DummyScorer
class DummyScorer:
    def score_batch(self, seqs: list[str]) -> np.ndarray:
        L = len(seqs[0])
        gcs = np.array([(s.count("G")+s.count("C"))/L for s in seqs], dtype=np.float32)
        return 1.0 - np.abs(gcs - 0.5)



class HumanKnowledgeScorer:
    """
    Simple human knowledge-based sequence scorer:
      - GC content optimally in 40-60% range
      - More common human "good" motifs are better (CRE-like, ETS, SP1-like, NF-κB-like)
      - Penalizes long poly-A/T runs and complete absence of CpG
    Returns values approximately in 0~1 range, higher is better.
    """

    def __init__(self,
                 target_gc_low: float = 0.40,
                 target_gc_high: float = 0.60,
                 good_motifs: list[str] | None = None,
                 bad_run_len: int = 6):
        self.target_gc_low = target_gc_low
        self.target_gc_high = target_gc_high
        # Some approximate consensus motifs common in human enhancers (very rough, but sufficient for V1)
        if good_motifs is None:
            good_motifs = [
                "TGACGTCA",  # CREB / AP-1 like
                "GGAA",      # ETS core
                "GGGCGG",    # SP1-like GC box
                "GCGCGC",    # CpG-rich
                "GGGGA",     # NF-κB-like (极粗略)
            ]
        self.good_motifs = [m.upper() for m in good_motifs]
        self.bad_run_len = bad_run_len

    def _gc_score(self, s: str) -> float:
        L = len(s)
        gc = (s.count("G") + s.count("C")) / L
        center = 0.5 * (self.target_gc_low + self.target_gc_high)
        half_span = 0.5 * (self.target_gc_high - self.target_gc_low)
        if half_span <= 0:
            return 0.0
        # Greater deviation from center means larger penalty, beyond range becomes 0
        score = 1.0 - abs(gc - center) / half_span
        return max(0.0, min(1.0, score))

    def _motif_score(self, s: str) -> float:
        L = len(s)
        total_hits = 0
        for m in self.good_motifs:
            k = len(m)
            for i in range(L - k + 1):
                if s[i:i+k] == m:
                    total_hits += 1
        # More hits means score closer to 1, even a few hits provide significant improvement
        if total_hits == 0:
            return 0.0
        return float(1.0 - np.exp(-total_hits / 3.0))

    def _bad_penalty(self, s: str) -> float:
        # Penalty for long poly-A/T runs
        bad = 0.0
        for base in ("A", "T"):
            if re.search(base * self.bad_run_len, s):
                bad += 1.0
        # Also penalize complete absence of CpG
        if "CG" not in s:
            bad += 0.5
        # Larger bad value means smaller coefficient multiplied to the previous score
        return float(np.exp(-bad))

    def score_batch(self, seqs: list[str]) -> np.ndarray:
        if not seqs:
            return np.zeros(0, dtype=np.float32)
        scores = []
        for s in seqs:
            s = s.upper()
            gc = self._gc_score(s)
            mot = self._motif_score(s)
            bad_factor = self._bad_penalty(s)
            # 0.5 * GC + 0.5 * motif, then multiply by bad_factor
            base_score = 0.5 * gc + 0.5 * mot
            final = base_score * bad_factor
            # Prevent extreme cases from going out of bounds
            final = max(0.0, min(1.0, final))
            scores.append(final)
        return np.array(scores, dtype=np.float32)













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
