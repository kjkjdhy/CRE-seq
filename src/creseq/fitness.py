import numpy as np


def compute_fitness(
    seqs,
    scorer,
    motif_pen,
    shape_pen,
    lambda_motif: float = 0.5,
    lambda_shape: float = 0.5,
) -> np.ndarray:
    """
    Compute final fitness for a batch of sequences.

    Base score comes from the external scorer (e.g. PARM / DeepSTARR).
    We subtract:
      - lambda_motif * motif_pen
      - lambda_shape * shape_pen
    so that larger penalties reduce the fitness.
    """
    # Base scores from ML model
    s = scorer.score_batch(seqs).astype(np.float32)

    # Subtract motif and DNAshape penalties
    s = s - lambda_motif * motif_pen - lambda_shape * shape_pen

    return s
