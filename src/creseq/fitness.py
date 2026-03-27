import numpy as np

from creseq.motif import motif_penalty
from creseq.syntax import syntax_penalty
from creseq.shape import shape_penalty


def compute_fitness(
    seqs,
    scorer,
    lambda_motif=1.0,
    lambda_syntax=1.0,
    lambda_shape=0.0,
):
    scores = scorer.score_batch(seqs).astype(np.float32)

    pen_motif = motif_penalty(seqs).astype(np.float32)
    pen_syntax = syntax_penalty(seqs).astype(np.float32)
    pen_shape = shape_penalty(seqs).astype(np.float32)

    return (
        scores
        - lambda_motif * pen_motif
        - lambda_syntax * pen_syntax
        - lambda_shape * pen_shape
    )