import numpy as np
from creseq.motif import motif_penalty
from creseq.syntax import syntax_penalty

def compute_fitness(seqs, scorer, lambda_motif=1.0, lambda_syntax=1.0):
    scores = scorer.score_batch(seqs).astype(np.float32)
    pen_motif = motif_penalty(seqs)
    pen_syntax = syntax_penalty(seqs)
    return scores - lambda_motif * pen_motif - lambda_syntax * pen_syntax
