import numpy as np
def compute_fitness(seqs, scorer, motif_pen, shape_pen, lambda_motif=0.5, lambda_shape=0.5):
    s=scorer.score_batch(seqs).astype(np.float32)
    return s - lambda_motif*motif_pen - lambda_shape*shape_pen
