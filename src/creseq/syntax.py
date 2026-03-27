import numpy as np
from typing import List, Optional, Tuple


IUPAC = {
    "A": {"A"},
    "C": {"C"},
    "G": {"G"},
    "T": {"T"},
    "W": {"A", "T"},
    "S": {"C", "G"},
    "M": {"A", "C"},
    "K": {"G", "T"},
    "R": {"A", "G"},
    "Y": {"C", "T"},
    "B": {"C", "G", "T"},
    "D": {"A", "G", "T"},
    "H": {"A", "C", "T"},
    "V": {"A", "C", "G"},
    "N": {"A", "C", "G", "T"},
}


def _iupac_match(subseq: str, motif: str) -> bool:
    if len(subseq) != len(motif):
        return False
    for base, code in zip(subseq, motif):
        if base not in IUPAC.get(code, {code}):
            return False
    return True


def _find_first_motif(
    seq: str,
    motifs: List[str],
    start: int,
    end: int,
) -> Optional[Tuple[int, str]]:
    start = max(0, start)
    end = min(len(seq), end)

    for p in range(start, end):
        for motif in motifs:
            mlen = len(motif)
            if p + mlen <= len(seq) and _iupac_match(seq[p:p + mlen], motif):
                return p, motif
    return None


def syntax_penalty(
    seqs: List[str],
    inr_window: Tuple[int, int] = (-2, 2),
    dpe_window: Tuple[int, int] = (28, 33),
    valid_distance: Tuple[int, int] = (28, 32),
    missing_motif_penalty: float = 1.0,
    bad_spacing_penalty: float = 1.0,
) -> np.ndarray:
    """
    Syntax penalty based on:
    1) presence of an Inr motif near TSS
    2) presence of a DPE motif downstream
    3) valid spacing between Inr and DPE

    TSS is approximated as the center of the sequence.
    """

    n = len(seqs)
    if n == 0:
        return np.array([], dtype=np.float32)

    penalties = np.zeros(n, dtype=np.float32)

    # Common simplified promoter grammar motifs
    inr_motifs = ["YYANWYY"]
    dpe_motifs = ["RGWYVT"]

    for i, s in enumerate(seqs):
        s = s.upper()
        L = len(s)
        tss = L // 2

        inr_start = tss + inr_window[0]
        inr_end = tss + inr_window[1] + 1

        dpe_start = tss + dpe_window[0]
        dpe_end = tss + dpe_window[1] + 1

        inr_match = _find_first_motif(s, inr_motifs, inr_start, inr_end)
        dpe_match = _find_first_motif(s, dpe_motifs, dpe_start, dpe_end)

        penalty = 0.0

        if inr_match is None:
            penalty += missing_motif_penalty
        if dpe_match is None:
            penalty += missing_motif_penalty

        if inr_match is not None and dpe_match is not None:
            inr_pos, _ = inr_match
            dpe_pos, _ = dpe_match
            dist = dpe_pos - inr_pos

            if not (valid_distance[0] <= dist <= valid_distance[1]):
                penalty += bad_spacing_penalty

        penalties[i] = penalty

    return penalties