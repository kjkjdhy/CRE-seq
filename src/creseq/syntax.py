from __future__ import annotations

import numpy as np
from typing import List, Dict, Tuple, Optional

# Mapping from bases to row indices in PWMs
_BASE2ROW = {"A": 0, "C": 1, "G": 2, "T": 3}

# Reverse-complement mapping
_RC = str.maketrans("ACGT", "TGCA")

# IUPAC → allowed bases
_IUPAC = {
    "A": ["A"],
    "C": ["C"],
    "G": ["G"],
    "T": ["T"],
    "R": ["A", "G"],
    "Y": ["C", "T"],
    "S": ["G", "C"],
    "W": ["A", "T"],
    "K": ["G", "T"],
    "M": ["A", "C"],
    "B": ["C", "G", "T"],
    "D": ["A", "G", "T"],
    "H": ["A", "C", "T"],
    "V": ["A", "C", "G"],
    "N": ["A", "C", "G", "T"],
}


def rc(seq: str) -> str:
    """Return reverse-complement of a DNA sequence."""
    return seq.translate(_RC)[::-1]


def pwm_from_consensus(consensus: str, pseudocount: float = 0.1) -> np.ndarray:
    """
    Build a PWM from an IUPAC consensus sequence.

    Supports all standard IUPAC DNA codes. For ambiguous characters, weight is
    distributed uniformly among the allowed bases.
    """
    consensus = consensus.upper()
    L = len(consensus)
    pwm = np.full((4, L), pseudocount, dtype=np.float32)

    for j, ch in enumerate(consensus):
        allowed = _IUPAC.get(ch)
        if allowed is None:
            raise ValueError(f"IUPAC character {ch!r} not supported")
        w = 1.0 / len(allowed)
        for b in allowed:
            pwm[_BASE2ROW[b], j] += w

    pwm /= pwm.sum(axis=0, keepdims=True)
    return pwm


def _log_odds_pwm(pwm: np.ndarray, bg: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Convert a PWM into log-odds scores relative to a background distribution.
    """
    if bg is None:
        bg = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)[:, None]
    eps = 1e-6
    return np.log((pwm + eps) / (bg + eps))


def _score_window(seq: str, lod: np.ndarray, i: int) -> float:
    """
    Score a window of length L starting at position i using a log-odds matrix.
    """
    L = lod.shape[1]
    s = 0.0
    for j in range(L):
        s += lod[_BASE2ROW[seq[i + j]], j]
    return float(s)


def scan_pwm_positions(
    seq: str,
    pwm: np.ndarray,
    min_frac_of_max: float = 0.85,
    both_strands: bool = True,
) -> List[int]:
    """
    Scan a single sequence with a PWM and return positions above a threshold.

    Positions are 0-based coordinates on the forward strand. For reverse-strand
    hits, we report the coordinate of the corresponding forward-strand window.
    """
    L = pwm.shape[1]
    lod = _log_odds_pwm(pwm)

    # Maximum possible LOD for this PWM (choose best base at each column)
    max_score = float(np.sum(np.max(lod, axis=0)))
    thr = max_score * min_frac_of_max

    hits: List[int] = []
    n = len(seq)
    rc_seq = rc(seq)

    for i in range(0, n - L + 1):
        sc_f = _score_window(seq, lod, i)
        if sc_f >= thr:
            hits.append(i)
            continue

        if both_strands:
            # Reverse-complement coordinate for the same genomic window
            sc_r = _score_window(rc_seq, lod, n - L - i)
            if sc_r >= thr:
                hits.append(i)

    return hits


def _helical_phase_penalty(
    d: int,
    period: float = 10.5,
    tol: float = 2.0,
) -> float:
    """
    Helical phasing penalty based on distance d between two motifs.

    period : helical period (bp), usually ~10.5.
    tol    : tolerance window (bp) around integer multiples of the period
             where penalty is zero.

    Returns
    -------
    float
        0 when d is within tol of k * period for some integer k; grows linearly
        once outside that window.
    """
    mod = d % period
    dist = min(mod, period - mod)
    if dist <= tol:
        return 0.0
    return float((dist - tol) / tol)


def pair_spacing_penalty_pwm(
    seqs: List[str],
    pwm_A: np.ndarray,
    pwm_B: np.ndarray,
    min_dist: int,
    max_dist: int,
    require_order: bool = False,
    min_frac_of_max: float = 0.85,
    both_strands: bool = True,
    use_helical_phase: bool = True,
    helical_period: float = 10.5,
    helical_tol: float = 2.0,
    miss_weight: float = 1.0,
    phase_weight: float = 1.0,
) -> np.ndarray:
    """
    Compute syntax penalties for a pair of PWMs (A, B) over a batch of sequences.

    Rules
    -----
    - If either A or B has no hit in a sequence: penalty = miss_weight.
    - Else, consider all motif-pair distances falling within [min_dist, max_dist].
      * If none are in this window: penalty = miss_weight.
      * If use_helical_phase:
            penalty = phase_weight * min(helical_phase_penalty(d)) over valid d.
        Otherwise:
            penalty = 0.
    """
    n = len(seqs)
    pen = np.zeros(n, dtype=np.float32)

    for i, s in enumerate(seqs):
        As = scan_pwm_positions(s, pwm_A, min_frac_of_max, both_strands)
        Bs = scan_pwm_positions(s, pwm_B, min_frac_of_max, both_strands)

        if len(As) == 0 or len(Bs) == 0:
            pen[i] = miss_weight
            continue

        valid_ds: List[int] = []
        for a in As:
            for b in Bs:
                d = (b - a) if require_order else abs(b - a)
                if min_dist <= d <= max_dist:
                    valid_ds.append(d)

        if not valid_ds:
            pen[i] = miss_weight
            continue

        if use_helical_phase:
            phase_pen = min(
                _helical_phase_penalty(d, helical_period, helical_tol)
                for d in valid_ds
            )
            pen[i] = phase_weight * float(phase_pen)
        else:
            pen[i] = 0.0

    return pen


def compute_syntax_penalty(
    seqs: List[str],
    rules: List[Dict],
    lambda_per_rule: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Compute total syntax penalty for a batch of sequences.

    Parameters
    ----------
    seqs : List[str]
        Input sequences.
    rules : List[dict]
        Each rule describes a motif pair and spacing / phasing constraints.
        Required / optional fields:
            "A", "B"            : consensus strings for motif A and B
            "min_dist", "max_dist"
            "require_order"     : if True, B must be downstream of A
            "min_frac"          : PWM score threshold as fraction of max
            "both"              : search both strands if True
            "helical"           : enable helical phasing term
            "period", "tol"     : helical period and tolerance
            "miss_w"            : weight when motifs are missing
            "phase_w"           : weight for helical term
            (optional) "pwmA", "pwmB" : precomputed PWMs
    lambda_per_rule : Optional[List[float]]
        Per-rule scaling factors. If None, all are set to 1.0.

    Returns
    -------
    np.ndarray, shape (n_seq,)
        Combined syntax penalty across all rules.
    """
    n = len(seqs)
    total = np.zeros(n, dtype=np.float32)

    if lambda_per_rule is None:
        lambda_per_rule = [1.0] * len(rules)

    cached_pwms: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}

    for lam, rule in zip(lambda_per_rule, rules):
        key = (rule.get("A", ""), rule.get("B", ""))

        if "pwmA" in rule and "pwmB" in rule:
            pwmA, pwmB = rule["pwmA"], rule["pwmB"]
        else:
            if key not in cached_pwms:
                pwmA = pwm_from_consensus(key[0])
                pwmB = pwm_from_consensus(key[1])
                cached_pwms[key] = (pwmA, pwmB)
            else:
                pwmA, pwmB = cached_pwms[key]

        pen = pair_spacing_penalty_pwm(
            seqs,
            pwmA,
            pwmB,
            min_dist=rule.get("min_dist", 6),
            max_dist=rule.get("max_dist", 30),
            require_order=rule.get("require_order", False),
            min_frac_of_max=rule.get("min_frac", 0.85),
            both_strands=rule.get("both", True),
            use_helical_phase=rule.get("helical", True),
            helical_period=rule.get("period", 10.5),
            helical_tol=rule.get("tol", 2.0),
            miss_weight=rule.get("miss_w", 1.0),
            phase_weight=rule.get("phase_w", 1.0),
        )

        total += lam * pen

    return total
