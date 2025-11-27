import numpy as np
from typing import List, Dict

# ---------------------------------------------------------------------
# Simple, self-contained DNAshape surrogate (Option B)
# ---------------------------------------------------------------------
# This module does NOT depend on DNAshapeR or any external R package.
# Instead, it produces useful "shape-like" structural proxies from raw DNA
# sequences, using stable rule-based heuristics:
#
#   1) MGW surrogate:
#        - computed from local GC fraction in sliding windows
#        - preferred when GC/AT balance is moderate (GC fraction ~0.5)
#
#   2) Roll surrogate:
#        - derived from dinucleotide-dependent bending tendencies
#        - mapped to [0,1] and converted to base-level Roll profile
#
#   3) Homopolymer detection:
#        - penalizes long AAAAAA / TTTTTT / CCCCC sequences
#
# The returned values are lightweight summaries:
#       mgw_mean : float per sequence
#       roll_var : float per sequence
#       homorun  : float per sequence
#
# These are sufficient for downstream fitness functions and fully compatible
# with your existing GA pipeline.
# ---------------------------------------------------------------------

# Dinucleotide-level inherent Roll angles (approximate). Only the RELATIVE
# differences matter because values are normalized afterward.
_ROLL_DEG = {
    "AA": 0.6, "TT": 0.6,
    "AT": -1.2,
    "TA": 3.0,
    "CA": 1.7, "TG": 1.7,
    "GT": 0.6, "AC": 0.6,
    "CT": 1.1, "AG": 1.1,
    "GA": -0.6, "TC": -0.6,
    "CG": 0.6,
    "GC": -1.5,
    "GG": 0.6, "CC": 0.6,
}

_DINUC_MIN = min(_ROLL_DEG.values())
_DINUC_MAX = max(_ROLL_DEG.values())
_DINUC_SPAN = float(_DINUC_MAX - _DINUC_MIN) if _DINUC_MAX != _DINUC_MIN else 1.0


def _sliding_gc_fraction(s: str, w: int = 5) -> np.ndarray:
    """
    Compute GC fraction in sliding windows of size w.

    Returns
    -------
    np.ndarray of shape (len(s) - w + 1,)
        GC fraction for each window. If the sequence is shorter than w,
        a single averaged GC fraction is returned.
    """
    L = len(s)
    arr = np.fromiter((1 if c in "GC" else 0 for c in s), dtype=np.float32)

    if L < w:
        return np.array([arr.mean()], dtype=np.float32)

    kernel = np.ones(w, dtype=np.float32)
    return np.convolve(arr, kernel, mode="valid") / float(w)


def _max_homopolymer_run(s: str) -> float:
    """
    Compute the longest homopolymer run length (e.g. 'AAAAA' → 5).
    """
    if not s:
        return 0.0

    longest = 1
    current = 1

    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            current += 1
            longest = max(longest, current)
        else:
            current = 1

    return float(longest)


def _roll_profile(s: str) -> np.ndarray:
    """
    Convert a DNA sequence into a base-level Roll profile in the range [0,1].

    Steps
    -----
    1) Assign a Roll magnitude to each dinucleotide (length L-1)
    2) Normalize all step-level values to [0,1]
    3) Convert step-level to base-level (length L)
       - endpoints copy nearest step
       - interior bases use average of adjacent steps
    """
    L = len(s)
    if L <= 1:
        return np.zeros(L, dtype=np.float32)

    # Step-level roll values
    step_vals = np.empty(L - 1, dtype=np.float32)
    for i in range(L - 1):
        dinuc = s[i : i + 2]
        step_vals[i] = _ROLL_DEG.get(dinuc, 0.0)

    # Normalize to [0,1]
    norm = (step_vals - _DINUC_MIN) / _DINUC_SPAN
    norm = np.clip(norm, 0.0, 1.0)

    # Convert to base-level profile
    base_roll = np.empty(L, dtype=np.float32)
    base_roll[0] = norm[0]
    base_roll[-1] = norm[-1]
    if L > 2:
        base_roll[1:-1] = 0.5 * (norm[:-1] + norm[1:])

    return base_roll


def compute_dnashape(seqs: List[str]) -> Dict[str, np.ndarray]:
    """
    Compute shape-like structural summaries for a batch of sequences.

    Parameters
    ----------
    seqs : List[str]
        All sequences must have identical length and contain only A/C/G/T.

    Returns
    -------
    dict with keys "mgw_mean", "roll_var", "homorun" (each np.ndarray)
        mgw_mean : mean GC-balance-based MGW surrogate, ∈ [0,1]
        roll_var : variance of normalized Roll profile, small values preferred
        homorun  : length of longest homopolymer run
    """
    n = len(seqs)
    if n == 0:
        return {
            "mgw_mean": np.zeros(0, dtype=np.float32),
            "roll_var": np.zeros(0, dtype=np.float32),
            "homorun": np.zeros(0, dtype=np.float32),
        }

    L = len(seqs[0])
    mgw_mean = np.zeros(n, dtype=np.float32)
    roll_var = np.zeros(n, dtype=np.float32)
    homorun = np.zeros(n, dtype=np.float32)

    for i, s in enumerate(seqs):
        s = s.upper()
        assert len(s) == L, "All sequences must have the same length"

        # MGW surrogate based on local GC balance
        gc_local = _sliding_gc_fraction(s, w=5)
        # Ideal GC fraction ~0.5 → score=1; deviation reduces the score.
        mgw_local = 1.0 - np.abs(gc_local - 0.5) * 2.0
        mgw_local = np.clip(mgw_local, 0.0, 1.0)
        mgw_mean[i] = float(mgw_local.mean())

        # Roll profile variance
        roll = _roll_profile(s)
        roll_var[i] = float(roll.var())

        # Longest homopolymer run
        homorun[i] = _max_homopolymer_run(s)

    return {
        "mgw_mean": mgw_mean,
        "roll_var": roll_var,
        "homorun": homorun,
    }


def shape_penalty(
    shape: Dict[str, np.ndarray],
    mgw_target: float = 1.0,
    roll_var_max: float = 0.02,
    homorun_max: float = 6.0,
) -> np.ndarray:
    """
    Convert DNAshape structural summaries into a per-sequence penalty.

    Parameters
    ----------
    shape : dict
        Must contain the keys "mgw_mean", "roll_var", "homorun".
    mgw_target : float
        Target MGW score; lower MGW gets penalized.
    roll_var_max : float
        Maximum allowed variance of Roll; higher variance is penalized.
    homorun_max : float
        Maximum allowed homopolymer run length.

    Returns
    -------
    np.ndarray of shape (n_seq,)
        Larger values indicate worse structure (higher penalty).
    """
    mgw = shape.get("mgw_mean")
    rv = shape.get("roll_var")
    hr = shape.get("homorun")

    if mgw is None or rv is None or hr is None:
        raise ValueError("shape_penalty expects keys 'mgw_mean', 'roll_var', 'homorun'")

    # MGW penalty: only penalize if below target
    p_mgw = np.maximum(0.0, mgw_target - mgw)

    # Roll variance penalty: only penalize when above threshold
    p_roll = np.maximum(0.0, rv - roll_var_max)

    # Homopolymer penalty
    p_homo = np.maximum(0.0, hr - homorun_max)

    return (p_mgw + p_roll + p_homo).astype(np.float32)
