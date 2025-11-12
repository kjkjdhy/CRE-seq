from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple, Optional

_BASE2ROW = {"A":0, "C":1, "G":2, "T":3}
_RC = str.maketrans("ACGT", "TGCA")

def rc(seq: str) -> str:
    return seq.translate(_RC)[::-1]

def pwm_from_consensus(consensus: str, pseudocount: float = 0.1) -> np.ndarray:
    consensus = consensus.upper()
    L = len(consensus)
    pwm = np.full((4, L), pseudocount, dtype=np.float32)
    for j, ch in enumerate(consensus):
        pwm[_BASE2ROW[ch], j] += 1.0
    pwm /= pwm.sum(axis=0, keepdims=True)
    return pwm

def _log_odds_pwm(pwm: np.ndarray, bg: Optional[np.ndarray] = None) -> np.ndarray:
    if bg is None:
        bg = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)[:, None]
    eps = 1e-6
    return np.log((pwm + eps) / (bg + eps))

def _score_window(seq: str, lod: np.ndarray, i: int) -> float:
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
    L = pwm.shape[1]
    lod = _log_odds_pwm(pwm)  # 4 x L
    max_score = float(np.sum(np.max(lod, axis=0)))
    thr = max_score * min_frac_of_max

    hits = []
    n = len(seq)
    rc_seq = rc(seq)
    for i in range(0, n - L + 1):
        sc_f = _score_window(seq, lod, i)
        if sc_f >= thr:
            hits.append(i)
            continue
        if both_strands:
            sc_r = _score_window(rc_seq, lod, n - L - i)  # 等效RC坐标
            if sc_r >= thr:
                hits.append(i)
    return hits

def _helical_phase_penalty(d: int, period: float = 10.5, tol: float = 2.0) -> float:
    mod = d % period
    dist = min(mod, period - mod)   # 距最近 k*period 的距离
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
    n = len(seqs)
    pen = np.zeros(n, dtype=np.float32)

    for i, s in enumerate(seqs):
        As = scan_pwm_positions(s, pwm_A, min_frac_of_max, both_strands)
        Bs = scan_pwm_positions(s, pwm_B, min_frac_of_max, both_strands)
        if len(As) == 0 or len(Bs) == 0:
            pen[i] = miss_weight
            continue

        valid_ds = []
        for a in As:
            for b in Bs:
                d = (b - a) if require_order else abs(b - a)
                if min_dist <= d <= max_dist:
                    valid_ds.append(d)

        if not valid_ds:
            pen[i] = miss_weight
            continue

        if use_helical_phase:
            phase_pen = min(_helical_phase_penalty(d, helical_period, helical_tol) for d in valid_ds)
            pen[i] = phase_weight * float(phase_pen)
        else:
            pen[i] = 0.0
    return pen

def compute_syntax_penalty(
    seqs: List[str],
    rules: List[Dict],
    lambda_per_rule: Optional[List[float]] = None,
) -> np.ndarray:
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
                cached_pwms[key] = (pwm_from_consensus(key[0]), pwm_from_consensus(key[1]))
            pwmA, pwmB = cached_pwms[key]

        pen = pair_spacing_penalty_pwm(
            seqs,
            pwmA, pwmB,
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
