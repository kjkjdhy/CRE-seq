import numpy as np
from typing import List, Dict
def _count(s: str, kmer: str) -> int:
    k=len(kmer); return sum(1 for i in range(len(s)-k+1) if s[i:i+k]==kmer)
def scan_motifs(seqs: List[str], wanted=None, unwanted=None) -> Dict[str,np.ndarray]:
    wanted=wanted or []; unwanted=unwanted or []; n=len(seqs)
    pos=np.zeros(n,dtype=np.float32); neg=np.zeros(n,dtype=np.float32)
    for i,s in enumerate(seqs):
        if wanted: pos[i]=sum(_count(s,k) for k in wanted)
        if unwanted: neg[i]=sum(_count(s,k) for k in unwanted)
    return {"counts_pos":pos,"counts_neg":neg}
def motif_penalty(m: Dict[str,np.ndarray], target_min_pos=1.0, target_max_neg=0.0)->np.ndarray:
    p1=np.maximum(0.0, target_min_pos - m.get("counts_pos",0.0))
    p2=np.maximum(0.0, m.get("counts_neg",0.0) - target_max_neg)
    return (p1+p2).astype(np.float32)



from typing import Tuple

def _find_sites(seq: str, kmer: str, strand: str = "+") -> list[int]:
    # Simplified: only scan forward strand; for both strands, scan the RC of seq as well
    k = len(kmer)
    hits = []
    for i in range(len(seq)-k+1):
        if seq[i:i+k] == kmer:
            hits.append(i)
    return hits

def pair_spacing_penalty(
    seqs: list[str],
    pair: Tuple[str, str],
    min_dist: int,
    max_dist: int,
    require_order: bool = False,   # True: A在左B在右；False: 无序
    require_same_strand: bool = False,  # 先留空位，后续接双链/方向
) -> np.ndarray:
    """For each sequence, if (A,B) does not have a pair with "valid spacing and order", penalize 1, otherwise 0."""
    A, B = pair
    n = len(seqs)
    pen = np.ones(n, dtype=np.float32)
    for i, s in enumerate(seqs):
        As = _find_sites(s, A)
        Bs = _find_sites(s, B)
        ok = False
        if require_order:
            for a in As:
                for b in Bs:
                    d = b - a
                    if min_dist <= d <= max_dist:
                        ok = True; break
                if ok: break
        else:
            # Unordered: just check if distance falls within window
            for a in As:
                for b in Bs:
                    d = abs(b - a)
                    if min_dist <= d <= max_dist:
                        ok = True; break
                if ok: break
        pen[i] = 0.0 if ok else 1.0
    return pen



