import numpy as np
from typing import List, Dict

IUPAC = {
    "A": {"A"}, "C": {"C"}, "G": {"G"}, "T": {"T"},
    "R": {"A", "G"}, "Y": {"C", "T"}, "S": {"G", "C"},
    "W": {"A", "T"}, "K": {"G", "T"}, "M": {"A", "C"},
    "B": {"C", "G", "T"}, "D": {"A", "G", "T"},
    "H": {"A", "C", "T"}, "V": {"A", "C", "G"},
    "N": {"A", "C", "G", "T"},
}

def _match(seq: str, pat: str, pos: int) -> bool:
    if pos < 0 or pos + len(pat) > len(seq):
        return False
    for i, c in enumerate(pat):
        if seq[pos + i] not in IUPAC[c]:
            return False
    return True

def motif_penalty(seqs: List[str]) -> np.ndarray:
    n = len(seqs)
    L = len(seqs[0])
    tss = L // 2
    pen = np.zeros(n, dtype=np.float32)

    for i, s in enumerate(seqs):
        s = s.upper()
        ok = True

        tata_ok = False
        for p in range(tss - 33, tss - 26):
            if _match(s, "TATAWAWR", p):
                tata_ok = True
                break

        inr_ok = False
        for p in range(tss - 2, tss + 1):
            if _match(s, "YYANWYY", p):
                inr_ok = True
                break

        dpe_pos = None
        for p in range(tss + 28, tss + 33):
            if _match(s, "RGWYV", p):
                dpe_pos = p
                break

        if not inr_ok:
            ok = False

        if dpe_pos is None:
            ok = False

        if inr_ok and dpe_pos is not None:
            inr_center = tss
            dist = dpe_pos - inr_center
            if not (28 <= dist <= 32):
                ok = False

        if not ok:
            pen[i] = 1.0

    return pen
