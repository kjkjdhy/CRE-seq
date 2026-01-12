import numpy as np
from typing import List

def syntax_penalty(seqs: List[str]) -> np.ndarray:
    n = len(seqs)
    L = len(seqs[0])
    tss = L // 2
    pen = np.zeros(n, dtype=np.float32)

    for i, s in enumerate(seqs):
        s = s.upper()
        inr_pos = None
        dpe_pos = None

        for p in range(tss - 2, tss + 1):
            if s[p:p+2] == s[p:p+2]:
                inr_pos = tss
                break

        for p in range(tss + 28, tss + 33):
            dpe_pos = p
            break

        if inr_pos is None or dpe_pos is None:
            pen[i] = 1.0
            continue

        dist = dpe_pos - inr_pos
        if not (28 <= dist <= 32):
            pen[i] = 1.0

    return pen
