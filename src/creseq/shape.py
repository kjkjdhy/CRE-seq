import numpy as np
from typing import List, Dict, Optional, Union

def _longest_homopolymer_run(s: str) -> int:
    if not s:
        return 0
    best = 1
    cur = 1
    prev = s[0]
    for ch in s[1:]:
        if ch == prev:
            cur += 1
        else:
            if cur > best:
                best = cur
            cur = 1
            prev = ch
    if cur > best:
        best = cur
    return best

def compute_dnashape(seqs: List[str]) -> Dict[str, np.ndarray]:
    n = len(seqs)
    L = len(seqs[0]) if n > 0 else 0

    mgw = np.zeros((n, L), dtype=np.float32)
    roll = np.zeros((n, L), dtype=np.float32)
    helt = np.zeros((n, L), dtype=np.float32)
    prot = np.zeros((n, L), dtype=np.float32)
    homorun = np.zeros((n,), dtype=np.float32)

    for i, s in enumerate(seqs):
        s = s.upper()
        homorun[i] = float(_longest_homopolymer_run(s))

        gc_mask = np.fromiter(((c == "G") or (c == "C") for c in s), count=L, dtype=np.int8).astype(np.float32)
        if L >= 5:
            win = 5
            pad = win // 2
            x = np.pad(gc_mask, (pad, pad), mode="edge")
            k = np.ones((win,), dtype=np.float32) / float(win)
            mgw_i = np.convolve(x, k, mode="valid")
        else:
            mgw_i = gc_mask
        mgw[i, :] = mgw_i[:L]

        r = np.zeros((L,), dtype=np.float32)
        h = np.zeros((L,), dtype=np.float32)
        p = np.zeros((L,), dtype=np.float32)

        for j in range(L - 1):
            a = s[j]
            b = s[j + 1]

            if (a == "A" and b == "T") or (a == "T" and b == "A"):
                r[j] = -0.20
            elif (a == "C" and b == "G") or (a == "G" and b == "C"):
                r[j] = 0.20
            else:
                r[j] = 0.0

            if (a == "A" and b == "A") or (a == "T" and b == "T"):
                h[j] = 0.15
            elif (a == "G" and b == "G") or (a == "C" and b == "C"):
                h[j] = -0.15
            else:
                h[j] = 0.0

            if a in ("A", "T") and b in ("A", "T"):
                p[j] = 0.10
            elif a in ("G", "C") and b in ("G", "C"):
                p[j] = -0.10
            else:
                p[j] = 0.0

        r[L - 1] = r[L - 2] if L >= 2 else 0.0
        h[L - 1] = h[L - 2] if L >= 2 else 0.0
        p[L - 1] = p[L - 2] if L >= 2 else 0.0

        roll[i, :] = r
        helt[i, :] = h
        prot[i, :] = p

    return {
        "MGW": mgw,
        "Roll": roll,
        "HelT": helt,
        "ProT": prot,
        "homorun": homorun,
    }

def _region(x: np.ndarray, center: int, left: int, right: int) -> np.ndarray:
    L = x.shape[1]
    a = max(0, center - left)
    b = min(L, center + right)
    if b <= a:
        return x[:, :0]
    return x[:, a:b]

def shape_penalty(
    seqs_or_shape: Union[List[str], Dict[str, np.ndarray]],
    shapes: Optional[Dict[str, np.ndarray]] = None,
    tss_index: Optional[int] = None,
    mgw_target: float = 0.55,
    mgw_tol: float = 0.08,
    roll_mean_target: float = 0.0,
    roll_mean_tol: float = 0.06,
    helt_mean_target: float = 0.0,
    helt_mean_tol: float = 0.06,
    prot_mean_target: float = 0.0,
    prot_mean_tol: float = 0.06,
    homorun_max: float = 6.0,
    window_left: int = 40,
    window_right: int = 40,
) -> np.ndarray:
    if isinstance(seqs_or_shape, dict):
        shp = seqs_or_shape
        n = int(shp["MGW"].shape[0])
        L = int(shp["MGW"].shape[1])
    else:
        seqs = seqs_or_shape
        shp = shapes if shapes is not None else compute_dnashape(seqs)
        n = len(seqs)
        L = int(shp["MGW"].shape[1])

    if tss_index is None:
        tss = L // 2
    else:
        tss = int(tss_index)

    mgw = shp["MGW"]
    roll = shp["Roll"]
    helt = shp["HelT"]
    prot = shp["ProT"]
    hr = shp.get("homorun", np.zeros((n,), dtype=np.float32)).astype(np.float32)

    mgw_w = _region(mgw, tss, window_left, window_right)
    roll_w = _region(roll, tss, window_left, window_right)
    helt_w = _region(helt, tss, window_left, window_right)
    prot_w = _region(prot, tss, window_left, window_right)

    def mean_or_zero(a: np.ndarray) -> np.ndarray:
        if a.shape[1] == 0:
            return np.zeros((n,), dtype=np.float32)
        return a.mean(axis=1).astype(np.float32)

    mgw_mean = mean_or_zero(mgw_w)
    roll_mean = mean_or_zero(roll_w)
    helt_mean = mean_or_zero(helt_w)
    prot_mean = mean_or_zero(prot_w)

    pen = np.zeros((n,), dtype=np.float32)

    pen += np.maximum(0.0, np.abs(mgw_mean - float(mgw_target)) - float(mgw_tol))
    pen += np.maximum(0.0, np.abs(roll_mean - float(roll_mean_target)) - float(roll_mean_tol))
    pen += np.maximum(0.0, np.abs(helt_mean - float(helt_mean_target)) - float(helt_mean_tol))
    pen += np.maximum(0.0, np.abs(prot_mean - float(prot_mean_target)) - float(prot_mean_tol))

    pen += np.maximum(0.0, hr - float(homorun_max)) / float(max(homorun_max, 1.0))

    return pen.astype(np.float32)
