import numpy as np
from typing import List, Dict
def compute_dnashape(seqs: List[str]) -> Dict[str,np.ndarray]:
    n=len(seqs); L=len(seqs[0])
    def sliding_gc(s,w=5):
        arr=np.fromiter((1 if c in "GC" else 0 for c in s), dtype=np.int8)
        if L<w: return np.array([arr.mean()], dtype=np.float32)
        return (np.convolve(arr, np.ones(w,dtype=np.int32), "valid")/w).astype(np.float32)
    def max_run(s):
        m=1;cur=1
        for i in range(1,len(s)):
            if s[i]==s[i-1]: cur+=1; m=max(m,cur)
            else: cur=1
        return float(m)
    mgw_mean=np.zeros(n,dtype=np.float32); roll_var=np.zeros(n,dtype=np.float32); homorun=np.zeros(n,dtype=np.float32)
    for i,s in enumerate(seqs):
        g=sliding_gc(s,5); mgw_mean[i]=1.0-np.abs(g.mean()-0.5); roll_var[i]=g.var(); homorun[i]=max_run(s)
    return {"mgw_mean":mgw_mean,"roll_var":roll_var,"homorun":homorun}
def shape_penalty(shape: Dict[str,np.ndarray], mgw_target=1.0, roll_var_max=0.02, homorun_max=6.0)->np.ndarray:
    return (np.maximum(0.0, mgw_target-shape["mgw_mean"]) + \
            np.maximum(0.0, shape["roll_var"]-roll_var_max) + \
            np.maximum(0.0, shape["homorun"]-homorun_max)).astype(np.float32)
