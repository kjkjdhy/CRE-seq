from __future__ import annotations
import random, numpy as np
from typing import List, Tuple
ALPHABET = np.array(list("ACGT"))
def set_seed(seed: int = 1): random.seed(seed); np.random.seed(seed)
def init_population(n: int, length: int) -> List[str]:
    arr = np.random.choice(ALPHABET, size=(n, length)); return ["".join(r) for r in arr]
def mutate(seq: str, p: float = 0.01) -> str:
    s=list(seq)
    for i,b in enumerate(s):
        if np.random.rand() < p:
            choices = ALPHABET[ALPHABET != b]; s[i]=np.random.choice(choices)
    return "".join(s)
def crossover(a: str, b: str, rate: float = 0.5) -> Tuple[str,str]:
    if np.random.rand()>rate or len(a)!=len(b): return a,b
    cut=np.random.randint(1,len(a)); return a[:cut]+b[cut:], b[:cut]+a[cut:]
def tournament_select(pop: List[str], fit: np.ndarray, k: int = 3) -> str:
    idx=np.random.choice(len(pop), size=k, replace=False); return pop[idx[np.argmax(fit[idx])]]
def evolve_one_gen(pop: List[str], fit: np.ndarray, mut_p=0.01, cx_rate=0.5) -> List[str]:
    new_pop=[pop[int(np.argmax(fit))]]
    while len(new_pop)<len(pop):
        p1=tournament_select(pop,fit); p2=tournament_select(pop,fit)
        c1,c2=crossover(p1,p2,cx_rate); new_pop.extend([mutate(c1,mut_p), mutate(c2,mut_p)])
    return new_pop[:len(pop)]
