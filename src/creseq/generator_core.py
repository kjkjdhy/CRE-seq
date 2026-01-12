# creseq/generator_core.py
from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

ALPHABET = np.array(list("ACGT"))


def set_seed(seed: int = 1) -> None:
    random.seed(seed)
    np.random.seed(seed)


def init_population(n: int, length: int) -> List[str]:
    arr = np.random.choice(ALPHABET, size=(n, length))
    return ["".join(r) for r in arr]


def mutate(seq: str, p: float = 0.01) -> str:
    s = list(seq)
    for i, b in enumerate(s):
        if np.random.rand() < p:
            choices = ALPHABET[ALPHABET != b]
            s[i] = str(np.random.choice(choices))
    return "".join(s)


def crossover(a: str, b: str, rate: float = 0.5) -> Tuple[str, str]:
    if np.random.rand() > rate or len(a) != len(b):
        return a, b
    cut = np.random.randint(1, len(a))
    return a[:cut] + b[cut:], b[:cut] + a[cut:]


def tournament_select(pop: List[str], fit: np.ndarray, k: int = 3) -> str:
    idx = np.random.choice(len(pop), size=k, replace=False)
    return pop[int(idx[np.argmax(fit[idx])])]


def evolve_one_gen(
    pop: List[str],
    fit: np.ndarray,
    mut_p: float = 0.01,
    cx_rate: float = 0.5,
) -> List[str]:
    # Elitism: keep the current best
    new_pop = [pop[int(np.argmax(fit))]]
    while len(new_pop) < len(pop):
        p1 = tournament_select(pop, fit)
        p2 = tournament_select(pop, fit)
        c1, c2 = crossover(p1, p2, cx_rate)
        new_pop.extend([mutate(c1, mut_p), mutate(c2, mut_p)])
    return new_pop[: len(pop)]


def _write_fasta(path: Path, seqs: List[str], scores: Optional[np.ndarray] = None) -> None:
    with path.open("w") as f:
        for i, s in enumerate(seqs):
            if scores is None:
                hdr = f">seq_{i}"
            else:
                hdr = f">seq_{i} fitness={float(scores[i]):.6f}"
            f.write(hdr + "\n")
            f.write(s + "\n")


def run_ga(
    scorer,
    seq_length: int = 200,
    pop_size: int = 64,
    n_gen: int = 80,
    seed: int = 1,
    outdir: Optional[Path] = None,
    fitness_fn: Optional[Callable] = None,
    mut_p: float = 0.01,
    cx_rate: float = 0.5,
) -> pd.DataFrame:
    """
    Core GA loop.
    - scorer: your PARM scorer instance
    - fitness_fn: function with signature fitness_fn(seqs: List[str], scorer) -> np.ndarray
      (This is where penalty is applied, via creseq/fitness.py)
    - Returns a pandas DataFrame (history) and optionally writes outputs to outdir.
    """
    if fitness_fn is None:
        raise ValueError("fitness_fn must be provided (e.g., creseq.fitness.compute_fitness).")

    set_seed(seed)

    if outdir is not None:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

    pop = init_population(pop_size, seq_length)

    # If available, we can log components for debugging (score, penalties) WITHOUT changing your run script.
    # This is optional: if compute_fitness_components doesn't exist, we just log best_fitness/mean_fitness.
    try:
        from creseq.fitness import compute_fitness_components  # type: ignore
        _HAS_COMPONENTS = True
    except Exception:
        compute_fitness_components = None
        _HAS_COMPONENTS = False

    history_rows: List[Dict] = []

    for gen in range(n_gen):
        if _HAS_COMPONENTS and compute_fitness_components is not None and fitness_fn.__name__ == "compute_fitness":
            fit, comps = compute_fitness_components(pop, scorer)
            score = comps.get("score", None)
            pen_motif = comps.get("pen_motif", None)
            pen_shape = comps.get("pen_shape", None)
            pen_syntax = comps.get("pen_syntax", None)
        else:
            fit = np.asarray(fitness_fn(pop, scorer), dtype=np.float32)
            score = pen_motif = pen_shape = pen_syntax = None

        best_idx = int(np.argmax(fit))
        best_fit = float(fit[best_idx])
        mean_fit = float(np.mean(fit))

        row: Dict = {
            "gen": gen,
            "best_fitness": best_fit,
            "mean_fitness": mean_fit,
            "best_seq": pop[best_idx],
        }

        # Optional logging columns (only if compute_fitness_components is available)
        if score is not None:
            row["best_score"] = float(np.asarray(score)[best_idx])
        if pen_motif is not None:
            row["best_pen_motif"] = float(np.asarray(pen_motif)[best_idx])
        if pen_shape is not None:
            row["best_pen_shape"] = float(np.asarray(pen_shape)[best_idx])
        if pen_syntax is not None:
            row["best_pen_syntax"] = float(np.asarray(pen_syntax)[best_idx])

        history_rows.append(row)

        # Evolve
        pop = evolve_one_gen(pop, fit, mut_p=mut_p, cx_rate=cx_rate)

    history = pd.DataFrame(history_rows)

    # Write outputs if requested
    if outdir is not None:
        history.to_csv(outdir / "history.csv", index=False)

        # Evaluate final population once more and save best + population snapshot
        final_fit = np.asarray(fitness_fn(pop, scorer), dtype=np.float32)
        best_idx = int(np.argmax(final_fit))
        best_seq = pop[best_idx]
        best_fit = float(final_fit[best_idx])

        _write_fasta(outdir / "final_population.fa", pop, final_fit)
        _write_fasta(outdir / "final_best.fa", [best_seq], np.array([best_fit], dtype=np.float32))

    return history
