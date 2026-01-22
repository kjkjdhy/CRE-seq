#!/usr/bin/env python3

import os
from pathlib import Path
import numpy as np

from creseq.generator_core import run_ga
from creseq.score_adapter import ParmScorer
from creseq.fitness import compute_fitness


def main():
    # =========================
    # Reproducibility
    # =========================
    seed = 1
    np.random.seed(seed)

    # =========================
    # Experiment parameters
    # =========================
    seq_length = 120
    population_size = 64
    n_generations = 80
    mutation_rate = 0.01
    crossover_rate = 0.5

    # =========================
    # PARM scorer (K562)
    # =========================
    parm_model_dir = os.environ.get("PARM_MODEL_DIR", None)
    if parm_model_dir is None:
        raise RuntimeError(
            "Please set PARM_MODEL_DIR, e.g.\n"
            "export PARM_MODEL_DIR=/path/to/PARM/pre_trained_models/K562/"
        )

    scorer = ParmScorer(model_dir=parm_model_dir)

    # =========================
    # Output directories
    # =========================
    out_root = Path("results/main_experiment")
    baseline_dir = out_root / "baseline"
    penalty_dir = out_root / "with_penalties"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    penalty_dir.mkdir(parents=True, exist_ok=True)

    # =========================
    # Baseline fitness (no penalties)
    # =========================
    def fitness_baseline(seq: str):
        return compute_fitness(
            seq=seq,
            scorer=scorer,
            penalties=False
        )

    run_ga(
        seq_length=seq_length,
        population_size=population_size,
        n_generations=n_generations,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        fitness_fn=fitness_baseline,
        outdir=str(baseline_dir),
        seed=seed,
    )

    # =========================
    # Fitness with penalties
    # =========================
    def fitness_with_penalties(seq: str):
        return compute_fitness(
            seq=seq,
            scorer=scorer,
            penalties=True
        )

    run_ga(
        seq_length=seq_length,
        population_size=population_size,
        n_generations=n_generations,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        fitness_fn=fitness_with_penalties,
        outdir=str(penalty_dir),
        seed=seed,
    )

    print(f"[DONE] Results written to {out_root.resolve()}")


if __name__ == "__main__":
    main()
