import argparse
import os
from pathlib import Path
from functools import partial

import numpy as np

from creseq.generator_core import run_ga
from creseq.score_adapter import ParmScorer
from creseq.fitness import compute_fitness


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--length", type=int, default=200)
    parser.add_argument("--pop_size", type=int, default=64)
    parser.add_argument("--n_gen", type=int, default=80)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument(
        "--parm_model_dir",
        type=str,
        default=os.environ.get("PARM_MODEL_DIR", None),
        help=(
            "Path to PARM pre-trained model directory "
            "(e.g., ~/software/PARM/pre_trained_models/K562). "
            "If not provided, will read from env var PARM_MODEL_DIR."
        ),
    )

    parser.add_argument("--lambda_motif", type=float, default=1.0)
    parser.add_argument("--lambda_syntax", type=float, default=1.0)
    parser.add_argument("--lambda_shape", type=float, default=0.0)

    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run pure scorer-only baseline with all penalties disabled.",
    )

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)

    if args.parm_model_dir is None:
        raise ValueError(
            "PARM model directory not provided. "
            "Use --parm_model_dir or set environment variable PARM_MODEL_DIR."
        )

    parm_model_dir = Path(args.parm_model_dir).expanduser().resolve()
    if not parm_model_dir.exists():
        raise FileNotFoundError(f"PARM model directory not found: {parm_model_dir}")

    scorer = ParmScorer(model_dir=str(parm_model_dir))

    if args.baseline:
        lambda_motif = 0.0
        lambda_syntax = 0.0
        lambda_shape = 0.0
    else:
        lambda_motif = args.lambda_motif
        lambda_syntax = args.lambda_syntax
        lambda_shape = args.lambda_shape

    fitness_fn = partial(
        compute_fitness,
        lambda_motif=lambda_motif,
        lambda_syntax=lambda_syntax,
        lambda_shape=lambda_shape,
    )

    history = run_ga(
        scorer=scorer,
        seq_length=args.length,
        pop_size=args.pop_size,
        n_gen=args.n_gen,
        seed=args.seed,
        outdir=outdir,
        fitness_fn=fitness_fn,
    )

    history.to_csv(outdir / "history.csv", index=False)

    with open(outdir / "run_config.txt", "w") as f:
        f.write(f"seed={args.seed}\n")
        f.write(f"length={args.length}\n")
        f.write(f"pop_size={args.pop_size}\n")
        f.write(f"n_gen={args.n_gen}\n")
        f.write(f"parm_model_dir={parm_model_dir}\n")
        f.write(f"baseline={args.baseline}\n")
        f.write(f"lambda_motif={lambda_motif}\n")
        f.write(f"lambda_syntax={lambda_syntax}\n")
        f.write(f"lambda_shape={lambda_shape}\n")


if __name__ == "__main__":
    main()