import argparse
from pathlib import Path
import numpy as np
import os
from pathlib import Path
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
    help="Path to PARM pre-trained model directory (e.g., ~/software/PARM/pre_trained_models/K562). "
         "If not provided, will read from env var PARM_MODEL_DIR."
)


    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)

    # -------- scorer --------
    if args.parm_model_dir is None:
     raise ValueError(
        "PARM model directory not provided. "
        "Use --parm_model_dir or set environment variable PARM_MODEL_DIR."
    )

parm_model_dir = Path(args.parm_model_dir).expanduser().resolve()
if not parm_model_dir.exists():
    raise FileNotFoundError(f"PARM model directory not found: {parm_model_dir}")

scorer = ParmScorer(model_dir=str(parm_model_dir))


    # -------- run GA --------
history = run_ga(
        scorer=scorer,
        seq_length=args.length,
        pop_size=args.pop_size,
        n_gen=args.n_gen,
        seed=args.seed,
        outdir=outdir,
        fitness_fn=compute_fitness,
    )

    # -------- save raw results only --------
history.to_csv(outdir / "history.csv", index=False)


if __name__ == "__main__":
    main()
