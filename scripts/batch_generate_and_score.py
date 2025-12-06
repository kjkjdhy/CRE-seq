# scripts/batch_generate_and_score.py
from __future__ import annotations
import pathlib
import re
import subprocess

import numpy as np
import pandas as pd

from creseq.score_adapter import ParmScorer


N_RUNS = 50


GENS = 80
POP = 96
LAMBDA_MOTIF = 0.3
LAMBDA_SHAPE = 0.2
LAMBDA_SYNTAX = 1.0


PARM_MODEL_DIR = "/Users/heyangdong/software/PARM/pre_trained_models/K562/"

# =========================================================

OUT_BASE = pathlib.Path("generator/outputs")
FA_NAME = "final_best.fa"
HEADER_RE = re.compile(r"fitness=([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")


def run_ga_once(seed: int, existing_dirs: set[str]) -> pathlib.Path:
    
    cmd = [
        "python",
        "scripts/run_ga.py",
        "--gens",
        str(GENS),
        "--pop",
        str(POP),
        "--lambda_motif",
        str(LAMBDA_MOTIF),
        "--lambda_shape",
        str(LAMBDA_SHAPE),
        "--lambda_syntax",
        str(LAMBDA_SYNTAX),
        "--seed",
        str(seed),
    ]
    print(f"\n=== Running GA with seed={seed} ===")
    subprocess.run(cmd, check=True)


    current_dirs = {p.name for p in OUT_BASE.iterdir() if p.is_dir()}
    new_dirs = current_dirs - existing_dirs
    if len(new_dirs) != 1:
        raise RuntimeError(
            f"Expected exactly 1 new run directory, found {len(new_dirs)}: {new_dirs}"
        )
    run_id = next(iter(new_dirs))
    print(f"seed={seed} -> new run_id={run_id}")
    return OUT_BASE / run_id


def parse_final_best(run_dir: pathlib.Path) -> tuple[str, str, float]:
    """从 run_dir/final_best.fa 里取出 (run_id, seq, fitness)。"""
    fa = run_dir / FA_NAME
    if not fa.exists():
        raise FileNotFoundError(f"{fa} not found")

    with fa.open() as f:
        header = f.readline().strip()
        seq = f.readline().strip().upper()

    m = HEADER_RE.search(header)
    if m:
        fitness = float(m.group(1))
    else:
        fitness = float("nan")

    return run_dir.name, seq, fitness


def main() -> None:
    OUT_BASE.mkdir(parents=True, exist_ok=True)


    existing_dirs = {p.name for p in OUT_BASE.iterdir() if p.is_dir()}

    rows = []

    for i in range(1, N_RUNS + 1):
        run_dir = run_ga_once(seed=i, existing_dirs=existing_dirs)
        existing_dirs.add(run_dir.name)
        run_id, seq, fit = parse_final_best(run_dir)
        rows.append(
            {
                "run_id": run_id,
                "seed": i,
                "sequence": seq,
                "fitness_ga": fit,
            }
        )

    df = pd.DataFrame(rows).sort_values("run_id")
    print(f"\nCollected {df.shape[0]} champion sequences from GA runs.")

  #Scoring with PARM K562

    seqs = df["sequence"].astype(str).str.upper().tolist()
    Ls = {len(s) for s in seqs}
    if len(Ls) != 1:
        raise ValueError(f"Ours: sequences have different lengths: {sorted(Ls)}")

    print(f"\nScoring {len(seqs)} champion sequences with PARM K562...")
    scorer = ParmScorer(model_dir=PARM_MODEL_DIR)
    scores = np.asarray(scorer.score_batch(seqs), dtype=np.float32)
    df["parm_k562_score"] = scores

    print(
        f"[OURS champions] n={len(scores)}, "
        f"min={scores.min():.4f}, "
        f"median={np.median(scores):.4f}, "
        f"p90={np.percentile(scores, 90):.4f}, "
        f"max={scores.max():.4f}"
    )

    out_path = pathlib.Path("my_generator_batch_with_parm.tsv")
    df.to_csv(out_path, sep="\t", index=False)
    print(f"\nSaved champion sequences + PARM scores to {out_path}")


if __name__ == "__main__":
    main()
