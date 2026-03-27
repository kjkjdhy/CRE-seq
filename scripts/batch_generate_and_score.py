# scripts/batch_generate_and_score.py
from __future__ import annotations
import pathlib
import re
import subprocess

import numpy as np
import pandas as pd
import os
from creseq.score_adapter import ParmScorer


N_RUNS = 50

GENS = 100
POP = 96
LAMBDA_MOTIF = 0.3
LAMBDA_SHAPE = 0.2
LAMBDA_SYNTAX = 1.0

PARM_MODEL_DIR = os.environ.get("PARM_MODEL_DIR")

# =========================================================

OUT_BASE = pathlib.Path("generator/outputs")
FA_NAME = "final_best.fa"
HEADER_RE = re.compile(r"fitness=([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")


def run_ga_once(seed: int, existing_dirs: set[str], baseline: bool) -> pathlib.Path:
    outdir = OUT_BASE / f"{'baseline' if baseline else 'penalty'}_seed_{seed:03d}"

    cmd = [
        "python",
        "scripts/run_ga.py",
        "--outdir",
        str(outdir),
        "--n_gen",
        str(GENS),
        "--pop_size",
        str(POP),
        "--seed",
        str(seed),
        "--parm_model_dir",
        PARM_MODEL_DIR,
    ]

    if baseline:
        cmd.append("--baseline")
    else:
        cmd += [
            "--lambda_motif", str(LAMBDA_MOTIF),
            "--lambda_syntax", str(LAMBDA_SYNTAX),
            "--lambda_shape", str(LAMBDA_SHAPE),
        ]

    print(f"\n=== Running {'baseline' if baseline else 'penalty'} GA with seed={seed} ===")
    subprocess.run(cmd, check=True)

    return outdir


def parse_final_best(run_dir: pathlib.Path) -> tuple[str, str, float]:
    fa = run_dir / FA_NAME
    if not fa.exists():
        raise FileNotFoundError(f"{fa} not found")

    with fa.open() as f:
        header = f.readline().strip()
        seq = f.readline().strip().upper()

    m = HEADER_RE.search(header)
    fitness = float(m.group(1)) if m else float("nan")

    return run_dir.name, seq, fitness


def main() -> None:
    OUT_BASE.mkdir(parents=True, exist_ok=True)

    rows = []

    for i in range(1, N_RUNS + 1):
        # baseline
        run_dir = run_ga_once(seed=i, existing_dirs=set(), baseline=True)
        run_id, seq, fit = parse_final_best(run_dir)
        rows.append({
            "type": "baseline",
            "run_id": run_id,
            "seed": i,
            "sequence": seq,
            "fitness_ga": fit,
        })

        # penalty
        run_dir = run_ga_once(seed=i, existing_dirs=set(), baseline=False)
        run_id, seq, fit = parse_final_best(run_dir)
        rows.append({
            "type": "penalty",
            "run_id": run_id,
            "seed": i,
            "sequence": seq,
            "fitness_ga": fit,
        })

    df = pd.DataFrame(rows).sort_values(["type", "run_id"])
    print(f"\nCollected {df.shape[0]} champion sequences from GA runs.")

    # Scoring with PARM K562
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