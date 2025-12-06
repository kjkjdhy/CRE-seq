# scripts/run_ga.py
from __future__ import annotations

import argparse
import csv
import json
import time
import pathlib

import numpy as np

from creseq.generator_core import set_seed, init_population, evolve_one_gen
from creseq.score_adapter import ParmScorer
from creseq.motif import scan_motifs, motif_penalty
from creseq.shape import compute_dnashape, shape_penalty
from creseq.fitness import compute_fitness
from creseq.syntax import compute_syntax_penalty


# ================================
# Core promoter syntax rules
# ================================
rules = [

    # ---------------------------------------------------------
    # 1. Core promoter motif grammar (TATA / Inr / DPE / Pause)
    # ---------------------------------------------------------

    # TATA ~ -30 relative to Inr (spacing ~ 28–32 bp)
    {
        "A": "TATAAA",
        "B": "YYANWYY",      # Inr consensus
        "min_dist": 28,
        "max_dist": 32,
        "require_order": True,    # TATA upstream, Inr downstream
        "helical": True,
        "period": 10.5,
        "tol": 2.0,
        "miss_w": 1.0,
        "phase_w": 1.0,
        "min_frac": 0.85,
        "both": True,
    },

    # Inr → DPE: spacing ~ 28–34 bp downstream
    {
        "A": "YYANWYY",
        "B": "RGWYV",        # DPE consensus
        "min_dist": 28,
        "max_dist": 34,
        "require_order": True,
        "helical": False,    # DPE usually does not require strict helical orientation
        "miss_w": 1.0,
        "phase_w": 0.0,
        "min_frac": 0.85,
        "both": True,
    },

    # Pause element: downstream of TSS at +30–50 bp
    {
        "A": "YYANWYY",      # Inr
        "B": "KCGRWCG",      # Pause consensus (PB-like)
        "min_dist": 30,
        "max_dist": 50,
        "require_order": True,
        "helical": False,    # distance-driven
        "miss_w": 1.0,
        "phase_w": 0.0,
        "min_frac": 0.85,
        "both": True,
    },

    # ---------------------------------------------------------
    # 2. TF → PIC / TSS spatial phase
    # ---------------------------------------------------------

    # NF-Y prefers downstream region with strong helical alignment
    {
        "A": "CCAAT",         # NF-Y motif
        "B": "YYANWYY",       # Inr
        "min_dist": 40,
        "max_dist": 90,
        "require_order": True,
        "helical": True,
        "period": 10.5,
        "tol": 2.0,
        "miss_w": 1.0,
        "phase_w": 1.0,
        "min_frac": 0.85,
        "both": True,
    },

    # YY1: close to TSS, ~10–40 bp, helical-sensitive
    {
        "A": "CCATNTT",       # YY1 consensus
        "B": "YYANWYY",
        "min_dist": 10,
        "max_dist": 40,
        "require_order": False,
        "helical": True,
        "period": 10.5,
        "tol": 2.0,
        "miss_w": 1.0,
        "phase_w": 1.0,
        "min_frac": 0.85,
        "both": True,
    },

    # SP1: GC-box, broadly upstream but phase matters
    {
        "A": "GGGCGG",
        "B": "YYANWYY",
        "min_dist": 10,
        "max_dist": 100,
        "require_order": False,
        "helical": True,
        "period": 10.5,
        "tol": 2.0,
        "miss_w": 1.0,
        "phase_w": 1.0,
        "min_frac": 0.85,
        "both": True,
    },

    # ETS family: upstream, helical-sensitive
    {
        "A": "CCGGAAGT",      # ETS consensus
        "B": "YYANWYY",
        "min_dist": 10,
        "max_dist": 120,
        "require_order": False,
        "helical": True,
        "period": 10.5,
        "tol": 2.0,
        "miss_w": 1.0,
        "phase_w": 1.0,
        "min_frac": 0.85,
        "both": True,
    },

    # ---------------------------------------------------------
    # 3. TF–TF cooperative alignment rules (optional)
    # ---------------------------------------------------------

    # NF-Y → SP1 cooperation (mild, distance-driven)
    {
        "A": "CCAAT",
        "B": "GGGCGG",
        "min_dist": 0,
        "max_dist": 100,
        "require_order": False,
        "helical": False,
        "miss_w": 0.5,
        "phase_w": 0.0,
        "min_frac": 0.85,
        "both": True,
    },

    # YY1 ↔ SP1 helical compatibility
    {
        "A": "CCATNTT",
        "B": "GGGCGG",
        "min_dist": 0,
        "max_dist": 80,
        "require_order": False,
        "helical": True,
        "period": 10.5,
        "tol": 2.0,
        "miss_w": 1.0,
        "phase_w": 1.0,
        "min_frac": 0.85,
        "both": True,
    },
]


def save_fasta(path: pathlib.Path, seqs: list[str], scores: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for i, (s, sc) in enumerate(zip(seqs, scores)):
            f.write(f">seq_{i}|fitness={float(sc):.6f}\n{s}\n")


def save_csv(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="CRE-seq: motif/shape/syntax-aware GA generator"
    )

    # Basic hyperparameters
    ap.add_argument("--pop", type=int, default=64)
    ap.add_argument("--length", type=int, default=200)
    ap.add_argument("--gens", type=int, default=50)
    ap.add_argument("--mut_p", type=float, default=0.01)
    ap.add_argument("--cx_rate", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=1)

    # Scoring and penalty weights
    ap.add_argument("--lambda_motif", type=float, default=0.5)
    ap.add_argument("--lambda_shape", type=float, default=0.5)
    ap.add_argument("--lambda_syntax", type=float, default=0.5)

    # DeepSTARR (unused in this script; ParmScorer is used instead)
    ap.add_argument("--deepstarr", type=str, default="")

    # Output
    ap.add_argument("--outdir", type=str, default="generator/outputs")
    ap.add_argument("--save_per_gen", action="store_true")
    args = ap.parse_args()

    # Run directory & save configuration
    run_id = time.strftime("%Y%m%d-%H%M%S")
    outdir = pathlib.Path(args.outdir) / run_id
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "config.json").write_text(json.dumps(vars(args), indent=2))

    # Random seed
    set_seed(args.seed)

    # Initialize population
    pop = init_population(args.pop, args.length)

    # Scorer (PARM K562 model)
    scorer = ParmScorer(
        model_dir="/Users/heyangdong/software/PARM/pre_trained_models/K562/"
    )

    # Basic motif rules
    wanted = ["TATAAA", "YYANWYY", "RGWYV", "KCGRWCG"]
    unwanted = ["TTTTTT"]  # Avoid long poly-T

    history: list[dict] = []

    for g in range(args.gens):
        # ----- Compute features and penalties ----- #
        m = scan_motifs(pop, wanted=wanted, unwanted=unwanted)
        shp = compute_dnashape(pop)

        pen_m = motif_penalty(
            m,
            target_min_pos=1.0,
            target_max_neg=0.0,
        )
        pen_s = shape_penalty(
            shp,
            mgw_target=1.0,
            roll_var_max=0.02,
            homorun_max=6.0,
        )

        # Base fitness (scorer - motif - shape)
        fit = compute_fitness(
            pop,
            scorer,
            pen_m,
            pen_s,
            lambda_motif=args.lambda_motif,
            lambda_shape=args.lambda_shape,
        )

        # Spatial syntax penalty (core promoter rules)
        pen_syntax = compute_syntax_penalty(pop, rules)
        fit = fit - args.lambda_syntax * pen_syntax

        # ----- Logging ----- #
        best = int(np.argmax(fit))
        print(
            f"Gen {g:03d} | best_fit={fit[best]:.4f} | "
            f"seq[:20]={pop[best][:20]}..."
        )

        history.append(
            {
                "gen": g,
                "best_fitness": float(fit[best]),
                "best_seq": pop[best],
                # Motif counts and penalties
                "counts_pos": float(m["counts_pos"][best]),
                "counts_neg": float(m["counts_neg"][best]),
                "motif_pen": float(pen_m[best]),
                # DNAshape summaries and penalty
                "mgw_mean": float(shp["mgw_mean"][best]),
                "roll_var": float(shp["roll_var"][best]),
                "homorun": float(shp["homorun"][best]),
                "shape_pen": float(pen_s[best]),
                # Spatial syntax penalty
                "syntax_pen": float(pen_syntax[best]),
            }
        )

        if args.save_per_gen:
            save_fasta(outdir / f"best_gen_{g:03d}.fa", [pop[best]], fit[[best]])

        # ----- Evolve to next generation ----- #
        pop = evolve_one_gen(pop, fit, mut_p=args.mut_p, cx_rate=args.cx_rate)

    # ----- Final evaluation + save ----- #
    m = scan_motifs(pop, wanted=wanted, unwanted=unwanted)
    shp = compute_dnashape(pop)
    pen_m = motif_penalty(
        m,
        target_min_pos=1.0,
        target_max_neg=0.0,
    )
    pen_s = shape_penalty(
        shp,
        mgw_target=1.0,
        roll_var_max=0.02,
        homorun_max=6.0,
    )
    fit = compute_fitness(
        pop,
        scorer,
        pen_m,
        pen_s,
        lambda_motif=args.lambda_motif,
        lambda_shape=args.lambda_shape,
    )
    pen_syntax = compute_syntax_penalty(pop, rules)
    fit = fit - args.lambda_syntax * pen_syntax

    best = int(np.argmax(fit))
    print("\n=== FINAL BEST ===")
    print(pop[best])
    print(f"fitness = {fit[best]:.4f}")

    save_fasta(outdir / "final_best.fa", [pop[best]], fit[[best]])
    save_csv(outdir / "history.csv", history)


if __name__ == "__main__":
    main()
