# scripts/run_ga.py
from __future__ import annotations
import argparse, csv, json, time, pathlib
import numpy as np


from creseq.generator_core import set_seed, init_population, evolve_one_gen
from creseq.score_adapter import ParmScorer
from creseq.motif import scan_motifs, motif_penalty
from creseq.shape import compute_dnashape, shape_penalty
from creseq.fitness import compute_fitness
from creseq.syntax import compute_syntax_penalty

def save_fasta(path: pathlib.Path, seqs: list[str], scores: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for i, (s, sc) in enumerate(zip(seqs, scores)):
            f.write(f">seq_{i}|fitness={float(sc):.6f}\n{s}\n")

def save_csv(path: pathlib.Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)

def main():
    ap = argparse.ArgumentParser(description="CRE-seq: motif/shape/syntax-aware GA generator")
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

    # DeepSTARR (leave empty to use DummyScorer)
    ap.add_argument("--deepstarr", type=str, default="")  # SavedModel directory

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

    # Scorer
    scorer = ParmScorer(
    model_dir="/Users/heyangdong/software/PARM/pre_trained_models/K562/"
)


    # Basic motif rules (example, can be modified per task)
    wanted = ["TGACGTCA"]  # CRE-like
    unwanted = ["TTTTTT"]  # Avoid long poly-T

    # Danko-style "spatial syntax" rules (can be modified per cell line/task)
    # For rule field descriptions, see src/creseq/syntax.py: compute_syntax_penalty docstring
    syntax_rules = [
        # Example 1: CRE-like and ETS (heterotypic cooperation), 6-30bp, prefer same phase (10.5±2)
        {"A": "TGACGTCA", "B": "GGAA", "min_dist": 6, "max_dist": 30,
         "require_order": False, "min_frac": 0.85, "both": True,
         "helical": True, "period": 10.5, "tol": 2.0, "miss_w": 1.0, "phase_w": 1.0},

        # Example 2: CRE-like self-repeat (homotypic), encourage 10~11bp periodic repeat
        {"A": "TGACGTCA", "B": "TGACGTCA", "min_dist": 9, "max_dist": 40,
         "require_order": False, "min_frac": 0.85, "both": True,
         "helical": True, "period": 10.5, "tol": 2.0, "miss_w": 0.5, "phase_w": 1.0},
    ]

    history = []
    for g in range(args.gens):
        # —— Compute features and penalties —— #
        m = scan_motifs(pop, wanted=wanted, unwanted=unwanted)
        shp = compute_dnashape(pop)

        pen_m = motif_penalty(m, target_min_pos=1.0, target_max_neg=0.0)
        pen_s = shape_penalty(shp, mgw_target=1.0, roll_var_max=0.02, homorun_max=6.0)

        # Base fitness (DeepSTARR/Dummy - motif - shape)
        fit = compute_fitness(pop, scorer, pen_m, pen_s,
                              lambda_motif=args.lambda_motif,
                              lambda_shape=args.lambda_shape)

        # Spatial syntax penalty (pairing + spacing window + helical phase)
        pen_syntax = compute_syntax_penalty(pop, syntax_rules)
        fit = fit - args.lambda_syntax * pen_syntax

        # —— Logging —— #
                # —— Logging —— #
        best = int(np.argmax(fit))
        print(f"Gen {g:03d} | best_fit={fit[best]:.4f} | seq[:20]={pop[best][:20]}...")

        history.append({
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
        })

        if args.save_per_gen:
            save_fasta(outdir / f"best_gen_{g:03d}.fa", [pop[best]], fit[[best]])

        # —— Evolve to next generation —— #
        pop = evolve_one_gen(pop, fit, mut_p=args.mut_p, cx_rate=args.cx_rate)

    # —— Final evaluation + save —— #
    m = scan_motifs(pop, wanted=wanted, unwanted=unwanted)
    shp = compute_dnashape(pop)
    pen_m = motif_penalty(m, target_min_pos=1.0, target_max_neg = 0.0)
    pen_s = shape_penalty(shp, mgw_target=1.0, roll_var_max=0.02, homorun_max=6.0)
    fit = compute_fitness(pop, scorer, pen_m, pen_s,
                          lambda_motif=args.lambda_motif,
                          lambda_shape=args.lambda_shape)
    pen_syntax = compute_syntax_penalty(pop, syntax_rules)
    fit = fit - args.lambda_syntax * pen_syntax

    best = int(np.argmax(fit))
    print("\n=== FINAL BEST ===")
    print(pop[best])
    print(f"fitness = {fit[best]:.4f}")

    save_fasta(outdir / "final_best.fa", [pop[best]], fit[[best]])
    save_csv(outdir / "history.csv", history)

if __name__ == "__main__":
    main()
