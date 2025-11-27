Command:
  python scripts/run_ga.py --gens 50 --pop 64 --length 200 \
    --lambda_motif 0.5 --lambda_shape 0.5 --lambda_syntax 0.5 \
    --save_per_gen --outdir generator/runs/human_k_v1

Scorer:
  HumanKnowledgeScorer (GC 40–60%, motif list = [TGACGTCA, GGAA, ...])

Notes:
  final_best.fa header: fitness ~0.735
