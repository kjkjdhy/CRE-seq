CRE-seq is a DNA sequence generation framework for designing promoter-like cis-regulatory elements (CREs) with high predicted transcriptional activity. The core idea is to combine evolutionary optimization with explicit promoter grammar constraints, rather than relying solely on black-box generative models. 
The framework uses a genetic algorithm to iteratively optimize DNA sequences based on a learned activity predictor, while optionally enforcing biologically motivated grammar rules such as motif presence and relative spacing between core promoter elements. This allows CRE-seq to explore high-scoring sequence space while maintaining interpretable and controllable sequence structure. 
Cre-seq framework is very flexible. The weights of all the penalties are adjustable. In the batch_generate_and_score file, lambda_motif, lambda_syntax and lambda_shape each represent the weight of those three penalties. If you want to only use the baseline version and only wish to compare the MPRA-trained activity scorers themselves without any structural constraint, you could just set them all to zero. You could also try out different types of constraints in respective files.  
Overall, this project is intended as a mechanistic, interpretable alternative to purely data-driven and black-box CRE design approaches, and as a flexible framework for studying how promoter grammar influences sequence optimization.

## External models

CRE-seq relies on the PARM (Promoter Activity Regulatory Model) as a surrogate for predicting regulatory activity. Due to model size and licensing constraints, pretrained PARM models are not included in this repository.

You can find the official PARM implementation and instructions for obtaining pretrained models on GitHub:

https://github.com/vansteensellab/PARM

After downloading, set the environment variable pointing to the model directory, e.g.:

```bash
export PARM_MODEL_DIR=/path/to/PARM/pre_trained_models/K562
```


## Reproducing the main CRE-seq experiment

To reproduce the main CRE-seq optimization experiment (baseline vs. penalty-aware GA, scored with PARM K562), run:

```
export PARM_MODEL_DIR=/path/to/PARM/pre_trained_models/K562/
python scripts/run_experiment.py
```
This command runs two genetic algorithm optimizations with identical settings, differing only in whether penalties are applied during fitness evaluation.

Outputs are written to:
```
results/main_experiment/
├── baseline/
│   ├── history.csv
│   └── final_best.fa
└── with_penalties/
    ├── history.csv
    └── final_best.fa
```
File descriptions:

- history.csv: Per-generation optimization statistics (e.g. best fitness over generations), used to compare convergence behavior between baseline and penalty-aware runs.
- final_best.fa: FASTA file containing the highest-scoring sequence(s) obtained at the end of optimization, used for downstream sequence analysis.
