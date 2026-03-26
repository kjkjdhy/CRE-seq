## CRE-seq: Interpretable DNA Regulatory Sequence Design via Evolutionary Optimization

### TL;DR
CRE-seq designs DNA regulatory sequences with high transcriptional activity using a genetic algorithm.  
It combines a deep learning activity predictor with explicit promoter grammar constraints.  
This enables efficient, interpretable, and controllable sequence optimization.

## Background Introduction
In genetics, designing DNA regulatory sequences with high transcriptional activity helps uncover the regulatory grammar that links sequence to gene expression, a central problem in gene regulation. It also enables practical applications in synthetic biology and gene therapy by creating sequences that drive strong and controllable gene expression.

One of the most common and powerful approaches for this task is the genetic algorithm (GA), which iteratively optimizes discrete sequences by selecting high-fitness sequences from the previous generation and generating new sequences through crossover and mutation. This process requires a scoring function to evaluate sequence “fitness.”

In the context of regulatory sequence design, this scorer is typically a deep learning model trained on large MPRA datasets, which contain millions of DNA sequences paired with their measured transcriptional activity.

## Project Description
This project explores an engineering approach to improve the convergence efficiency of this discrete optimization process.

CRE-seq is a DNA sequence generation framework for designing promoter-like cis-regulatory elements (CREs) with high predicted transcriptional activity. The core idea is to combine evolutionary optimization with explicit promoter grammar constraints, rather than relying solely on black-box generative models.

The framework uses a genetic algorithm to iteratively optimize DNA sequences based on a learned activity predictor, while optionally enforcing biologically motivated constraints such as motif presence and relative spacing between core promoter elements. This enables CRE-seq to explore high-scoring sequence space while maintaining interpretable and controllable structure.

CRE-seq is designed to be flexible. The weights of different constraints are configurable: in `batch_generate_and_score`, `lambda_motif`, `lambda_syntax`, and `lambda_shape` control the strength of each penalty. Setting all of them to zero recovers a pure baseline that relies only on the activity predictor, allowing direct comparison between constrained and unconstrained optimization.

Overall, this project serves as a mechanistic and interpretable complement to purely data-driven CRE design approaches, and as a flexible framework for studying how promoter grammar influences sequence optimization.

## Results
![ECDF](results/Figure.png)


## Step-by-step reproduction guide

### 1. Clone the repository

```bash
git clone https://github.com/kjkjdhy/CRE-seq.git
cd CRE-seq
```

---

### 2. Create and activate a Python environment

```bash
conda create -n cre-seq python=3.9 -y
conda activate cre-seq
pip install -r requirements.txt
```

---

### 3. Download the external PARM model (required)

CRE-seq uses PARM (Promoter Activity Regulatory Model) as a surrogate model for predicting regulatory activity.  
Due to licensing constraints, pretrained PARM models are **not included** in this repository.

Download PARM and pretrained models from:

https://github.com/vansteensellab/PARM

After downloading, locate the pretrained model directory, for example:

```text
PARM/pre_trained_models/K562/
```

---

### 4. Set the PARM model path

Set an environment variable pointing to the pretrained PARM model directory:

```bash
export PARM_MODEL_DIR=/path/to/PARM/pre_trained_models/K562
```

(Replace the path above with your actual local path.)

---

### 5. Run the main CRE-seq experiment

```bash
python scripts/run_experiment.py
```

This command runs **two genetic algorithm optimizations** with identical settings:

- A **baseline** run (no grammar penalties)  
- A **penalty-aware** run (with motif, syntax, and DNA shape constraints)

The two runs differ **only** in whether penalties are applied during fitness evaluation.

---

### 6. Output files and directory structure

After the run finishes, results are written to:

```text
results/main_experiment/
├── baseline/
│   ├── history.csv
│   └── final_best.fa
└── with_penalties/
    ├── history.csv
    └── final_best.fa
```

---

### 7. Output file descriptions

- `history.csv`  
  Per-generation optimization statistics (e.g., best fitness per generation), used to compare convergence behavior between baseline and penalty-aware runs.

- `final_best.fa`  
  FASTA file containing the highest-scoring sequence(s) obtained at the end of optimization, used for downstream analysis.

---

This setup is sufficient to fully reproduce the main CRE-seq experiment.
