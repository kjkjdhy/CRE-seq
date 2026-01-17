CRE-seq is a DNA sequence generation framework for designing promoter-like cis-regulatory elements (CREs) with high predicted transcriptional activity. The core idea is to combine evolutionary optimization with explicit promoter grammar constraints, rather than relying solely on black-box generative models. 
The framework uses a genetic algorithm to iteratively optimize DNA sequences based on a learned activity predictor, while optionally enforcing biologically motivated grammar rules such as motif presence and relative spacing between core promoter elements. This allows CRE-seq to explore high-scoring sequence space while maintaining interpretable and controllable sequence structure. 
Cre-seq framework is very flexible. The weights of all the penalties are adjustable. In the batch_generate_and_score file, lambda_motif, lambda_syntax and lambda_shape each represent the weight of those three penalties. If you want to only use the baseline version and only wish to compare the MPRA-trained activity scorers themselves without any structural constraint, you could just set them all to zero. You could also try out different types of constraints in respective files.  
Overall, this project is intended as a mechanistic, interpretable alternative to purely data-driven and black-box CRE design approaches, and as a flexible framework for studying how promoter grammar influences sequence optimization.


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

https://github.com/vansteenselab/PARM

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
  Per-generation optimization statistics (e.g. best fitness per generation), used to compare convergence
  behavior between baseline and penalty-aware runs.

- `final_best.fa`  
  FASTA file containing the highest-scoring sequence(s) obtained at the end of optimization, used for
  downstream sequence analysis.

---

This step-by-step setup is sufficient to fully reproduce the main CRE-seq experiment reported in the paper.
