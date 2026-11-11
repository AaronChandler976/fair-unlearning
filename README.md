# Efficient and Fair Machine Unlearning: Newton-Based Data Removal, Demographic Fairness, and Multi-Dataset Evaluation

An implementation and experimental study of **Fair Machine Unlearning**, inspired by the paper *Fair Machine Unlearning: Data Removal while Mitigating Disparities* (AISTATS 2024).

This project investigates how machine learning models can remove the influence of selected training data while maintaining predictive performance and monitoring fairness across demographic groups.

## Project Overview

Machine unlearning aims to remove the influence of specified training samples without retraining a model from scratch.

This project implements a Newton-based unlearning approach for fairness-aware logistic regression and compares its results against complete retraining on the remaining dataset.

We evaluate:

- Predictive performance (Accuracy)
- Fairness (Average Equalized Odds Difference — AEOD)
- Parameter distance between Newton unlearning and retraining
- Runtime and computational efficiency
- Effects of random and demographic-targeted data deletion

Experiments are conducted across three datasets: Adult Census Income, COMPAS Recidivism, and HSLS:09.

## Methodology

### 1. Fair Logistic Regression Objective

The training objective combines binary cross-entropy, a fairness regularization term, and L2 regularization:

\[
L(\theta,D)=L_{\mathrm{BCE}}+\gamma L_{\mathrm{fair}}+\frac{\lambda}{2}\|\theta\|^2
\]

The pairwise fairness penalty measures differences between protected groups, considering samples with matching ground-truth labels:

\[
L_{\mathrm{fair}}=(v^T\theta)^2
\]

Where:

- \(L_{\mathrm{BCE}}\): Binary cross-entropy loss.
- \(L_{\mathrm{fair}}\): Pairwise fairness penalty.
- \(\gamma\): Fairness regularization strength.
- \(\lambda\): L2 regularization coefficient.
- \(v\): Group-based fairness vector.

### 2. Newton-Based Machine Unlearning

After training the original model, selected training records are removed.

The gradient and Hessian of the objective are evaluated on the remaining dataset:

\[
g'=\nabla L(\theta_D^*,D')
\]

\[
H'=\nabla^2 L(\theta_D^*,D')
\]

The Newton unlearning update is:

\[
\theta_{\mathrm{unlearned}}
=
\theta_D^*-(H'+\alpha I)^{-1}g'
\]

Here, \(\alpha\) is a small damping parameter used for numerical stability.

The implementation uses `torch.linalg.solve()` rather than explicitly calculating the matrix inverse.

### 3. Retraining Baseline

For every deletion experiment, a separate model is trained from scratch using only the remaining training records.

This retrained model serves as the reference for evaluating the Newton unlearning approximation.

### 4. Evaluation Metrics

| Metric | Description |
|---|---|
| Accuracy | Measures predictive correctness. |
| AEOD | Measures average differences in group-specific true-positive and false-positive rates. |
| Parameter Distance | L2 distance between Newton-unlearned and retrained model parameters. |
| Runtime | Compares Newton unlearning and full retraining execution time. |

Average Equalized Odds Difference is calculated as:

\[
AEOD=\frac{1}{2}\left(
|TPR_a-TPR_b|+|FPR_a-FPR_b|
\right)
\]

Lower AEOD indicates smaller differences in error rates between the evaluated demographic groups.

## Datasets

### Adult Census Income

- **Task:** Predict whether annual income exceeds $50,000.
- **Protected attribute:** Sex (female and male).
- **Features:** Demographic, educational, and employment-related variables.
- **Training records:** 30,162.
- **Test records:** 15,060.
- **Model input dimensions:** 102.

### COMPAS Recidivism

- **Task:** Predict two-year recidivism.
- **Protected attribute:** Race (Caucasian and African-American).
- **Features:** Age, prior offenses, juvenile offense counts, sex, and charge degree.
- **Training records:** 4,222.
- **Test records:** 1,056.
- **Model input dimensions:** 9.

### HSLS:09 (High School Longitudinal Study of 2009)

- **Source:** National Center for Education Statistics (NCES).
- **Task:** Predict whether a student had ever attended college by February 2016.
- **Protected attribute:** Sex (male and female).
- **Features:** High-school GPA and family income.
- **Usable records:** 17,319.
- **Training records:** 13,855.
- **Test records:** 3,464.
- **Model input dimensions:** 16.

HSLS preprocessing excludes records with missing sex or unknown outcomes, as well as imputed target outcomes in the primary experiment. Numerical and categorical preprocessing is fitted on the training split to avoid data leakage.

The HSLS experiment is an unweighted predictive evaluation, not a survey-weighted population estimate.

## Experiments

Two deletion strategies are evaluated across all three datasets.

### Random Data Deletion

Randomly remove 1%, 5%, and 10% of the training records.

| Dataset | Deletion Rates | Status |
|---|---|---|
| Adult | 1%, 5%, 10% | Completed |
| COMPAS | 1%, 5%, 10% | Completed |
| HSLS:09 | 1%, 5%, 10% | Completed |

### Demographic-Targeted Data Deletion

Remove selected percentages of records within each protected group.

| Dataset | Protected Groups | Deletion Rates | Status |
|---|---|---|---|
| Adult | Female / Male | 1%, 5%, 10% | Completed |
| COMPAS | Caucasian / African-American | 1%, 5%, 10% | Completed |
| HSLS:09 | Male / Female | 1%, 5%, 10% | Completed |

Deletion percentages are calculated within each selected demographic group, not across the entire training dataset.

In total, the project evaluates 27 deletion scenarios: nine random-deletion scenarios and eighteen demographic-targeted scenarios.

## Experimental Results

The following results summarize the recorded experiments. Accuracy, AEOD, and parameter distance are means across deletion scenarios. Runtime and speedup are reported using the recorded median measurements.

### Adult Census Income

Original baseline:

- Accuracy: 84.46%
- AEOD: 0.0602

| Strategy | Newton Accuracy | Retrained Accuracy | Newton AEOD | Retrained AEOD | Parameter Distance |
|---|---:|---:|---:|---:|---:|
| Random deletion | 84.42% | 84.42% | 0.0600 | 0.0601 | 0.125457 |
| Group deletion | 84.44% | 84.43% | 0.0593 | 0.0593 | 0.105667 |

### COMPAS Recidivism

Original baseline:

- Accuracy: 65.25%
- AEOD: 0.1469

| Strategy | Newton Accuracy | Retrained Accuracy | Newton AEOD | Retrained AEOD | Parameter Distance |
|---|---:|---:|---:|---:|---:|
| Random deletion | 65.44% | 65.44% | 0.1413 | 0.1413 | 0.000134 |
| Group deletion | 65.34% | 65.34% | 0.1461 | 0.1461 | 0.000231 |

### HSLS:09

Original baseline:

- Accuracy: 80.34%
- AEOD: 0.0389

| Strategy | Newton Accuracy | Retrained Accuracy | Newton AEOD | Retrained AEOD | Parameter Distance |
|---|---:|---:|---:|---:|---:|
| Random deletion | 80.34% | 80.34% | 0.0389 | 0.0389 | 0.014933 |
| Group deletion | 80.33% | 80.33% | 0.0391 | 0.0391 | 0.018120 |

### Runtime Comparison

| Dataset | Strategy | Newton Median (s) | Retraining Median (s) | Observed Speedup |
|---|---|---:|---:|---:|
| Adult | Random | 0.6235 | 6.5463 | 10.50× |
| Adult | Group | 0.6813 | 5.9842 | 8.40× |
| COMPAS | Random | 0.0225 | 0.0731 | 4.17× |
| COMPAS | Group | 0.0316 | 0.1472 | 4.96× |
| HSLS | Random | 0.0812 | 1.0815 | 16.11× |
| HSLS | Group | 0.0634 | 0.7314 | 12.21× |

### Preliminary Observations

1. Newton unlearning closely approximates full retraining in predictive accuracy and fairness metrics across the evaluated datasets.
2. Newton unlearning achieved observed speedups ranging from approximately 4.17× to 16.11× over retraining.
3. Parameter distance varies across datasets and generally increases with larger deletion rates in the individual experiments.
4. Fairness outcomes depend on the dataset, demographic group, and deletion percentage.
5. Demographic-targeted deletion does not consistently improve fairness.
6. Newton and retrained models frequently produce identical test metrics to the reported precision, despite having different parameter values.

These results demonstrate empirical similarity to retraining under the evaluated conditions. They do not establish a formal guarantee of certified data removal.

## Visualizations

Ten experimental plots have been generated.

### Random Deletion Experiments

- `accuracy_vs_deletion.png`
- `aeod_vs_deletion.png`
- `runtime_comparison.png`
- `parameter_distance.png`

### Demographic-Targeted Deletion Experiments

**Adult:**
- `adult_group_accuracy.png`
- `adult_group_aeod.png`

**COMPAS:**
- `compas_group_accuracy.png`
- `compas_group_aeod.png`

**HSLS:09:**
- `hsls_group_accuracy.png`
- `hsls_group_aeod.png`

All visualizations are stored in:

`experiments/results/plots/`

## Repository Structure

```text
fair-unlearning/
│
├── data/
│   └── raw/
│       ├── adult/
│       ├── compas/
│       └── hsls/
│
├── src/
│   ├── data/
│   │   ├── adult.py
│   │   ├── compas.py
│   │   └── hsls.py
│   │
│   ├── objectives/
│   │   ├── fair_loss.py
│   │   ├── derivatives.py
│   │   └── train.py
│   │
│   └── unlearning/
│       └── newton.py
│
├── tests/
│   ├── test_fair_loss.py
│   ├── test_derivatives.py
│   ├── test_unlearning.py
│   ├── test_retraining_comparison.py
│   ├── test_adult.py
│   ├── test_adult_training.py
│   ├── test_adult_unlearning.py
│   ├── test_compas.py
│   └── test_hsls_training.py
│
├── experiments/
│   ├── run_adult_deletion_rates.py
│   ├── run_adult_group_deletion.py
│   ├── run_compas_deletion_rates.py
│   ├── run_compas_group_deletion.py
│   ├── run_hsls_deletion_rates.py
│   ├── run_hsls_group_deletion.py
│   ├── plot_results.py
│   │
│   └── results/
│       ├── adult_deletion_rates.csv
│       ├── adult_group_deletion.csv
│       ├── compas_deletion_rates.csv
│       ├── compas_group_deletion.csv
│       ├── hsls_deletion_rates.csv
│       ├── hsls_group_deletion.csv
│       │
│       └── plots/
│           ├── accuracy_vs_deletion.png
│           ├── aeod_vs_deletion.png
│           ├── runtime_comparison.png
│           ├── parameter_distance.png
│           ├── adult_group_accuracy.png
│           ├── adult_group_aeod.png
│           ├── compas_group_accuracy.png
│           ├── compas_group_aeod.png
│           ├── hsls_group_accuracy.png
│           └── hsls_group_aeod.png
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Running the Experiments

### Installation

Clone the repository:

```bash
git clone https://github.com/AhmedSabz/fair-unlearning.git
cd fair-unlearning
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Run Unit Tests

```bash
python -m tests.test_fair_loss
python -m tests.test_derivatives
python -m tests.test_unlearning
python -m tests.test_retraining_comparison
```

### Run Adult Experiments

```bash
python -m tests.test_adult
python -m tests.test_adult_training
python -m experiments.run_adult_deletion_rates
python -m experiments.run_adult_group_deletion
```

### Run COMPAS Experiments

```bash
python -m tests.test_compas
python -m experiments.run_compas_deletion_rates
python -m experiments.run_compas_group_deletion
```

### Run HSLS Experiments

```bash
python -m tests.test_hsls_training
python -m experiments.run_hsls_deletion_rates
python -m experiments.run_hsls_group_deletion
```

Download the datasets from their respective official sources and place them in the corresponding `data/raw/` directories before running the experiments.

### Generate Visualizations

```bash
python -m experiments.plot_results
```

Results are saved in `experiments/results/`, and visualizations are saved in `experiments/results/plots/`.

## Technologies

- Python
- PyTorch
- NumPy
- pandas
- scikit-learn
- Matplotlib
- Automatic Differentiation
- Newton Optimization
- Logistic Regression
- Algorithmic Fairness
- Machine Unlearning

## Current Project Status

| Component | Status |
|---|---|
| Fairness objective | Completed |
| Gradient and Hessian computation | Completed |
| Newton unlearning implementation | Completed |
| Retraining baseline | Completed |
| Adult dataset experiments | Completed |
| COMPAS dataset experiments | Completed |
| HSLS:09 dataset integration | Completed |
| Random deletion experiments | Completed |
| Demographic-targeted deletion experiments | Completed |
| Cross-dataset evaluation | Completed |
| Ten experimental visualizations | Completed |
| Consolidated experimental results | Completed |
| Repeated-seed statistical evaluation | Future work |
| Final research report and presentation | In progress |

## Limitations and Future Work

Current limitations:

- The implementation uses a single Newton update, which approximates the retrained optimum.
- Experiments use specific train/test splits and deletion seeds; results may vary with different random seeds.
- Runtime measurements depend on hardware, system load, and implementation details.
- HSLS results are unweighted predictive evaluations rather than survey-weighted population estimates.
- Dataset preprocessing and experimental protocols may differ from those in the original research paper.
- Empirical similarity to retraining does not establish formal certified deletion or privacy guarantees.

Potential extensions:

1. Evaluate multiple deletion seeds and report confidence intervals.
2. Repeat runtime measurements and report robust timing statistics.
3. Investigate numerical stability and approximation error at larger deletion rates.
4. Evaluate additional protected attributes and intersectional demographic groups.
5. Investigate objective perturbation and formal privacy/unlearning guarantees.
6. Extend the method to nonlinear models and larger datasets.

## References

Oesterling et al. (2024). *Fair Machine Unlearning: Data Removal while Mitigating Disparities*. Proceedings of AISTATS 2024.

National Center for Education Statistics (NCES). *High School Longitudinal Study of 2009 (HSLS:09)*. https://nces.ed.gov/surveys/hsls09/

---

**Author:** Ahmed Sabzwari

**Research Focus:** Machine Learning Systems · Algorithmic Fairness · Optimization · Responsible AI

**Disclaimer:** This repository is an independent experimental implementation inspired by the referenced research. Dataset preprocessing and experimental protocols may differ from the original paper. Results should be interpreted as empirical findings rather than proof of certified machine unlearning.