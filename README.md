# Efficient and Fair Machine Unlearning: Newton-Based Data Removal, Demographic Fairness, and Multi-Dataset Evaluation.

An implementation and experimental study of **Fair Machine Unlearning**, based on the paper *Fair Machine Unlearning: Data Removal while Mitigating Disparities* (AISTATS 2024).

The project investigates how machine learning models can remove the influence of selected training data while maintaining predictive performance and monitoring fairness across demographic groups.

## Project Overview

Machine unlearning aims to remove the influence of specified training samples without having to retrain a model from scratch.

This project implements a Newton-based unlearning approach for fair logistic regression and compares its results against complete retraining on the remaining dataset.

We evaluate:

- Predictive performance (Accuracy)
- Fairness (Average Equalized Odds Difference — AEOD)
- Parameter distance between Newton unlearning and retraining
- Runtime and computational efficiency
- Effects of random and demographic-targeted data deletion

## Methodology

### 1. Fair Logistic Regression Objective

The training objective combines binary cross-entropy, a fairness regularization term, and L2 regularization:

\[
L(\theta,D)=L_{BCE}+\gamma L_{fair}+\frac{\lambda}{2}\|\theta\|^2
\]

The pairwise fairness penalty measures differences between protected groups, considering samples with matching labels.

\[
L_{fair}=(v^T\theta)^2
\]

Where:
- \(L_{BCE}\): Binary cross-entropy loss.
- \(L_{fair}\): Pairwise fairness penalty.
- \(\gamma\): Fairness regularization strength.
- \(\lambda\): L2 regularization coefficient.
- \(v\): Group-based fairness vector.

### 2. Newton-Based Machine Unlearning

After training the original model, selected training records are removed.

The model is updated using the gradient and Hessian of the objective evaluated on the remaining dataset:

\[
g'=\nabla L(\theta_D^*,D')
\]

\[
H'=\nabla^2 L(\theta_D^*,D')
\]

The Newton unlearning update is:

\[
\theta_{unlearned}=\theta_D^*-(H'+\alpha I)^{-1}g'
\]

Here, \(\alpha\) is a small damping parameter used for numerical stability.

The implementation uses `torch.linalg.solve()` rather than explicitly calculating the matrix inverse.

### 3. Retraining Baseline

For every deletion experiment, a separate model is trained from scratch using only the remaining training records.

This retrained model serves as the reference for evaluating the Newton unlearning approximation.

## Datasets

### Adult Census Income

- Binary classification task: Predict whether annual income exceeds $50,000.
- Protected attribute: Sex (female and male).
- Features: Demographic, educational, and employment-related variables.
- Training records after preprocessing: 30,162.
- Test records: 15,060.
- Model input dimensions: 102.

### COMPAS Recidivism

- Binary classification task: Predict two-year recidivism.
- Protected attribute: Race (Caucasian and African-American).
- Features: Age, prior offenses, juvenile offense counts, sex, and charge degree.
- Training records after preprocessing: 4,222.
- Test records: 1,056.
- Model input dimensions: 9.

### HSLS:09 (Planned)

The High School Longitudinal Study of 2009 from NCES is the next planned dataset.

The intended experiment will investigate educational outcomes and demographic fairness using student-level data.

The public-use student dataset and codebook are being prepared. The exact target variable, protected attributes, and preprocessing pipeline have not yet been finalized.

## Experiments

### Random Data Deletion

Randomly remove 1%, 5%, and 10% of training records.

| Dataset | Deletion rates | Completed |
|---|---|---|
| Adult | 1%, 5%, 10% | Yes |
| COMPAS | 1%, 5%, 10% | Yes |
| HSLS:09 | TBD | Planned |

### Demographic-Targeted Data Deletion

Remove selected percentages of records within each protected group.

| Dataset | Protected groups | Deletion rates | Completed |
|---|---|---|---|
| Adult | Female / Male | 1%, 5%, 10% | Yes |
| COMPAS | Caucasian / African-American | 1%, 5%, 10% | Yes |
| HSLS:09 | TBD | TBD | Planned |

Deletion percentages are calculated within each selected demographic group, not across the entire training dataset.

## Preliminary Results

### Adult Census Income

Original baseline:
- Accuracy: 84.46%
- AEOD: 0.0602

Random deletion experiments:

| Deletion | Newton Accuracy | Newton AEOD | Parameter Distance |
|---|---:|---:|---:|
| 1% | 84.46% | 0.0592 | 0.0431 |
| 5% | 84.41% | 0.0605 | 0.1182 |
| 10% | 84.40% | 0.0604 | 0.2151 |

### COMPAS Recidivism

Original baseline:
- Accuracy: 65.25%
- AEOD: 0.1469

Random deletion experiments:

| Deletion | Newton Accuracy | Newton AEOD | Parameter Distance |
|---|---:|---:|---:|
| 1% | 65.53% | 0.1466 | 0.0000262 |
| 5% | 65.44% | 0.1374 | 0.0000693 |
| 10% | 65.34% | 0.1400 | 0.0003065 |

### Preliminary Observations

1. Newton unlearning produces accuracy and fairness metrics close to those of retraining across the tested deletion rates.
2. Parameter distance generally increases as more records are removed.
3. Fairness outcomes depend on the dataset, demographic group, and deletion percentage.
4. Demographic-targeted deletion does not consistently improve fairness.
5. Newton unlearning was faster than retraining in the measured experiments, although runtime varies with system load and implementation conditions.

These results are preliminary and should not be interpreted as a formal guarantee of certified data removal.

## Visualizations

Eight plots have been generated.

**Random deletion experiments:**

- `accuracy_vs_deletion.png`
- `aeod_vs_deletion.png`
- `runtime_comparison.png`
- `parameter_distance.png`

**Demographic-targeted deletion experiments:**

- `adult_group_accuracy.png`
- `adult_group_aeod.png`
- `compas_group_accuracy.png`
- `compas_group_aeod.png`

Visualizations are stored in:

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
│   │   └── compas.py
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
│   └── test_compas_training.py
│
├── experiments/
│   ├── run_adult_deletion_rates.py
│   ├── run_adult_group_deletion.py
│   ├── run_compas_deletion_rates.py
│   ├── run_compas_group_deletion.py
│   ├── plot_results.py
│   ├── plot_group_results.py
│   └── results/
│       ├── adult_deletion_rates.csv
│       ├── adult_group_deletion.csv
│       ├── compas_deletion_rates.csv
│       ├── compas_group_deletion.csv
│       └── plots/
│
├── README.md
└── .gitignore
```

## Running the Experiments

Install the required Python packages:

```bash
pip install torch pandas numpy scikit-learn matplotlib
```

Run the unit tests:

```bash
python -m tests.test_fair_loss
python -m tests.test_derivatives
python -m tests.test_unlearning
python -m tests.test_retraining_comparison
```

Run Adult experiments:

```bash
python -m tests.test_adult
python -m tests.test_adult_training
python -m experiments.run_adult_deletion_rates
python -m experiments.run_adult_group_deletion
```

Run COMPAS experiments:

```bash
python -m tests.test_compas
python -m tests.test_compas_training
python -m experiments.run_compas_deletion_rates
python -m experiments.run_compas_group_deletion
```

Generate visualizations:

```bash
python -m experiments.plot_results
python -m experiments.plot_group_results
```

## Current Project Status

| Component | Status |
|---|---|
| Fairness objective | Completed |
| Gradient and Hessian computation | Completed |
| Newton unlearning | Completed |
| Retraining baseline | Completed |
| Adult dataset experiments | Completed |
| COMPAS dataset experiments | Completed |
| Random deletion experiments | Completed |
| Demographic deletion experiments | Completed |
| Result visualizations | Completed |
| HSLS:09 dataset integration | In progress |
| Cross-dataset analysis | Pending |
| Final report and presentation | Pending |

## Future Work

1. Complete HSLS:09 public-use dataset integration.
2. Identify appropriate educational outcome and protected-group variables.
3. Run random and demographic-targeted deletion experiments on HSLS.
4. Consolidate results across all datasets.
5. Investigate numerical stability and approximation error at larger deletion rates.
6. Compare fairness and predictive performance across datasets.
7. Finalize the research report and presentation.

## References

Oesterling et al. (2024). *Fair Machine Unlearning: Data Removal while Mitigating Disparities*. Proceedings of AISTATS 2024.

NCES. *High School Longitudinal Study of 2009 (HSLS:09)*. https://nces.ed.gov/surveys/hsls09/

---

**Disclaimer:** This repository is an experimental implementation inspired by the referenced research. Dataset preprocessing and experimental protocols may differ from the original paper. Results should be interpreted as preliminary empirical findings rather than proof of certified machine unlearning.
