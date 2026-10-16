# Financial Fraud Detection Using Machine Learning

## Overview

This repository contains two machine learning experiments for detecting fraudulent financial transactions using the PaySim synthetic mobile-money transaction dataset.

The project explores feature engineering, data preprocessing, class-imbalance handling, model training, and evaluation.

### Experiments

**1. SMOTE Experiment**
- Compares model performance before and after applying SMOTE.
- Investigates the impact of oversampling the minority fraud class.
- Evaluates classification performance using relevant metrics.

**2. Undersampling Experiment**
- Applies feature engineering and preprocessing.
- Uses random undersampling on the training set.
- Evaluates Logistic Regression, Random Forest, and Decision Tree models.
- Keeps the test set at its original class distribution.

## Repository Structure

```text
financial-fraud-detection/
├── README.md
├── requirements.txt
├── .gitignore
├── smote-experiment/
│   ├── README.md
│   ├── before_smote.ipynb
│   ├── after_smote.ipynb
│   └── preprocess.py
└── undersampling-experiment/
    ├── README.md
    ├── config.py
    ├── export_preprocessed_csv.py
    ├── src/
    │   └── preprocessing.py
    ├── notebooks/
    │   ├── logistic_regrassion.ipynb
    │   └── RF&DT.ipynb
    └── processed_data/
        ├── metadata.json
        ├── feature_names.json
        └── feature_engineering_params.json
```

## Dataset

The experiments use PaySim, a synthetic dataset representing mobile-money transactions.

The original dataset is not included in this repository. Obtain it separately and configure the data paths before running the notebooks or preprocessing scripts.

## Selected Results: Undersampling Experiment

The following results were recorded on the held-out test set containing 1,272,524 transactions, including 1,643 fraudulent transactions.

| Model configuration | Fraud Precision | Fraud Recall | Fraud F1-score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9951 | 0.9976 | 0.9964 | 0.999855 | 0.998259 |
| Random Forest (`max_depth=20`, 200 trees) | 0.9868 | 0.9976 | 0.9921 | 0.999374 | 0.998697 |
| Decision Tree (regularized) | 0.9377 | 0.9988 | 0.9673 | 0.999389 | 0.996886 |

These are selected experimental results, not a guarantee of performance on real financial transactions.

## Technologies

- Python
- Pandas and NumPy
- Scikit-learn
- Imbalanced-learn
- Matplotlib and Seaborn
- Jupyter Notebook

## Installation

Create a virtual environment and install the required dependencies:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

The virtual environment is local and should not be committed to GitHub.

## Reproducibility

1. Obtain the PaySim dataset separately.
2. Configure the paths required by the relevant experiment.
3. Install the dependencies listed in `requirements.txt`.
4. Follow the instructions in the README inside the experiment directory.
5. Run the notebooks or preprocessing scripts as appropriate.

## Important Notes

- SMOTE and undersampling are investigated in separate experiments.
- Resampling should be applied to training data only; the test set should remain untouched.
- Cross-validation results may be optimistic if resampling is performed before creating the cross-validation folds.
- Feature engineering and preprocessing must be checked for data leakage.
- PaySim is synthetic, and results on this dataset may not generalize to real-world financial transactions.

## Disclaimer

This project is intended for educational and experimental purposes. It is not a production-ready fraud detection system.
