# Fraud Detection Using Undersampling

## Overview

This experiment evaluates machine learning models for financial fraud detection using the PaySim synthetic transaction dataset.

The workflow includes feature engineering, categorical encoding, feature scaling, random undersampling of the training data, and model evaluation on an untouched test set.

## Workflow

1. Load the PaySim dataset.
2. Engineer additional transaction and balance-related features.
3. Encode categorical transaction types.
4. Split the data into stratified training and test sets.
5. Fit the scaler on the training data and transform the test data.
6. Apply random undersampling to the training data only.
7. Train and evaluate the classification models.

## Feature Engineering

The preprocessing pipeline creates additional features, including:

- Log-transformed transaction amount.
- Differences between original and final account balances.
- Transaction amount relative to account balances.
- Indicators for zeroed or unchanged balances.
- Large-transaction and full-balance-transfer indicators.
- Time-based features derived from transaction steps.
- Transaction amount deciles.

The pipeline also encodes transaction types and scales the features.

## Training and Test Data

According to the exported experiment metadata:

| Property | Value |
|---|---:|
| Training samples after undersampling | 72,270 |
| Non-fraud training samples | 65,700 |
| Fraud training samples | 6,570 |
| Test samples | 1,272,524 |
| Non-fraud test samples | 1,270,881 |
| Fraud test samples | 1,643 |
| Number of features | 26 |
| Test split | 20% |

The undersampling strategy is `0.1`, giving a non-fraud-to-fraud ratio of approximately 10:1 in the resampled training set.

The test set retains its original class distribution.

## Models

The notebooks evaluate the following algorithms and configurations:

- Logistic Regression.
- Random Forest with different tree counts, maximum depths, and class-weight settings.
- Decision Tree with different class-weight and regularization settings.

### Selected Results

The following results are from the recorded test-set evaluations.

| Model | Fraud Precision | Fraud Recall | Fraud F1-score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9951 | 0.9976 | 0.9964 | 0.999855 | 0.998259 |
| Random Forest, 200 trees, depth 20 | 0.9868 | 0.9976 | 0.9921 | 0.999374 | 0.998697 |
| Regularized Decision Tree | 0.9377 | 0.9988 | 0.9673 | 0.999389 | 0.996886 |

These configurations were selected from the experiments; they do not represent an exhaustive hyperparameter search.

## Evaluation Metrics

- Accuracy.
- Precision and recall for fraudulent transactions.
- F1-score.
- Confusion matrix.
- ROC-AUC.
- Precision-Recall AUC (PR-AUC).
- Training time.

Precision and recall are particularly important because missing a fraudulent transaction and incorrectly flagging a legitimate transaction have different consequences.

## Files

- `config.py`: experiment configuration.
- `export_preprocessed_csv.py`: exports processed training and test datasets.
- `src/preprocessing.py`: preprocessing and feature-engineering functions.
- `notebooks/logistic_regrassion.ipynb`: Logistic Regression experiments.
- `notebooks/RF&DT.ipynb`: Random Forest and Decision Tree experiments.
- `processed_data/`: metadata and feature descriptions for the exported dataset.

The actual CSV datasets are not included in the repository.

## Reproducibility and Limitations

Obtain the PaySim dataset separately, configure the input and output paths, install the dependencies, and follow the preprocessing and notebook workflow.

The feature-engineering thresholds and binning parameters should be fitted using training data only to avoid test-distribution leakage. The preprocessing implementation should be reviewed for this before interpreting the results as a fully leakage-free evaluation.

The dataset is synthetic, and high test scores do not establish equivalent performance on real financial transactions.

## Disclaimer

This project is for educational and experimental purposes and is not a production-ready financial fraud detection system.
