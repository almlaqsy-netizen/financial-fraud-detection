# Fraud Detection Before and After SMOTE

## Overview

This experiment investigates the impact of Synthetic Minority Over-sampling Technique (SMOTE) on machine learning-based financial fraud detection using the PaySim dataset.

The experiment compares model behavior before and after applying SMOTE to address class imbalance.

## Objectives

- Examine the impact of class imbalance on fraud detection.
- Compare classification performance before and after SMOTE.
- Evaluate fraud precision, recall, F1-score, and confusion matrices.
- Study the trade-off between detecting fraudulent transactions and generating false positives.

## Notebooks

### `before_smote.ipynb`

Contains the baseline experiment before applying SMOTE.

### `after_smote.ipynb`

Contains the experiment after applying SMOTE.

## Evaluation

The notebooks investigate model performance using classification metrics and confusion matrices. Refer to the notebook outputs for the actual configurations and recorded results.

## Methodological Note

SMOTE should be applied only to the training data, never to the held-out test set.

If SMOTE is applied before cross-validation folds are created, information from synthetic samples can leak across the folds and make cross-validation results overly optimistic. For a more reliable evaluation, apply SMOTE separately within each training fold, for example through an imbalanced-learn pipeline.

## Dataset

The experiment uses the PaySim synthetic mobile-money transaction dataset. The original dataset is not included in this repository.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn
- Matplotlib
- Jupyter Notebook

## Disclaimer

This experiment is for educational purposes. Results on a synthetic dataset do not guarantee equivalent performance on real-world financial transactions.
