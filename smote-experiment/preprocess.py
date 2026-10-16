from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


RANDOM_STATE = 42


def main():
    outdir = Path("artifacts")
    outdir.mkdir(parents=True, exist_ok=True)

    print("Reading dataset...")
    df = pd.read_csv("PS_20174392719_1491204439457_log.csv")

    print("\n" + "=" * 60)
    print("GENERAL OVERVIEW")
    print("=" * 60)

    print("\nOriginal shape:")
    print(df.shape)

    print("\nRows and columns:")
    print("Rows   :", df.shape[0])
    print("Columns:", df.shape[1])

    print("\nFirst 10 rows:")
    print(df.head(10))

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nUnique values count for each column:")
    print(df.nunique())

    print("\nUnique values in 'type':")
    print(df["type"].unique())

    print("\nNumber of unique values in 'type':")
    print(df["type"].nunique())

    print("\nMissing values per column:")
    print(df.isnull().sum())

    print("\nDuplicated rows:")
    print(df.duplicated().sum())

    print("\nDescriptive statistics:")
    print(df.describe())

    print("\nFraud distribution before preprocessing:")
    print(df["isFraud"].value_counts())
    print(df["isFraud"].value_counts(normalize=True) * 100)

    print("\nTransaction type distribution:")
    print(df["type"].value_counts())

    print("\nFraud by transaction type:")
    print(pd.crosstab(df["type"], df["isFraud"]))

    # Drop columns removed by the paper
    drop_cols = ["step", "nameOrig", "nameDest", "isFlaggedFraud"]
    existing_drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=existing_drop_cols)

    # Keep only final six predictors + target
    feature_cols = [
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
    ]
    target_col = "isFraud"

    required_cols = feature_cols + [target_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[required_cols].copy()

    print("\nSelected final columns:")
    print(df.columns.tolist())

    # Encode type
    encoder = LabelEncoder()
    df["type"] = encoder.fit_transform(df["type"].astype(str))

    # Save encoding map
    type_mapping = pd.DataFrame({
        "original_type": encoder.classes_,
        "encoded_type": range(len(encoder.classes_))
    })
    type_mapping.to_csv(outdir / "type_mapping.csv", index=False)

    X = df[feature_cols]
    y = df[target_col].astype(int)

    print("\nFinal feature shape:", X.shape)
    print("\nFraud distribution after preprocessing:")
    print(y.value_counts())
    print(y.value_counts(normalize=True) * 100)

    # 90:10 stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.10,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print("\n" + "=" * 60)
    print("TRAIN / TEST SPLIT")
    print("=" * 60)

    print("Train shape:", X_train.shape)
    print("Test shape :", X_test.shape)

    print("\ny_train distribution:")
    print(y_train.value_counts())
    print(y_train.value_counts(normalize=True) * 100)

    print("\ny_test distribution:")
    print(y_test.value_counts())
    print(y_test.value_counts(normalize=True) * 100)

    # Save as CSV instead of joblib
    train_df = X_train.copy()
    train_df[target_col] = y_train.values

    test_df = X_test.copy()
    test_df[target_col] = y_test.values

    train_df.to_csv(outdir / "train_processed.csv", index=False)
    test_df.to_csv(outdir / "test_processed.csv", index=False)

    print(f"\nSaved CSV files to: {outdir.resolve()}")
    print("Files created:")
    print("- train_processed.csv")
    print("- test_processed.csv")
    print("- type_mapping.csv")


if __name__ == "__main__":
    main()


#     (.venv) PS C:\Users\Lenovo\Desktop\fraud> & c:\Users\Lenovo\Desktop\fraud\.venv\Scripts\python.exe c:/Users/Lenovo/Desktop/fraud/preprocess.py
# Reading dataset...

# ============================================================
# GENERAL OVERVIEW
# ============================================================

# Original shape:
# (6362620, 11)

# Rows and columns:
# Rows   : 6362620
# Columns: 11

# First 10 rows:
#    step      type    amount     nameOrig  oldbalanceOrg  ...     nameDest oldbalanceDest  newbalanceDest  isFraud  isFlaggedFraud
# 0     1   PAYMENT   9839.64  C1231006815      170136.00  ...  M1979787155            0.0            0.00        0               0
# 1     1   PAYMENT   1864.28  C1666544295       21249.00  ...  M2044282225            0.0            0.00        0               0
# 2     1  TRANSFER    181.00  C1305486145         181.00  ...   C553264065            0.0            0.00        1               0
# 3     1  CASH_OUT    181.00   C840083671         181.00  ...    C38997010        21182.0            0.00        1               0
# 4     1   PAYMENT  11668.14  C2048537720       41554.00  ...  M1230701703            0.0            0.00        0               0
# 5     1   PAYMENT   7817.71    C90045638       53860.00  ...   M573487274            0.0            0.00        0               0
# 6     1   PAYMENT   7107.77   C154988899      183195.00  ...   M408069119            0.0            0.00        0               0
# 7     1   PAYMENT   7861.64  C1912850431      176087.23  ...   M633326333            0.0            0.00        0               0
# 8     1   PAYMENT   4024.36  C1265012928        2671.00  ...  M1176932104            0.0            0.00        0               0
# 9     1     DEBIT   5337.77   C712410124       41720.00  ...   C195600860        41898.0        40348.79        0               0

# [10 rows x 11 columns]

# Column names:
# ['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest', 'isFraud', 'isFlaggedFraud']

# Data types:
# step                int64
# type               object
# amount            float64
# nameOrig           object
# oldbalanceOrg     float64
# newbalanceOrig    float64
# nameDest           object
# oldbalanceDest    float64
# newbalanceDest    float64
# isFraud             int64
# isFlaggedFraud      int64
# dtype: object

# Unique values count for each column:
# step                  743
# type                    5
# amount            5316900
# nameOrig          6353307
# oldbalanceOrg     1845844
# newbalanceOrig    2682586
# nameDest          2722362
# oldbalanceDest    3614697
# newbalanceDest    3555499
# isFraud                 2
# isFlaggedFraud          2
# dtype: int64

# Unique values in 'type':
# ['PAYMENT' 'TRANSFER' 'CASH_OUT' 'DEBIT' 'CASH_IN']

# Number of unique values in 'type':
# 5

# Missing values per column:
# step              0
# type              0
# amount            0
# nameOrig          0
# oldbalanceOrg     0
# newbalanceOrig    0
# nameDest          0
# oldbalanceDest    0
# newbalanceDest    0
# isFraud           0
# isFlaggedFraud    0
# dtype: int64

# Duplicated rows:
# 0

# Descriptive statistics:
#                step        amount  oldbalanceOrg  newbalanceOrig  oldbalanceDest  newbalanceDest       isFraud  isFlaggedFraud
# count  6.362620e+06  6.362620e+06   6.362620e+06    6.362620e+06    6.362620e+06    6.362620e+06  6.362620e+06    6.362620e+06
# mean   2.433972e+02  1.798619e+05   8.338831e+05    8.551137e+05    1.100702e+06    1.224996e+06  1.290820e-03    2.514687e-06
# std    1.423320e+02  6.038582e+05   2.888243e+06    2.924049e+06    3.399180e+06    3.674129e+06  3.590480e-02    1.585775e-03
# min    1.000000e+00  0.000000e+00   0.000000e+00    0.000000e+00    0.000000e+00    0.000000e+00  0.000000e+00    0.000000e+00
# 25%    1.560000e+02  1.338957e+04   0.000000e+00    0.000000e+00    0.000000e+00    0.000000e+00  0.000000e+00    0.000000e+00
# 50%    2.390000e+02  7.487194e+04   1.420800e+04    0.000000e+00    1.327057e+05    2.146614e+05  0.000000e+00    0.000000e+00
# 75%    3.350000e+02  2.087215e+05   1.073152e+05    1.442584e+05    9.430367e+05    1.111909e+06  0.000000e+00    0.000000e+00
# max    7.430000e+02  9.244552e+07   5.958504e+07    4.958504e+07    3.560159e+08    3.561793e+08  1.000000e+00    1.000000e+00

# Fraud distribution before preprocessing:
# isFraud
# 0    6354407
# 1       8213
# Name: count, dtype: int64
# isFraud
# 0    99.870918
# 1     0.129082
# Name: proportion, dtype: float64

# Transaction type distribution:
# type
# CASH_OUT    2237500
# PAYMENT     2151495
# CASH_IN     1399284
# TRANSFER     532909
# DEBIT         41432
# Name: count, dtype: int64

# Fraud by transaction type:
# isFraud         0     1
# type                   
# CASH_IN   1399284     0
# CASH_OUT  2233384  4116
# DEBIT       41432     0
# PAYMENT   2151495     0
# TRANSFER   528812  4097

# Selected final columns:
# ['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'isFraud']

# Final feature shape: (6362620, 6)

# Fraud distribution after preprocessing:
# isFraud
# 0    6354407
# 1       8213
# Name: count, dtype: int64
# isFraud
# 0    99.870918
# 1     0.129082
# Name: proportion, dtype: float64

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================
# Train shape: (5726358, 6)
# Test shape : (636262, 6)

# y_train distribution:
# isFraud
# 0    5718966
# 1       7392
# Name: count, dtype: int64
# isFraud
# 0    99.870913
# 1     0.129087
# Name: proportion, dtype: float64

# y_test distribution:
# isFraud
# 0    635441
# 1       821
# Name: count, dtype: int64
# isFraud
# 0    99.870965
# 1     0.129035
# Name: proportion, dtype: float64

# Saved CSV files to: C:\Users\Lenovo\Desktop\fraud\artifacts
# Files created:
# - train_processed.csv
# - test_processed.csv
# - type_mapping.csv