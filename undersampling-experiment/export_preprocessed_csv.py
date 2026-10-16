#!/usr/bin/env python3
"""
تنفيذ ال preprocessssssssssssssss
"""
import argparse
import json
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATA_FILE, RANDOM_SEED, TEST_SIZE
from data_loader import load_data
from preprocessing import (
    engineer_features,
    encode_categorical,
    prepare_features,
    scale_features,
    handle_imbalance,
)


def export_preprocessed_csv(
    input_path=DATA_FILE,
    output_dir=None,
    sample_size=None,
    test_size=TEST_SIZE,
    imbalance_method="undersample",
    sampling_strategy=0.1,
):
    """Run preprocessing and save train.csv/test.csv plus metadata."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if output_dir is None:
        output_dir = os.path.join(base_dir, "processed_data")
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("EXPORT PREPROCESSED CSV FILES")
    print("=" * 70)
    print(f"Input: {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Sample size: {sample_size if sample_size else 'full dataset'}")
    print(f"Test size: {test_size}")
    print(f"Imbalance method: {imbalance_method}")

    df = load_data(filepath=input_path, sample_size=sample_size)

    # 1) Feature engineering
    df = engineer_features(df)

    # 2) Encode categorical variables
    df, label_encoder = encode_categorical(df)

    # 3) Separate features and target
    X, y, feature_names = prepare_features(df)

    # 4) Train/test split with stratification
    print(f"\nSplitting data with stratify=y...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    print(f"Train before resampling: {X_train.shape}")
    print(f"Test: {X_test.shape}")

    # 5) Scale after split: fit only on train, transform train/test
    print("\nScaling features using StandardScaler...")
    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_test, feature_names
    )

    # # 6) Handle imbalance only on training set
    # if imbalance_method and imbalance_method.lower() != "none":
    #     X_train_final, y_train_final = handle_imbalance(
    #         X_train_scaled,
    #         y_train,
    #         method=imbalance_method,
    #         sampling_strategy=sampling_strategy,
    #     )
    # else:
    #     X_train_final, y_train_final = X_train_scaled, y_train
    
    # 6) Handle imbalance only on training set
    if imbalance_method and imbalance_method.lower() != "none":
        if imbalance_method.lower() != "undersample":
            raise ValueError(
                f"Unsupported imbalance method: {imbalance_method}"
            )

        X_train_final, y_train_final = handle_imbalance(
            X_train_scaled,
            y_train,
            sampling_strategy=sampling_strategy,
        )
    else:
        X_train_final, y_train_final = X_train_scaled, y_train


    # 7) Combine X and y into exportable CSV files
    train_df = X_train_final.copy()
    train_df["isFraud"] = np.asarray(y_train_final).astype(int)

    test_df = X_test_scaled.copy()
    test_df["isFraud"] = np.asarray(y_test).astype(int)

    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")
    feature_path = os.path.join(output_dir, "feature_names.json")
    metadata_path = os.path.join(output_dir, "metadata.json")
    scaler_path = os.path.join(output_dir, "scaler.pkl")
    encoder_path = os.path.join(output_dir, "label_encoder.pkl")

    print("\nSaving files...")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    with open(feature_path, "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    with open(encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_path": input_path,
        "sample_size": sample_size if sample_size else "full",
        "test_size": test_size,
        "random_seed": RANDOM_SEED,
        "imbalance_method": imbalance_method,
        "sampling_strategy": sampling_strategy,
        "train_shape": list(train_df.shape),
        "test_shape": list(test_df.shape),
        "target_column": "isFraud",
        "feature_count": len(feature_names),
        "train_target_distribution": train_df["isFraud"].value_counts().to_dict(),
        "test_target_distribution": test_df["isFraud"].value_counts().to_dict(),
        "files": {
            "train_csv": "train.csv",
            "test_csv": "test.csv",
            "feature_names": "feature_names.json",
            "scaler": "scaler.pkl",
            "label_encoder": "label_encoder.pkl",
        },
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nDone.")
    print(f"Train CSV: {train_path}")
    print(f"Test CSV:  {test_path}")
    print(f"Feature names: {feature_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Scaler: {scaler_path}")
    print(f"Label encoder: {encoder_path}")
    print("\nFinal shapes:")
    print(f"  train.csv: {train_df.shape}")
    print(f"  test.csv:  {test_df.shape}")
    print("\nTarget distribution:")
    print(f"  train: {train_df['isFraud'].value_counts().to_dict()}")
    print(f"  test:  {test_df['isFraud'].value_counts().to_dict()}")

    return train_path, test_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run preprocessing and export train.csv/test.csv for Colab."
    )
    parser.add_argument(
        "--input",
        default=DATA_FILE,
        help="Path to raw Synthetic_Financial_datasets_log.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where train.csv and test.csv will be saved",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Optional number of rows for a quick test, e.g. 500000",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=TEST_SIZE,
        help="Test split size, default from config.py",
    )
    parser.add_argument(
        "--imbalance-method",
        choices=["undersample", "none"],
        default="undersample",
        help="How to handle class imbalance on training data only",
    )
    parser.add_argument(
        "--sampling-strategy",
        type=float,
        default=0.1,
        help="Sampling strategy for undersample/oversample; default 0.1",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_preprocessed_csv(
        input_path=args.input,
        output_dir=args.output_dir,
        sample_size=args.sample,
        test_size=args.test_size,
        imbalance_method=args.imbalance_method,
        sampling_strategy=args.sampling_strategy,
    )
