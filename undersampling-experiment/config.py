"""
Configuration settings for the Financial Fraud Detection project.
"""
import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
GRAPHS_DIR = os.path.join(BASE_DIR, 'graphs')

# Data file
DATA_FILE = os.path.join(DATA_DIR, 'PS_20174392719_1491204439457_log.csv')

# Random seed for reproducibility
RANDOM_SEED = 42

# Model parameters
TEST_SIZE = 0.2
CV_FOLDS = 5

# Sampling parameters (for handling class imbalance)
UNDERSAMPLE_RATIO = 0.1  # Ratio of majority class to keep
