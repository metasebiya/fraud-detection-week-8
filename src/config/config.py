# src/config/config.py

import os

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATHS = {
    "fraud_data": PROJECT_ROOT / "data" / "raw" / "Fraud_Data.csv",
    "ip_to_country": PROJECT_ROOT / "data" / "raw" / "IpAddress_to_Country.csv",
    "creditcard_data": PROJECT_ROOT / "data" / "raw" / "creditcard.csv",
}
CONFIG = {
    "data_paths": DATA_PATHS,
    "test_size": 0.2,
    "random_state": 42,
    "missing_value_strategy": "drop",
    "imbalance_strategy": "smote"
}