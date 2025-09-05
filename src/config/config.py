import os
from pathlib import Path
from dotenv import load_dotenv
from collections import OrderedDict

# Load environment variables from .env
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# def get_loggable_params():
#     """Return a flattened dict of config values safe to log to MLflow."""
#     params = {}
#     for k, v in CONFIG.items():
#         if k in ("mlflow_username", "mlflow_password"):
#             continue
#         if isinstance(v, dict):
#             for subk, subv in v.items():
#                 params[f"{k}.{subk}"] = str(subv)
#         else:
#             params[k] = str(v)
#     return params


def get_loggable_params():
    params = OrderedDict()
    for k, v in CONFIG.items():
        if k in ("mlflow_username", "mlflow_password"):
            continue
        if isinstance(v, dict):
            for subk, subv in v.items():
                params[f"{k}.{subk}"] = str(subv)
        else:
            params[k] = str(v)
    return params

def require_env(key: str, default=None):
    value = os.getenv(key, default)
    if value is None:
        raise EnvironmentError(f"Missing required environment variable: {key}")
    return value

def resolve_path(path_str):
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path

DATA_PATHS = {
    "fraud_data": resolve_path(require_env("FRAUD_DATA_PATH")),
    "ip_to_country": resolve_path(require_env("IP_TO_COUNTRY_PATH")),
    "creditcard_data": resolve_path(require_env("CREDITCARD_DATA_PATH")),
}

CONFIG = {
    # MLflow / DagsHub settings
    "mlflow_uri": require_env("MLFLOW_TRACKING_URI"),
    "experiment_name": require_env("MLFLOW_EXPERIMENT_NAME"),
    "run_name": require_env("MLFLOW_RUN_NAME"),

    # Auth (now included so pipeline can read them)
    "mlflow_username": require_env("MLFLOW_TRACKING_USERNAME"),
    "mlflow_password": require_env("MLFLOW_TRACKING_PASSWORD"),

    #Model Name
    "model_name": require_env("MODEL_NAME", "fraud-detection-model"),

    #Hugging Face
    "hf_repo_id": require_env("HF_REPO_ID"),  # e.g., "username/fraud-detection-model"

    # Output directory
    "output_dir": require_env("OUTPUT_DIR", "results"),
    "artifacts_dir": require_env("ARTIFACTS_DIR", "artifacts"),
    # Data paths
    "data_paths": DATA_PATHS,

    # Pipeline parameters
    "test_size": float(require_env("TEST_SIZE", 0.2)),
    "random_state": int(require_env("RANDOM_STATE", 42)),
    "missing_value_strategy": require_env("MISSING_VALUE_STRATEGY", "drop"),
    "imbalance_strategy": require_env("IMBALANCE_STRATEGY", "smote"),
}
