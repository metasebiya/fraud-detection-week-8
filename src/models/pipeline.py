import os
import sys # Import the sys module
import logging
from datetime import datetime
import joblib # For saving models (though MLflow will handle it, good to keep for general use)
import mlflow # Import MLflow
import mlflow.sklearn # For logging scikit-learn models
import mlflow.lightgbm # For logging LightGBM models
import io
from contextlib import redirect_stdout
from train import ModelTrainer


# Dynamically add the 'src' directory to sys.path
# This allows importing modules from 'src' when pipeline.py is in 'src/models/'
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir) # Insert at the beginning to prioritize local modules

# --- DIAGNOSTIC PRINTS ---
# These prints will help you verify the paths Python is using.
print(f"DEBUG: Current script directory: {current_dir}")
print(f"DEBUG: Calculated src directory to add: {src_dir}")
print(f"DEBUG: Current sys.path after modification: {sys.path}")
# --- END DIAGNOSTIC PRINTS ---

# Import our modularized components
from src.data_loader import DataLoader
from src.data_cleaner import DataCleaner
from src.data_transformer import DataTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MLOPsPipeline:
    """
    Orchestrates the entire machine learning pipeline for fraud detection,
    from data loading and preprocessing to model training and evaluation,
    with MLflow integration for tracking and registration.
    """
    def __init__(self, data_paths, output_base_dir="../data"):
        """
        Initializes the MLOpsPipeline with data paths and output directories.

        Args:
            data_paths (dict): Dictionary of raw data file paths.
                               Example: {'fraud_data': 'Fraud_Data.csv', ...}
            output_base_dir (str): Base directory for saving processed data and models.
        """
        self.data_paths = data_paths
        self.output_base_dir = output_base_dir
        self.raw_data_dir = os.path.join(output_base_dir, "raw")
        self.processed_data_dir = os.path.join(output_base_dir, "processed")
        self.models_dir = os.path.join(output_base_dir, "models")
        self.results_dir = os.path.join(output_base_dir, "results")

        # Ensure output directories exist
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        self.data_loader = DataLoader()
        self.data_cleaner = DataCleaner()
        self.data_transformer = DataTransformer()
        self.model_trainer = ModelTrainer()

        # Set MLflow tracking URI to a SQLite database for model registry support
        # This will create an 'mlruns.db' file inside your 'mlruns' directory.
        mlflow_db_path = os.path.join(self.output_base_dir, "mlruns", "mlruns.db")
        mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_path}")
        logging.info(f"MLflow tracking URI set to: {mlflow.get_tracking_uri()}")

        logging.info("MLOPsPipeline initialized.")

    def run_pipeline(self):
        """
        Executes the full fraud detection ML pipeline with MLflow tracking.
        """
        logging.info("Starting ML Pipeline execution...")
        start_time = datetime.now()

        # Set MLflow experiment name
        mlflow.set_experiment("Fraud Detection Pipeline")

        # Start an MLflow run for the entire pipeline execution
        with mlflow.start_run(run_name=f"Pipeline Run {start_time.strftime('%Y%m%d_%H%M%S')}") as run:
            mlflow.log_param("pipeline_start_time", start_time.strftime('%Y-%m-%d %H:%M:%S'))
            logging.info(f"MLflow Run ID: {run.info.run_id}")
            logging.info(f"MLflow Experiment ID: {run.info.experiment_id}")

            try:
                # --- 1. Data Loading ---
                logging.info("Step 1: Loading raw data...")
                loaded_datasets = self.data_loader.load_data(self.data_paths)
                if any(df is None for df in loaded_datasets.values()):
                    logging.error("Critical data loading error. Exiting pipeline.")
                    return None

                # --- 2. Data Cleaning ---
                logging.info("Step 2: Cleaning data...")
                cleaned_datasets = self.data_cleaner.clean_all_datasets(loaded_datasets)
                if any(df is None for df in cleaned_datasets.values()):
                    logging.error("Critical data cleaning error. Exiting pipeline.")
                    return None

                # Save cleaned raw data
                logging.info(f"Saving cleaned raw data to {self.raw_data_dir}...")
                for name, item in cleaned_datasets.items():
                    if item is not None:
                        file_name = f"{name}_cleaned.csv"
                        file_path = os.path.join(self.raw_data_dir, file_name)
                        item.to_csv(file_path, index=False)
                        logging.info(f"Saved cleaned '{name}' to {file_path}")
                    else:
                        logging.warning(f"Skipping saving for '{name}' as it was not loaded or cleaned successfully.")
                mlflow.log_param("data_cleaned", "True")


                # --- 3. Data Transformation (Feature Engineering, Scaling, Encoding, Imbalance Handling) ---
                logging.info("Step 3: Transforming data for ML...")
                prepared_data = self.data_transformer.transform_data_for_ml(
                    cleaned_datasets['fraud_data'].copy(),
                    cleaned_datasets['ip_to_country'].copy(),
                    cleaned_datasets['creditcard_data'].copy()
                )
                if prepared_data is None:
                    logging.error("Critical data transformation error. Exiting pipeline.")
                    return None
                logging.info("Data transformation complete. Prepared data for ML.")
                mlflow.log_param("data_transformed", "True")


                # --- 4. Model Training and Evaluation ---
                logging.info("Step 4: Training and evaluating models...")
                all_model_metrics, trained_models = self.model_trainer.run_all_models(prepared_data)
                logging.info("Model training and evaluation complete.")

                # --- 5. Log Models and Metrics with MLflow ---
                logging.info("Logging models and metrics to MLflow...")

                # Log Logistic Regression for Fraud Data
                with mlflow.start_run(nested=True, run_name="LR_Fraud_Data"):
                    mlflow.log_params(self.model_trainer.logistic_regression_model.get_params())
                    mlflow.log_metrics(all_model_metrics['lr_fraud'])
                    mlflow.sklearn.log_model(
                        sk_model=trained_models['lr_fraud'],
                        artifact_path="lr_fraud_model",
                        registered_model_name="LogisticRegressionFraudModel"
                    )
                    logging.info("Logistic Regression Fraud model logged to MLflow.")

                # Log LightGBM for Fraud Data
                with mlflow.start_run(nested=True, run_name="LGBM_Fraud_Data"):
                    mlflow.log_params(self.model_trainer.lgbm_model.get_params())
                    mlflow.log_metrics(all_model_metrics['lgbm_fraud'])
                    mlflow.lightgbm.log_model( # Use mlflow.lightgbm for LGBM models
                        lgb_model=trained_models['lgbm_fraud'],
                        artifact_path="lgbm_fraud_model",
                        registered_model_name="LightGBMFraudModel"
                    )
                    logging.info("LightGBM Fraud model logged to MLflow.")

                # Log Logistic Regression for Credit Card Data
                with mlflow.start_run(nested=True, run_name="LR_CreditCard_Data"):
                    mlflow.log_params(trained_models['lr_creditcard'].get_params()) # Get params from the specific instance
                    mlflow.log_metrics(all_model_metrics['lr_creditcard'])
                    mlflow.sklearn.log_model(
                        sk_model=trained_models['lr_creditcard'],
                        artifact_path="lr_creditcard_model",
                        registered_model_name="LogisticRegressionCreditCardModel"
                    )
                    logging.info("Logistic Regression Credit Card model logged to MLflow.")

                # Log LightGBM for Credit Card Data
                with mlflow.start_run(nested=True, run_name="LGBM_CreditCard_Data"):
                    mlflow.log_params(trained_models['lgbm_creditcard'].get_params()) # Get params from the specific instance
                    mlflow.log_metrics(all_model_metrics['lgbm_creditcard'])
                    mlflow.lightgbm.log_model( # Use mlflow.lightgbm for LGBM models
                        lgb_model=trained_models['lgbm_creditcard'],
                        artifact_path="lgbm_creditcard_model",
                        registered_model_name="LightGBMCreditCardModel"
                    )
                    logging.info("LightGBM Credit Card model logged to MLflow.")

                # --- 6. Save Results (to file) ---
                logging.info(f"Saving evaluation results to {self.results_dir}...")
                results_file = os.path.join(self.results_dir, f"model_evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
                with open(results_file, 'w') as f:
                    f.write(f"ML Pipeline Run: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    f.write("--- Summary of All Model Metrics ---\n")
                    for key, metrics in all_model_metrics.items():
                        f.write(f"{key}: F1-Score={metrics['f1_score']:.4f}, AUC-PR={metrics['auc_pr']:.4f}\n")
                    f.write("\n--- Justification for Best Model ---\n")
                    # Re-run justification to capture it in the log file
                    f_temp = io.StringIO()
                    with redirect_stdout(f_temp):
                        self.model_trainer._justify_best_model(all_model_metrics)
                    f.write(f_temp.getvalue())
                logging.info(f"Evaluation results saved to {results_file}")


                end_time = datetime.now()
                duration = end_time - start_time
                mlflow.log_param("pipeline_end_time", end_time.strftime('%Y-%m-%d %H:%M:%S'))
                mlflow.log_param("pipeline_duration", str(duration))
                logging.info(f"ML Pipeline completed successfully in {duration}.")
                return all_model_metrics

            except Exception as e:
                logging.exception(f"An error occurred during pipeline execution: {e}")
                mlflow.log_param("pipeline_status", "Failed")
                raise # Re-raise the exception to indicate failure
                # return None # Or return None if you want to suppress the re-raise

if __name__ == '__main__':
    # Define your dataset paths relative to where you run the script
    DATA_PATHS = {
        'fraud_data': '../data/raw/Fraud_Data.csv',
        'ip_to_country': '../data/raw/IpAddress_to_Country.csv',
        'creditcard_data': '../data/raw/creditcard.csv'
    }

    # Base output directory (e.g., a 'data' folder at the root of your project)
    OUTPUT_BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

    pipeline = MLOPsPipeline(DATA_PATHS, OUTPUT_BASE_DIR)
    results = pipeline.run_pipeline()

    if results:
        logging.info("Pipeline executed successfully. Check output directories for artifacts and MLflow UI.")
        logging.info("To view MLflow UI, run 'mlflow ui' in your terminal from the project root directory (where the 'data' folder is).")
    else:
        logging.error("Pipeline execution failed.")
