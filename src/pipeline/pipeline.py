import os
import json
import logging
import pandas as pd
import numpy as np
from scipy import sparse
from prefect import flow, task
import mlflow
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
from datetime import datetime

from src.processing.data_processor import Processor
from src.models.model_trainer import ModelTrainer
from src.models.model_explainer import ModelExplainer
from src.config.config import get_loggable_params, CONFIG

# --- Configure logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Build plots directory from config
PLOTS_DIR = Path(CONFIG["artifacts_dir"]) / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

class FraudDetectionPipeline:
    def __init__(self):
        self.config = CONFIG
        self.processor = Processor()
        self.trainer = ModelTrainer()
        self.explainer = ModelExplainer(output_dir=str(Path(self.config["artifacts_dir"])))

    def _save_and_log_plot(self, fig, filename):
        plot_path = PLOTS_DIR / filename
        fig.savefig(plot_path, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(str(plot_path))
        logger.info(f"📊 Saved and logged plot: {plot_path}")

    @task
    def process_data(self):
        logger.info("🚀 Starting data processing...")
        processed = self.processor.run_pipeline(self.config)
        if hasattr(self.processor, "last_plot") and self.processor.last_plot:
            self._save_and_log_plot(self.processor.last_plot, "data_processing_overview.png")
        return processed

    @task
    def train_models(self, processed):
        logger.info("📦 Training models...")
        results, models, model_dataset_map, _ = self.trainer.run_all_models(processed)
        if hasattr(self.trainer, "plots"):
            for name, fig in self.trainer.plots.items():
                self._save_and_log_plot(fig, f"training_{name}.png")
        return results, models, model_dataset_map

    @task
    def explain_and_log(self, processed, models, model_dataset_map):
        logger.info("📊 Generating model explanations...")
        for model_key, model in models.items():
            dataset_key = model_dataset_map[model_key]
            X_bg = processed[dataset_key][f"X_train_{dataset_key}_preprocessed"]
            feature_names = processed[dataset_key][f"feature_names_{dataset_key}"]

            if isinstance(feature_names, (np.ndarray, pd.Index)):
                feature_names = feature_names.tolist()
            if len(feature_names) == 1 and isinstance(feature_names[0], (list, np.ndarray)):
                feature_names = list(feature_names[0])
            feature_names = [str(f) for f in feature_names]

            if sparse.issparse(X_bg):
                X_bg = X_bg.toarray()

            if len(feature_names) != X_bg.shape[1]:
                logger.warning(
                    f"Feature name mismatch for {model_key}: "
                    f"X has {X_bg.shape[1]} cols, names={len(feature_names)}. Truncating."
                )
                feature_names = feature_names[:X_bg.shape[1]]

            X_df = pd.DataFrame(X_bg, columns=feature_names)

            summary_path, force_path, plot_figs = self.explainer.explain_model(
                model=model,
                X=X_df,
                feature_names=feature_names,
                model_name=model_key,
                dataset_name=dataset_key
            )

            mlflow.log_artifact(summary_path)
            if force_path:
                mlflow.log_artifact(force_path)
            if plot_figs:
                for name, fig in plot_figs.items():
                    self._save_and_log_plot(fig, f"explain_{model_key}_{name}.png")

    @task
    def log_metrics(self, results):
        logger.info("📈 Logging model metrics...")
        for model_name, metrics in results.items():
            mlflow.log_metrics({
                f"{model_name}_precision": metrics["precision"],
                f"{model_name}_recall": metrics["recall"],
                f"{model_name}_f1": metrics["f1_score"],
                f"{model_name}_roc_auc": metrics["roc_auc"],
                f"{model_name}_aucpr": metrics["auc_pr"]
            })

    @task
    def save_best_model_for_docker(self, results, models):
        logger.info("🏆 Selecting best model for Docker deployment...")
        best_model_key = max(results, key=lambda m: results[m]["f1_score"])
        best_model = models[best_model_key]

        # Save locally in artifacts dir
        local_model_path = Path(self.config["artifacts_dir"]) / "model.joblib"
        local_model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, local_model_path)
        logger.info(f"💾 Saved best model locally at {local_model_path}")

        # Log to MLflow for tracking
        artifact_path = "model"
        mlflow.sklearn.log_model(best_model, artifact_path=artifact_path)
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/{artifact_path}"
        logger.info(f"✅ Logged model to MLflow at {model_uri}")

        # Write pointer file
        pointer = {
            "model_key": best_model_key,
            "run_id": run_id,
            "model_uri": model_uri,
            "saved_at": datetime.utcnow().isoformat() + "Z"
        }
        pointer_path = Path(self.config["artifacts_dir"]) / "production_model.json"
        with open(pointer_path, "w") as f:
            json.dump(pointer, f, indent=2)
        logger.info(f"📌 Updated production model pointer at {pointer_path}")

    @flow(name="Fraud Detection Pipeline")
    def orchestrated_pipeline(self):
        logger.info("🔑 Setting MLflow/DagsHub authentication...")
        os.environ["MLFLOW_TRACKING_USERNAME"] = self.config["mlflow_username"]
        os.environ["MLFLOW_TRACKING_PASSWORD"] = self.config["mlflow_password"]

        mlflow.set_tracking_uri(self.config["mlflow_uri"])
        mlflow.set_experiment(self.config["experiment_name"])

        with mlflow.start_run(run_name=self.config.get("run_name", "fraud_detection_pipeline")):
            mlflow.log_params(get_loggable_params())

            processed = self.process_data()
            results, models, model_dataset_map = self.train_models(processed)
            self.explain_and_log(processed, models, model_dataset_map)
            self.log_metrics(results)
            self.save_best_model_for_docker(results, models)

        logger.info("✅ Pipeline completed successfully.")
        return results
