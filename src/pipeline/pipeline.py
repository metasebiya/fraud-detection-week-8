from prefect import flow, task
import mlflow
from src.processing.data_processor import Processor
from src.models.model_trainer import ModelTrainer
from src.models.model_explainer import ModelExplainer
from src.config.config import CONFIG
import pandas as pd
import numpy as np
class FraudDetectionPipeline:
    def __init__(self, config):
        self.config = config
        self.processor = Processor()
        self.trainer = ModelTrainer()
        self.explainer = ModelExplainer(output_dir="artifacts")

    from prefect import task

    @task
    def run_pipeline_task(self):
        import pandas as pd
        import numpy as np
        import mlflow

        with mlflow.start_run(run_name="fraud_detection_pipeline"):
            mlflow.log_params(self.config.get("params", {}))

            # Step 1: Data processing
            processed = Processor.run_pipeline(self.config)

            # Step 2: Model training
            results, models = self.trainer.run_all_models(processed)

            # Step 3: SHAP explanation + artifact logging
            for model_name, model in models.items():
                dataset_key = model_name.split("_")[1]

                X_train_resampled = processed[dataset_key][f"X_train_{dataset_key}_resampled"]
                feature_names = processed[dataset_key][f"feature_names_{dataset_key}"]

                # Flatten and sanitize feature names
                if isinstance(feature_names, (np.ndarray, pd.Index)):
                    feature_names = feature_names.tolist()
                if len(feature_names) == 1 and isinstance(feature_names[0], (list, np.ndarray)):
                    feature_names = list(feature_names[0])
                feature_names = [str(f) for f in feature_names]

                # Ensure shape alignment
                if X_train_resampled.shape[1] != len(feature_names):
                    # If mismatch, fallback to using DataFrame of pre-SMOTE X
                    print(
                        f"⚠️ Feature mismatch for {model_name} ({dataset_key}): "
                        f"X has {X_train_resampled.shape[1]} columns, "
                        f"but feature_names has {len(feature_names)}"
                    )
                    # Attempt to use original preprocessed data before SMOTE for SHAP
                    X_pre_smote = processed[dataset_key][f"X_test_{dataset_key}"]
                    X_df = pd.DataFrame(X_pre_smote, columns=feature_names[:X_pre_smote.shape[1]])
                else:
                    # Normal case
                    X_df = pd.DataFrame(X_train_resampled, columns=feature_names)

                print(f"X_train_resampled shape: {X_train_resampled.shape}")
                print(f"Feature names count: {len(feature_names)}")

                # SHAP explanations
                summary_path, force_path = self.explainer.explain_model(
                    model=model,
                    X=X_df,
                    feature_names=feature_names,
                    model_name=model_name,
                    dataset_name=dataset_key
                )

                # Log SHAP plots to MLflow
                mlflow.log_artifact(summary_path)
                mlflow.log_artifact(force_path)

            # Step 4: Log metrics
            for model_name, metrics in results.items():
                mlflow.log_metrics({
                    f"{model_name}_precision": metrics["precision"],
                    f"{model_name}_recall": metrics["recall"],
                    f"{model_name}_f1": metrics["f1_score"],
                    f"{model_name}_roc_auc": metrics["roc_auc"],
                    f"{model_name}_aucpr": metrics["auc_pr"]
                })

            return results

    @flow(name="Fraud Detection Pipeline")
    def orchestrated_pipeline(self):
        return self.run_pipeline_task()
