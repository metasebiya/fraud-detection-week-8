from src.pipeline.pipeline import FraudDetectionPipeline

config = {
    "mlflow_uri": "https://dagshub.com/metibizu/froud-detection-mlflow.mlflow",
    "experiment_name": "fraud-detection-mlflow",
    "run_name": "FraudDetection_Run",
    "output_dir": "results",
    "data_paths": {
        "fraud_data": "data/raw/Fraud_Data.csv",
        "ip_to_country": "data/raw/IpAddress_to_Country.csv",
        "creditcard_data": "data/raw/creditcard.csv"
    }
}

pipeline = FraudDetectionPipeline(config)
pipeline.orchestrated_pipeline()
