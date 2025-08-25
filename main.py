# main.py

from src.pipeline.pipeline import FraudDetectionPipeline
from config.config import CONFIG  # Load your config dictionary

def main():
    pipeline = FraudDetectionPipeline(config=CONFIG)
    results = pipeline.orchestrated_pipeline()
    print("✅ Pipeline completed. Results:")
    print(results)

if __name__ == "__main__":
    main()
