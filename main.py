from src.config.config import CONFIG
from src.pipeline.pipeline import FraudDetectionPipeline

def main():
    pipeline = FraudDetectionPipeline()
    results = pipeline.orchestrated_pipeline()
    print(results)

if __name__ == "__main__":
    main()
