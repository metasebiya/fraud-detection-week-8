from src.processing.data_processor import Processor
from src.data.data_loader import DataLoader
from src.data.data_cleaner import DataCleaner
from src.data.data_transformer import DataTransformer

config = {
    "data_paths": {
        "fraud_data": "../data/raw/Fraud_Data.csv",
        "ip_to_country": "../data/raw/IpAddress_to_Country.csv",
        "creditcard_data": "../data/raw/creditcard.csv"
    }
}

processed = Processor.run_pipeline(config)

print("\n--- Pipeline Complete ---")
print("Fraud and Credit Card datasets are ready for model training and evaluation.")
