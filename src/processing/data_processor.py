import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import numpy as np
from pathlib import Path

from src.data.data_loader import DataLoader
from src.data.data_cleaner import DataCleaner
from src.data.data_transformer import DataTransformer
from src.config.config import CONFIG  # <-- to get artifacts_dir

class Processor:
    def __init__(self):
        self.loader = DataLoader()
        self.cleaner = DataCleaner()
        self.transformer = DataTransformer()
        self.plots = {}  # store generated figures here

    def perform_eda(self, df, df_name):
        """
        Performs basic EDA and stores matplotlib figures in self.plots.
        """
        print(f"\n--- Exploratory Data Analysis (EDA) for {df_name} ---")
        print("\nDescriptive Statistics for numerical features:")
        print(df.describe())

        if 'class' in df.columns or 'Class' in df.columns:
            target_col = 'class' if 'class' in df.columns else 'Class'
            print(f"\nClass Distribution in {df_name}:")
            print(df[target_col].value_counts())
            print(f"Fraudulent transactions: {df[target_col].value_counts()[1]} "
                  f"({df[target_col].value_counts(normalize=True)[1]:.2%})")

            # Class distribution plot
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(x=target_col, data=df, ax=ax)
            ax.set_title(f'Class Distribution in {df_name}')
            ax.set_xlabel(f'{target_col} (0: Non-Fraud, 1: Fraud)')
            ax.set_ylabel('Number of Transactions')
            self.plots[f"{df_name}_class_distribution"] = fig

            # Fraud_Data.csv specific categorical features
            if df_name == 'Fraud_Data.csv':
                categorical_cols = ['source', 'browser', 'sex']
                for col in categorical_cols:
                    if col in df.columns:
                        fraud_rate = df.groupby(col)[target_col].mean().sort_values(ascending=False)
                        print(f"\nFraud rate by {col} in {df_name}:\n{fraud_rate}")
                        fig, ax = plt.subplots(figsize=(8, 5))
                        sns.barplot(x=col, y=target_col, data=df, ax=ax)
                        ax.set_title(f'Fraud Rate by {col} in {df_name}')
                        ax.set_ylabel('Fraud Rate')
                        self.plots[f"{df_name}_fraud_rate_by_{col}"] = fig

                # Numerical distributions
                numerical_cols = ['purchase_value', 'age']
                for col in numerical_cols:
                    if col in df.columns:
                        fig, ax = plt.subplots(figsize=(12, 5))
                        sns.histplot(data=df, x=col, hue=target_col, kde=True, bins=50, ax=ax)
                        ax.set_title(f'{col} Distribution by Class in {df_name}')
                        self.plots[f"{df_name}_{col}_distribution"] = fig

            elif df_name == 'creditcard.csv':
                if 'Amount' in df.columns:
                    fig, ax = plt.subplots(figsize=(12, 5))
                    sns.histplot(data=df, x='Amount', hue=target_col, kde=True, bins=50, ax=ax)
                    ax.set_title(f'Transaction Amount Distribution by Class in {df_name}')
                    self.plots[f"{df_name}_amount_distribution"] = fig

    def run_pipeline(self, config):
        data_loader = DataLoader()
        data_cleaner = DataCleaner()
        data_transformer = DataTransformer()

        # Load data
        loaded_dfs = data_loader.load_data(config["data_paths"])
        fraud_df = loaded_dfs.get("fraud_data")
        ip_to_country = loaded_dfs.get("ip_to_country")
        creditcard_df = loaded_dfs.get("creditcard_data")

        if not all([fraud_df is not None, ip_to_country is not None, creditcard_df is not None]):
            raise ValueError("One or more datasets failed to load.")

        # Perform EDA and store plots
        self.perform_eda(fraud_df, "Fraud_Data.csv")
        self.perform_eda(creditcard_df, "creditcard.csv")

        # Clean data
        fraud_df = data_cleaner.handle_missing_values(
            fraud_df, 'ip_address',
            strategy=config.get("missing_value_strategy", "drop")
        )
        fraud_df = data_cleaner.convert_time_columns(fraud_df, ['signup_time', 'purchase_time'])
        fraud_df = data_cleaner.remove_duplicates(fraud_df, "Fraud_Data.csv")
        ip_to_country = data_cleaner.remove_duplicates(ip_to_country, "IpAddress_to_Country.csv")
        creditcard_df = data_cleaner.remove_duplicates(creditcard_df, "creditcard.csv")

        # Merge geolocation
        fraud_df = data_cleaner.convert_ip_to_int_columns(fraud_df, 'ip_address', 'ip_address_int')
        ip_to_country = data_cleaner.convert_ip_to_int_columns(ip_to_country, 'lower_bound_ip_address', 'lower_bound_ip_address_int')
        ip_to_country = data_cleaner.convert_ip_to_int_columns(ip_to_country, 'upper_bound_ip_address', 'upper_bound_ip_address_int')
        fraud_df = data_transformer.merge_geolocation_data(fraud_df, ip_to_country)

        # Feature engineering
        fraud_df = data_transformer.engineer_fraud_features(fraud_df)

        # Prepare for modeling
        X_fraud, y_fraud, preprocessor_fraud, fraud_base_features = data_transformer.prepare_fraud_data_for_modeling(fraud_df)
        X_cc, y_cc, preprocessor_cc, cc_base_features = data_transformer.prepare_creditcard_data_for_modeling(creditcard_df)

        def preprocess(X, y, preprocessor, base_features, name):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=config.get("test_size", 0.2),
                random_state=config.get("random_state", 42),
                stratify=y
            )

            pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
            X_train_proc = pipeline.fit_transform(X_train)
            X_test_proc = pipeline.transform(X_test)

            X_train_res, y_train_res = data_transformer.handle_class_imbalance(
                X_train_proc, y_train,
                dataset_name=name,
                strategy=config.get("imbalance_strategy", "smote")
            )

            feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
            if isinstance(feature_names, (np.ndarray, pd.Index)):
                feature_names = feature_names.tolist()
            if len(feature_names) == 1 and isinstance(feature_names[0], (list, np.ndarray)):
                feature_names = list(feature_names[0])
            feature_names = [str(f) for f in feature_names]

            return {
                f"X_train_{name}_preprocessed": X_train_proc,
                f"y_train_{name}_preprocessed": y_train,
                f"X_train_{name}_resampled": X_train_res,
                f"y_train_{name}_resampled": y_train_res,
                f"X_test_{name}": X_test_proc,
                f"y_test_{name}": y_test,
                f"feature_names_{name}": feature_names
            }

        fraud_processed = preprocess(X_fraud, y_fraud, preprocessor_fraud, fraud_base_features, "fraud")
        cc_processed = preprocess(X_cc, y_cc, preprocessor_cc, cc_base_features, "creditcard")

        return {
            "fraud": fraud_processed,
            "creditcard": cc_processed,
            "plots": self.plots  # <-- return all EDA plots
        }
